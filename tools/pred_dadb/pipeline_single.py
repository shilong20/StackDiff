"""
【作用概述】
单张单层图像的“原子 -> 四元环/质心 -> Re 链 -> da/db -> 原点”流程实现（库形式）。

核心输入/输出：
- 输入：单张单层图路径（128x128），以及原子检测/四元环提取相关超参数；
        可选直接传入已提取的原子点云 `atoms_points_px`（避免批处理时重复检测）。
- 输出：dict（可 JSON 序列化）：
  - `origin_xy_512` / `da_vec_512` / `db_vec_512`（默认 out_scale=4）
  - 同时包含 128 坐标系下的同名字段（`*_128`）与失败原因 `reason`（若失败）。

【关联说明】文件/模块：
- tools/pred_dadb/atoms.py（原子检测：detect_atoms_from_image）
- tools/pred_dadb/cycles.py（四元环提取与模板吸附平移生长）
- tools/pred_dadb/centroid_chain.py（质心点云 -> Re 链方向 -> da/db 推导）
- tools/pred_dadb/pipeline_bilayer_root.py（批量双层分类入口：调用本模块）

【命令行用法】本文件不直接运行（由 pipeline_bilayer_root.py 调用）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.pred_dadb.atoms import AtomDetectConfig, detect_atoms_from_image  # noqa: E402
from tools.pred_dadb.centroid_chain import solve_re_chain_from_centroids  # noqa: E402
from tools.pred_dadb.cycles import (  # noqa: E402
    _cycle_center,
    _cycle_edges,
    _cycle_score_re4,
    _enumerate_cycles_by_dense_triangles,
    _filter_flip_cycles_by_dominant_long_diag,
    _global_nn_median,
    _infer_sideA_A_from_highT1_path,
    _is_cycle_good_by_color,
    _propagate_cycles_atom_template,
    _unit,
)


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=np.float64).reshape(2)
    v = np.asarray(v, dtype=np.float64).reshape(2)
    nu = float(np.hypot(u[0], u[1]))
    nv = float(np.hypot(v[0], v[1]))
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    c = float(np.dot(u, v) / (nu * nv))
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def _jsonify(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, Path):
        return str(x.as_posix())
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for (k, v) in x.items()}
    if isinstance(x, np.ndarray):
        return _jsonify(x.tolist())
    if isinstance(x, np.generic):
        try:
            return x.item()
        except Exception:
            return float(x)
    return str(x)


def _pick_origin_from_center_cycle(
    cyc_pts_ordered: np.ndarray, *, img_wh: Tuple[int, int]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    在一个有序四元环点序 (p00,p10,p11,p01) 中选原点：
    - 先算 4 个内角，挑钝角顶点（>90°）；若无，则取最大内角顶点；
    - 在钝角顶点集合中选离图像中心最近者。
    """
    pts = np.asarray(cyc_pts_ordered, dtype=np.float64).reshape(4, 2)
    neigh = {0: (1, 3), 1: (0, 2), 2: (1, 3), 3: (0, 2)}
    angs = []
    for i in range(4):
        j, k = neigh[i]
        angs.append(_angle_deg(pts[j] - pts[i], pts[k] - pts[i]))
    angs = np.asarray(angs, dtype=np.float64).reshape(4)

    obt = [int(i) for i in range(4) if float(angs[i]) > 90.0 + 1e-6]
    if not obt:
        obt = [int(np.argmax(angs))]

    w, h = int(img_wh[0]), int(img_wh[1])
    center = np.array([(w - 1) * 0.5, (h - 1) * 0.5], dtype=np.float64)
    d2 = [float(np.sum((pts[i] - center) ** 2)) for i in obt]
    origin_vidx = int(obt[int(np.argmin(d2))])
    origin = pts[origin_vidx].astype(np.float64)

    dbg = {
        "cycle_angles_deg": [float(x) for x in angs],
        "obtuse_vidxs": obt,
        "origin_vidx": int(origin_vidx),
    }
    return origin, dbg


def _align_dadb_to_origin_corner(
    *,
    cyc_pts_ordered: np.ndarray,
    origin_vidx: int,
    chain_dir: np.ndarray,
    da_vec_px: np.ndarray,
    db_vec_px: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    用“原点所在四元环的两条相邻边方向”修正 da/db 的符号，保证位于同一侧：
    - 与 chain_dir 更平行的那条边视为 db 边方向；
    - 另一条边视为 da 边方向；
    - 若 da/db 与各自边方向点积为负，则翻转。
    """
    pts = np.asarray(cyc_pts_ordered, dtype=np.float64).reshape(4, 2)
    neigh = {0: (1, 3), 1: (0, 2), 2: (1, 3), 3: (0, 2)}
    j, k = neigh[int(origin_vidx)]
    e1 = (pts[j] - pts[int(origin_vidx)]).astype(np.float64)
    e2 = (pts[k] - pts[int(origin_vidx)]).astype(np.float64)
    u1 = _unit(e1)
    u2 = _unit(e2)

    ch = _unit(np.asarray(chain_dir, dtype=np.float64).reshape(2))
    if abs(float(np.dot(u1, ch))) >= abs(float(np.dot(u2, ch))):
        u_db_edge, u_da_edge = u1, u2
    else:
        u_db_edge, u_da_edge = u2, u1

    da = np.asarray(da_vec_px, dtype=np.float64).reshape(2)
    db = np.asarray(db_vec_px, dtype=np.float64).reshape(2)
    if float(np.hypot(db[0], db[1])) > 1e-12:
        if float(np.dot(_unit(db), u_db_edge)) < 0.0:
            db = -db
    if float(np.hypot(da[0], da[1])) > 1e-12:
        if float(np.dot(_unit(da), u_da_edge)) < 0.0:
            da = -da

    dbg = {
        "u_db_edge": [float(u_db_edge[0]), float(u_db_edge[1])],
        "u_da_edge": [float(u_da_edge[0]), float(u_da_edge[1])],
        "angle_edge_deg": float(_angle_deg(u_db_edge, u_da_edge)),
        "angle_db_da_deg_after": float(_angle_deg(db, da)),
    }
    return da, db, dbg


def _extract_cycles_one_image(
    *,
    pts: np.ndarray,
    img: Image.Image,
    img_path: Path,
    dist_factor: float,
    strict_color_filter: bool,
    reject_flip: bool,
    flip_diag_tol_deg: float,
    use_theory_dadb: bool,
    da_A: float,
    db_A: float,
    theory_len_tol: float,
    theory_perp_tol: float,
    theory_max_hop: int,
    max_components: int,
    no_intersect: bool,
) -> Tuple[List[Tuple[int, int, int, int]], Dict[str, Any]]:
    """
    单张图的四元环提取（致密三角形枚举 -> seed -> atom_template 生长），不落盘。
    """
    P = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    tree = cKDTree(P)
    nn_med = _global_nn_median(tree, P)
    if (not np.isfinite(nn_med)) or float(nn_med) <= 0:
        return [], {"ok": False, "reason": "bad_nn"}

    dist_limit = float(dist_factor) * float(nn_med)
    cand, dbg_enum = _enumerate_cycles_by_dense_triangles(P, dist_limit=float(dist_limit))
    if not cand:
        return [], {"ok": False, "reason": "no_candidates", "dbg": {"nn_med": float(nn_med), "dist_limit": float(dist_limit), **dbg_enum}}

    scores = []
    for cyc in cand:
        s, sd = _cycle_score_re4(P, cyc)
        scores.append((float(s), tuple(int(x) for x in cyc), sd))
    scores.sort(reverse=True, key=lambda t: t[0])
    score_map = {tuple(int(x) for x in cyc): float(s) for (s, cyc, _sd) in scores}

    def _keep_cyc(cyc: Tuple[int, int, int, int]) -> bool:
        cyc = tuple(int(x) for x in cyc)
        if score_map.get(cyc, 0.0) <= 0.20:
            return False
        if bool(strict_color_filter):
            return _is_cycle_good_by_color(P, cyc)
        return True

    cents = np.stack([_cycle_center(P, c) for c in cand], axis=0).astype(np.float64)
    keep_mask = np.array([_keep_cyc(cyc) for cyc in cand], dtype=bool)
    cents_keep = cents[keep_mask]
    cand_keep = [cand[i] for i in range(len(cand)) if bool(keep_mask[i])]
    if not cand_keep:
        return [], {"ok": False, "reason": "no_candidates_after_filter"}

    flip_dbg = None
    if bool(reject_flip):
        cand_keep, cents_keep2, flip_dbg = _filter_flip_cycles_by_dominant_long_diag(P, cand_keep, centroids=cents_keep, tol_deg=float(flip_diag_tol_deg))
        if cents_keep2 is not None:
            cents_keep = cents_keep2
    if not cand_keep:
        return [], {"ok": False, "reason": "no_candidates_after_flip_filter"}

    seed_score, seed_cyc, seed_sd = None, None, None
    keep_set = set(tuple(int(x) for x in c) for c in cand_keep)
    for s, cyc, sd in scores:
        if tuple(int(x) for x in cyc) not in keep_set:
            continue
        seed_score, seed_cyc, seed_sd = float(s), tuple(int(x) for x in cyc), sd
        break
    if seed_cyc is None:
        return [], {"ok": False, "reason": "no_seed"}

    # 方向来自 seed 的两条边（单位向量）
    vs, _ = _cycle_edges(P, seed_cyc)
    u0 = _unit(vs[0])
    u1 = _unit(vs[1])

    # 理论步长（Å -> px）
    sideA_A = _infer_sideA_A_from_highT1_path(img_path)
    if sideA_A is None or float(sideA_A) <= 1e-6:
        use_theory_dadb = False

    used_global = np.zeros((P.shape[0],), dtype=bool)
    accepted_polys: List[List[np.ndarray]] = []
    picked: List[Tuple[int, int, int, int]] = []
    comps_dbg: List[Dict[str, Any]] = []

    def _choose_steps_atom_template(seed_cycle: Tuple[int, int, int, int], u0t: np.ndarray, u1t: np.ndarray, da_len_px: float, db_len_px: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        # 用“生长出的环数量”仲裁 swap
        sa1 = np.asarray(_unit(u0t) * float(da_len_px), dtype=np.float64)
        sb1 = np.asarray(_unit(u1t) * float(db_len_px), dtype=np.float64)
        sa2 = np.asarray(_unit(u0t) * float(db_len_px), dtype=np.float64)
        sb2 = np.asarray(_unit(u1t) * float(da_len_px), dtype=np.float64)

        used_tmp = np.zeros((P.shape[0],), dtype=bool)
        acc_tmp: List[List[np.ndarray]] = []
        c1, _ = _propagate_cycles_atom_template(
            P,
            seed_cycle=seed_cycle,
            step_a=sa1,
            step_b=sb1,
            nn_med=float(nn_med),
            snap_r_mul=0.35,
            snap_r_min=2.0,
            match_k=8,
            center_merge_tol=2.0,
            template_score_min=0.20,
            strict_color_filter=bool(strict_color_filter),
            no_intersect=False,
            used_global=used_tmp,
            accepted_polys=acc_tmp,
            max_hop=2,
        )
        n1 = int(len(c1))

        used_tmp2 = np.zeros((P.shape[0],), dtype=bool)
        acc_tmp2: List[List[np.ndarray]] = []
        c2, _ = _propagate_cycles_atom_template(
            P,
            seed_cycle=seed_cycle,
            step_a=sa2,
            step_b=sb2,
            nn_med=float(nn_med),
            snap_r_mul=0.35,
            snap_r_min=2.0,
            match_k=8,
            center_merge_tol=2.0,
            template_score_min=0.20,
            strict_color_filter=bool(strict_color_filter),
            no_intersect=False,
            used_global=used_tmp2,
            accepted_polys=acc_tmp2,
            max_hop=2,
        )
        n2 = int(len(c2))
        if n2 > n1:
            return sa2, sb2, {"swap": True, "n_cycles": int(n2), "n_cycles_alt": int(n1)}
        return sa1, sb1, {"swap": False, "n_cycles": int(n1), "n_cycles_alt": int(n2)}

    def _run_one(seed_cycle: Tuple[int, int, int, int], sa: np.ndarray, sb: np.ndarray, meta: Dict[str, Any]) -> None:
        nonlocal picked
        new_cycles, dbg_t = _propagate_cycles_atom_template(
            P,
            seed_cycle=seed_cycle,
            step_a=np.asarray(sa, dtype=np.float64),
            step_b=np.asarray(sb, dtype=np.float64),
            nn_med=float(nn_med),
            snap_r_mul=0.35,
            snap_r_min=2.0,
            match_k=8,
            center_merge_tol=2.0,
            template_score_min=0.20,
            strict_color_filter=bool(strict_color_filter),
            no_intersect=bool(no_intersect),
            used_global=used_global,
            accepted_polys=accepted_polys,
            max_hop=int(theory_max_hop),
        )
        picked.extend([tuple(int(x) for x in c) for c in new_cycles])
        comps_dbg.append({**meta, "picked_in_component": int(len(new_cycles)), "template_dbg": dbg_t})

    if bool(use_theory_dadb) and sideA_A is not None:
        w = float(img.size[0])
        A_to_px = w / float(sideA_A)
        da_len_px = float(da_A) * A_to_px
        db_len_px = float(db_A) * A_to_px
        sa0, sb0, swap_dbg = _choose_steps_atom_template(seed_cyc, u0, u1, da_len_px, db_len_px)
        _run_one(seed_cyc, sa0, sb0, {"seed": list(seed_cyc), "seed_score": float(seed_score), "seed_dbg": seed_sd, "swap_dbg": swap_dbg})
    else:
        # 无理论标尺时，仅返回 seed
        picked = [seed_cyc]
        comps_dbg.append({"seed": list(seed_cyc), "seed_score": float(seed_score), "step_method": "seed_only_fallback"})

    # 补全更多 seed（用于处理缺失/多晶域）
    for _ in range(int(max(1, max_components)) - 1):
        next_seed = None
        for s, cyc, sd in scores:
            cyc = tuple(int(x) for x in cyc)
            if cyc not in keep_set:
                continue
            if any(bool(used_global[int(i)]) for i in cyc):
                continue
            next_seed = (float(s), cyc, sd)
            break
        if next_seed is None:
            break
        ns, ncyc, nsd = next_seed
        nvs, _nls = _cycle_edges(P, ncyc)
        nu0 = _unit(nvs[0])
        nu1 = _unit(nvs[1])
        if bool(use_theory_dadb) and sideA_A is not None:
            w = float(img.size[0])
            A_to_px = w / float(sideA_A)
            da_len_px = float(da_A) * A_to_px
            db_len_px = float(db_A) * A_to_px
            sa, sb, swap_dbg = _choose_steps_atom_template(ncyc, nu0, nu1, da_len_px, db_len_px)
            _run_one(ncyc, sa, sb, {"seed": list(ncyc), "seed_score": float(ns), "seed_dbg": nsd, "swap_dbg": swap_dbg})

    cycles = [tuple(int(x) for x in c) for c in picked]
    dbg = {"ok": True, "nn_med": float(nn_med), "dist_limit": float(dist_limit), "flip_filter": flip_dbg, "n_candidates": int(len(cand_keep)), "n_cycles": int(len(cycles)), "components": comps_dbg[:8], **dbg_enum}
    return cycles, {"ok": True, "dbg": dbg, "seed_cycle": list(seed_cyc)}


def run_pipeline_one_image_dadb(
    img_path: Path,
    *,
    out_scale: float = 4.0,
    atom_cfg: Optional[AtomDetectConfig] = None,
    atoms_points_px: Optional[np.ndarray] = None,
    use_highpass: Optional[bool] = None,
    dist_factor: float = 1.15,
    strict_color_filter: bool = False,
    reject_flip: bool = True,
    flip_diag_tol_deg: float = 25.0,
    use_theory_dadb: bool = True,
    da_A: float = 6.4,
    db_A: float = 6.6,
    theory_len_tol: float = 0.30,
    theory_perp_tol: float = 1.5,
    theory_max_hop: int = 3,
    max_components: int = 3,
    no_intersect: bool = True,
    include_debug: bool = False,
) -> Dict[str, Any]:
    """
    对单张图像运行当前四步流程并返回结果 dict（已做 JSON 序列化兼容处理）。

    返回字段：
    - 成功：ok=True，包含 origin/da/db 的 128 与 512 坐标系（按 out_scale 缩放）；
    - 失败：ok=False，包含 reason。

    说明：
    - 若传入 atoms_points_px，则跳过 `detect_atoms_from_image`，直接使用给定点云。
    """
    img_path = Path(img_path)
    if not img_path.exists():
        return {"ok": False, "reason": "missing_image", "image": str(img_path.as_posix())}

    img = Image.open(str(img_path)).convert("L")
    arr = np.array(img)

    if use_highpass is None:
        use_hp = "data/HighT1" in str(img_path.as_posix())
    else:
        use_hp = bool(use_highpass)

    if atom_cfg is None:
        atom_cfg = AtomDetectConfig(use_highpass=bool(use_hp))
    elif bool(atom_cfg.use_highpass) != bool(use_hp):
        atom_cfg = AtomDetectConfig(**{**atom_cfg.__dict__, "use_highpass": bool(use_hp)})

    if atoms_points_px is not None:
        pts = np.asarray(atoms_points_px, dtype=np.float64).reshape(-1, 2)
        dbg_atoms = {"provided": True, "n_points": int(pts.shape[0])}
    else:
        atoms, dbg_atoms = detect_atoms_from_image(arr, cfg=atom_cfg, return_debug=True)
        pts = np.asarray(atoms, dtype=np.float64).reshape(-1, 2)

    dbg_atoms_small = None
    if isinstance(dbg_atoms, dict):
        dbg_atoms_small = {
            "thr": float(dbg_atoms.get("thr")) if dbg_atoms.get("thr") is not None else None,
            "area_thr": float(dbg_atoms.get("area_thr")) if dbg_atoms.get("area_thr") is not None else None,
            "n_contours": int(dbg_atoms.get("n_contours")) if dbg_atoms.get("n_contours") is not None else None,
            "provided": bool(atoms_points_px is not None),
        }

    out: Dict[str, Any] = {
        "ok": False,
        "image": str(img_path.as_posix()),
        "image_size": [int(img.size[0]), int(img.size[1])],
        "out_scale": float(out_scale),
        "atom_detect": {
            "provided": bool(atoms_points_px is not None),
            "use_highpass": bool(use_hp),
            "bg_sigma_px": float(atom_cfg.bg_sigma_px),
            "bina_thre": float(atom_cfg.bina_thre),
            "min_area_threshold": float(atom_cfg.min_area_threshold),
            "min_distance_px": float(atom_cfg.min_distance_px),
            "max_points": int(atom_cfg.max_points),
        },
        "n_atoms": int(pts.shape[0]),
        "dbg_atoms": dbg_atoms_small,
    }

    if pts.shape[0] < 10:
        out["reason"] = "too_few_atoms"
        if include_debug:
            out["dbg_atoms_full"] = dbg_atoms
        return _jsonify(out)

    cycles, cyc_meta = _extract_cycles_one_image(
        pts=pts,
        img=img,
        img_path=img_path,
        dist_factor=float(dist_factor),
        strict_color_filter=bool(strict_color_filter),
        reject_flip=bool(reject_flip),
        flip_diag_tol_deg=float(flip_diag_tol_deg),
        use_theory_dadb=bool(use_theory_dadb),
        da_A=float(da_A),
        db_A=float(db_A),
        theory_len_tol=float(theory_len_tol),
        theory_perp_tol=float(theory_perp_tol),
        theory_max_hop=int(theory_max_hop),
        max_components=int(max_components),
        no_intersect=bool(no_intersect),
    )
    if not cyc_meta.get("ok", False) or not cycles:
        out["reason"] = cyc_meta.get("reason", "cycles_failed")
        if include_debug:
            out["dbg_cycles"] = cyc_meta
        return _jsonify(out)

    centroids = []
    for cyc in cycles:
        p = np.stack([pts[int(i)] for i in cyc], axis=0)
        centroids.append(np.mean(p, axis=0))
    centroids = np.asarray(centroids, dtype=np.float64).reshape(-1, 2)
    if centroids.shape[0] < 2:
        out["reason"] = "too_few_centroids"
        if include_debug:
            out["dbg_cycles"] = cyc_meta
        return _jsonify(out)

    seed_cycle = tuple(int(x) for x in cyc_meta.get("seed_cycle", cycles[0]))
    seed_pts = np.stack([pts[int(i)] for i in seed_cycle], axis=0)
    e0 = (seed_pts[1] - seed_pts[0]).astype(np.float64)
    e1 = (seed_pts[3] - seed_pts[0]).astype(np.float64)

    chain_dir, dbg_chain = solve_re_chain_from_centroids(centroids, seed_edge_dirs=(e0, e1), k_neighbors=6, max_candidates=6)
    dadb = (dbg_chain.get("dadb_from_centroid_edges") or {}) if isinstance(dbg_chain, dict) else {}
    if not (isinstance(dadb, dict) and isinstance(dadb.get("da_vec_px"), list) and isinstance(dadb.get("db_vec_px"), list)):
        out["reason"] = "dadb_failed"
        if include_debug:
            out["dbg_chain"] = dbg_chain
            out["dbg_cycles"] = cyc_meta
        return _jsonify(out)

    da_vec = np.array([float(dadb["da_vec_px"][0]), float(dadb["da_vec_px"][1])], dtype=np.float64)
    db_vec = np.array([float(dadb["db_vec_px"][0]), float(dadb["db_vec_px"][1])], dtype=np.float64)

    w, h = img.size
    img_center = np.array([(w - 1) * 0.5, (h - 1) * 0.5], dtype=np.float64)
    ci = int(np.argmin(np.sum((centroids - img_center[None, :]) ** 2, axis=1)))
    center_cycle = cycles[ci]
    center_pts = np.stack([pts[int(i)] for i in center_cycle], axis=0)
    origin, dbg_origin = _pick_origin_from_center_cycle(center_pts, img_wh=img.size)
    origin_vidx = int(dbg_origin["origin_vidx"])
    da_vec2, db_vec2, dbg_align = _align_dadb_to_origin_corner(
        cyc_pts_ordered=center_pts,
        origin_vidx=origin_vidx,
        chain_dir=np.asarray(chain_dir, dtype=np.float64).reshape(2),
        da_vec_px=da_vec,
        db_vec_px=db_vec,
    )

    s = float(out_scale)
    out.update(
        {
            "ok": True,
            "n_cycles": int(len(cycles)),
            "seed_cycle": [int(x) for x in seed_cycle],
            "center_cycle": [int(x) for x in center_cycle],
            "origin_xy_128": [float(origin[0]), float(origin[1])],
            "da_vec_128": [float(da_vec2[0]), float(da_vec2[1])],
            "db_vec_128": [float(db_vec2[0]), float(db_vec2[1])],
            "origin_xy_512": [float(origin[0] * s), float(origin[1] * s)],
            "da_vec_512": [float(da_vec2[0] * s), float(da_vec2[1] * s)],
            "db_vec_512": [float(db_vec2[0] * s), float(db_vec2[1] * s)],
            "angle_db_da_deg": float(_angle_deg(db_vec2, da_vec2)),
        }
    )
    if include_debug:
        out["dbg_cycles"] = cyc_meta.get("dbg")
        out["dbg_chain"] = dbg_chain
        out["dbg_origin"] = dbg_origin
        out["dbg_align"] = dbg_align
    return _jsonify(out)


__all__ = ["run_pipeline_one_image_dadb"]

