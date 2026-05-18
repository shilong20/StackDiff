"""
【作用概述】在 Re 原子点云（N×2）上提取不重叠的 Re4 四元环（4-cycle/平行四边形），并提供基于 seed 的模板吸附平移生长工具。
核心输入/输出：
- 输入：原子点云 `pts`（像素坐标系 x-right/y-down），以及（可选）理论 da/db 长度换算到像素后的平移步长。
- 输出：四元环（4 个点索引或 4 个点坐标）、质心、以及用于后续链方向/da-db 推导的辅助统计。

【关联说明】文件/模块：
- tools/pred_dadb/pipeline_single.py（单图 pipeline：调用本模块提取四元环与质心）
- tools/pred_dadb/pipeline_bilayer_root.py（批量双层分类：间接依赖四元环提取）
- tools/pred_dadb/vis_scripts/*（部分可视化脚本会复用 `_infer_sideA_A_from_highT1_path`）

【命令行用法】本文件不直接运行（仅作为库模块）。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tools.pred_dadb.centroid_chain import _axis_angle_deg

_RE_FOV_NM = re.compile(r"_(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)$")


def _infer_sideA_A_from_highT1_path(img_path: Path) -> Optional[float]:
    """
    从 HighT1 文件所在目录名推断视野尺寸（nm），并换算为 Å。
    例如：..._2.79x2.79 -> sideA_A ≈ 2.79nm * 10 = 27.9Å
    """
    for part in [img_path.parent.name, img_path.parent.parent.name]:
        m = _RE_FOV_NM.search(str(part))
        if not m:
            continue
        a = float(m.group(1))
        b = float(m.group(2))
        nm = 0.5 * (a + b)
        return float(nm * 10.0)
    return None


def _global_nn_median(tree, pts: np.ndarray) -> float:
    """
    全局最近邻距离的中位数（排除自身）。
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2:
        return float("nan")
    d, _ = tree.query(pts, k=2)
    if d.ndim != 2 or d.shape[1] < 2:
        return float("nan")
    return float(np.median(d[:, 1]))


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(2)
    n = float(np.hypot(v[0], v[1]))
    if n < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return (v / n).astype(np.float64)


def _seg_intersect(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> bool:
    """
    严格线段相交（不含端点接触）。假设 disjoint 点集，因此端点接触通常不发生。
    """
    a1 = np.asarray(a1, dtype=np.float64).reshape(2)
    a2 = np.asarray(a2, dtype=np.float64).reshape(2)
    b1 = np.asarray(b1, dtype=np.float64).reshape(2)
    b2 = np.asarray(b2, dtype=np.float64).reshape(2)

    def cross(u: np.ndarray, v: np.ndarray) -> float:
        return float(u[0] * v[1] - u[1] * v[0])

    r = a2 - a1
    s = b2 - b1
    rxs = cross(r, s)
    q_p = b1 - a1
    qpxr = cross(q_p, r)

    if abs(rxs) < 1e-12:
        # 平行（含共线）：按“不相交”处理（避免把近似共线边误判为交叉）
        return False

    t = cross(q_p, s) / rxs
    u = qpxr / rxs
    eps = 1e-9
    return (eps < t < 1.0 - eps) and (eps < u < 1.0 - eps)


def _poly_edges(poly: Sequence[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray]]:
    p = [np.asarray(x, dtype=np.float64).reshape(2) for x in poly]
    return [(p[i], p[(i + 1) % len(p)]) for i in range(len(p))]


def _accept_no_intersection(new_poly: Sequence[np.ndarray], polys: Sequence[Sequence[np.ndarray]]) -> bool:
    ne = _poly_edges(new_poly)
    for old in polys:
        oe = _poly_edges(old)
        for a1, a2 in ne:
            for b1, b2 in oe:
                if _seg_intersect(a1, a2, b1, b2):
                    return False
    return True


def _enumerate_cycles_by_dense_triangles(
    pts: np.ndarray,
    *,
    dist_limit: float,
) -> Tuple[List[Tuple[int, int, int, int]], Dict]:
    """
    在“短键图”中枚举菱形四元环：
    - 对每条短键边 (i,j)，找共同邻居 CN=N(i)∩N(j)；
    - 对 CN 中任意两点 {k,l}，构造四元环 (k,i,l,j)（外周顺序）。
    返回去重后的候选 cycles 列表（未做 disjoint 筛选）。
    """
    from scipy.spatial import cKDTree

    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=float(dist_limit))
    adj: List[set[int]] = [set() for _ in range(int(pts.shape[0]))]
    for i, j in pairs:
        i = int(i)
        j = int(j)
        adj[i].add(j)
        adj[j].add(i)

    cycles: List[Tuple[int, int, int, int]] = []
    seen = set()
    for i, j in pairs:
        i = int(i)
        j = int(j)
        cn = adj[i].intersection(adj[j])
        if len(cn) < 2:
            continue
        cn_list = sorted(int(x) for x in cn)
        for a in range(len(cn_list)):
            for b in range(a + 1, len(cn_list)):
                k = cn_list[a]
                l = cn_list[b]
                cyc = (int(k), int(i), int(l), int(j))
                key = tuple(sorted(cyc))
                if key in seen:
                    continue
                seen.add(key)
                cycles.append(cyc)

    dbg = {"n_pairs": int(len(pairs)), "n_cycles_raw": int(len(cycles)), "dist_limit": float(dist_limit)}
    return cycles, dbg


def _cycle_edges(pts: np.ndarray, cyc: Tuple[int, int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    cyc 顺序约定：(p0,p1,p2,p3) 沿外周。
    返回：
      - edges: (4,2) 各边向量 p{i+1}-p{i}
      - lens: (4,) 各边长度
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    p0, p1, p2, p3 = [int(x) for x in cyc]
    vs = np.stack([pts[p1] - pts[p0], pts[p2] - pts[p1], pts[p3] - pts[p2], pts[p0] - pts[p3]], axis=0).astype(np.float64)
    ls = np.hypot(vs[:, 0], vs[:, 1]).astype(np.float64)
    return vs, ls


def _cycle_center(pts: np.ndarray, cyc: Tuple[int, int, int, int]) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    idxs = np.asarray([int(x) for x in cyc], dtype=np.int32)
    return np.mean(pts[idxs], axis=0).astype(np.float64)


def _cycle_long_diag_unit(pts: np.ndarray, cyc: Tuple[int, int, int, int]) -> Tuple[np.ndarray, float, float]:
    """
    返回四元环“长对角线”的单位方向，以及两条对角线长度。
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    p0, p1, p2, p3 = [int(x) for x in cyc]
    d0 = (pts[p2] - pts[p0]).astype(np.float64)
    d1 = (pts[p3] - pts[p1]).astype(np.float64)
    l0 = float(np.hypot(d0[0], d0[1]))
    l1 = float(np.hypot(d1[0], d1[1]))
    if l1 > l0:
        d0, d1 = d1, d0
        l0, l1 = l1, l0
    return _unit(d0), float(l0), float(l1)


def _filter_flip_cycles_by_dominant_long_diag(
    pts: np.ndarray,
    cycles: Sequence[Tuple[int, int, int, int]],
    centroids: Optional[np.ndarray] = None,
    *,
    tol_deg: float = 25.0,
    min_cluster_size: int = 4,
) -> Tuple[List[Tuple[int, int, int, int]], Optional[np.ndarray], Dict]:
    """
    剔除“flip 四元环”：长对角线方向落在与主簇近似正交的一簇。
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    cyc_list = [tuple(int(x) for x in c) for c in cycles]
    if len(cyc_list) < int(max(1, min_cluster_size)):
        return (
            list(cyc_list),
            None if centroids is None else np.asarray(centroids, dtype=np.float64).reshape(-1, 2),
            {"enabled": True, "skipped": True, "reason": "too_few_cycles"},
        )

    diag_units: List[np.ndarray] = []
    for cyc in cyc_list:
        u, _l0, _l1 = _cycle_long_diag_unit(pts, cyc)
        diag_units.append(u)

    tol = float(max(1.0, tol_deg))

    best_i = 0
    best_cnt = -1
    for i, ui in enumerate(diag_units):
        cnt = 0
        for uj in diag_units:
            if float(_axis_angle_deg(ui, uj)) <= tol:
                cnt += 1
        if cnt > best_cnt:
            best_cnt = int(cnt)
            best_i = int(i)

    if best_cnt < int(max(2, min_cluster_size // 2)):
        return (
            list(cyc_list),
            None if centroids is None else np.asarray(centroids, dtype=np.float64).reshape(-1, 2),
            {"enabled": True, "skipped": True, "reason": "no_dominant_cluster", "best_cnt": int(best_cnt)},
        )

    ref = diag_units[best_i]
    acc = np.zeros((2,), dtype=np.float64)
    inliers = 0
    for u in diag_units:
        if float(_axis_angle_deg(ref, u)) <= tol:
            uu = u if float(np.dot(u, ref)) >= 0.0 else -u
            acc += uu
            inliers += 1
    if float(np.hypot(acc[0], acc[1])) > 1e-9:
        ref = _unit(acc)

    keep_idx: List[int] = []
    angles: List[float] = []
    for i, u in enumerate(diag_units):
        a = float(_axis_angle_deg(u, ref))
        angles.append(a)
        if a <= tol:
            keep_idx.append(int(i))

    filt_cycles = [cyc_list[i] for i in keep_idx]
    filt_centroids = None
    if centroids is not None:
        C = np.asarray(centroids, dtype=np.float64).reshape(-1, 2)
        if C.shape[0] == len(cyc_list):
            filt_centroids = C[np.asarray(keep_idx, dtype=np.int32)]

    dbg = {
        "enabled": True,
        "skipped": False,
        "tol_deg": float(tol),
        "n_in": int(len(cyc_list)),
        "n_keep": int(len(filt_cycles)),
        "n_reject": int(len(cyc_list) - len(filt_cycles)),
        "dominant_cnt": int(best_cnt),
        "ref_long_diag": [float(ref[0]), float(ref[1])],
        "angles_deg_stats": {
            "min": float(np.min(angles)) if angles else None,
            "p50": float(np.median(angles)) if angles else None,
            "max": float(np.max(angles)) if angles else None,
        },
    }
    return filt_cycles, filt_centroids, dbg


def _cycle_score_re4(pts: np.ndarray, cyc: Tuple[int, int, int, int]) -> Tuple[float, Dict]:
    """
    Re4 菱形几何评分（越大越好）。
    """
    vs, ls = _cycle_edges(pts, cyc)
    m = float(np.mean(ls))
    if m < 1e-6:
        return -1e9, {"reason": "degenerate"}
    cv = float(np.std(ls) / m)

    par0 = float(_axis_angle_deg(vs[0], -vs[2]))
    par1 = float(_axis_angle_deg(vs[1], -vs[3]))
    acute = float(_axis_angle_deg(vs[0], vs[1]))

    u0 = _unit(vs[0])
    angs = [float(_axis_angle_deg(u0, vs[i])) for i in range(4)]
    i1 = int(np.argmax(angs))
    u1 = _unit(vs[i1])
    sep = float(_axis_angle_deg(u0, u1))
    if sep < 25.0:
        return -1e9, {"reason": "one_direction", "sep_deg": float(sep)}

    grp = []
    for i in range(4):
        a0 = float(_axis_angle_deg(vs[i], u0))
        a1 = float(_axis_angle_deg(vs[i], u1))
        grp.append(0 if a0 <= a1 else 1)
    n0 = int(sum(1 for g in grp if g == 0))
    n1 = 4 - n0
    if n0 == 0 or n1 == 0:
        return -1e9, {"reason": "one_color", "n0": n0, "n1": n1, "sep_deg": float(sep)}
    opp_ok = int(grp[0] == grp[2]) + int(grp[1] == grp[3])

    pen = 0.0
    pen += (cv / 0.12) ** 2
    pen += (par0 / 8.0) ** 2 + (par1 / 8.0) ** 2
    pen += ((acute - 60.0) / 15.0) ** 2
    pen += ((sep - 60.0) / 20.0) ** 2
    pen += ((abs(n0 - 2)) / 1.0) ** 2
    pen += ((2 - opp_ok) / 1.0) ** 2
    score = float(1.0 / (1.0 + pen))

    dbg = {
        "score": float(score),
        "cv": float(cv),
        "par_deg": [float(par0), float(par1)],
        "acute_deg": float(acute),
        "sep_deg": float(sep),
        "grp": grp,
        "n0": n0,
        "n1": n1,
        "opp_ok": int(opp_ok),
    }
    return score, dbg


def _is_cycle_good_by_color(
    pts: np.ndarray,
    cyc: Tuple[int, int, int, int],
    *,
    par_max_deg: float = 6.0,
    acute_range_deg: Tuple[float, float] = (45.0, 75.0),
    sep_min_deg: float = 35.0,
) -> bool:
    s, sd = _cycle_score_re4(pts, cyc)
    if not np.isfinite(s) or s <= 0:
        return False
    par0, par1 = sd.get("par_deg", [999.0, 999.0])
    if float(par0) > float(par_max_deg) or float(par1) > float(par_max_deg):
        return False
    acute = float(sd.get("acute_deg", 0.0))
    if not (float(acute_range_deg[0]) <= acute <= float(acute_range_deg[1])):
        return False
    sep = float(sd.get("sep_deg", 0.0))
    if sep < float(sep_min_deg):
        return False
    if int(sd.get("n0", 0)) != 2 or int(sd.get("n1", 0)) != 2:
        return False
    if int(sd.get("opp_ok", 0)) != 2:
        return False
    return True


def _select_disjoint_cycles_scored(
    pts: np.ndarray,
    candidates: Sequence[Tuple[int, int, int, int]],
    *,
    score_map: Dict[Tuple[int, int, int, int], float],
    used_global: np.ndarray,
    accepted_polys: List[List[np.ndarray]],
    no_intersect: bool,
) -> Tuple[List[Tuple[int, int, int, int]], Dict]:
    """
    在已有 used/accepted_polys 基础上，按评分优先做 disjoint 选择。
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    used_global = np.asarray(used_global, dtype=bool).reshape(-1)

    center = np.array([(pts[:, 0].min() + pts[:, 0].max()) * 0.5, (pts[:, 1].min() + pts[:, 1].max()) * 0.5], dtype=np.float64)

    scored = []
    for cyc in candidates:
        idxs = tuple(int(x) for x in cyc)
        s = float(score_map.get(idxs, score_map.get(tuple(sorted(idxs)), 0.0)))
        poly = pts[np.asarray(idxs, dtype=np.int32)]
        c = np.mean(poly, axis=0)
        d2 = float(np.sum((c - center) ** 2))
        # 菱形质量只用作 tie-break
        vs, ls = _cycle_edges(pts, idxs)
        m = float(np.mean(ls))
        q = float(np.std(ls) / m) if m > 1e-9 else 1e9
        scored.append((-s, d2, q, idxs))
    scored.sort()

    out: List[Tuple[int, int, int, int]] = []
    n_reject_used = 0
    n_reject_inter = 0
    for _ns, _d2, _q, cyc in scored:
        if any(bool(used_global[int(i)]) for i in cyc):
            n_reject_used += 1
            continue
        poly = [pts[int(cyc[0])], pts[int(cyc[1])], pts[int(cyc[2])], pts[int(cyc[3])]]
        if bool(no_intersect):
            if not _accept_no_intersection(poly, accepted_polys):
                n_reject_inter += 1
                continue
            accepted_polys.append([x.copy() for x in poly])
        out.append(tuple(int(x) for x in cyc))
        for i in cyc:
            used_global[int(i)] = True

    dbg = {"n_candidates": int(len(candidates)), "n_cycles": int(len(out)), "reject_used": int(n_reject_used), "reject_intersections": int(n_reject_inter)}
    return out, dbg


def _template_match_4pts_unique(
    tree,
    pts: np.ndarray,
    pred_pts: np.ndarray,
    *,
    snap_r: float,
    match_k: int,
    used_global: np.ndarray,
) -> Optional[List[int]]:
    """
    在全原子点云上做“模板吸附”匹配：给定 4 个预测点 pred_pts（4×2），
    在半径 snap_r 内找到 4 个互不相同且未被占用的真实原子索引。
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    pred_pts = np.asarray(pred_pts, dtype=np.float64).reshape(4, 2)
    used_global = np.asarray(used_global, dtype=bool).reshape(-1)

    k = int(max(1, match_k))
    r = float(max(1e-6, snap_r))

    cand_lists: List[List[Tuple[int, float]]] = []
    for p in pred_pts:
        d, idx = tree.query(p, k=k, distance_upper_bound=r)
        d = np.asarray(d, dtype=np.float64).reshape(-1)
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        one: List[Tuple[int, float]] = []
        for jj, dd in zip(idx.tolist(), d.tolist()):
            j = int(jj)
            if j < 0 or j >= pts.shape[0]:
                continue
            if not np.isfinite(dd):
                continue
            if float(dd) > r:
                continue
            if bool(used_global[j]):
                continue
            one.append((j, float(dd)))
        one.sort(key=lambda t: t[1])
        if not one:
            return None
        cand_lists.append(one)

    order = sorted(range(4), key=lambda i: len(cand_lists[i]))
    cand_lists_ord = [cand_lists[i] for i in order]

    best: Optional[List[int]] = None
    best_cost = 1e30
    chosen: List[int] = []
    chosen_set = set()

    def dfs(i: int, cost: float) -> None:
        nonlocal best, best_cost
        if cost >= best_cost:
            return
        if i >= 4:
            best_cost = float(cost)
            best = list(chosen)
            return
        for j, dd in cand_lists_ord[i]:
            if j in chosen_set:
                continue
            chosen.append(int(j))
            chosen_set.add(int(j))
            dfs(i + 1, cost + float(dd))
            chosen_set.remove(int(j))
            chosen.pop()

    dfs(0, 0.0)
    if best is None:
        return None

    out = [0, 0, 0, 0]
    for pos, orig_i in enumerate(order):
        out[orig_i] = int(best[pos])
    return out


def _propagate_cycles_atom_template(
    pts: np.ndarray,
    *,
    seed_cycle: Tuple[int, int, int, int],
    step_a: np.ndarray,
    step_b: np.ndarray,
    nn_med: float,
    snap_r_mul: float,
    snap_r_min: float,
    match_k: int,
    center_merge_tol: float,
    template_score_min: float,
    strict_color_filter: bool,
    no_intersect: bool,
    used_global: np.ndarray,
    accepted_polys: List[List[np.ndarray]],
    max_hop: int = 3,
) -> Tuple[List[Tuple[int, int, int, int]], Dict]:
    """
    在“全原子点云”上用 seed 模板做配准 + 平移生长，返回新增 cycles（点索引四元组）。
    """
    from scipy.spatial import cKDTree

    P = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    seed_cycle = tuple(int(x) for x in seed_cycle)
    tree = cKDTree(P)
    used_global = np.asarray(used_global, dtype=bool).reshape(-1)

    seed_pts = P[np.asarray(seed_cycle, dtype=np.int32)]
    c_seed = np.mean(seed_pts, axis=0).astype(np.float64)
    offsets = (seed_pts - c_seed[None, :]).astype(np.float64)

    snap_r = float(max(float(snap_r_min), float(snap_r_mul) * float(nn_med)))
    cm_tol = float(max(0.5, center_merge_tol))
    max_hop = int(max(1, max_hop))

    a = np.asarray(step_a, dtype=np.float64).reshape(2)
    b = np.asarray(step_b, dtype=np.float64).reshape(2)
    ang = float(_axis_angle_deg(a, b))
    c = (a + b).astype(np.float64) if ang < 90.0 else (b - a).astype(np.float64)

    base_dirs = [a, -a, b, -b, c, -c]
    dirs: List[np.ndarray] = []
    for dv in base_dirs:
        for n in range(1, max_hop + 1):
            dirs.append((float(n) * dv).astype(np.float64))

    visited_centers: List[np.ndarray] = []

    def _seen_center(c0: np.ndarray) -> bool:
        c0 = np.asarray(c0, dtype=np.float64).reshape(2)
        for cc in visited_centers:
            if float(np.hypot(*(cc - c0))) <= cm_tol:
                return True
        return False

    out_cycles: List[Tuple[int, int, int, int]] = []
    if any(bool(used_global[int(i)]) for i in seed_cycle):
        return [], {"ok": False, "reason": "seed_points_used"}

    seed_poly = [P[int(seed_cycle[0])], P[int(seed_cycle[1])], P[int(seed_cycle[2])], P[int(seed_cycle[3])]]
    if bool(no_intersect):
        if not _accept_no_intersection(seed_poly, accepted_polys):
            return [], {"ok": False, "reason": "seed_intersect"}
        accepted_polys.append([x.copy() for x in seed_poly])
    for i in seed_cycle:
        used_global[int(i)] = True
    out_cycles.append(seed_cycle)
    visited_centers.append(np.asarray(c_seed, dtype=np.float64).reshape(2))

    q: List[np.ndarray] = [np.asarray(c_seed, dtype=np.float64).reshape(2)]
    n_try = 0
    n_match_ok = 0
    n_reject_score = 0
    n_reject_used = 0
    n_reject_inter = 0

    while q:
        c_curr = np.asarray(q.pop(0), dtype=np.float64).reshape(2)
        for dv in dirs:
            c_pred = (c_curr + dv).astype(np.float64)
            if _seen_center(c_pred):
                continue

            n_try += 1
            pred_pts = (c_pred[None, :] + offsets).astype(np.float64)
            match = _template_match_4pts_unique(tree, P, pred_pts, snap_r=snap_r, match_k=int(match_k), used_global=used_global)
            if match is None:
                continue

            cyc = tuple(int(x) for x in match)
            if any(bool(used_global[int(i)]) for i in cyc):
                n_reject_used += 1
                continue

            s, _sd = _cycle_score_re4(P, cyc)
            if (not np.isfinite(s)) or float(s) < float(template_score_min):
                n_reject_score += 1
                continue
            if bool(strict_color_filter) and (not _is_cycle_good_by_color(P, cyc)):
                n_reject_score += 1
                continue

            poly = [P[int(cyc[0])], P[int(cyc[1])], P[int(cyc[2])], P[int(cyc[3])]]
            if bool(no_intersect):
                if not _accept_no_intersection(poly, accepted_polys):
                    n_reject_inter += 1
                    continue
                accepted_polys.append([x.copy() for x in poly])

            for i in cyc:
                used_global[int(i)] = True
            out_cycles.append(cyc)
            c_real = np.mean(P[np.asarray(cyc, dtype=np.int32)], axis=0).astype(np.float64)
            visited_centers.append(np.asarray(c_real, dtype=np.float64).reshape(2))
            q.append(np.asarray(c_real, dtype=np.float64).reshape(2))
            n_match_ok += 1

    dbg = {
        "ok": True,
        "snap_r": float(snap_r),
        "center_merge_tol": float(cm_tol),
        "angle_ab_deg": float(ang),
        "n_dirs": int(len(dirs)),
        "n_try": int(n_try),
        "n_match_ok": int(n_match_ok),
        "reject_score": int(n_reject_score),
        "reject_used": int(n_reject_used),
        "reject_intersections": int(n_reject_inter),
        "n_cycles": int(len(out_cycles)),
    }
    return out_cycles, dbg


__all__ = [
    "_axis_angle_deg",
    "_infer_sideA_A_from_highT1_path",
    "_global_nn_median",
    "_unit",
    "_enumerate_cycles_by_dense_triangles",
    "_cycle_edges",
    "_cycle_center",
    "_cycle_score_re4",
    "_is_cycle_good_by_color",
    "_filter_flip_cycles_by_dominant_long_diag",
    "_select_disjoint_cycles_scored",
    "_propagate_cycles_atom_template",
]

