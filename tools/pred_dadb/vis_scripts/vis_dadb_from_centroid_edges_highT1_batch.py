#!/usr/bin/env python3
"""
【作用概述】对一批 HighT1 单层图（通常是 20 张抽样）可视化基于“质心相邻边 + Re 链方向”估计的 da/db：
- db：方向取 Re 链方向；长度取“与链方向平行的质心相邻边(蓝边)”的边长均值
- da：方向取另一条主轴方向；长度取另一组蓝边的边长均值
并对 da 做符号翻转，使得夹角尽量接近 120°。

核心输入/输出：
- 输入：extract_cycles_highT1_from_atoms.py 的输出目录 `--cycles_dir`
  - 需要包含 `index_cycles.json` 与每张图的 `*_cycles.json`
  - `index_cycles.json` 内应包含 `center_cycle_ordered_points`（优先）或 `seed_cycle_ordered_points`
- 输出：在 `--out_dir` 下为每张 ok 图片写出：
  - `<stem>_dadb_overlay.png`：原图上叠加 da/db 箭头
    - 原点：离图像中心最近的“中心四元环”的钝角顶点
      - 中心四元环：最终四元环集合中“质心最靠近图像中心”的那个（由 extract_cycles_highT1_from_atoms.py 写入）
      - 钝角顶点：该四元环 4 个内角中 > 90° 的点（若极端畸变导致无法分出，则取最大内角点）
    - 橙色：db（Re 链方向）
    - 青色：da
  - `index_dadb_overlay.json`：每张图的 da/db（px 与 Å）、原点坐标等调试信息

【关联说明】文件/模块：
- tools/pred_dadb/centroid_chain.py（solve_re_chain_from_centroids / da/db 估计）
- tools/pred_dadb/extract_cycles_highT1_from_atoms.py（cycles 输出格式；index_cycles.json 写入 center_cycle_ordered_points）

【命令行用法】
python tools/pred_dadb/vis_dadb_from_centroid_edges_highT1_batch.py --cycles_dir tools/pred_dadb/vis_highT1_cycles/<stamp>
（参数：--scale=输出缩放倍数；--arrow_len_mul=箭头长度系数，分别乘以 |db| 与 |da| 的像素长度）
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.pred_dadb.cycles import _infer_sideA_A_from_highT1_path  # noqa: E402
from tools.pred_dadb.centroid_chain import solve_re_chain_from_centroids  # noqa: E402


def _load_cycles_json(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    centroids = []
    for cyc in data:
        pts = np.array([[float(p["x"]), float(p["y"])] for p in cyc], dtype=np.float64).reshape(4, 2)
        centroids.append(np.mean(pts, axis=0))
    return np.asarray(centroids, dtype=np.float64).reshape(-1, 2)


def _draw_arrow(
    dr: ImageDraw.ImageDraw,
    *,
    origin_xy: Tuple[float, float],
    v: np.ndarray,
    length_px: float,
    scale: int,
    color: Tuple[int, int, int, int],
    width: int,
) -> None:
    vv = np.asarray(v, dtype=np.float64).reshape(2)
    n = float(np.hypot(vv[0], vv[1]))
    if n < 1e-12:
        return
    uu = vv / n
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    x2 = ox + float(uu[0]) * float(length_px) * float(scale)
    y2 = oy + float(uu[1]) * float(length_px) * float(scale)
    dr.line((ox, oy, x2, y2), fill=color, width=int(max(1, width)))
    r = max(2, int(round(width * 1.3)))
    dr.ellipse((x2 - r, y2 - r, x2 + r, y2 + r), fill=color)


def _seed_edges_from_index_item(it: Dict[str, Any], *, fallback_centroids: np.ndarray, img_wh: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    v = it.get("seed_cycle_ordered_points")
    seed = None
    if isinstance(v, list) and len(v) == 4:
        try:
            seed = np.asarray([[float(p["x"]), float(p["y"])] for p in v], dtype=np.float64).reshape(4, 2)
        except Exception:
            seed = None
    if seed is None:
        # 兜底：取离中心最近的质心对应的四元环，但这里只需要边方向，直接用质心近似方向会不稳；
        # 因此若缺 seed_ordered_points，直接返回两个固定轴（会使 drop 策略退化，但不致崩）。
        w, h = int(img_wh[0]), int(img_wh[1])
        center = np.array([(w - 1) * 0.5, (h - 1) * 0.5], dtype=np.float64)
        _ = int(np.argmin(np.sum((np.asarray(fallback_centroids) - center[None, :]) ** 2, axis=1)))
        return np.array([1.0, 0.0], dtype=np.float64), np.array([0.0, 1.0], dtype=np.float64)
    e0 = (seed[1] - seed[0]).astype(np.float64)
    e1 = (seed[3] - seed[0]).astype(np.float64)
    return e0, e1


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


def _cycle_points_from_index_item(it: Dict[str, Any], *, key: str) -> Optional[np.ndarray]:
    v = it.get(key)
    if not (isinstance(v, list) and len(v) == 4):
        return None
    try:
        return np.asarray([[float(p["x"]), float(p["y"])] for p in v], dtype=np.float64).reshape(4, 2)
    except Exception:
        return None


def _pick_origin_from_center_cycle(
    it: Dict[str, Any], *, img_wh: Tuple[int, int]
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    从 index_cycles.json 的 item 中选择原点：
    - 优先使用 center_cycle_ordered_points（中心四元环）
    - 若缺失则回退到 seed_cycle_ordered_points
    在该四元环中选取钝角顶点（>90°），且在钝角点中离图像中心最近者。
    返回 (origin_xy 或 None, dbg)。
    """
    pts = _cycle_points_from_index_item(it, key="center_cycle_ordered_points")
    src = "center_cycle_ordered_points"
    if pts is None:
        pts = _cycle_points_from_index_item(it, key="seed_cycle_ordered_points")
        src = "seed_cycle_ordered_points"
    if pts is None:
        return None, {"origin_src": None, "reason": "missing_cycle_points"}

    # ordered parallelogram points: [p00, p10, p11, p01]
    neigh = {0: (1, 3), 1: (0, 2), 2: (1, 3), 3: (0, 2)}
    angs = []
    for i in range(4):
        j, k = neigh[i]
        v1 = pts[j] - pts[i]
        v2 = pts[k] - pts[i]
        angs.append(_angle_deg(v1, v2))
    angs = np.asarray(angs, dtype=np.float64).reshape(4)

    # pick obtuse vertices
    obt = [int(i) for i in range(4) if float(angs[i]) > 90.0 + 1e-6]
    if not obt:
        obt = [int(np.argmax(angs))]

    w, h = int(img_wh[0]), int(img_wh[1])
    center = np.array([(w - 1) * 0.5, (h - 1) * 0.5], dtype=np.float64)
    d2 = [float(np.sum((pts[i] - center) ** 2)) for i in obt]
    best_local = int(obt[int(np.argmin(d2))])
    origin = pts[best_local].astype(np.float64)

    dbg = {
        "origin_src": src,
        "cycle_points_ordered": [[float(p[0]), float(p[1])] for p in pts],
        "cycle_angles_deg": [float(x) for x in angs],
        "obtuse_vidxs": obt,
        "origin_vidx": best_local,
    }
    return origin, dbg


def _align_dadb_to_origin_corner(
    *,
    it: Dict[str, Any],
    origin_xy: np.ndarray,
    img_wh: Tuple[int, int],
    db_vec_px: np.ndarray,
    da_vec_px: np.ndarray,
    chain_dir: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    用“原点所在四元环的两条相邻边方向”来修正 da/db 的符号：
    - db（链方向）对齐到与 chain_dir 更平行的那条边（dot>0）
    - da 对齐到另一条边（dot>0）
    这样可保证 (origin, da, db) 张成的夹角与该顶点的内角同侧。
    """
    pts = _cycle_points_from_index_item(it, key="center_cycle_ordered_points")
    if pts is None:
        pts = _cycle_points_from_index_item(it, key="seed_cycle_ordered_points")
    if pts is None:
        return da_vec_px, db_vec_px, {"aligned": False, "reason": "missing_cycle_points"}

    # find which vertex is the origin in this ordered cycle (exact match with tolerance)
    d2 = np.sum((pts - origin_xy.reshape(1, 2)) ** 2, axis=1)
    vidx = int(np.argmin(d2))
    if float(d2[vidx]) > 1e-6:
        # 原点应该就是四元环的顶点；若不一致，仍尝试用最近顶点做对齐。
        pass

    neigh = {0: (1, 3), 1: (0, 2), 2: (1, 3), 3: (0, 2)}
    j, k = neigh[vidx]
    e1 = (pts[j] - pts[vidx]).astype(np.float64)
    e2 = (pts[k] - pts[vidx]).astype(np.float64)
    ne1 = float(np.hypot(e1[0], e1[1]))
    ne2 = float(np.hypot(e2[0], e2[1]))
    if ne1 < 1e-9 or ne2 < 1e-9:
        return da_vec_px, db_vec_px, {"aligned": False, "reason": "degenerate_edge"}

    u1 = e1 / ne1
    u2 = e2 / ne2
    chain_u = np.asarray(chain_dir, dtype=np.float64).reshape(2)
    nch = float(np.hypot(chain_u[0], chain_u[1]))
    chain_u = chain_u / nch if nch > 1e-12 else np.array([1.0, 0.0], dtype=np.float64)

    # decide which edge is "db-side" (more parallel to chain)
    if abs(float(np.dot(u1, chain_u))) >= abs(float(np.dot(u2, chain_u))):
        u_db_edge, u_da_edge = u1, u2
    else:
        u_db_edge, u_da_edge = u2, u1

    db = np.asarray(db_vec_px, dtype=np.float64).reshape(2)
    da = np.asarray(da_vec_px, dtype=np.float64).reshape(2)
    ndb = float(np.hypot(db[0], db[1]))
    nda = float(np.hypot(da[0], da[1]))
    if ndb > 1e-12:
        db_u = db / ndb
        if float(np.dot(db_u, u_db_edge)) < 0.0:
            db = -db
    if nda > 1e-12:
        da_u = da / nda
        if float(np.dot(da_u, u_da_edge)) < 0.0:
            da = -da

    dbg = {
        "aligned": True,
        "origin_vidx": vidx,
        "u_db_edge": [float(u_db_edge[0]), float(u_db_edge[1])],
        "u_da_edge": [float(u_da_edge[0]), float(u_da_edge[1])],
        "angle_edge_deg": float(_angle_deg(u_db_edge, u_da_edge)),
        "angle_db_da_deg_after": float(_angle_deg(db, da)),
    }
    return da, db, dbg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles_dir", required=True, help="extract_cycles_highT1_from_atoms.py 输出目录（含 index_cycles.json）")
    ap.add_argument("--out_dir", default="", help="输出目录（默认 tools/pred_dadb/vis_highT1_dadb/<stamp>/）")
    ap.add_argument("--scale", type=int, default=4, help="输出缩放倍数")
    ap.add_argument(
        "--arrow_len_mul",
        type=float,
        default=1.0,
        help="箭头长度系数（分别绘制为 arrow_len_mul*|db| 与 arrow_len_mul*|da|；设为 1 表示画真实长度）",
    )
    args = ap.parse_args()

    cycles_dir = Path(str(args.cycles_dir))
    idx_path = cycles_dir / "index_cycles.json"
    if not idx_path.exists():
        raise FileNotFoundError(str(idx_path))
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    items = idx.get("items", [])
    if not items:
        raise ValueError(f"index_cycles.json 里没有 items：{idx_path}")

    if str(args.out_dir).strip():
        out_dir = Path(str(args.out_dir))
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = _REPO_ROOT / "tools" / "pred_dadb" / "vis_highT1_dadb" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    out_index: Dict[str, Any] = {"cycles_dir": str(cycles_dir.as_posix()), "items": []}

    scale = int(max(1, args.scale))
    for it in items:
        stem = str(it.get("stem", ""))
        rel_img = Path(str(it.get("path", "")))
        rec: Dict[str, Any] = {"stem": stem, "path": str(rel_img.as_posix())}

        if not bool(it.get("ok", False)):
            rec["ok"] = False
            rec["reason"] = it.get("reason", "cycles_failed")
            out_index["items"].append(rec)
            continue

        img_path = (_REPO_ROOT / rel_img).resolve()
        cyc_json = cycles_dir / str(it.get("cycles_json", ""))
        if (not img_path.exists()) or (not cyc_json.exists()):
            rec["ok"] = False
            rec["reason"] = "missing_files"
            out_index["items"].append(rec)
            continue

        img = Image.open(str(img_path)).convert("L")
        centroids = _load_cycles_json(cyc_json)
        if centroids.shape[0] < 2:
            rec["ok"] = False
            rec["reason"] = "too_few_centroids"
            out_index["items"].append(rec)
            continue

        origin, origin_dbg = _pick_origin_from_center_cycle(it, img_wh=img.size)
        if origin is None:
            rec["ok"] = False
            rec["reason"] = "no_origin_cycle"
            rec["dbg_origin"] = origin_dbg
            out_index["items"].append(rec)
            continue
        rec["dbg_origin"] = origin_dbg

        e0, e1 = _seed_edges_from_index_item(it, fallback_centroids=centroids, img_wh=img.size)
        chain_v, dbg = solve_re_chain_from_centroids(centroids, seed_edge_dirs=(e0, e1), k_neighbors=6, max_candidates=6)

        dadb = dbg.get("dadb_from_centroid_edges")
        mode = "from_centroid_edges"
        da = None
        db = None
        ang = None

        if isinstance(dadb, dict) and isinstance(dadb.get("da_vec_px"), list) and isinstance(dadb.get("db_vec_px"), list):
            da = np.array([float(dadb["da_vec_px"][0]), float(dadb["da_vec_px"][1])], dtype=np.float64)
            db = np.array([float(dadb["db_vec_px"][0]), float(dadb["db_vec_px"][1])], dtype=np.float64)
            ang = float(dadb.get("angle_db_da_deg", 0.0))
        else:
            # 数据不足时：用理论 da/db 长度（6.4Å/6.6Å）做兜底，仅用于“画出来给你人工看”。
            # 方向仍遵循：db=链方向；da=另一条主轴方向，并选取使夹角接近 120° 的符号。
            sideA_A = _infer_sideA_A_from_highT1_path(img_path)
            main_axes = dbg.get("main_axes_deg") or []
            best_angle = dbg.get("best_angle_deg")
            if sideA_A is not None and float(sideA_A) > 1e-9 and isinstance(main_axes, list) and len(main_axes) >= 2 and isinstance(best_angle, (int, float)):
                A_to_px = float(img.width) / float(sideA_A)
                da_len_px0 = float(6.4 * A_to_px)
                db_len_px0 = float(6.6 * A_to_px)

                def sep(u: float, v: float) -> float:
                    d = abs(float(u) - float(v)) % 180.0
                    return float(min(d, 180.0 - d))

                a0 = float(main_axes[0])
                a1 = float(main_axes[1])
                ang_da_deg = a1 if sep(float(best_angle), a0) <= sep(float(best_angle), a1) else a0

                da_u0 = np.array([float(np.cos(np.radians(ang_da_deg))), float(np.sin(np.radians(ang_da_deg)))], dtype=np.float64)
                db_u = np.asarray(chain_v, dtype=np.float64).reshape(2)
                ndb = float(np.hypot(db_u[0], db_u[1]))
                db_u = db_u / ndb if ndb > 1e-12 else np.array([1.0, 0.0], dtype=np.float64)

                def angle_between(u: np.ndarray, v: np.ndarray) -> float:
                    nu = float(np.hypot(u[0], u[1]))
                    nv = float(np.hypot(v[0], v[1]))
                    if nu < 1e-12 or nv < 1e-12:
                        return 0.0
                    c = float(np.dot(u, v) / (nu * nv))
                    c = max(-1.0, min(1.0, c))
                    return float(np.degrees(np.arccos(c)))

                a0d = angle_between(db_u, da_u0)
                a1d = angle_between(db_u, -da_u0)
                da_u = da_u0 if abs(a0d - 120.0) <= abs(a1d - 120.0) else -da_u0

                da = da_u * float(da_len_px0)
                db = db_u * float(db_len_px0)
                ang = float(angle_between(db, da))
                mode = "theory_fallback"

        if da is None or db is None or ang is None:
            rec["ok"] = False
            rec["reason"] = "no_dadb"
            out_index["items"].append(rec)
            continue

        # 用原点所在四元环的内角方向修正符号，避免 “同一角但在相反侧” 的 180° 歧义。
        da, db, dbg_align = _align_dadb_to_origin_corner(
            it=it,
            origin_xy=origin,
            img_wh=img.size,
            db_vec_px=db,
            da_vec_px=da,
            chain_dir=np.asarray(chain_v, dtype=np.float64).reshape(2),
        )
        rec["dbg_align"] = dbg_align
        ang = float(_angle_deg(db, da))

        len_db = float(np.hypot(db[0], db[1]))
        len_da = float(np.hypot(da[0], da[1]))
        # 画“真实长度”的箭头：_draw_arrow 内部会把 v 归一化，因此这里传入的 length_px 决定可视化长度。
        # 通过 --arrow_len_mul 可以整体放大/缩小，但不会把 da 画成 db 的长度。
        arrow_len_db = float(args.arrow_len_mul) * len_db if np.isfinite(len_db) and len_db > 1e-6 else float(args.arrow_len_mul) * 30.0
        arrow_len_da = float(args.arrow_len_mul) * len_da if np.isfinite(len_da) and len_da > 1e-6 else float(args.arrow_len_mul) * 30.0

        base = img.convert("RGB").resize((img.width * scale, img.height * scale), resample=Image.BILINEAR).convert("RGBA")
        dr = ImageDraw.Draw(base, "RGBA")

        # origin atom marker
        rr = max(2, int(round(scale * 1.2)))
        ox, oy = float(origin[0]) * scale, float(origin[1]) * scale
        dr.ellipse((ox - rr, oy - rr, ox + rr, oy + rr), fill=(255, 80, 80, 235))

        # arrows
        # 注意：origin_xy 已经在缩放坐标系中，因此这里需要把箭头长度也按 scale 放大。
        _draw_arrow(
            dr,
            origin_xy=(ox, oy),
            v=db,
            length_px=arrow_len_db,
            scale=scale,
            color=(255, 180, 60, 240),
            width=max(1, int(round(scale * 0.9))),
        )
        _draw_arrow(
            dr,
            origin_xy=(ox, oy),
            v=da,
            length_px=arrow_len_da,
            scale=scale,
            color=(80, 220, 220, 235),
            width=max(1, int(round(scale * 0.8))),
        )

        # title
        bar_h = int(round(scale * 12))
        dr.rectangle((0, 0, img.width * scale, bar_h), fill=(0, 0, 0, 170))
        dr.text(
            (6, 2),
            f"{stem}  mode={mode}  |db|={len_db:.2f}px  |da|={len_da:.2f}px  ang={float(ang):.1f}deg  arrow_mul={float(args.arrow_len_mul):.2f}",
            fill=(255, 255, 255, 230),
        )

        out_png = out_dir / f"{stem}_dadb_overlay.png"
        base.convert("RGB").save(str(out_png))

        # px -> Å (HighT1)
        sideA_A = _infer_sideA_A_from_highT1_path(img_path)
        if sideA_A is not None and float(sideA_A) > 1e-9:
            px_to_A = float(sideA_A) / float(img.width)
            rec["sideA_A"] = float(sideA_A)
            rec["px_to_A"] = float(px_to_A)
            rec["da_len_A"] = float(len_da * px_to_A)
            rec["db_len_A"] = float(len_db * px_to_A)

        rec.update(
            {
                "ok": True,
                "n_centroids": int(centroids.shape[0]),
                "origin_xy": [float(origin[0]), float(origin[1])],
                "dadb_mode": str(mode),
                "da_vec_px": [float(da[0]), float(da[1])],
                "db_vec_px": [float(db[0]), float(db[1])],
                "angle_db_da_deg": float(ang),
                "out_png": out_png.name,
            }
        )
        out_index["items"].append(rec)

    (out_dir / "index_dadb_overlay.json").write_text(json.dumps(out_index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"out_dir={out_dir}")
    print(f"index={out_dir / 'index_dadb_overlay.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
