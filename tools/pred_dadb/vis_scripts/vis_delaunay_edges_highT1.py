#!/usr/bin/env python3
"""
【作用概述】在 HighT1 单层图上可视化“质心 Delaunay 三角网格边（灰色）”并叠加当前链方向判定所用的“短近邻边”（蓝色）：
1) 从四元环 `*_cycles.json` 计算质心点云；
2) 生成两条候选链方向（main_axes_deg）；
3) 用 neighbor_min_edge 判据把“最短一档近邻边”按方向分类（得到两组蓝边）；
4) 比较两组蓝边的边长均值（mean），均值更小的方向视作 Re 链方向；
5) 可视化：灰色画 Delaunay 网格边用于对照；蓝色画 neighbor_min_edge 的短边分类结果。

核心输入/输出：
- 输入：
  - `--cycles_dir`：`tools/pred_dadb/extract_cycles_highT1_from_atoms.py` 输出目录（含 `index_cycles.json` 与 `*_cycles.json`）
  - `--stem`：要查看的样本 stem（与 index_cycles.json 对应）
- 输出：
  - `<stem>_delaunay_edges.png`：两联图（左=候选 a0；右=候选 a1）
    - 灰色：Delaunay 全部边
    - 蓝色：该候选方向在每个三角形中被选中的边（按被选频次加粗）
    - 橙色箭头：最终选择的 Re 链方向（由均值最小判据得到）
  - 控制台打印：a0/a1 的 mean_len、最终 best_angle、以及被选边数量等

【关联说明】文件/模块：
- tools/pred_dadb/centroid_chain.py（`solve_re_chain_from_centroids` / neighbor_min_edge 逻辑）
- tools/pred_dadb/extract_cycles_highT1_from_atoms.py（四元环质心来源）

【命令行用法】
python tools/pred_dadb/vis_delaunay_edges_highT1.py --cycles_dir tools/pred_dadb/vis_highT1_cycles/<stamp> --stem "<stem>"
（参数：--angle_tol_deg=选边角度容差；--scale=放大倍数）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.pred_dadb.centroid_chain import solve_re_chain_from_centroids  # noqa: E402


def _load_cycles(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    centroids = []
    cycles_pts = []
    for cyc in data:
        pts = np.array([[float(p["x"]), float(p["y"])] for p in cyc], dtype=np.float64).reshape(4, 2)
        cycles_pts.append(pts)
        centroids.append(np.mean(pts, axis=0))
    C = np.asarray(centroids, dtype=np.float64).reshape(-1, 2)
    P = np.asarray(cycles_pts, dtype=np.float64).reshape(-1, 4, 2)
    return C, P


def _parse_edge_map(m: Dict) -> Dict[Tuple[int, int], int]:
    out: Dict[Tuple[int, int], int] = {}
    for k, v in (m or {}).items():
        try:
            a, b = str(k).split("-", 1)
            out[(int(a), int(b))] = int(v)
        except Exception:
            continue
    return out


def _draw_arrow(
    dr: ImageDraw.ImageDraw,
    *,
    origin_xy: Tuple[float, float],
    v: np.ndarray,
    length_px: float,
    color: Tuple[int, int, int, int],
    width: int,
) -> None:
    vv = np.asarray(v, dtype=np.float64).reshape(2)
    n = float(np.hypot(vv[0], vv[1]))
    if n < 1e-9:
        return
    vv = vv / n
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    x2 = ox + float(vv[0]) * float(length_px)
    y2 = oy + float(vv[1]) * float(length_px)
    dr.line((ox, oy, x2, y2), fill=color, width=int(max(1, width)))
    r = max(2, int(round(width * 1.2)))
    dr.ellipse((x2 - r, y2 - r, x2 + r, y2 + r), fill=color)


def _render_panel(
    img: Image.Image,
    centroids: np.ndarray,
    tri_edges: List[Tuple[int, int]],
    picked_edges: Dict[Tuple[int, int], int],
    *,
    title: str,
    chain_dir: np.ndarray,
    scale: int,
) -> Image.Image:
    img = img.convert("L")
    base = img.convert("RGB").resize((img.width * scale, img.height * scale), resample=Image.BILINEAR).convert("RGBA")
    dr = ImageDraw.Draw(base, "RGBA")

    C = np.asarray(centroids, dtype=np.float64).reshape(-1, 2)

    # draw triangulation edges
    for a, b in tri_edges:
        p0 = (float(C[a, 0]) * scale, float(C[a, 1]) * scale)
        p1 = (float(C[b, 0]) * scale, float(C[b, 1]) * scale)
        dr.line((p0, p1), fill=(180, 180, 180, 80), width=max(1, int(round(scale * 0.15))))

    # draw picked edges (thicker by frequency)
    for (a, b), cnt in picked_edges.items():
        p0 = (float(C[a, 0]) * scale, float(C[a, 1]) * scale)
        p1 = (float(C[b, 0]) * scale, float(C[b, 1]) * scale)
        w = max(1, int(round(scale * (0.25 + 0.18 * min(cnt, 6)))))
        dr.line((p0, p1), fill=(80, 170, 255, 220), width=w)

    # draw centroids
    r = max(2, int(round(scale * 0.9)))
    for x, y in C:
        dr.ellipse((float(x) * scale - r, float(y) * scale - r, float(x) * scale + r, float(y) * scale + r), fill=(255, 235, 80, 200))

    # draw final chain direction arrow at centroid mean
    cc = np.mean(C, axis=0) if C.size else np.array([(img.width - 1) * 0.5, (img.height - 1) * 0.5], dtype=np.float64)
    origin = (float(cc[0]) * scale, float(cc[1]) * scale)
    _draw_arrow(
        dr,
        origin_xy=origin,
        v=np.asarray(chain_dir, dtype=np.float64).reshape(2),
        length_px=40.0 * scale,
        color=(255, 180, 60, 240),
        width=max(1, int(round(scale * 0.8))),
    )

    # title bar
    dr.rectangle((0, 0, img.width * scale, int(round(scale * 10))), fill=(0, 0, 0, 160))
    dr.text((6, 2), title, fill=(255, 255, 255, 220))
    return base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles_dir", required=True, help="extract_cycles_highT1_from_atoms.py 输出目录（含 index_cycles.json）")
    ap.add_argument("--stem", required=True, help="要可视化的 stem（与 index_cycles.json 对应）")
    ap.add_argument("--out", default="", help="输出路径（默认写到 cycles_dir 下）")
    ap.add_argument("--angle_tol_deg", type=float, default=18.0, help="选边角度容差（度）")
    ap.add_argument("--max_edge_factor", type=float, default=2.0, help="三角形边长上限：max_edge_factor * nn_med（用于剔除边缘超长边）")
    ap.add_argument("--scale", type=int, default=4, help="输出缩放倍数")
    args = ap.parse_args()

    cycles_dir = Path(str(args.cycles_dir))
    idx_path = cycles_dir / "index_cycles.json"
    if not idx_path.exists():
        raise FileNotFoundError(str(idx_path))
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    items = idx.get("items", [])
    stem = str(args.stem)
    it = next((x for x in items if str(x.get("stem", "")) == stem), None)
    if it is None:
        raise ValueError(f"stem 不在 index_cycles.json 中：{stem}")

    img_path = (_REPO_ROOT / Path(str(it["path"]))).resolve()
    cyc_json = cycles_dir / str(it["cycles_json"])
    if not img_path.exists() or not cyc_json.exists():
        raise FileNotFoundError("missing image/cycles json")

    img = Image.open(str(img_path)).convert("L")
    centroids, cycles_pts = _load_cycles(cyc_json)
    if centroids.shape[0] < 4:
        raise RuntimeError("too_few_centroids")

    # seed edges：选质心最靠近图像中心的四元环，用其两条边方向做 cand3 剔除（与批处理保持一致）
    img_center = np.array([(img.width - 1) * 0.5, (img.height - 1) * 0.5], dtype=np.float64)
    seed_i = int(np.argmin(np.sum((centroids - img_center[None, :]) ** 2, axis=1)))
    seed = cycles_pts[int(seed_i)]
    e0 = (seed[1] - seed[0]).astype(np.float64)  # p10-p00
    e1 = (seed[3] - seed[0]).astype(np.float64)  # p01-p00

    chain_v, dbg = solve_re_chain_from_centroids(centroids, seed_edge_dirs=(e0, e1), k_neighbors=6, max_candidates=6)
    main_axes = dbg.get("main_axes_deg") or []
    if len(main_axes) < 2:
        raise RuntimeError("no_main_axes")
    a0 = float(main_axes[0])
    a1 = float(main_axes[1])

    # 当前 solve_re_chain_from_centroids 已精简为 neighbor_min_edge；这里复用其“被选边”调试信息。
    ndbg = dbg.get("neighbor_debug") or {}
    edge_src = ndbg if isinstance(ndbg, dict) and ("picked_edges0" in ndbg or "picked_edges1" in ndbg) else (ndbg.get("dbg") or {})
    e0 = _parse_edge_map(edge_src.get("picked_edges0"))
    e1 = _parse_edge_map(edge_src.get("picked_edges1"))

    # tri edges list
    tri = Delaunay(np.asarray(centroids, dtype=np.float64))
    simplices = np.asarray(tri.simplices, dtype=np.int64).reshape(-1, 3)

    # 与核心算法一致：用最近邻距离中位数估计最大允许边长，过滤掉包含超长边的三角形。
    nn_med = float("nan")
    try:
        tree = cKDTree(np.asarray(centroids, dtype=np.float64))
        d, _ = tree.query(np.asarray(centroids, dtype=np.float64), k=min(2, int(centroids.shape[0])))
        if d.ndim == 2 and d.shape[1] >= 2:
            nn_med = float(np.median(d[:, 1]))
    except Exception:
        nn_med = float("nan")
    max_edge_len = float(args.max_edge_factor) * float(nn_med) if np.isfinite(nn_med) and nn_med > 0 else float("inf")

    edge_set = set()
    for i, j, k in simplices:
        ii, jj, kk = int(i), int(j), int(k)
        edges = [(ii, jj), (jj, kk), (kk, ii)]
        ls = []
        for a, b in edges:
            v = np.asarray(centroids[b], dtype=np.float64) - np.asarray(centroids[a], dtype=np.float64)
            ls.append(float(np.hypot(v[0], v[1])))
        if ls and float(np.max(ls)) > float(max_edge_len):
            continue
        for a, b in edges:
            key = (min(a, b), max(a, b))
            edge_set.add(key)
    tri_edges = sorted(edge_set)

    mean0 = ndbg.get("mean_len0") if isinstance(ndbg, dict) else None
    mean1 = ndbg.get("mean_len1") if isinstance(ndbg, dict) else None
    best = dbg.get("best_angle_deg")
    chosen = dbg.get("chosen_by")
    title0 = f"a0={a0:.1f}deg mean_len={mean0:.2f}  (picked_edges={len(e0)})"
    title1 = f"a1={a1:.1f}deg mean_len={mean1:.2f}  (picked_edges={len(e1)})"
    if isinstance(best, (int, float)):
        title0 += f" | best={float(best):.1f} ({chosen})"
        title1 += f" | best={float(best):.1f} ({chosen})"

    p0 = _render_panel(img, centroids, tri_edges, e0, title=title0, chain_dir=chain_v, scale=int(args.scale))
    p1 = _render_panel(img, centroids, tri_edges, e1, title=title1, chain_dir=chain_v, scale=int(args.scale))

    out = Image.new("RGBA", (p0.width + p1.width, max(p0.height, p1.height)), (0, 0, 0, 0))
    out.alpha_composite(p0, (0, 0))
    out.alpha_composite(p1, (p0.width, 0))

    out_path = Path(str(args.out)) if str(args.out).strip() else (cycles_dir / f"{stem}_delaunay_edges.png")
    out.convert("RGB").save(str(out_path))

    print("stem=", stem)
    print("main_axes_deg=", [a0, a1])
    print("delaunay_mean_len0=", mean0, "delaunay_mean_len1=", mean1)
    print("best_angle_deg=", best, "chosen_by=", chosen)
    print("out=", str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
