#!/usr/bin/env python3
"""
【作用概述】在 HighT1 单层图上演示“旋转坐标排序法”的分行效果：对四元环质心点云按候选角度旋转，
用“行内抖动阈值 eps”把点分成多行，并用不同颜色可视化每一行。

本脚本用于回答一个非常具体的问题：rotate-sort 里用 `eps` 把 y' 分段成多行是否可靠，以及
`eps` 取值对“同一行被误拆分”的影响。默认会把两条最终备选方向（main_axes）都做一次可视化，
便于你对比两个方向下的“分行一致性”。

核心输入/输出：
- 输入：
  - `--cycles_dir`：`tools/pred_dadb/extract_cycles_highT1_from_atoms.py` 的输出目录（含 `index_cycles.json` 与 `*_cycles.json`）。
  - `--stem`：要演示的样本 stem（例如 `..._r04c02_2.79x2.79_0`）。
- 输出：
  - `<stem>_rows_compare.png`：四联图（2 行 × 2 列）
    - 每一行对应一个候选方向（main_axes_deg 的两条）：
      1) 原图坐标系：质心按“行”着色
      2) 旋转坐标系：质心按“行”着色，并画出每行中心的水平参考线
  - 同时在控制台打印：cand3/dropped/main_axes/best_angle 以及两条候选各自的 eps/n_rows 等调试信息。

【关联说明】文件/模块：
- tools/pred_dadb/centroid_chain.py（`solve_re_chain_from_centroids` 与 `_kmeans_1d_two_clusters`）
- tools/pred_dadb/vis_chain_direction_highT1_from_cycles.py（同一套“起点四元环两条边剔除候选”的链方向流程）

【命令行用法】
python tools/pred_dadb/vis_rows_rotate_sort_highT1.py --cycles_dir tools/pred_dadb/vis_highT1_cycles/<stamp> --stem "<stem>"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.pred_dadb.centroid_chain import _kmeans_1d_two_clusters, solve_re_chain_from_centroids  # noqa: E402


def _load_cycles_json(path: Path) -> Tuple[np.ndarray, np.ndarray]:
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


def _rows_by_rotate_sort(
    centroids: np.ndarray,
    ang_deg: float,
    *,
    intra_row_clip_quantile: float = 0.95,
    eps_mul: float = 1.5,
    eps_min: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, float, float, Dict]:
    """
    返回：
    - row_id: (N,) 每个质心所属行编号（0..n_rows-1，按 y' 从小到大排序）
    - row_centers: (n_rows,) 每行中心 y'（旋转坐标系）
    - eps: 行内阈值
    - jit: 行内抖动中心估计
    - debug
    """
    C = np.asarray(centroids, dtype=np.float64).reshape(-1, 2)
    n = int(C.shape[0])
    if n < 4:
        raise ValueError("too_few_centroids")

    th = float(np.radians(float(ang_deg)))
    c = float(np.cos(th))
    s = float(np.sin(th))
    cc = np.mean(C, axis=0)
    dx = C[:, 0] - float(cc[0])
    dy = C[:, 1] - float(cc[1])

    # rotate by -theta
    xprime = (c * dx + s * dy).astype(np.float64)
    yprime = (-s * dx + c * dy).astype(np.float64)

    order = np.argsort(yprime)
    y_sorted = yprime[order]
    diffs = np.diff(y_sorted)
    if diffs.size < 2:
        raise ValueError("too_few_diffs")

    hi = float(np.quantile(diffs, float(intra_row_clip_quantile)))
    diffs_clip = np.minimum(diffs, hi)
    labels, centers = _kmeans_1d_two_clusters(diffs_clip)
    if labels.size == 0:
        raise ValueError("kmeans_empty")

    k_jit = int(np.argmin(centers))
    jit = float(max(centers[k_jit], 1e-6))
    eps = float(max(float(eps_mul) * jit, float(eps_min)))

    groups_idx: List[np.ndarray] = []
    start = 0
    for i, d in enumerate(diffs):
        if float(d) > eps:
            groups_idx.append(order[start : i + 1])
            start = i + 1
    groups_idx.append(order[start:])

    row_centers = np.array([float(np.mean(yprime[g])) for g in groups_idx if g.size], dtype=np.float64)
    if row_centers.size < 1:
        raise ValueError("no_rows")
    perm = np.argsort(row_centers)
    row_centers = row_centers[perm]
    groups_idx = [groups_idx[int(i)] for i in perm]

    row_id = -np.ones((n,), dtype=np.int64)
    for rid, g in enumerate(groups_idx):
        row_id[np.asarray(g, dtype=np.int64)] = int(rid)

    dbg = {
        "angle_deg": float(ang_deg),
        "n": int(n),
        "diff_centers": [float(centers[0]), float(centers[1])],
        "jit": float(jit),
        "eps": float(eps),
        "eps_mul": float(eps_mul),
        "eps_min": float(eps_min),
        "n_rows": int(row_centers.size),
        "row_centers": [float(x) for x in row_centers.tolist()],
        "xprime_range": [float(np.min(xprime)), float(np.max(xprime))],
        "yprime_range": [float(np.min(yprime)), float(np.max(yprime))],
    }
    return row_id, row_centers, eps, jit, dbg


def _palette(n: int) -> List[Tuple[int, int, int, int]]:
    base = [
        (255, 80, 80, 235),
        (80, 220, 255, 235),
        (120, 255, 160, 235),
        (255, 220, 80, 235),
        (200, 120, 255, 235),
        (255, 160, 80, 235),
        (80, 255, 220, 235),
        (180, 180, 255, 235),
    ]
    if n <= len(base):
        return base[:n]
    out = list(base)
    while len(out) < n:
        out.extend(base)
    return out[:n]


def _draw_two_panel_rows(
    img: Image.Image,
    centroids: np.ndarray,
    row_id: np.ndarray,
    row_centers: np.ndarray,
    *,
    ang_deg: float,
    title: str,
    out_path: Path,
) -> None:
    img = img.convert("L")
    C = np.asarray(centroids, dtype=np.float64).reshape(-1, 2)
    rid = np.asarray(row_id, dtype=np.int64).reshape(-1)
    n_rows = int(np.max(rid)) + 1 if rid.size else 0
    cols = _palette(max(1, n_rows))

    W, H = img.size
    scale = 4
    panel_w = W * scale
    panel_h = H * scale
    out = Image.new("RGBA", (panel_w * 2, panel_h), (0, 0, 0, 0))
    left = img.convert("RGB").resize((panel_w, panel_h), resample=Image.BILINEAR).convert("RGBA")
    right = Image.new("RGBA", (panel_w, panel_h), (20, 20, 20, 255))
    out.alpha_composite(left, (0, 0))
    out.alpha_composite(right, (panel_w, 0))

    dr = ImageDraw.Draw(out, "RGBA")

    # 左：原图坐标系，按行着色
    r = max(2, int(round(scale * 1.1)))
    for (x, y), k in zip(C, rid):
        k = int(max(0, k))
        col = cols[k % len(cols)]
        ix = int(round(float(x) * scale))
        iy = int(round(float(y) * scale))
        dr.ellipse((ix - r, iy - r, ix + r, iy + r), fill=col)

    # 右：旋转坐标系（-theta），把点映射到 panel，并画 row_center 水平线
    th = float(np.radians(float(ang_deg)))
    c = float(np.cos(th))
    s = float(np.sin(th))
    cc = np.mean(C, axis=0)
    dx = C[:, 0] - float(cc[0])
    dy = C[:, 1] - float(cc[1])
    xprime = (c * dx + s * dy).astype(np.float64)
    yprime = (-s * dx + c * dy).astype(np.float64)

    # 映射到右侧 panel：自适应缩放到 [m, W-m]
    margin = 8.0
    xr0, xr1 = float(np.min(xprime)), float(np.max(xprime))
    yr0, yr1 = float(np.min(yprime)), float(np.max(yprime))
    span = float(max(xr1 - xr0, yr1 - yr0, 1e-6))
    # 以 span 为共同尺度，保持等比例
    sx = (panel_w - 2 * margin) / span
    sy = (panel_h - 2 * margin) / span
    sc = float(min(sx, sy))

    def map_pt(xp: float, yp: float) -> Tuple[float, float]:
        xx = (xp - 0.5 * (xr0 + xr1)) * sc + panel_w * 0.5
        yy = (yp - 0.5 * (yr0 + yr1)) * sc + panel_h * 0.5
        return (float(xx) + panel_w, float(yy))

    # row centers line
    for k, yc in enumerate(np.asarray(row_centers, dtype=np.float64).reshape(-1)):
        _x0, y0 = map_pt(float(xr0), float(yc))
        _x1, y1 = map_pt(float(xr1), float(yc))
        dr.line((_x0, y0, _x1, y1), fill=(255, 255, 255, 70), width=max(1, int(round(scale * 0.3))))

    rp = max(2, int(round(scale * 1.0)))
    for xp, yp, k in zip(xprime, yprime, rid):
        k = int(max(0, k))
        col = cols[k % len(cols)]
        xx, yy = map_pt(float(xp), float(yp))
        dr.ellipse((xx - rp, yy - rp, xx + rp, yy + rp), fill=col)

    # 标题条（不依赖字体文件）
    dr.rectangle((0, 0, panel_w * 2, int(round(scale * 10))), fill=(0, 0, 0, 160))
    dr.text((6, 2), f"{title} | angle={ang_deg:.2f}deg | n_rows={n_rows}", fill=(255, 255, 255, 220))

    out.convert("RGB").save(str(out_path))


def _draw_compare_rows(
    img: Image.Image,
    centroids: np.ndarray,
    a0: float,
    row0: Tuple[np.ndarray, np.ndarray, float, float, Dict],
    a1: float,
    row1: Tuple[np.ndarray, np.ndarray, float, float, Dict],
    *,
    out_path: Path,
) -> None:
    """
    输出 2 行 × 2 列：每行一个角度，左原图坐标系，右旋转坐标系。
    """
    img = img.convert("L")
    W, H = img.size
    scale = 4
    panel_w = W * scale
    panel_h = H * scale
    out = Image.new("RGBA", (panel_w * 2, panel_h * 2), (0, 0, 0, 0))

    # 逐行画两联
    for row_idx, (a, pack) in enumerate([(float(a0), row0), (float(a1), row1)]):
        row_id, row_centers, eps, _jit, dbg = pack
        # 临时输出到内存，再贴到总图
        tmp_path = out_path.with_suffix(f".tmp_row{row_idx}.png")
        _draw_two_panel_rows(
            img,
            centroids,
            row_id,
            row_centers,
            ang_deg=float(a),
            title=f"rotate-sort rows (eps={eps:.3f}, eps_mul={dbg.get('eps_mul')}, eps_min={dbg.get('eps_min')})",
            out_path=tmp_path,
        )
        row_img = Image.open(str(tmp_path)).convert("RGBA")
        out.alpha_composite(row_img, (0, row_idx * panel_h))
        # 清理临时文件：不删除，移动到 trash
        try:
            trash = _REPO_ROOT / "trash"
            trash.mkdir(parents=True, exist_ok=True)
            tmp_path.replace(trash / tmp_path.name)
        except Exception:
            pass

    # 顶部总标题
    dr = ImageDraw.Draw(out, "RGBA")
    dr.rectangle((0, 0, panel_w * 2, int(round(scale * 10))), fill=(0, 0, 0, 160))
    dr.text((6, 2), f"rows compare (two candidate angles): a0={a0:.2f}deg, a1={a1:.2f}deg", fill=(255, 255, 255, 220))
    out.convert("RGB").save(str(out_path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles_dir", required=True, help="extract_cycles_highT1_from_atoms.py 输出目录（含 index_cycles.json）")
    ap.add_argument("--stem", required=True, help="要演示的图片 stem（与 index_cycles.json 对应）")
    ap.add_argument("--out", default="", help="输出路径（默认写到 cycles_dir 同级 vis_highT1_chain 目录旁）")
    ap.add_argument("--eps_mul", type=float, default=6.0, help="eps = max(eps_mul*jit, eps_min)；调大可减少“同一行被误拆分”")
    ap.add_argument("--eps_min", type=float, default=6.0, help="eps 下限（像素）")
    args = ap.parse_args()

    cycles_dir = Path(str(args.cycles_dir))
    idx_path = cycles_dir / "index_cycles.json"
    if not idx_path.exists():
        raise FileNotFoundError(str(idx_path))
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    items = idx.get("items", [])
    stem = str(args.stem)

    hit = None
    for it in items:
        if str(it.get("stem", "")) == stem:
            hit = it
            break
    if hit is None:
        raise ValueError(f"stem 不在 index_cycles.json 中：{stem}")

    rel_img = Path(str(hit["path"]))
    img_path = (_REPO_ROOT / rel_img).resolve()
    cyc_json = cycles_dir / str(hit["cycles_json"])
    if not cyc_json.exists():
        raise FileNotFoundError(str(cyc_json))

    img = Image.open(str(img_path)).convert("L")
    centroids, cycles_pts = _load_cycles_json(cyc_json)
    if centroids.shape[0] < 4:
        raise RuntimeError("too_few_centroids")

    # 起点四元环：质心最近图像中心
    img_center = np.array([(img.width - 1) * 0.5, (img.height - 1) * 0.5], dtype=np.float64)
    seed_i = int(np.argmin(np.sum((centroids - img_center[None, :]) ** 2, axis=1)))
    seed = cycles_pts[int(seed_i)]
    e0 = (seed[1] - seed[0]).astype(np.float64)  # p10-p00
    e1 = (seed[3] - seed[0]).astype(np.float64)  # p01-p00

    chain_v, dbg = solve_re_chain_from_centroids(centroids, seed_edge_dirs=(e0, e1), k_neighbors=6, max_candidates=6)
    best_ang = float(dbg.get("best_angle_deg", 0.0))

    if str(args.out).strip():
        out_path = Path(str(args.out))
    else:
        out_path = cycles_dir / f"{stem}_rows_compare.png"

    main_axes = dbg.get("main_axes_deg") or []
    if len(main_axes) < 2:
        raise RuntimeError("未得到两条 main_axes_deg，无法做对比可视化")
    a0 = float(main_axes[0])
    a1 = float(main_axes[1])

    pack0 = _rows_by_rotate_sort(centroids, a0, eps_mul=float(args.eps_mul), eps_min=float(args.eps_min))
    pack1 = _rows_by_rotate_sort(centroids, a1, eps_mul=float(args.eps_mul), eps_min=float(args.eps_min))
    _draw_compare_rows(img, centroids, a0, pack0, a1, pack1, out_path=out_path)

    print("stem=", stem)
    print("cand3_deg=", dbg.get("cand3_deg"))
    print("dropped_candidate_deg=", dbg.get("dropped_candidate_deg"))
    print("main_axes_deg=", dbg.get("main_axes_deg"))
    print("best_angle_deg=", best_ang)
    print("best_chain_dir_xy=", [float(chain_v[0]), float(chain_v[1])])
    _row0 = pack0[4]
    _row1 = pack1[4]
    print("rows_debug_a0=", json.dumps(_row0, ensure_ascii=False))
    print("rows_debug_a1=", json.dumps(_row1, ensure_ascii=False))
    print("out=", str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
