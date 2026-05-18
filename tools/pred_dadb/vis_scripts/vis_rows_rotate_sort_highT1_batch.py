#!/usr/bin/env python3
"""
【作用概述】对一批 HighT1 单层图（通常是 20 张抽样）批量演示“旋转坐标排序法”的分行效果：
对每张图的四元环质心点云在两条候选主轴方向（main_axes）下分别做分行，并把最终选择的 Re 链方向标出来。

核心输入/输出：
- 输入：`tools/pred_dadb/extract_cycles_highT1_from_atoms.py` 的输出目录 `--cycles_dir`
  - 需要包含 `index_cycles.json` 与每张图的 `*_cycles.json`
  - `index_cycles.json` 里提供原图相对路径
- 输出：在 `--out_dir` 下为每张 ok 图片写出：
  - `<stem>_rows2_chain.png`：四联图（2 行 × 2 列）
    - 每一行对应一个候选角度（main_axes_deg 的两条）
    - 左列：原图坐标系（质心按“行”着色 + 橙色箭头=最终 Re 链方向；若该行角度被选中则标题标记 SELECTED）
    - 右列：旋转坐标系（-θ）（质心按“行”着色 + 行中心水平线）
  - `index_rows.json`：每张图的 angles/eps/n_rows/是否选中等调试信息

分行算法说明（与主仲裁一致）：
- 对候选角 θ，把质心点云旋转 -θ；
- 对旋转后的 y' 排序并做差分 diffs；
- 用 1D k-means(k=2) 从 diffs 中估计“行内抖动尺度” jit；
- 设 eps = max(eps_mul*jit, eps_min)，用 eps 在 y' 上做分段得到多行；
- 每行用不同颜色着色，并在旋转坐标系里画出行中心的水平线。

【关联说明】文件/模块：
- tools/pred_dadb/centroid_chain.py（`solve_re_chain_from_centroids`、`_kmeans_1d_two_clusters`）
- tools/pred_dadb/extract_cycles_highT1_from_atoms.py（四元环提取，质心来源）

【命令行用法】
python tools/pred_dadb/vis_rows_rotate_sort_highT1_batch.py --cycles_dir tools/pred_dadb/vis_highT1_cycles/<stamp>
（参数：--eps_mul/--eps_min 调大可以减少“同一行被误拆分”）
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def _rows_by_rotate_sort(
    centroids: np.ndarray,
    ang_deg: float,
    *,
    intra_row_clip_quantile: float = 0.95,
    eps_mul: float,
    eps_min: float,
) -> Tuple[np.ndarray, np.ndarray, float, float, Dict]:
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

    # 行距估计：相邻 row_center 差分的中位数（与主仲裁一致，配合一次温和裁剪）
    spacing = 0.0
    if row_centers.size >= 2:
        row_diffs = np.diff(row_centers)
        if row_diffs.size:
            med = float(np.median(row_diffs))
            if med > 1e-6:
                keep2 = (row_diffs > 0.5 * med) & (row_diffs < 1.8 * med)
                rd = row_diffs[keep2] if np.any(keep2) else row_diffs
                spacing = float(np.median(rd))

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
        "row_spacing_px": float(spacing),
        "xprime_range": [float(np.min(xprime)), float(np.max(xprime))],
        "yprime_range": [float(np.min(yprime)), float(np.max(yprime))],
    }
    return row_id, row_centers, eps, jit, dbg


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

def _axis_angle_deg(a: float, b: float) -> float:
    """
    轴向角差：a/b 以度表示，折叠到 [0,90]。
    """
    d = abs(float(a) - float(b))
    d = min(d, 180.0 - d)
    d = min(d, 180.0 - d)
    return float(d)

def _same_axis_angle(a: float, b: float, *, tol_deg: float = 1e-3) -> bool:
    return _axis_angle_deg(float(a), float(b)) <= float(tol_deg)


def _render_two_panel(
    img: Image.Image,
    centroids: np.ndarray,
    row_id: np.ndarray,
    row_centers: np.ndarray,
    *,
    ang_deg: float,
    eps: float,
    eps_mul: float,
    eps_min: float,
    spacing_px: float,
    tag: str,
    selected: bool,
    chain_dir: np.ndarray,
    global_span: Optional[float] = None,
) -> Image.Image:
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

    # 左：最终 Re 链方向（橙色箭头）
    cc = np.mean(C, axis=0) if C.size else np.array([(W - 1) * 0.5, (H - 1) * 0.5], dtype=np.float64)
    origin = (float(cc[0]) * scale, float(cc[1]) * scale)
    _render_len = 42.0 * scale
    _draw_arrow(
        dr,
        origin_xy=origin,
        v=np.asarray(chain_dir, dtype=np.float64).reshape(2),
        length_px=_render_len,
        color=(255, 180, 60, 240),
        width=max(1, int(round(scale * 1.0))),
    )

    # 右：旋转坐标系（-theta），把点映射到 panel，并画 row_center 水平线
    th = float(np.radians(float(ang_deg)))
    c = float(np.cos(th))
    s = float(np.sin(th))
    cc2 = np.mean(C, axis=0)
    dx = C[:, 0] - float(cc2[0])
    dy = C[:, 1] - float(cc2[1])
    xprime = (c * dx + s * dy).astype(np.float64)
    yprime = (-s * dx + c * dy).astype(np.float64)

    margin = 8.0
    xr0, xr1 = float(np.min(xprime)), float(np.max(xprime))
    yr0, yr1 = float(np.min(yprime)), float(np.max(yprime))
    span_local = float(max(xr1 - xr0, yr1 - yr0, 1e-6))
    span = float(max(span_local, float(global_span) if global_span is not None else 0.0, 1e-6))
    sx = (panel_w - 2 * margin) / span
    sy = (panel_h - 2 * margin) / span
    sc = float(min(sx, sy))

    def map_pt(xp: float, yp: float) -> Tuple[float, float]:
        xx = (xp - 0.5 * (xr0 + xr1)) * sc + panel_w * 0.5
        yy = (yp - 0.5 * (yr0 + yr1)) * sc + panel_h * 0.5
        return (float(xx) + panel_w, float(yy))

    for yc in np.asarray(row_centers, dtype=np.float64).reshape(-1):
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
    tag2 = f"{tag} / SELECTED" if bool(selected) else f"{tag} / alt"
    dr.text(
        (6, 2),
        f"theta={ang_deg:.2f}deg ({tag2}) | spacing={spacing_px:.2f}px | eps={eps:.2f} (mul={eps_mul:.1f}, min={eps_min:.1f}) | rows={n_rows}",
        fill=(255, 255, 255, 220),
    )

    return out


def _render_compare(
    img: Image.Image,
    centroids: np.ndarray,
    *,
    a0: float,
    pack0: Tuple[np.ndarray, np.ndarray, float, float, Dict],
    a1: float,
    pack1: Tuple[np.ndarray, np.ndarray, float, float, Dict],
    best_angle: float,
    chain_dir: np.ndarray,
    eps_mul: float,
    eps_min: float,
    spacing_map: Dict[float, float],
    title_suffix: str = "",
) -> Image.Image:
    W, H = img.size
    scale = 4
    panel_w = W * scale
    panel_h = H * scale
    out = Image.new("RGBA", (panel_w * 2, panel_h * 2), (0, 0, 0, 0))

    # 为了避免“看起来上面更大但其实是缩放造成的错觉”，两个角度的旋转坐标系统一用同一尺度（global_span）。
    # 用各自 x'/y' 的跨度最大值作为 span，再取二者的 max。
    span0 = float(max(pack0[4].get("xprime_range")[1] - pack0[4].get("xprime_range")[0], pack0[4].get("yprime_range")[1] - pack0[4].get("yprime_range")[0], 1e-6))
    span1 = float(max(pack1[4].get("xprime_range")[1] - pack1[4].get("xprime_range")[0], pack1[4].get("yprime_range")[1] - pack1[4].get("yprime_range")[0], 1e-6))
    global_span = float(max(span0, span1))

    for row_idx, (a, pack, tag) in enumerate([(float(a0), pack0, "a0"), (float(a1), pack1, "a1")]):
        row_id, row_centers, eps, _jit, _dbg = pack
        selected = _same_axis_angle(float(a), float(best_angle), tol_deg=1e-3)
        spacing_px = float(spacing_map.get(float(a), _dbg.get("row_spacing_px", 0.0)))
        row_img = _render_two_panel(
            img,
            centroids,
            row_id,
            row_centers,
            ang_deg=float(a),
            eps=float(eps),
            eps_mul=float(eps_mul),
            eps_min=float(eps_min),
            spacing_px=float(spacing_px),
            tag=str(tag),
            selected=bool(selected),
            chain_dir=np.asarray(chain_dir, dtype=np.float64).reshape(2),
            global_span=float(global_span),
        )
        out.alpha_composite(row_img, (0, row_idx * panel_h))

    dr = ImageDraw.Draw(out, "RGBA")
    dr.rectangle((0, 0, panel_w * 2, int(round(scale * 10))), fill=(0, 0, 0, 160))
    suf = f" | {title_suffix}" if str(title_suffix).strip() else ""
    dr.text((6, 2), f"rows compare: a0={a0:.2f}deg, a1={a1:.2f}deg | best={best_angle:.2f}deg{suf}", fill=(255, 255, 255, 220))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles_dir", required=True, help="extract_cycles_highT1_from_atoms.py 输出目录（含 index_cycles.json）")
    ap.add_argument("--out_dir", default="", help="输出目录（默认 tools/pred_dadb/vis_highT1_chain_rows/<stamp>/）")
    ap.add_argument("--eps_mul", type=float, default=6.0, help="eps = max(eps_mul*jit, eps_min)")
    ap.add_argument("--eps_min", type=float, default=6.0, help="eps 下限（像素）")
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
        out_dir = _REPO_ROOT / "tools" / "pred_dadb" / "vis_highT1_chain_rows" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    out_index: Dict[str, Any] = {
        "cycles_dir": str(cycles_dir.as_posix()),
        "eps_mul": float(args.eps_mul),
        "eps_min": float(args.eps_min),
        "items": [],
    }

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
        cyc_json = cycles_dir / str(it["cycles_json"])
        if not cyc_json.exists() or not img_path.exists():
            rec["ok"] = False
            rec["reason"] = "missing_files"
            out_index["items"].append(rec)
            continue

        centroids, cycles_pts = _load_cycles_json(cyc_json)
        if centroids.shape[0] < 4:
            rec["ok"] = False
            rec["reason"] = "too_few_centroids"
            rec["n_centroids"] = int(centroids.shape[0])
            out_index["items"].append(rec)
            continue

        img = Image.open(str(img_path)).convert("L")
        img_center = np.array([(img.width - 1) * 0.5, (img.height - 1) * 0.5], dtype=np.float64)
        seed_i = int(np.argmin(np.sum((centroids - img_center[None, :]) ** 2, axis=1)))
        seed = cycles_pts[int(seed_i)]
        e0 = (seed[1] - seed[0]).astype(np.float64)
        e1 = (seed[3] - seed[0]).astype(np.float64)

        chain_v, dbg = solve_re_chain_from_centroids(centroids, seed_edge_dirs=(e0, e1), k_neighbors=6, max_candidates=6)
        main_axes = dbg.get("main_axes_deg") or []
        if len(main_axes) < 2:
            rec["ok"] = False
            rec["reason"] = "no_main_axes"
            out_index["items"].append(rec)
            continue
        a0 = float(main_axes[0])
        a1 = float(main_axes[1])
        best_angle = float(dbg.get("best_angle_deg", a0))

        pack0 = _rows_by_rotate_sort(centroids, a0, eps_mul=float(args.eps_mul), eps_min=float(args.eps_min))
        pack1 = _rows_by_rotate_sort(centroids, a1, eps_mul=float(args.eps_mul), eps_min=float(args.eps_min))

        spacing_map = {}

        out_img = _render_compare(
            img,
            centroids,
            a0=a0,
            pack0=pack0,
            a1=a1,
            pack1=pack1,
            best_angle=best_angle,
            chain_dir=np.asarray(chain_v, dtype=np.float64).reshape(2),
            eps_mul=float(args.eps_mul),
            eps_min=float(args.eps_min),
            spacing_map=spacing_map,
            title_suffix=f"chosen_by={dbg.get('chosen_by')}",
        )

        out_png = out_dir / f"{stem}_rows2_chain.png"
        out_img.convert("RGB").save(str(out_png))

        rec.update(
            {
                "ok": True,
                "n_centroids": int(centroids.shape[0]),
                "cand3_deg": dbg.get("cand3_deg"),
                "dropped_candidate_deg": dbg.get("dropped_candidate_deg"),
                "main_axes_deg": [a0, a1],
                "best_angle_deg": best_angle,
                "chain_dir": [float(chain_v[0]), float(chain_v[1])],
                "chosen_by": dbg.get("chosen_by"),
                "rows_a0": pack0[4],
                "rows_a1": pack1[4],
                "out_png": out_png.name,
            }
        )
        out_index["items"].append(rec)

    (out_dir / "index_rows.json").write_text(json.dumps(out_index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"out_dir={out_dir}")
    print(f"index_rows={out_dir / 'index_rows.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
