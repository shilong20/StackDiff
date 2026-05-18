#!/usr/bin/env python3
"""
【作用概述】对单张 HighT1 单层图，叠加可视化“已提取的四元环(Re4 parallelogram)”以及“质心点云得到的三个候选主方向(cand3)”。
用于快速诊断：四元环是否正确、cand3 是否包含两条与四元环边平行的主轴、以及被剔除方向是否合理。

核心输入/输出：
- 输入：`extract_cycles_highT1_from_atoms.py` 的输出目录 `--cycles_dir`（含 index_cycles.json 与 *_cycles.json）
- 输出：在 `--out_dir` 下写出一张 `<stem>_cycles_cand3.png`
  - 白色/半透明：四元环边
  - 黄色点：四元环质心
  - 三条箭头：cand3 三个候选方向（被删除 dropped 的箭头更细）

说明：
- 若 `index_cycles.json` 的 item 中存在 `seed_cycle_ordered_points`，则直接复用其两条临边方向作为 seed 边方向；
  否则退回为“质心最靠近图像中心”的四元环作为 seed。

【关联说明】文件/模块：
- tools/pred_dadb/centroid_chain.py（solve_re_chain_from_centroids）
- tools/pred_dadb/extract_cycles_highT1_from_atoms.py（四元环提取输出格式）

【命令行用法】
python tools/pred_dadb/vis_one_highT1_cycles_candidates.py \
  --cycles_dir tools/pred_dadb/vis_highT1_cycles/<stamp> \
  --stem "<stem>" \
  --out_dir tools/pred_dadb/vis_debug_cycles_candidates
（参数：--scale=输出放大倍数；--arrow_len=箭头长度(px,以原图尺度计)）
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.pred_dadb.centroid_chain import solve_re_chain_from_centroids  # noqa: E402


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


def _draw_arrow(
    dr: ImageDraw.ImageDraw,
    *,
    origin_xy: Tuple[float, float],
    ang_deg: float,
    length_px: float,
    scale: int,
    color: Tuple[int, int, int, int],
    width: int,
) -> None:
    r = float(np.radians(float(ang_deg)))
    v = np.array([float(np.cos(r)), float(np.sin(r))], dtype=np.float64)
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    x2 = ox + float(v[0]) * float(length_px) * float(scale)
    y2 = oy + float(v[1]) * float(length_px) * float(scale)
    dr.line((ox, oy, x2, y2), fill=color, width=int(max(1, width)))
    rr = max(2, int(round(width * 1.2)))
    dr.ellipse((x2 - rr, y2 - rr, x2 + rr, y2 + rr), fill=color)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles_dir", required=True, help="extract_cycles_highT1_from_atoms.py 输出目录（含 index_cycles.json）")
    ap.add_argument("--stem", required=True, help="要可视化的 stem（与 index_cycles.json 中 stem 精确匹配）")
    ap.add_argument("--out_dir", default="", help="输出目录（默认 tools/pred_dadb/vis_debug_cycles_candidates/<stamp>/）")
    ap.add_argument("--scale", type=int, default=4, help="输出缩放倍数")
    ap.add_argument("--arrow_len", type=float, default=35.0, help="箭头长度（以原图像素计）")
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
        raise ValueError(f"stem 不存在于 index_cycles.json：{stem}")
    if not bool(it.get("ok", False)):
        raise ValueError(f"该 stem 在 cycles 输出里标记为 ok=false：{stem} reason={it.get('reason')}")

    img_path = (_REPO_ROOT / Path(str(it["path"]))).resolve()
    cyc_json = cycles_dir / str(it["cycles_json"])
    if not img_path.exists():
        raise FileNotFoundError(str(img_path))
    if not cyc_json.exists():
        raise FileNotFoundError(str(cyc_json))

    img = Image.open(str(img_path)).convert("L")
    centroids, cycles_pts = _load_cycles_json(cyc_json)
    if centroids.shape[0] < 1:
        raise ValueError("没有 cycles/centroids")

    # seed edges：优先复用 extract_cycles 阶段保存的“基准四元环有序点”（p00,p10,p11,p01）。
    seed_ordered = it.get("seed_cycle_ordered_points")
    seed_pts = None
    if isinstance(seed_ordered, list) and len(seed_ordered) == 4:
        try:
            seed_pts = np.asarray([[float(p["x"]), float(p["y"])] for p in seed_ordered], dtype=np.float64).reshape(4, 2)
        except Exception:
            seed_pts = None
    if seed_pts is None:
        # 旧数据兼容：退回到“质心最近图像中心的四元环”
        img_center = np.array([(img.width - 1) * 0.5, (img.height - 1) * 0.5], dtype=np.float64)
        seed_i = int(np.argmin(np.sum((centroids - img_center[None, :]) ** 2, axis=1)))
        seed_pts = np.asarray(cycles_pts[int(seed_i)], dtype=np.float64).reshape(4, 2)

    e0 = (seed_pts[1] - seed_pts[0]).astype(np.float64)
    e1 = (seed_pts[3] - seed_pts[0]).astype(np.float64)

    # 只需要 cand3 + dropped 信息
    _v, dbg = solve_re_chain_from_centroids(centroids, seed_edge_dirs=(e0, e1), k_neighbors=6, max_candidates=6)
    cand3 = dbg.get("cand3_deg") or []
    dropped = dbg.get("dropped_candidate_deg")
    best = dbg.get("best_angle_deg")

    scale = int(max(1, args.scale))
    out = img.convert("RGB").resize((img.width * scale, img.height * scale), resample=Image.BILINEAR).convert("RGBA")
    dr = ImageDraw.Draw(out, "RGBA")

    # draw cycles
    for cyc in cycles_pts:
        pts = [(float(x) * scale, float(y) * scale) for x, y in cyc]
        # cycle 点顺序未必是四边形顺序，这里直接画闭合 polyline（能看出是否绑错链）
        dr.line(pts + [pts[0]], fill=(245, 245, 245, 160), width=max(1, int(round(scale * 0.18))))

    # draw centroids
    r = max(2, int(round(scale * 0.9)))
    for x, y in centroids:
        dr.ellipse((float(x) * scale - r, float(y) * scale - r, float(x) * scale + r, float(y) * scale + r), fill=(255, 235, 80, 210))

    # arrows at mean centroid
    cc = np.mean(centroids, axis=0)
    origin = (float(cc[0]) * scale, float(cc[1]) * scale)
    for a in cand3[:3]:
        a = float(a)
        is_drop = (dropped is not None) and (min(abs(a - float(dropped)), 180.0 - abs(a - float(dropped))) < 1e-6)
        is_best = (best is not None) and (min(abs(a - float(best)), 180.0 - abs(a - float(best))) < 1e-6)
        if is_best:
            col = (255, 170, 60, 235)
            w = max(2, int(round(scale * 0.9)))
        elif is_drop:
            col = (160, 160, 160, 220)
            w = max(1, int(round(scale * 0.35)))
        else:
            col = (120, 220, 140, 230)
            w = max(1, int(round(scale * 0.65)))
        _draw_arrow(dr, origin_xy=origin, ang_deg=a, length_px=float(args.arrow_len), scale=scale, color=col, width=w)

    # title bar
    bar_h = int(round(scale * 12))
    dr.rectangle((0, 0, img.width * scale, bar_h), fill=(0, 0, 0, 170))
    t = f"{stem}  cand3={[(float(x)) for x in cand3[:3]]}  dropped={dropped}  best={best}"
    dr.text((6, 2), t, fill=(255, 255, 255, 230))

    if str(args.out_dir).strip():
        out_dir = Path(str(args.out_dir))
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = _REPO_ROOT / "tools" / "pred_dadb" / "vis_debug_cycles_candidates" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_cycles_cand3.png"
    out.convert("RGB").save(str(out_path))
    print(f"out_png={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
