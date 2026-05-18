#!/usr/bin/env python3
"""
【作用概述】对一批 HighT1 单层图（通常是 20 张抽样）批量可视化“质心 Delaunay 边（灰色）对照 + Re 链方向判定（橙色）”。

注意：当前 `solve_re_chain_from_centroids` 已精简为 neighbor_min_edge 判据（不再使用 Delaunay 仲裁），
因此本脚本里的“蓝边”来自 neighbor_min_edge 的边分类结果；灰边仍然画 Delaunay 网格边，便于你对照检查连边是否合理。

核心输入/输出：
- 输入：`tools/pred_dadb/extract_cycles_highT1_from_atoms.py` 的输出目录 `--cycles_dir`
  - 需要包含 `index_cycles.json` 与每张图的 `*_cycles.json`
  - `index_cycles.json` 里提供原图相对路径
- 输出：在 `--out_dir` 下为每张 ok 图片写出：
  - `<stem>_delaunay_chain.png`：左右两联图
    - 左：候选方向 a0（main_axes[0]）对应的 Delaunay 选边
    - 右：候选方向 a1（main_axes[1]）对应的 Delaunay 选边
    - 灰色：过滤后的 Delaunay 三角网格边（仅用于对照）
    - 蓝色：neighbor_min_edge 分类得到的“短近邻边”（频次越高越粗）
    - 黄色点：质心
    - 橙色箭头：最终 Re 链方向（neighbor_min_edge）
  - `index_delaunay_chain.json`：每张图的候选角、均值边长、best_angle、过滤统计等调试信息

【关联说明】文件/模块：
- tools/pred_dadb/centroid_chain.py（`solve_re_chain_from_centroids` / neighbor_min_edge 逻辑）
- tools/pred_dadb/extract_cycles_highT1_from_atoms.py（四元环提取，质心来源）

【命令行用法】
python tools/pred_dadb/vis_delaunay_chain_highT1_batch.py --cycles_dir tools/pred_dadb/vis_highT1_cycles/<stamp>
（参数：--max_edge_factor=过滤边缘超长边强度）
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay, cKDTree

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


def _tri_edges_filtered(centroids: np.ndarray, *, max_edge_factor: float) -> Tuple[List[Tuple[int, int]], Dict]:
    C = np.asarray(centroids, dtype=np.float64).reshape(-1, 2)
    tri = Delaunay(C)
    simplices = np.asarray(tri.simplices, dtype=np.int64).reshape(-1, 3)

    tree = cKDTree(C)
    d, _ = tree.query(C, k=min(2, int(C.shape[0])))
    nn_med = float(np.median(d[:, 1])) if d.ndim == 2 and d.shape[1] >= 2 else float("nan")
    max_edge_len = float(max_edge_factor) * float(nn_med) if np.isfinite(nn_med) and nn_med > 0 else float("inf")

    edge_set = set()
    skipped = 0
    for i, j, k in simplices:
        ii, jj, kk = int(i), int(j), int(k)
        edges = [(ii, jj), (jj, kk), (kk, ii)]
        ls = []
        for a, b in edges:
            v = C[b] - C[a]
            ls.append(float(np.hypot(v[0], v[1])))
        if ls and float(np.max(ls)) > float(max_edge_len):
            skipped += 1
            continue
        for a, b in edges:
            edge_set.add((min(a, b), max(a, b)))

    dbg = {
        "n_triangles": int(simplices.shape[0]),
        "skipped_long_edge_triangles": int(skipped),
        "nn_med": None if not np.isfinite(nn_med) else float(nn_med),
        "max_edge_len": None if not np.isfinite(max_edge_len) else float(max_edge_len),
        "max_edge_factor": float(max_edge_factor),
        "n_edges": int(len(edge_set)),
    }
    return sorted(edge_set), dbg


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

    # triangulation edges
    for a, b in tri_edges:
        p0 = (float(C[a, 0]) * scale, float(C[a, 1]) * scale)
        p1 = (float(C[b, 0]) * scale, float(C[b, 1]) * scale)
        dr.line((p0, p1), fill=(180, 180, 180, 80), width=max(1, int(round(scale * 0.15))))

    # picked edges
    for (a, b), cnt in picked_edges.items():
        p0 = (float(C[a, 0]) * scale, float(C[a, 1]) * scale)
        p1 = (float(C[b, 0]) * scale, float(C[b, 1]) * scale)
        w = max(1, int(round(scale * (0.25 + 0.18 * min(int(cnt), 6)))))
        dr.line((p0, p1), fill=(80, 170, 255, 220), width=w)

    # centroids
    r = max(2, int(round(scale * 0.9)))
    for x, y in C:
        dr.ellipse((float(x) * scale - r, float(y) * scale - r, float(x) * scale + r, float(y) * scale + r), fill=(255, 235, 80, 200))

    # chain dir
    cc = np.mean(C, axis=0) if C.size else np.array([(img.width - 1) * 0.5, (img.height - 1) * 0.5], dtype=np.float64)
    origin = (float(cc[0]) * scale, float(cc[1]) * scale)
    _draw_arrow(dr, origin_xy=origin, v=chain_dir, length_px=40.0 * scale, color=(255, 180, 60, 240), width=max(1, int(round(scale * 0.8))))

    # title
    dr.rectangle((0, 0, img.width * scale, int(round(scale * 10))), fill=(0, 0, 0, 160))
    dr.text((6, 2), title, fill=(255, 255, 255, 220))
    return base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles_dir", required=True, help="extract_cycles_highT1_from_atoms.py 输出目录（含 index_cycles.json）")
    ap.add_argument("--out_dir", default="", help="输出目录（默认 tools/pred_dadb/vis_highT1_chain_delaunay/<stamp>/）")
    ap.add_argument("--max_edge_factor", type=float, default=2.0, help="过滤三角形超长边：max_edge_factor * nn_med")
    ap.add_argument("--scale", type=int, default=4, help="输出缩放倍数")
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
        out_dir = _REPO_ROOT / "tools" / "pred_dadb" / "vis_highT1_chain_delaunay" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    out_index: Dict[str, Any] = {
        "cycles_dir": str(cycles_dir.as_posix()),
        "max_edge_factor": float(args.max_edge_factor),
        "scale": int(args.scale),
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
        if not img_path.exists() or not cyc_json.exists():
            rec["ok"] = False
            rec["reason"] = "missing_files"
            out_index["items"].append(rec)
            continue

        img = Image.open(str(img_path)).convert("L")
        centroids, cycles_pts = _load_cycles_json(cyc_json)
        if centroids.shape[0] < 4:
            rec["ok"] = False
            rec["reason"] = "too_few_centroids"
            out_index["items"].append(rec)
            continue

        # seed edges：质心最近图像中心的四元环
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
        best = float(dbg.get("best_angle_deg", a0))
        # 兼容：当前 `solve_re_chain_from_centroids` 已精简为 neighbor_min_edge。
        ndbg = dbg.get("neighbor_debug") or {}
        edge_src = ndbg if isinstance(ndbg, dict) and ("picked_edges0" in ndbg or "picked_edges1" in ndbg) else (ndbg.get("dbg") or {})
        e_map0 = _parse_edge_map(edge_src.get("picked_edges0"))
        e_map1 = _parse_edge_map(edge_src.get("picked_edges1"))

        tri_edges, tri_dbg = _tri_edges_filtered(centroids, max_edge_factor=float(args.max_edge_factor))

        mean0 = ndbg.get("mean_len0") if isinstance(ndbg, dict) else None
        mean1 = ndbg.get("mean_len1") if isinstance(ndbg, dict) else None
        m0 = float(mean0) if isinstance(mean0, (int, float)) else float("nan")
        m1 = float(mean1) if isinstance(mean1, (int, float)) else float("nan")
        title0 = f"a0={a0:.1f}deg mean_len={m0:.2f} (edges={len(e_map0)})"
        title1 = f"a1={a1:.1f}deg mean_len={m1:.2f} (edges={len(e_map1)})"
        if abs((a0 - best) % 180.0) < 1e-3:
            title0 += " [SELECTED]"
        if abs((a1 - best) % 180.0) < 1e-3:
            title1 += " [SELECTED]"

        p0 = _render_panel(img, centroids, tri_edges, e_map0, title=title0, chain_dir=np.asarray(chain_v, dtype=np.float64).reshape(2), scale=int(args.scale))
        p1 = _render_panel(img, centroids, tri_edges, e_map1, title=title1, chain_dir=np.asarray(chain_v, dtype=np.float64).reshape(2), scale=int(args.scale))

        out_img = Image.new("RGBA", (p0.width + p1.width, max(p0.height, p1.height)), (0, 0, 0, 0))
        out_img.alpha_composite(p0, (0, 0))
        out_img.alpha_composite(p1, (p0.width, 0))

        out_png = out_dir / f"{stem}_delaunay_chain.png"
        out_img.convert("RGB").save(str(out_png))

        rec.update(
            {
                "ok": True,
                "n_centroids": int(centroids.shape[0]),
                "main_axes_deg": [a0, a1],
                "best_angle_deg": float(best),
                "chain_dir": [float(chain_v[0]), float(chain_v[1])],
                "neighbor_debug": ndbg,
                "triangulation_debug": tri_dbg,
                "out_png": out_png.name,
            }
        )
        out_index["items"].append(rec)

    (out_dir / "index_delaunay_chain.json").write_text(json.dumps(out_index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"out_dir={out_dir}")
    print(f"index={out_dir / 'index_delaunay_chain.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
