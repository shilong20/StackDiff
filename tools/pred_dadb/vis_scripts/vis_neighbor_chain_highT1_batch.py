#!/usr/bin/env python3
"""
【作用概述】对一批 HighT1 单层图（通常是 20 张抽样）批量可视化“相邻点连线 + 边长筛选 + 平行相似度聚类”的链方向判定，
用于替代 Delaunay 三角剖分法，避免边缘处三角剖分产生跨空洞长边的干扰。

核心输入/输出：
- 输入：`tools/pred_dadb/extract_cycles_highT1_from_atoms.py` 的输出目录 `--cycles_dir`
  - 需要包含 `index_cycles.json` 与每张图的 `*_cycles.json`
  - `index_cycles.json` 里提供原图相对路径
- 输出：在 `--out_dir` 下为每张 ok 图片写出：
 - `<stem>_neighbor_chain.png`：左右两联图
    - 左：候选方向 a0（main_axes[0]）的边分类结果
    - 右：候选方向 a1（main_axes[1]）的边分类结果
    - 灰色：候选边集合（kNN 去重边集合 -> 仅保留边长 ∈ [nn_min, 1.2*nn_min]，nn_min 为质心最近邻距离最小值）
    - 蓝色：被归入该候选方向的边（越粗表示被多个点的 kNN 重复命中次数越多）
    - 黄色点：质心
    - 橙色箭头：最终 Re 链方向（neighbor_min_edge 判据）
  - `index_neighbor_chain.json`：每张图的候选角、均值边长、best_angle、边筛选统计等调试信息

算法摘要（在 `solve_re_chain_from_centroids(...)` 内实现）：
1) 生成并筛选候选方向（cand3 -> drop -> main_axes 两条）；
2) 对质心做 kNN（k=6）连边并去重；
3) 用最近邻最小值 nn_min 建长度窗口 [nn_min, 1.2*nn_min]，只保留最短一档近邻连边；
4) 对剩余边按与 a0/a1 的轴向夹角分类：min(angle)<=18° 才归类，否则视为“不相似”忽略；
5) 统计两类边长的均值（mean），均值更小的方向为 Re 链方向。

【关联说明】文件/模块：
- tools/pred_dadb/centroid_chain.py（`solve_re_chain_from_centroids` / neighbor_min_edge 实现）
- tools/pred_dadb/extract_cycles_highT1_from_atoms.py（四元环提取，质心来源）

【命令行用法】
python tools/pred_dadb/vis_neighbor_chain_highT1_batch.py --cycles_dir tools/pred_dadb/vis_highT1_cycles/<stamp>
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
from scipy.spatial import cKDTree

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.pred_dadb.centroid_chain import solve_re_chain_from_centroids  # noqa: E402
from tools.pred_dadb.cycles import _infer_sideA_A_from_highT1_path  # noqa: E402


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


def _ordered_points_from_index_item(it: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    从 index_cycles.json 的 item 里读取 seed_cycle_ordered_points（若存在）。
    返回 (4,2) ndarray，顺序为 p00,p10,p11,p01。
    """
    v = it.get("seed_cycle_ordered_points")
    if not isinstance(v, list) or len(v) != 4:
        return None
    pts = []
    for p in v:
        if not isinstance(p, dict):
            return None
        if "x" not in p or "y" not in p:
            return None
        pts.append([float(p["x"]), float(p["y"])])
    return np.asarray(pts, dtype=np.float64).reshape(4, 2)


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


def _neighbor_edges_filtered(centroids: np.ndarray, *, k_neighbors: int = 6, min_len_factor: float = 0.6, max_len_factor: float = 1.6) -> Tuple[List[Tuple[int, int]], Dict]:
    C = np.asarray(centroids, dtype=np.float64).reshape(-1, 2)
    tree = cKDTree(C)
    d, idx = tree.query(C, k=min(int(max(2, k_neighbors + 1)), int(C.shape[0])))
    nn_med = float(np.median(d[:, 1])) if d.ndim == 2 and d.shape[1] >= 2 else float("nan")
    lo = float(min_len_factor) * float(nn_med)
    hi = float(max_len_factor) * float(nn_med)

    edge_set = set()
    for i in range(int(C.shape[0])):
        for j in idx[i, 1:]:
            a, b = int(i), int(j)
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            edge_set.add(key)

    kept = []
    for a, b in sorted(edge_set):
        v = C[b] - C[a]
        L = float(np.hypot(v[0], v[1]))
        if np.isfinite(lo) and np.isfinite(hi) and (L < lo or L > hi):
            continue
        kept.append((int(a), int(b)))

    dbg = {"k_neighbors": int(k_neighbors), "nn_med": float(nn_med), "len_lo": float(lo), "len_hi": float(hi), "n_edges_raw": int(len(edge_set)), "n_edges_kept": int(len(kept))}
    return kept, dbg


def _parse_edges_kept(v: Any) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    if not isinstance(v, list):
        return out
    for item in v:
        if not (isinstance(item, list) or isinstance(item, tuple)) or len(item) != 2:
            continue
        try:
            a = int(item[0])
            b = int(item[1])
        except Exception:
            continue
        if a == b:
            continue
        out.append((min(a, b), max(a, b)))
    # 去重 + 排序
    return sorted(set(out))


def _render_panel(
    img: Image.Image,
    centroids: np.ndarray,
    all_edges: List[Tuple[int, int]],
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

    # all edges (length-filtered) in gray
    for a, b in all_edges:
        p0 = (float(C[a, 0]) * scale, float(C[a, 1]) * scale)
        p1 = (float(C[b, 0]) * scale, float(C[b, 1]) * scale)
        dr.line((p0, p1), fill=(180, 180, 180, 70), width=max(1, int(round(scale * 0.12))))

    # picked edges in blue
    for (a, b), cnt in picked_edges.items():
        p0 = (float(C[a, 0]) * scale, float(C[a, 1]) * scale)
        p1 = (float(C[b, 0]) * scale, float(C[b, 1]) * scale)
        w = max(1, int(round(scale * (0.25 + 0.18 * min(int(cnt), 6)))))
        dr.line((p0, p1), fill=(80, 170, 255, 220), width=w)

    # centroids
    r = max(2, int(round(scale * 0.9)))
    for x, y in C:
        dr.ellipse((float(x) * scale - r, float(y) * scale - r, float(x) * scale + r, float(y) * scale + r), fill=(255, 235, 80, 200))

    # chain arrow
    cc = np.mean(C, axis=0) if C.size else np.array([(img.width - 1) * 0.5, (img.height - 1) * 0.5], dtype=np.float64)
    origin = (float(cc[0]) * scale, float(cc[1]) * scale)
    _draw_arrow(dr, origin_xy=origin, v=chain_dir, length_px=40.0 * scale, color=(255, 180, 60, 240), width=max(1, int(round(scale * 0.8))))

    dr.rectangle((0, 0, img.width * scale, int(round(scale * 10))), fill=(0, 0, 0, 160))
    dr.text((6, 2), title, fill=(255, 255, 255, 220))
    return base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles_dir", required=True, help="extract_cycles_highT1_from_atoms.py 输出目录（含 index_cycles.json）")
    ap.add_argument("--out_dir", default="", help="输出目录（默认 tools/pred_dadb/vis_highT1_chain_neighbor/<stamp>/）")
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
        out_dir = _REPO_ROOT / "tools" / "pred_dadb" / "vis_highT1_chain_neighbor" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    out_index: Dict[str, Any] = {"cycles_dir": str(cycles_dir.as_posix()), "items": []}

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
        # 少量质心时也允许可视化（方向可能不稳定，但便于人工排查）。
        if centroids.shape[0] < 2:
            rec["ok"] = False
            rec["reason"] = "too_few_centroids"
            out_index["items"].append(rec)
            continue

        # seed edges：优先复用 extract_cycles 阶段保存的“基准四元环有序点”（p00,p10,p11,p01）。
        # 若旧数据没有该字段，则退回用“最靠近图像中心的四元环”作为 seed（其 points 顺序仍然来自 cycles.json）。
        seed_ordered = _ordered_points_from_index_item(it)
        if seed_ordered is None:
            img_center = np.array([(img.width - 1) * 0.5, (img.height - 1) * 0.5], dtype=np.float64)
            seed_i = int(np.argmin(np.sum((centroids - img_center[None, :]) ** 2, axis=1)))
            seed_ordered = np.asarray(cycles_pts[int(seed_i)], dtype=np.float64).reshape(4, 2)

        # 直接复用有序四边形的两条临边（从同一顶点 p00 出发）
        e0 = (seed_ordered[1] - seed_ordered[0]).astype(np.float64)
        e1 = (seed_ordered[3] - seed_ordered[0]).astype(np.float64)

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

        # neighbor_debug 在“样本不足”时会是 {"reason": ..., "dbg": {...}}；
        # 在正常选择成功时是展开的 {...}。这里统一取 inner_dbg。
        ndbg = dbg.get("neighbor_debug") or {}
        ndbg_inner = ndbg.get("dbg") if isinstance(ndbg, dict) and isinstance(ndbg.get("dbg"), dict) else ndbg

        e_map0 = _parse_edge_map(ndbg_inner.get("picked_edges0"))
        e_map1 = _parse_edge_map(ndbg_inner.get("picked_edges1"))

        # 以算法内部实际使用的候选边集合为准（避免把“本应剔除的长边”画出来干扰人工检查）。
        all_edges = _parse_edges_kept(ndbg_inner.get("edges_kept"))
        if not all_edges:
            # 兜底：若 debug 里没有（例如 fallback 路径），则仅画长度窗口筛过的边。
            all_edges, _ = _neighbor_edges_filtered(centroids, k_neighbors=6, min_len_factor=0.6, max_len_factor=1.6)

        mean0 = ndbg_inner.get("mean_len0")
        mean1 = ndbg_inner.get("mean_len1")
        m0 = float(mean0) if isinstance(mean0, (int, float)) else float("nan")
        m1 = float(mean1) if isinstance(mean1, (int, float)) else float("nan")
        title0 = f"a0={a0:.1f}deg mean_len={m0:.2f} (picked={len(e_map0)})"
        title1 = f"a1={a1:.1f}deg mean_len={m1:.2f} (picked={len(e_map1)})"
        if abs((a0 - best) % 180.0) < 1e-3:
            title0 += " [SELECTED]"
        if abs((a1 - best) % 180.0) < 1e-3:
            title1 += " [SELECTED]"

        p0 = _render_panel(img, centroids, all_edges, e_map0, title=title0, chain_dir=np.asarray(chain_v, dtype=np.float64).reshape(2), scale=int(args.scale))
        p1 = _render_panel(img, centroids, all_edges, e_map1, title=title1, chain_dir=np.asarray(chain_v, dtype=np.float64).reshape(2), scale=int(args.scale))

        out_img = Image.new("RGBA", (p0.width + p1.width, max(p0.height, p1.height)), (0, 0, 0, 0))
        out_img.alpha_composite(p0, (0, 0))
        out_img.alpha_composite(p1, (p0.width, 0))

        out_png = out_dir / f"{stem}_neighbor_chain.png"
        out_img.convert("RGB").save(str(out_png))

        # 可选：把 da/db（像素）换算成 Å（HighT1 文件夹名通常含视野尺寸，如 _2.79x2.79）
        dadb = dbg.get("dadb_from_centroid_edges")
        if isinstance(dadb, dict):
            sideA_A = _infer_sideA_A_from_highT1_path(img_path)
            if sideA_A is not None and float(sideA_A) > 1e-9:
                px_to_A = float(sideA_A) / float(img.width)
                da_v = dadb.get("da_vec_px")
                db_v = dadb.get("db_vec_px")
                if isinstance(da_v, list) and len(da_v) == 2:
                    dadb["da_vec_A"] = [float(da_v[0]) * px_to_A, float(da_v[1]) * px_to_A]
                    dadb["da_len_A"] = float(np.hypot(float(da_v[0]), float(da_v[1])) * px_to_A)
                if isinstance(db_v, list) and len(db_v) == 2:
                    dadb["db_vec_A"] = [float(db_v[0]) * px_to_A, float(db_v[1]) * px_to_A]
                    dadb["db_len_A"] = float(np.hypot(float(db_v[0]), float(db_v[1])) * px_to_A)
                dadb["sideA_A"] = float(sideA_A)
                dadb["px_to_A"] = float(px_to_A)

        rec.update(
            {
                "ok": True,
                "n_centroids": int(centroids.shape[0]),
                "cand3_deg": dbg.get("cand3_deg"),
                "dropped_candidate_deg": dbg.get("dropped_candidate_deg"),
                "main_axes_deg": [a0, a1],
                "best_angle_deg": float(best),
                "chain_dir": [float(chain_v[0]), float(chain_v[1])],
                "dadb_from_centroid_edges": dbg.get("dadb_from_centroid_edges"),
                "neighbor_debug": ndbg,
                "out_png": out_png.name,
            }
        )
        out_index["items"].append(rec)

    (out_dir / "index_neighbor_chain.json").write_text(json.dumps(out_index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"out_dir={out_dir}")
    print(f"index={out_dir / 'index_neighbor_chain.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
