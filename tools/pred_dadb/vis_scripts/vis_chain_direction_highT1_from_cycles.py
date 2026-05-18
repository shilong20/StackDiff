#!/usr/bin/env python3
"""
【作用概述】基于 HighT1 单层图的“四元环质心点云”估计并可视化 Re 链方向（t1/链内方向）。

核心输入/输出：
- 输入：`tools/pred_dadb/extract_cycles_highT1_from_atoms.py` 的输出目录（包含 `index_cycles.json` 与每张图的 `*_cycles.json`）。
  `index_cycles.json` 里包含原图相对路径与 atoms_dir（可选，用于叠加原子点）。
- 输出：在 `--out_dir` 下为每张 ok 图片写出：
  - `<stem>_chain_overlay.png`：灰度底图 + 质心(黄点) + 三候选主方向(青/紫/绿箭头；被剔除者用红色标记) + 最终 Re 链方向(橙色箭头)（可选叠加原子点淡白）
  - `index_chain.json`：每张图的候选角度/best_angle 等调试信息

链方向算法（与仿真验证一致）：
1) 候选方向生成：对质心点云做 kNN（k=6），取邻接向量角度直方图，提取多个主峰候选；
2) 候选方向收敛为 3 条：取角度直方图 top-3 峰作为“三候选主方向”；
3) 用起点四元环（靠近图像中心的四元环）的两条边方向做一次过滤：
   剔除掉那条与两条边都最不平行的候选（通常是对角线/伪主轴），剩余两条进入仲裁；
4) 仲裁（neighbor_min_edge）：对质心做 kNN 连边并取最短一档近邻边（按 nn_min 构造长度窗口），
   比较两候选方向对应的边长均值，均值更小的方向作为 Re 链方向（链内更紧密）。

【关联说明】文件/模块：
- tools/pred_dadb/centroid_chain.py（solve_re_chain_from_centroids 等实现）
- tools/pred_dadb/atom_detect.py（原子点云导出，可选叠加）
- tools/pred_dadb/extract_cycles_highT1_from_atoms.py（四元环导出）

【命令行用法】
python tools/pred_dadb/vis_chain_direction_highT1_from_cycles.py --cycles_dir tools/pred_dadb/vis_highT1_cycles/<stamp>
（参数：--centroid_k=kNN 邻居数；--max_candidates=候选数；--no_atoms=不叠加原子点）
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

from tools.pred_dadb.centroid_chain import solve_re_chain_from_centroids  # noqa: E402


def _load_cycles_json(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    cycles.json: List[List[{x,y}]]，每个 cycle 为 4 个顶点坐标（像素）。
    返回：
      - centroids (M,2)
      - cycles_pts (M,4,2) 依次为 (p00,p10,p11,p01)
    """
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


def _load_atoms_json(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    pts = [(float(d["x"]), float(d["y"])) for d in data]
    return np.asarray(pts, dtype=np.float64).reshape(-1, 2)


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


def _draw_chain_overlay(
    img: Image.Image,
    centroids: np.ndarray,
    *,
    chain_dir: np.ndarray,
    cand_angles_deg: Optional[Sequence[float]],
    dropped_candidate_deg: Optional[float],
    atoms: Optional[np.ndarray],
    out_path: Path,
) -> None:
    img = img.convert("L")
    scale = 4
    base = img.convert("RGB").resize((img.width * scale, img.height * scale), resample=Image.BILINEAR).convert("RGBA")
    dr = ImageDraw.Draw(base, "RGBA")

    if atoms is not None and atoms.size:
        pts = np.asarray(atoms, dtype=np.float64).reshape(-1, 2)
        pr = max(1, int(round(scale * 0.6)))
        for x, y in pts:
            ix = int(round(float(x) * scale))
            iy = int(round(float(y) * scale))
            dr.ellipse((ix - pr, iy - pr, ix + pr, iy + pr), fill=(255, 255, 255, 35))

    C = np.asarray(centroids, dtype=np.float64).reshape(-1, 2)
    cr = max(2, int(round(scale * 1.0)))
    for x, y in C:
        ix = int(round(float(x) * scale))
        iy = int(round(float(y) * scale))
        dr.ellipse((ix - cr, iy - cr, ix + cr, iy + cr), fill=(255, 235, 80, 220))

    cc = np.mean(C, axis=0) if C.size else np.array([(img.width - 1) / 2, (img.height - 1) / 2], dtype=np.float64)
    origin = (float(cc[0]) * scale, float(cc[1]) * scale)
    L = 40.0 * scale

    # 三个候选主方向（青/紫/绿；被剔除的那个用红色半透明标记）
    if cand_angles_deg is not None:
        cols = [(80, 220, 255, 235), (200, 120, 255, 235), (120, 255, 160, 235)]  # cyan, purple, green
        for i, a in enumerate(list(cand_angles_deg)[:3]):
            rr = float(np.radians(float(a)))
            v0 = np.array([float(np.cos(rr)), float(np.sin(rr))], dtype=np.float64)
            col = cols[i % len(cols)]
            w = max(1, int(round(scale * 0.7)))
            if dropped_candidate_deg is not None:
                da = abs(float(a) - float(dropped_candidate_deg))
                da = min(da, 180.0 - da)
                if da < 1e-3:
                    col = (255, 80, 80, 190)  # dropped
                    w = max(1, int(round(scale * 0.55)))
            _draw_arrow(dr, origin_xy=origin, v=v0, length_px=L, color=col, width=w)

    # 最终链方向（橙色，更粗）
    v = np.asarray(chain_dir, dtype=np.float64).reshape(2)
    nv = float(np.hypot(v[0], v[1]))
    if nv > 1e-9:
        _draw_arrow(
            dr,
            origin_xy=origin,
            v=v,
            length_px=L,
            color=(255, 180, 60, 235),
            width=max(1, int(round(scale * 1.0))),
        )

    base.convert("RGB").save(str(out_path))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles_dir", required=True, help="extract_cycles_highT1_from_atoms.py 输出目录（含 index_cycles.json）")
    ap.add_argument("--out_dir", default="", help="输出目录（默认 tools/pred_dadb/vis_highT1_chain/<stamp>/）")
    ap.add_argument("--centroid_k", type=int, default=6, help="质心候选方向生成：kNN 邻居数")
    ap.add_argument("--max_candidates", type=int, default=6, help="最多候选方向数")
    ap.add_argument("--no_atoms", action="store_true", help="不叠加原子点（只画质心+箭头）")
    args = ap.parse_args()

    cycles_dir = Path(str(args.cycles_dir))
    idx_path = cycles_dir / "index_cycles.json"
    if not idx_path.exists():
        raise FileNotFoundError(str(idx_path))
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    items = idx.get("items", [])
    if not items:
        raise ValueError(f"index_cycles.json 里没有 items：{idx_path}")

    atoms_dir = Path(str(idx.get("atoms_dir", ""))) if str(idx.get("atoms_dir", "")).strip() else None
    if atoms_dir is not None and not atoms_dir.exists():
        atoms_dir = None

    if str(args.out_dir).strip():
        out_dir = Path(str(args.out_dir))
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = _REPO_ROOT / "tools" / "pred_dadb" / "vis_highT1_chain" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    out_index: Dict[str, Any] = {
        "cycles_dir": str(cycles_dir.as_posix()),
        "atoms_dir": str(atoms_dir.as_posix()) if atoms_dir is not None else "",
        "centroid_k": int(args.centroid_k),
        "max_candidates": int(args.max_candidates),
        "no_atoms": bool(args.no_atoms),
        "items": [],
    }

    for it in items:
        stem = str(it.get("stem", ""))
        rec: Dict[str, Any] = {"stem": stem, "path": it.get("path"), "ok_cycles": bool(it.get("ok", False))}
        if not it.get("ok", False):
            rec["ok"] = False
            rec["reason"] = it.get("reason", "cycles_failed")
            out_index["items"].append(rec)
            continue

        rel_img = Path(str(it["path"]))
        img_path = (_REPO_ROOT / rel_img).resolve()
        cyc_json = cycles_dir / str(it["cycles_json"])
        if not cyc_json.exists():
            raise FileNotFoundError(str(cyc_json))

        img = Image.open(str(img_path)).convert("L")
        centroids, cycles_pts = _load_cycles_json(cyc_json)
        if centroids.shape[0] < 4:
            rec["ok"] = False
            rec["reason"] = "too_few_centroids"
            rec["n_centroids"] = int(centroids.shape[0])
            out_index["items"].append(rec)
            continue

        # 起点四元环：质心最靠近图像中心的那个
        if cycles_pts.shape[0] >= 1:
            img_center = np.array([(img.width - 1) * 0.5, (img.height - 1) * 0.5], dtype=np.float64)
            seed_i = int(np.argmin(np.sum((centroids - img_center[None, :]) ** 2, axis=1)))
            seed = cycles_pts[int(seed_i)]
            # seed 两条边方向（用 p00 作为公共顶点）
            e0 = (seed[1] - seed[0]).astype(np.float64)  # p10-p00
            e1 = (seed[3] - seed[0]).astype(np.float64)  # p01-p00
            seed_edges = (e0, e1)
        else:
            seed_edges = None

        chain_v, dbg = solve_re_chain_from_centroids(
            centroids,
            seed_edge_dirs=seed_edges,
            k_neighbors=int(args.centroid_k),
            max_candidates=int(args.max_candidates),
        )

        cand_angles = (dbg.get("cand3_deg") or [])[:3]
        dropped = dbg.get("dropped_candidate_deg")

        atoms = None
        if not bool(args.no_atoms) and atoms_dir is not None:
            atoms_json = atoms_dir / f"{stem}_atoms.json"
            if atoms_json.exists():
                atoms = _load_atoms_json(atoms_json)

        out_png = out_dir / f"{stem}_chain_overlay.png"
        _draw_chain_overlay(img, centroids, chain_dir=chain_v, cand_angles_deg=cand_angles, dropped_candidate_deg=dropped, atoms=atoms, out_path=out_png)

        rec.update(
            {
                "ok": True,
                "n_centroids": int(centroids.shape[0]),
                "chain_dir": [float(chain_v[0]), float(chain_v[1])],
                "best_angle_deg": dbg.get("best_angle_deg"),
                "best_spacing_px": None,
                "candidates_deg": dbg.get("candidates_deg"),
                "cand3_deg": dbg.get("cand3_deg"),
                "dropped_candidate_deg": dbg.get("dropped_candidate_deg"),
                "main_axes_deg": dbg.get("main_axes_deg"),
                "spacings": None,
                "overlay_png": out_png.name,
            }
        )
        out_index["items"].append(rec)

    (out_dir / "index_chain.json").write_text(json.dumps(out_index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"out_dir={out_dir}")
    print(f"index_chain={out_dir / 'index_chain.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
