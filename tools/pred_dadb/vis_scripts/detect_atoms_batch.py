#!/usr/bin/env python3
"""
【作用概述】
从给定根目录（例如 data/HighT1）中抽样若干张单层图（默认匹配 *_0.png 与 *_1.png），
调用当前 pipeline 使用的原子检测算法提取 Re 原子二维像素坐标，并输出：
- 每张图的原子点云 json（*_atoms.json）
- 原子叠加可视化（*_atoms_overlay.png）
- 索引文件（atoms_index.json），便于后续批量复用点云或人工排查。

核心输入/输出：
- 输入：`--root <dir>`（递归搜索 png），以及原子检测参数（是否高通、阈值倍数、最小面积等）。
- 输出：`--out_dir <dir>`（默认生成 tools/pred_dadb/vis_atom_detect/<stamp>/）。

【关联说明】文件/模块：
- tools/pred_dadb/atoms.py（AtomDetectConfig / detect_atoms_from_image）
- tools/pred_dadb/pipeline_bilayer_root.py（批量双层 pipeline 默认也会落盘 atoms.json；本脚本用于带 overlay 的可视化抽样）

【命令行用法】
python tools/pred_dadb/vis_scripts/detect_atoms_batch.py --root data/HighT1 --num 20 --seed 0
python tools/pred_dadb/vis_scripts/detect_atoms_batch.py --root data/HighT1 --num 20 --seed 0 --no_highpass --bina_thre 1.7 --min_area 80
（参数：--num=抽样张数；--seed=随机种子；--use_highpass/--no_highpass=是否高通；--bina_thre=阈值倍数；--min_area=最小面积）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.pred_dadb.atoms import AtomDetectConfig, detect_atoms_from_image  # noqa: E402


def _safe_id_for_image(img_path: Path) -> str:
    rel = str(img_path.as_posix())
    h = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:10]
    return f"{h}_{img_path.stem}"


def _draw_overlay(img: Image.Image, pts: np.ndarray) -> Image.Image:
    out = img.convert("RGB")
    dr = ImageDraw.Draw(out)
    for x, y in np.asarray(pts, dtype=np.float64).reshape(-1, 2):
        r = 2.0
        dr.ellipse((x - r, y - r, x + r, y + r), outline=(0, 255, 0), width=1)
    return out


def _detect_one(img_path: Path, cfg: AtomDetectConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    img = Image.open(str(img_path))
    arr = np.array(img.convert("L"))
    pts, dbg = detect_atoms_from_image(arr, cfg=cfg, return_debug=True)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    dbg = dbg if isinstance(dbg, dict) else {}
    return pts, dbg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="根目录（递归搜索 png）")
    ap.add_argument("--num", type=int, default=20, help="抽样张数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="", help="输出目录（默认 tools/pred_dadb/vis_atom_detect/<stamp>/）")
    ap.add_argument("--glob", default="*_0.png,*_1.png", help="逗号分隔的 glob（相对每个子目录），默认 *_0.png,*_1.png")

    # atom detect config
    ap.add_argument("--use_highpass", action="store_true", help="强制启用高通（实验图建议开）")
    ap.add_argument("--no_highpass", action="store_true", help="强制禁用高通（仿真图建议关）")
    ap.add_argument("--bg_sigma", type=float, default=6.0)
    ap.add_argument("--bina_thre", type=float, default=1.8)
    ap.add_argument("--min_area", type=float, default=100.0)
    ap.add_argument("--min_dist", type=float, default=5.0)
    ap.add_argument("--max_points", type=int, default=4000)
    args = ap.parse_args()

    root = Path(str(args.root))
    if not root.exists():
        raise FileNotFoundError(str(root))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if str(args.out_dir).strip():
        out_dir = Path(str(args.out_dir))
    else:
        out_dir = _REPO_ROOT / "tools" / "pred_dadb" / "vis_atom_detect" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    globs = [g.strip() for g in str(args.glob).split(",") if g.strip()]
    imgs: List[Path] = []
    # 递归遍历，避免一次性 glob(**) 造成的路径拼接问题
    for g in globs:
        imgs.extend(sorted(root.rglob(g)))
    imgs = [p for p in imgs if p.is_file()]
    if not imgs:
        raise RuntimeError(f"no images found under {root} by globs={globs}")

    rnd = random.Random(int(args.seed))
    if int(args.num) > 0 and int(args.num) < len(imgs):
        imgs = rnd.sample(imgs, k=int(args.num))
    imgs = sorted(imgs)

    use_hp: Optional[bool] = None
    if bool(args.no_highpass):
        use_hp = False
    if bool(args.use_highpass):
        use_hp = True

    cfg = AtomDetectConfig(
        use_highpass=bool(True if use_hp is None else use_hp),
        bg_sigma_px=float(args.bg_sigma),
        bina_thre=float(args.bina_thre),
        min_area_threshold=float(args.min_area),
        min_distance_px=float(args.min_dist),
        max_points=int(args.max_points),
    )

    index: List[Dict[str, Any]] = []
    for p in imgs:
        pts, dbg = _detect_one(p, cfg)
        sid = _safe_id_for_image(p)
        atoms_fn = f"{sid}_atoms.json"
        ov_fn = f"{sid}_atoms_overlay.png"
        (out_dir / atoms_fn).write_text(
            json.dumps([{"x": float(x), "y": float(y)} for (x, y) in pts.tolist()], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _draw_overlay(Image.open(str(p)).convert("L"), pts).save(out_dir / ov_fn)
        index.append(
            {
                "image": str(p.as_posix()),
                "atoms_json": atoms_fn,
                "overlay_png": ov_fn,
                "n_points": int(pts.shape[0]),
                "thr": dbg.get("thr"),
                "area_thr": dbg.get("area_thr"),
                "use_highpass": bool(cfg.use_highpass),
                "bina_thre": float(cfg.bina_thre),
                "min_area_threshold": float(cfg.min_area_threshold),
            }
        )

    (out_dir / "atoms_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"out_dir={out_dir}")
    print(f"n_images={len(imgs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

