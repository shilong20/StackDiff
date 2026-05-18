#!/usr/bin/env python3
"""
【作用概述】把 generate_sample 导出的 labels（`MOIRE_SAVE_LABELS=1` 生成的 `.npz`）转换成“单层图像”：
将 `layer1_xy128`/`layer2_xy128` 点云光栅化为 128×128 灰度图（高斯点），并按 HighT1 目录风格写出：
每个样本一个文件夹，内部包含 `*_0.png` 与 `*_1.png`，可直接喂给
`src/tools/analysis/interlayer_analyzer_res2.py` 做 slip/twist 测试，而不需要走分解流程。

核心输入/输出：
- 输入：labels 目录（默认 `<output_dir>_labels/`），其中包含 `<stem>.npz`，字段至少有：
  - layer1_xy128: (N,2) float
  - layer2_xy128: (M,2) float
  - sideA: float（Å，最终 128×128 视野边长）
- 输出：输出根目录下若干子文件夹（folder 名末尾带 `_{a}x{b}` nm，便于解析 sideA），每个子文件夹包含：
  - `<stem>_0.png`：layer1 单层图（灰度 8bit）
  - `<stem>_1.png`：layer2 单层图（灰度 8bit）

【关联说明】文件/模块：
- generate_sample/batch_runner.py（MOIRE_SAVE_LABELS 导出 layer1_xy128/layer2_xy128/sideA）
- src/tools/analysis/interlayer_analyzer_res2.py（后续 slip/twist 解析入口）

【命令行用法】
  python tools/export_monolayer_pngs_from_labels.py --labels_dir data/experiments/ReS2_test2p8nm_slip_labels --out_root data/experiments/ReS2_test2p8nm_mono
（参数：--sigma_px=高斯点 sigma；--overwrite=覆盖已存在输出）
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from scipy import ndimage


def _points_to_gaussian_u8(points_xy: np.ndarray, *, size: int = 128, sigma_px: float = 1.6) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    img = np.zeros((int(size), int(size)), dtype=np.float32)
    if pts.size == 0:
        return img.astype(np.uint8)
    x = np.clip(np.round(pts[:, 0]).astype(np.int32), 0, size - 1)
    y = np.clip(np.round(pts[:, 1]).astype(np.int32), 0, size - 1)
    img[y, x] = 1.0
    if float(sigma_px) > 0.0:
        img = ndimage.gaussian_filter(img, sigma=float(sigma_px)).astype(np.float32)
    vmax = float(img.max())
    if vmax > 1e-9:
        img = img / vmax
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def _safe_name(s: str, *, max_len: int = 120) -> str:
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", s.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        return "sample"
    return s[: int(max_len)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_dir", required=True, help="包含 <stem>.npz 的 labels 目录（MOIRE_SAVE_LABELS=1 导出）")
    ap.add_argument("--out_root", required=True, help="输出根目录（每个样本一个子文件夹，内含 *_0/_1.png）")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--sigma_px", type=float, default=1.6)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    labels_dir = Path(str(args.labels_dir)).resolve()
    out_root = Path(str(args.out_root)).resolve()
    if not labels_dir.is_dir():
        raise FileNotFoundError(str(labels_dir))
    out_root.mkdir(parents=True, exist_ok=True)

    npz_paths = sorted([p for p in labels_dir.glob("*.npz") if p.is_file()])
    n = 0
    for npz_path in npz_paths:
        stem = npz_path.stem
        data = np.load(npz_path, allow_pickle=True)
        layer1 = np.asarray(data["layer1_xy128"], dtype=np.float32).reshape(-1, 2)
        layer2 = np.asarray(data["layer2_xy128"], dtype=np.float32).reshape(-1, 2)
        sideA = float(data["sideA"])  # Å

        side_nm = float(sideA) / 10.0
        suffix = f"{side_nm:.2f}x{side_nm:.2f}"
        folder = out_root / f"{_safe_name(stem)}_{suffix}"
        folder.mkdir(parents=True, exist_ok=True)

        out0 = folder / f"{stem}_0.png"
        out1 = folder / f"{stem}_1.png"
        if (out0.exists() or out1.exists()) and (not bool(args.overwrite)):
            continue

        img0 = _points_to_gaussian_u8(layer1, size=int(args.size), sigma_px=float(args.sigma_px))
        img1 = _points_to_gaussian_u8(layer2, size=int(args.size), sigma_px=float(args.sigma_px))
        Image.fromarray(img0).save(out0, format="PNG")
        Image.fromarray(img1).save(out1, format="PNG")
        n += 1

    print(f"done: exported={n} out_root={out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
