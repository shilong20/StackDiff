#!/usr/bin/env python3
"""
【作用概述】对已有的无噪声 PNG 数据集目录叠加 GT 级别的完整噪声模型（scan+poisson+gaussian），输出到新目录或覆盖原目录。
适用于跨材料泛化性实验等已有 clean 数据集需要统一添加噪声的场景。
【关联说明】文件/模块：src/data_prep/online_augmentor.py；内置噪声参数方案。
【命令行用法】python tools/add_gt_noise_to_datasets.py \
    --input_dirs data/experiments/MoS2_rotation data/experiments/MoS2_slip ... \
    [--output_parent data/experiments/cross_material] \
    [--inplace] [--seed 12345] [--num N]
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
SYS_SRC = ROOT_DIR / "src"
if str(SYS_SRC) not in sys.path:
    sys.path.append(str(SYS_SRC))

from data_prep.online_augmentor import (  # type: ignore  # noqa: E402
    add_gaussian_noise,
    add_poisson_noise_from_original,
    add_scan_noise_from_original,
)

SIGMA_GT = 0.129
SCAN_POISSON_PARAMS = {
    "pixel_size_A_orig": 0.12,
    "dwell_time_s": 3.0e-6,
    "sigma_jitter_A": 0.02,
    "width_orig_px": 3289,
    "line_freq_hz": 60.0,
    "phase_x": 0.0,
    "phase_y": math.pi / 2,
    "beam_current_A": 3.0e-11,
    "k": 7.8125,
}


def psnr01(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = float(np.mean(diff * diff))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def add_gt_noise(img01: np.ndarray) -> np.ndarray:
    sp = SCAN_POISSON_PARAMS
    out = add_scan_noise_from_original(
        img01,
        pixel_size_A_orig=sp["pixel_size_A_orig"],
        k=sp["k"],
        width_orig_px=sp["width_orig_px"],
        dwell_time_scan_s_orig=sp["dwell_time_s"],
        sigma_jitter_A=sp["sigma_jitter_A"],
        line_freq_hz=sp["line_freq_hz"],
        phase_x=sp["phase_x"],
        phase_y=sp["phase_y"],
    )
    out = add_poisson_noise_from_original(
        out,
        beam_current_A=sp["beam_current_A"],
        dwell_time_s=sp["dwell_time_s"],
        k=sp["k"],
    )
    out = add_gaussian_noise(out, sigma=SIGMA_GT, preserve_scale=True)
    return np.clip(out, 0.0, 1.0)


def process_dir(input_dir: Path, output_dir: Path, seed: int, num: int | None) -> None:
    paths = sorted([p for p in input_dir.glob("*.png") if p.is_file()])
    if num is not None:
        paths = paths[: max(0, num)]
    if not paths:
        print(f"  [skip] No PNG found in {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    psnrs = []
    for i, p in enumerate(paths):
        if (i + 1) % 100 == 0 or (i + 1) == len(paths):
            print(f"  [{i+1}/{len(paths)}] {p.name}")
        with Image.open(p) as im:
            base_u8 = np.asarray(im.convert("L"), dtype=np.uint8)
        base01 = base_u8.astype(np.float32) / 255.0

        noisy01 = add_gt_noise(base01.copy())
        psnrs.append(psnr01(base01, noisy01))

        noisy_u8 = (noisy01 * 255.0).astype(np.uint8)
        Image.fromarray(noisy_u8, mode="L").save(output_dir / p.name, format="PNG")

    vals = np.asarray(psnrs, dtype=np.float64)
    print(f"  PSNR(vs clean): mean={vals.mean():.2f} std={vals.std(ddof=0):.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="对已有 clean 数据集叠加 GT 级别噪声（scan+poisson+gaussian, sigma=0.129）"
    )
    ap.add_argument("--input_dirs", nargs="+", required=True, help="输入目录列表")
    ap.add_argument(
        "--output_parent",
        default=None,
        help="输出父目录（输出目录名与输入相同）。若不指定则用 --inplace。",
    )
    ap.add_argument("--inplace", action="store_true", help="直接覆盖输入目录中的文件")
    ap.add_argument("--seed", type=int, default=12345, help="随机种子")
    ap.add_argument("--num", type=int, default=None, help="限制每个目录处理前 N 张")
    args = ap.parse_args()

    if not args.inplace and args.output_parent is None:
        raise SystemExit("必须指定 --output_parent 或 --inplace。")

    for input_path in args.input_dirs:
        input_dir = Path(input_path).resolve()
        if not input_dir.is_dir():
            print(f"[skip] Not a directory: {input_dir}")
            continue

        if args.inplace:
            output_dir = input_dir
        else:
            output_dir = Path(args.output_parent).resolve() / input_dir.name

        print(f"\n[{input_dir.name}] {input_dir} -> {output_dir}")
        process_dir(input_dir, output_dir, seed=args.seed, num=args.num)

    print("\n完成。")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    raise SystemExit(main())
