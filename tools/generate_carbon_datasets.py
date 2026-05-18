#!/usr/bin/env python3
"""
【作用概述】基于一份"基准图像目录"（通常是无碳堆积的 128×128 PNG），为多个碳堆积强度 alpha 生成新数据集目录；
每张图都会确定性地施加碳堆积（prob=1.0），默认同时叠加 GT 级别噪声（scan+poisson+gaussian），输出文件名保持不变。
【关联说明】文件/模块：src/data_prep/online_augmentor.py（add_carbon_background, add_gaussian_noise 等）；
data/experiments/（默认输入/输出父目录）；docs/noise_robustness_params.md（噪声参数方案）。
【命令行用法】python tools/generate_carbon_datasets.py --input_dir data/experiments/ReS2_noise0_defect0
  --output_parent data/experiments
  （参数：--alphas=要测试的 alpha 列表；--cover=固定覆盖面积比；--no-noise=不加噪声；--seed=随机种子）。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
SYS_SRC = ROOT_DIR / "src"
if str(SYS_SRC) not in sys.path:
    sys.path.append(str(SYS_SRC))

from data_prep.online_augmentor import (  # type: ignore  # noqa: E402
    add_carbon_background,
    add_gaussian_noise,
    add_poisson_noise_from_original,
    add_scan_noise_from_original,
)

# GT 噪声参数（与 docs/noise_robustness_params.md 一致）
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


def _add_gt_noise(img01: np.ndarray) -> np.ndarray:
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


def _parse_float_list(raw: str) -> List[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("列表不能为空。")
    return [float(p) for p in parts]


def main() -> int:
    ap = argparse.ArgumentParser(description="按目标碳堆积 alpha 档次生成碳堆积+噪声的数据集目录")
    ap.add_argument("--input_dir", required=True, help="基准 PNG 目录（不会被修改）")
    ap.add_argument("--output_parent", required=True, help="输出父目录（例如 data/experiments）")
    ap.add_argument("--prefix", default="ReS2_carbon", help="输出目录名前缀（默认 ReS2_carbon）")
    ap.add_argument(
        "--alphas",
        default="0.1,0.2,0.3,0.5,0.7",
        help="碳堆积 alpha（混合强度）列表，逗号分隔",
    )
    ap.add_argument("--cover", type=float, default=0.5, help="碳堆积覆盖面积比（默认 0.5）")
    ap.add_argument("--sigma_mask", type=float, default=100, help="掩膜高斯模糊 sigma（默认 100）")
    ap.add_argument("--sigma_field", type=float, default=100, help="强度场高斯模糊 sigma（默认 100）")
    ap.add_argument("--coarse_down", type=int, default=100, help="粗粒度掩膜下采样倍率（默认 100）")
    ap.add_argument("--focus_enable", action="store_true", default=True, help="启用焦点聚焦（默认启用）")
    ap.add_argument("--no_focus", dest="focus_enable", action="store_false", help="禁用焦点聚焦")
    ap.add_argument("--focus_gain", type=float, default=5.0, help="焦点增益（默认 5.0）")
    ap.add_argument("--morph_kernel", type=int, default=11, help="形态学核大小（默认 11）")
    ap.add_argument("--morph_iter", type=int, default=1, help="形态学迭代次数（默认 1）")
    ap.add_argument("--no-noise", action="store_true", help="不添加噪声（默认添加 GT 级别的完整噪声模型）")
    ap.add_argument("--seed", type=int, default=12345, help="随机种子（默认 12345，保证可复现）")
    ap.add_argument("--num", type=int, default=None, help="限制处理前 N 张（默认处理全部）")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_parent = Path(args.output_parent).resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"input_dir 不存在或不是目录：{input_dir}")
    output_parent.mkdir(parents=True, exist_ok=True)

    alphas = _parse_float_list(str(args.alphas))

    paths = sorted([p for p in input_dir.glob("*.png") if p.is_file()])
    if args.num is not None:
        paths = paths[: max(0, int(args.num))]
    if not paths:
        raise SystemExit(f"input_dir 下未找到 PNG：{input_dir}")

    np.random.seed(int(args.seed))

    out_dirs: Dict[float, Path] = {}
    for alpha in alphas:
        alpha_str = f"{alpha:.1f}".replace(".", "p")
        out_dir = output_parent / f"{args.prefix}_alpha{alpha_str}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dirs[alpha] = out_dir

    base_cfg = {
        "cover": float(args.cover),
        "sigma_mask": float(args.sigma_mask),
        "sigma_field": float(args.sigma_field),
        "coarse_down": int(args.coarse_down),
        "focus_enable": bool(args.focus_enable),
        "focus_center": (np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)),
        "focus_radius_ratio": np.random.uniform(0.3, 0.7),
        "focus_gain": float(args.focus_gain),
        "morph_kernel": int(args.morph_kernel),
        "morph_iter": int(args.morph_iter),
    }

    noise_info = "no noise" if args.no_noise else f"GT noise (sigma={SIGMA_GT})"
    print(f"Noise mode: {noise_info}")

    for i, p in enumerate(paths):
        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{len(paths)}] {p.name}")
        with Image.open(p) as im:
            base_u8 = np.asarray(im.convert("L"), dtype=np.uint8)
        base01 = base_u8.astype(np.float32) / 255.0

        rng_state = np.random.get_state()

        for alpha in alphas:
            np.random.set_state(rng_state)
            cfg = dict(base_cfg)
            cfg["alpha"] = float(alpha)
            cfg["focus_center"] = (np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9))
            cfg["focus_radius_ratio"] = np.random.uniform(0.3, 0.7)

            out01 = add_carbon_background(base01.copy(), cfg)

            if not args.no_noise:
                out01 = _add_gt_noise(out01)

            out_u8 = (np.clip(out01, 0.0, 1.0) * 255.0).astype(np.uint8)
            out_path = out_dirs[alpha] / p.name
            Image.fromarray(out_u8, mode="L").save(out_path, format="PNG")

    print(f"\n碳堆积参数：cover={args.cover}, focus_enable={args.focus_enable}, noise={noise_info}")
    print("输出目录：")
    for alpha in alphas:
        print(f"- alpha={alpha:.1f} -> {out_dirs[alpha]}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    raise SystemExit(main())
