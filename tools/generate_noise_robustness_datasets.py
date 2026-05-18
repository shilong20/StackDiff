#!/usr/bin/env python3
"""
【作用概述】基于无噪声基准图目录，为 5 个噪声鲁棒性档次生成含完整噪声模型（扫描+泊松+高斯）的数据集目录；
PSNR(vs GT) 为整数，sigma_total 由校准表确定，输出文件名与输入保持一致。
【关联说明】文件/模块：src/data_prep/online_augmentor.py；docs/noise_robustness_params.md（校准参数）；
tools/generate_noise_datasets_by_psnr.py（旧版纯高斯方案）。
【命令行用法】python tools/generate_noise_robustness_datasets.py \
    --input_dir data/experiments/ReS2_noise0_defect0 \
    --output_parent data/experiments \
    [--levels 30,26,22,18,16] [--seed 12345] [--num N]
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
    add_gaussian_noise,
    add_poisson_noise_from_original,
    add_scan_noise_from_original,
)

# ---------------------------------------------------------------------------
# 校准表：PSNR(vs GT) → sigma_total（含最小扫描+泊松背景）
# GT 图使用 sigma_1 = 0.120（PSNR(vs clean) ≈ 20 dB）
# 详见 docs/noise_robustness_params.md
# ---------------------------------------------------------------------------
SIGMA_1 = 0.129  # GT 高斯噪声 sigma（校准于 1000 张 ReS2 图，平均 PSNR(vs clean)=20 dB）

# sigma_2: 在 GT 上叠加的额外高斯噪声
# sigma_total = sqrt(sigma_1^2 + sigma_2^2)，用于一步法从 clean 生成
# 校准于 100 张 ReS2 图（从 1000 张中采样），见 docs/noise_robustness_params.md
PSNR_VS_GT_TO_SIGMA2: Dict[int, float] = {
    35: 0.021,
    34: 0.023,
    33: 0.026,
    32: 0.030,
    31: 0.033,
    30: 0.038,
    29: 0.042,
    28: 0.048,
    27: 0.054,
    26: 0.060,
    25: 0.068,
    24: 0.077,
    23: 0.087,
    22: 0.098,
    21: 0.111,
    20: 0.126,
    19: 0.143,
    18: 0.163,
    17: 0.185,
    16: 0.211,
    15: 0.240,
    14: 0.275,
    13: 0.316,
    12: 0.364,
    11: 0.425,
}

PSNR_VS_GT_TO_SIGMA_TOTAL: Dict[int, float] = {
    k: round(math.sqrt(SIGMA_1**2 + v**2), 4) for k, v in PSNR_VS_GT_TO_SIGMA2.items()
}

# 最小扫描+泊松噪声参数（固定）
SCAN_POISSON_PARAMS = {
    "pixel_size_A_orig": 0.12,
    "dwell_time_s": 3.0e-6,
    "sigma_jitter_A": 0.02,
    "width_orig_px": 3289,
    "line_freq_hz": 60.0,
    "phase_x": 0.0,
    "phase_y": math.pi / 2,
    "beam_current_A": 3.0e-11,
    "k": 7.8125,  # 2 * 500 / 128
}


def psnr01(a01: np.ndarray, b01: np.ndarray) -> float:
    diff = a01.astype(np.float32) - b01.astype(np.float32)
    mse = float(np.mean(diff * diff))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def add_full_noise(img01: np.ndarray, sigma_gauss: float) -> np.ndarray:
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
    out = add_gaussian_noise(out, sigma=sigma_gauss, preserve_scale=True)
    return np.clip(out, 0.0, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="生成噪声鲁棒性实验数据集（扫描+泊松+高斯，sigma_total 由校准表确定）"
    )
    ap.add_argument("--input_dir", required=True, help="无噪声基准 PNG 目录")
    ap.add_argument("--output_parent", required=True, help="输出父目录（如 data/experiments）")
    ap.add_argument("--prefix", default="ReS2_noise", help="输出目录名前缀")
    ap.add_argument("--suffix", default="_defect0", help="输出目录名后缀")
    ap.add_argument(
        "--levels",
        default="30,26,22,18,15,12",
        help="PSNR(vs GT) 整数档次，逗号分隔（默认 30,26,22,18,15,12）",
    )
    ap.add_argument("--seed", type=int, default=12345, help="随机种子")
    ap.add_argument("--num", type=int, default=None, help="限制处理前 N 张")
    ap.add_argument("--also-gt", action="store_true", help="同时生成 GT 数据集（sigma_1）")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_parent = Path(args.output_parent).resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"input_dir 不存在或不是目录：{input_dir}")
    output_parent.mkdir(parents=True, exist_ok=True)

    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    for lvl in levels:
        if lvl not in PSNR_VS_GT_TO_SIGMA_TOTAL:
            raise SystemExit(f"不支持的 PSNR(vs GT) 档次：{lvl}（可选：{sorted(PSNR_VS_GT_TO_SIGMA_TOTAL.keys())}）")

    if args.also_gt:
        levels = [0] + levels  # 0 表示 GT

    paths = sorted([p for p in input_dir.glob("*.png") if p.is_file()])
    if args.num is not None:
        paths = paths[: max(0, int(args.num))]
    if not paths:
        raise SystemExit(f"input_dir 下未找到 PNG：{input_dir}")

    np.random.seed(int(args.seed))

    out_dirs: Dict[int, Path] = {}
    for lvl in levels:
        if lvl == 0:
            out_dir = output_parent / f"{args.prefix}_gt{args.suffix}"
        else:
            out_dir = output_parent / f"{args.prefix}_{lvl}{args.suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dirs[lvl] = out_dir

    stats_vs_clean: Dict[int, List[float]] = {lvl: [] for lvl in levels}
    stats_vs_gt: Dict[int, List[float]] = {lvl: [] for lvl in levels if lvl != 0}

    for i, p in enumerate(paths):
        if (i + 1) % 50 == 0 or (i + 1) == len(paths):
            print(f"[{i+1}/{len(paths)}] {p.name}")
        with Image.open(p) as im:
            base_u8 = np.asarray(im.convert("L"), dtype=np.uint8)
        base01 = base_u8.astype(np.float32) / 255.0

        gt01 = None
        if 0 in levels or stats_vs_gt:
            rng_state = np.random.get_state()
            gt01 = add_full_noise(base01.copy(), SIGMA_1)
            np.random.set_state(rng_state)
            np.random.bytes(128)  # advance RNG past the GT generation

        if 0 in levels and gt01 is not None:
            gt_u8 = (gt01 * 255.0).astype(np.uint8)
            Image.fromarray(gt_u8, mode="L").save(out_dirs[0] / p.name, format="PNG")
            stats_vs_clean[0].append(psnr01(base01, gt01))

        for lvl in levels:
            if lvl == 0:
                continue
            sigma_total = PSNR_VS_GT_TO_SIGMA_TOTAL[lvl]
            noisy01 = add_full_noise(base01.copy(), sigma_total)
            noisy_u8 = (noisy01 * 255.0).astype(np.uint8)
            Image.fromarray(noisy_u8, mode="L").save(out_dirs[lvl] / p.name, format="PNG")
            stats_vs_clean[lvl].append(psnr01(base01, noisy01))
            if gt01 is not None:
                stats_vs_gt[lvl].append(psnr01(gt01, noisy01))

    print("\n" + "=" * 70)
    print("PSNR 统计（MAX_I=1.0，preserve_scale=True）")
    print("=" * 70)
    for lvl in levels:
        vals_vc = np.asarray(stats_vs_clean[lvl], dtype=np.float64)
        sigma = SIGMA_1 if lvl == 0 else PSNR_VS_GT_TO_SIGMA_TOTAL[lvl]
        lvl_str = "GT" if lvl == 0 else str(lvl)
        line = (
            f"target_PSNR(vs GT)={lvl_str:>3s}  "
            f"sigma_total={sigma:.4f}  "
            f"PSNR(vs clean): mean={vals_vc.mean():.2f} std={vals_vc.std(ddof=0):.2f}"
        )
        if lvl in stats_vs_gt and stats_vs_gt[lvl]:
            vals_vg = np.asarray(stats_vs_gt[lvl], dtype=np.float64)
            line += f"  PSNR(vs GT): mean={vals_vg.mean():.2f} std={vals_vg.std(ddof=0):.2f}"
        print(line)

    print("\n输出目录：")
    for lvl in levels:
        print(f"- {out_dirs[lvl]}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    raise SystemExit(main())
