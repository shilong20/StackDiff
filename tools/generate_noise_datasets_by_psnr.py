#!/usr/bin/env python3
"""
【作用概述】基于一份“基准图像目录”（通常是无噪声或低噪声的 128×128 PNG），为多个目标 PSNR 档次生成“仅添加高斯噪声”的新数据集目录；输出文件名保持不变，并在生成过程中统计每个档次的平均/中位数 PSNR。
【关联说明】文件/模块：src/data_prep/online_augmentor.py（add_gaussian_noise）；tools/add_gaussian_noise.py（单图噪声工具）；generate_sample/Batch_generate.py（基准数据集可由其生成）。
【命令行用法】python tools/generate_noise_datasets_by_psnr.py --input_dir data/experiments/ReS2_noise0_defect0 --output_parent data/experiments（参数：--pairs=自定义 PSNR:sigma 列表；--seed=随机种子；--num=限制处理数量）。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
SYS_SRC = ROOT_DIR / "src"
if str(SYS_SRC) not in sys.path:
    sys.path.append(str(SYS_SRC))

from data_prep.online_augmentor import add_gaussian_noise  # type: ignore  # noqa: E402


def psnr01(a01: np.ndarray, b01: np.ndarray) -> float:
    """PSNR，输入为 [0,1] 浮点数组，MAX_I=1。"""
    diff = a01.astype(np.float32) - b01.astype(np.float32)
    mse = float(np.mean(diff * diff))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def parse_pairs(raw: str) -> Dict[int, float]:
    """
    解析形如：'25:0.0694,20:0.1302,15:0.2478,12:0.3737,10:0.5152'
    返回 {psnr_int: sigma_float}。
    """
    out: Dict[int, float] = {}
    for part in [p.strip() for p in raw.split(",") if p.strip()]:
        if ":" not in part:
            raise ValueError(f"无效 pairs 项（缺少冒号）：{part}")
        k, v = part.split(":", 1)
        psnr = int(float(k.strip()))
        sigma = float(v.strip())
        out[psnr] = sigma
    if not out:
        raise ValueError("pairs 为空。")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="按目标 PSNR 档次生成“仅高斯噪声”的数据集目录")
    ap.add_argument("--input_dir", required=True, help="基准 PNG 目录（不会被修改）")
    ap.add_argument("--output_parent", required=True, help="输出父目录（例如 data/experiments）")
    ap.add_argument("--prefix", default="ReS2_noise", help="输出目录名前缀（默认 ReS2_noise）")
    ap.add_argument("--suffix", default="_defect0", help="输出目录名后缀（默认 _defect0）")
    ap.add_argument(
        "--pairs",
        default="25:0.0694,20:0.1302,15:0.2478,12:0.3737,10:0.5152",
        help="目标 PSNR 与 gaussian_sigma 的映射（psnr:sigma，用逗号分隔）",
    )
    ap.add_argument("--seed", type=int, default=12345, help="随机种子（默认 12345，保证可复现）")
    ap.add_argument("--num", type=int, default=None, help="限制处理前 N 张（默认处理全部）")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_parent = Path(args.output_parent).resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"input_dir 不存在或不是目录：{input_dir}")
    output_parent.mkdir(parents=True, exist_ok=True)

    pairs = parse_pairs(str(args.pairs))
    psnr_levels = sorted(pairs.keys(), reverse=True)

    paths = sorted([p for p in input_dir.glob("*.png") if p.is_file()])
    if args.num is not None:
        paths = paths[: max(0, int(args.num))]
    if not paths:
        raise SystemExit(f"input_dir 下未找到 PNG：{input_dir}")

    # 固定随机性：使用 numpy 全局 RNG（与 add_gaussian_noise 内部一致）
    np.random.seed(int(args.seed))

    # 预创建输出目录
    out_dirs: Dict[int, Path] = {}
    for psnr in psnr_levels:
        out_dir = output_parent / f"{args.prefix}_{psnr}{args.suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dirs[psnr] = out_dir

    stats: Dict[int, List[float]] = {psnr: [] for psnr in psnr_levels}

    for i, p in enumerate(paths):
        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(paths)}] {p.name}")
        with Image.open(p) as im:
            base_u8 = np.asarray(im.convert("L"), dtype=np.uint8)
        base01 = base_u8.astype(np.float32) / 255.0

        # 对每个 PSNR 档次生成一份噪声版本（文件名不变）
        for psnr in psnr_levels:
            sigma = float(pairs[psnr])
            noisy01 = add_gaussian_noise(base01, sigma=sigma, preserve_scale=True)
            noisy01 = np.clip(noisy01, 0.0, 1.0)
            noisy_u8 = (noisy01 * 255.0).astype(np.uint8)
            out_path = out_dirs[psnr] / p.name
            Image.fromarray(noisy_u8, mode="L").save(out_path, format="PNG")
            stats[psnr].append(psnr01(base01, noisy01))

    print("\nPSNR(dB) 统计（MAX_I=1.0，preserve_scale=True）：")
    for psnr in psnr_levels:
        vals = np.asarray(stats[psnr], dtype=np.float64)
        print(
            f"target={psnr:>2d}dB sigma={pairs[psnr]:.4f} "
            f"mean={vals.mean():.3f} median={np.median(vals):.3f} std={vals.std(ddof=0):.3f}"
        )
    print("\n输出目录：")
    for psnr in psnr_levels:
        print(f"- {out_dirs[psnr]}")
    return 0


if __name__ == "__main__":
    # 避免某些环境下 OpenMP 过度占用线程
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    raise SystemExit(main())

