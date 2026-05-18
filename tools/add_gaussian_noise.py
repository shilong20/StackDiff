#!/usr/bin/env python3
"""
【作用概述】对单张图像添加高斯噪声并另存，便于快速做噪声强度对照实验；不会修改原文件，默认在同目录生成新文件。
【关联说明】文件/模块：无强耦合；可配合 generate_sample 或任意图像处理流程使用。
【命令行用法】python tools/add_gaussian_noise.py --image <img> --sigma 50（参数：--sigma=噪声标准差；--out=输出路径可选）。

对单张图像添加高斯噪声并保存。

只支持“单个图像路径 + 噪声强度（标准差）”的简单用法：

命令行示例：
    python tools/add_gaussian_noise.py \
  --image data/augmentation_examples/final_augmented.png \
  --sigma 50.0

函数示例：
    from tools.add_gaussian_noise import add_gaussian_noise_to_image
    add_gaussian_noise_to_image(\"path/to/img.png\", sigma=10.0)
    # 默认会在同目录生成 path/to/img_sigma10.png
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def add_gaussian_noise_to_array(
    img_arr: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    在数组上添加高斯噪声（零均值，标准差为 sigma，结果裁剪到 [0, 255]，返回 uint8）。
    """
    if sigma <= 0:
        return img_arr

    arr = img_arr.astype(np.float32)
    noise = np.random.normal(loc=0.0, scale=sigma, size=arr.shape).astype(np.float32)
    noisy = arr + noise
    noisy = np.clip(noisy, 0.0, 255.0)
    return noisy.astype(np.uint8)


def add_gaussian_noise_to_image(
    image_path: str,
    sigma: float,
    output_path: Optional[str] = None,
) -> str:
    """
    对单张图像添加高斯噪声并保存。

    Args:
        image_path: 输入图像路径。
        sigma: 噪声强度（标准差，像素值尺度，典型值 5~30）。
        output_path: 输出路径，若为 None 则在同目录下生成 *_noisy.xxx。

    Returns:
        实际保存的输出图像路径（字符串）。
    """
    src = Path(image_path)
    if not src.is_file():
        raise FileNotFoundError(f"输入图像不存在: {src}")

    if output_path is None:
        # 默认在文件名中加入 sigma 信息，例如 img_sigma10.png
        sigma_str = f"{sigma:g}"
        out = src.with_name(f"{src.stem}_sigma{sigma_str}{src.suffix}")
    else:
        out = Path(output_path)

    with Image.open(src) as im:
        # 保留灰度 / RGB，其他模式统一转为 RGB
        if im.mode not in ("L", "RGB"):
            im = im.convert("RGB")
        arr = np.array(im)
        noisy_arr = add_gaussian_noise_to_array(arr, sigma=sigma)
        noisy_img = Image.fromarray(noisy_arr)
        noisy_img.save(out)

    return str(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Add Gaussian noise to a single image")
    parser.add_argument("--image", type=str, required=True, help="输入图像路径")
    parser.add_argument("--sigma", type=float, required=True, help="高斯噪声标准差（像素值尺度）")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出图像路径（默认：在同目录生成 *_sigma<sigma>.xxx）",
    )
    args = parser.parse_args()

    try:
        out_path = add_gaussian_noise_to_image(args.image, sigma=args.sigma, output_path=args.output)
        print(f"已保存带噪声图像到: {out_path}")
        return 0
    except Exception as e:
        print(f"处理失败: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
