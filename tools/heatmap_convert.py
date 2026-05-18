#!/usr/bin/env python3
"""
【作用概述】将灰度图、二维 `.npy` 数组或两张图的差值转换为伪彩色热图 PNG；核心输入是标量图/差值图，核心输出是 RGB 热图，可选输出带 colorbar 的版本。
【关联说明】可复现 `SI/data_augmentation/generate_figures.py` 中 Figure 2 位移/差异热图的转换方式；依赖 Pillow、NumPy 与 Matplotlib。
【命令行用法】python tools/heatmap_convert.py --input after.png --reference before.png --mode absdiff --cmap YlOrBr --auto-vmax-frac 0.8 --output diff_heatmap.png（参数：--input=输入图或npy；--reference=差值参考图；--mode=value/absdiff/diff；--cmap=Matplotlib色图；--auto-vmax-frac=自动上限比例）。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_scalar(path: Path) -> np.ndarray:
    """读取图片或 `.npy` 为二维 float32 标量图。"""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix in IMAGE_SUFFIXES:
        with Image.open(path) as img:
            arr = np.asarray(img.convert("L"))
    else:
        raise ValueError(f"不支持的输入格式: {path}。支持图片或 .npy。")

    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"{path} 读取后不是二维标量图，shape={arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{path} 为空。")
    if np.nanmax(arr) > 1.5:
        arr = arr / 255.0
    return arr


def parse_percentiles(raw: str) -> Tuple[float, float]:
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    if len(parts) != 2:
        raise ValueError("--percentiles 需要形如 0,100 或 1,99")
    lo, hi = float(parts[0]), float(parts[1])
    if not (0.0 <= lo < hi <= 100.0):
        raise ValueError("--percentiles 必须满足 0 <= lo < hi <= 100")
    return lo, hi


def make_value_map(input_arr: np.ndarray, reference_arr: Optional[np.ndarray], mode: str) -> np.ndarray:
    if reference_arr is None:
        if mode != "value":
            raise ValueError("--mode 为 absdiff/diff 时必须提供 --reference。")
        return input_arr.astype(np.float32, copy=False)

    if input_arr.shape != reference_arr.shape:
        raise ValueError(f"--input 与 --reference 尺寸不一致: {input_arr.shape} vs {reference_arr.shape}")

    if mode == "absdiff":
        return np.abs(input_arr - reference_arr).astype(np.float32)
    if mode == "diff":
        return (input_arr - reference_arr).astype(np.float32)
    if mode == "value":
        return input_arr.astype(np.float32, copy=False)
    raise ValueError(f"未知 mode: {mode}")


def normalize_values(
    values: np.ndarray,
    *,
    percentiles: Tuple[float, float],
    vmin: Optional[float],
    vmax: Optional[float],
    auto_vmax_frac: float,
) -> Tuple[np.ndarray, float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("输入中没有有限数值。")

    lo_pct, hi_pct = percentiles
    lo = float(np.percentile(finite, lo_pct)) if vmin is None else float(vmin)
    hi = float(np.percentile(finite, hi_pct)) if vmax is None else float(vmax)
    if vmax is None:
        hi = lo + (hi - lo) * float(auto_vmax_frac)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-8

    cleaned = np.nan_to_num(values, nan=lo, posinf=hi, neginf=lo)
    normalized = (cleaned - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0), lo, hi


def apply_colormap(norm01: np.ndarray, cmap_name: str, reverse: bool = False) -> np.ndarray:
    cmap = matplotlib.colormaps[cmap_name]
    if reverse:
        cmap = cmap.reversed()
    rgb = cmap(norm01)[..., :3]
    return np.uint8(np.clip(rgb, 0.0, 1.0) * 255.0)


def save_colorbar_image(rgb: np.ndarray, output: Path, cmap_name: str, vmin: float, vmax: float, title: str) -> None:
    height, width = rgb.shape[:2]
    fig_w = max(3.0, width / 180.0)
    fig_h = max(3.0, height / 180.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    im = ax.imshow(rgb)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=matplotlib.colormaps[cmap_name], norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将标量图或两张图的差值转换为 Matplotlib 伪彩色热图。"
    )
    parser.add_argument("--input", required=True, type=Path, help="输入图片或二维 .npy。")
    parser.add_argument("--reference", type=Path, help="参考图片或二维 .npy；提供后可用 absdiff/diff 模式。")
    parser.add_argument("--output", required=True, type=Path, help="输出 RGB 热图 PNG。")
    parser.add_argument(
        "--mode",
        choices=["value", "absdiff", "diff"],
        default="value",
        help="value=直接转换 input；absdiff=abs(input-reference)；diff=input-reference。",
    )
    parser.add_argument("--cmap", default="YlOrBr", help="Matplotlib colormap 名称，例如 YlOrBr、viridis、magma、turbo。")
    parser.add_argument(
        "--percentiles",
        default="0,100",
        help="自动归一化使用的百分位范围，默认 0,100；可用 1,99 抑制极端值。",
    )
    parser.add_argument("--vmin", type=float, help="手动指定归一化下限。")
    parser.add_argument("--vmax", type=float, help="手动指定归一化上限。")
    parser.add_argument(
        "--auto-vmax-frac",
        type=float,
        default=1.0,
        help="未显式指定 --vmax 时，将自动上限压到 vmin+(vmax-vmin)*该比例；设为 0.8 可复现 Figure 2 的增强对比风格。",
    )
    parser.add_argument("--reverse", action="store_true", help="反转 colormap。")
    parser.add_argument("--with-colorbar", type=Path, help="额外输出一张带 colorbar 的图。")
    parser.add_argument("--title", default="", help="带 colorbar 输出图的标题。")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.reference is not None and not args.reference.is_file():
        raise FileNotFoundError(args.reference)
    if args.auto_vmax_frac <= 0:
        raise ValueError("--auto-vmax-frac 必须大于 0。")

    input_arr = load_scalar(args.input)
    reference_arr = load_scalar(args.reference) if args.reference is not None else None
    values = make_value_map(input_arr, reference_arr, args.mode)
    percentiles = parse_percentiles(args.percentiles)
    norm01, lo, hi = normalize_values(
        values,
        percentiles=percentiles,
        vmin=args.vmin,
        vmax=args.vmax,
        auto_vmax_frac=args.auto_vmax_frac,
    )
    rgb = apply_colormap(norm01, args.cmap, reverse=args.reverse)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(args.output)

    if args.with_colorbar is not None:
        args.with_colorbar.parent.mkdir(parents=True, exist_ok=True)
        save_colorbar_image(rgb, args.with_colorbar, args.cmap, lo, hi, args.title)

    print(f"saved: {args.output}")
    print(f"normalization: vmin={lo:.8g}, vmax={hi:.8g}, cmap={args.cmap}")


if __name__ == "__main__":
    main()
