"""
【作用概述】在仿真单层 ReS2 图像上可视化 Re 链方向估计结果，用于对比/验证：
1) 结构张量 + 投影周期仲裁（projection_periodicity）
2) Canny + HoughLines 的 top-k 峰值法（hough_peaks）

输入：若干单层灰度图路径（通常为 128x128 的 `_0.png/_1.png`）。
输出：每张输入图生成一张 512x512 的 overlay PNG（在图像中心绘制两种方法得到的链方向箭头与角度）。

【关联说明】文件/模块：
- tools/pred_dadb/chain_direction.py（链方向估计实现）
- data/experiments/ReS2_test2p8nm_*_monolayer（仿真单层图输出目录）

【命令行用法】
python tools/pred_dadb/vis_chain_hough_on_synth.py \
  --glob 'data/experiments/ReS2_test2p8nm_*_monolayer/*/*.png' \
  --out_dir tools/pred_dadb/vis_chain_hough_synth_2p8nm

参数说明：
- --glob：输入图片通配符（支持重复传入）
- --out_dir：输出目录（不存在会创建）
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import asdict
from typing import Iterable, List, Tuple

import cv2
import numpy as np

# Allow running as a script: `python tools/pred_dadb/vis_chain_hough_on_synth.py ...`
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.pred_dadb.chain_direction import (
    ChainDirectionConfig,
    HoughChainDirectionConfig,
    PointRadonChainDirectionConfig,
    ProjectionPeriodicityConfig,
    get_chain_direction_hough_peaks,
    get_chain_direction_point_radon_peaks,
    get_chain_direction_projection_periodicity,
)


def _read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _alpha_line(
    canvas_bgr: np.ndarray,
    p0: Tuple[int, int],
    p1: Tuple[int, int],
    color_bgr: Tuple[int, int, int],
    *,
    thickness: int = 2,
    alpha: float = 0.7,
    tip_length: float = 0.18,
) -> None:
    """在 BGR 图上画半透明箭头线。"""
    overlay = canvas_bgr.copy()
    cv2.arrowedLine(
        overlay,
        p0,
        p1,
        color_bgr,
        thickness=thickness,
        tipLength=float(tip_length),
        line_type=cv2.LINE_AA,
    )
    cv2.addWeighted(overlay, float(alpha), canvas_bgr, float(1.0 - alpha), 0.0, dst=canvas_bgr)


def _draw_chain_overlay(
    gray_u8: np.ndarray,
    *,
    ang_proj_deg: float,
    ang_hough_deg: float,
    ang_radon_deg: float,
    out_path: str,
    title: str,
) -> None:
    # enlarge for nicer inspection
    img = cv2.resize(gray_u8, (512, 512), interpolation=cv2.INTER_NEAREST)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cx, cy = 256, 256
    L = 180  # arrow length in px on 512 canvas

    def endpoint(angle_deg: float) -> Tuple[int, int]:
        # image coords: x-right/y-down; angle from +x axis
        rad = np.deg2rad(angle_deg)
        x = int(round(cx + L * float(np.cos(rad))))
        y = int(round(cy + L * float(np.sin(rad))))
        return x, y

    # projection_periodicity: purple
    _alpha_line(bgr, (cx, cy), endpoint(ang_proj_deg), (180, 0, 180), thickness=3, alpha=0.65)
    # hough_peaks: cyan
    _alpha_line(bgr, (cx, cy), endpoint(ang_hough_deg), (255, 255, 0), thickness=2, alpha=0.80)
    # point_radon_peaks: green
    _alpha_line(bgr, (cx, cy), endpoint(ang_radon_deg), (0, 220, 0), thickness=2, alpha=0.80)

    # small legend
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bgr, title, (8, 20), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        bgr,
        f"proj={ang_proj_deg:.1f}deg (purple)",
        (8, 42),
        font,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        bgr,
        f"hough={ang_hough_deg:.1f}deg (cyan)",
        (8, 64),
        font,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        bgr,
        f"radon={ang_radon_deg:.1f}deg (green)",
        (8, 86),
        font,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, bgr)


def _iter_globs(globs: Iterable[str]) -> List[str]:
    paths: List[str] = []
    for g in globs:
        paths.extend(glob.glob(g))
    # stable order for easier comparison
    paths = sorted(set(paths))
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", action="append", required=True, help="输入图片通配符，可重复传入。")
    ap.add_argument("--out_dir", required=True, help="输出目录。")
    ap.add_argument("--max_n", type=int, default=0, help="最多处理多少张；0 表示全量。")
    args = ap.parse_args()

    paths = _iter_globs(args.glob)
    if args.max_n and args.max_n > 0:
        paths = paths[: int(args.max_n)]

    out_dir = str(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    ch_cfg = ChainDirectionConfig(sigma_px=6.0)
    proj_cfg = ProjectionPeriodicityConfig()
    hough_cfg = HoughChainDirectionConfig()
    radon_cfg = PointRadonChainDirectionConfig(atom_use_highpass=False)

    # also dump config for reproducibility
    with open(os.path.join(out_dir, "configs.txt"), "w", encoding="utf-8") as f:
        f.write("ChainDirectionConfig:\n")
        f.write(repr(asdict(ch_cfg)) + "\n\n")
        f.write("ProjectionPeriodicityConfig:\n")
        f.write(repr(asdict(proj_cfg)) + "\n\n")
        f.write("HoughChainDirectionConfig:\n")
        f.write(repr(asdict(hough_cfg)) + "\n\n")
        f.write("PointRadonChainDirectionConfig:\n")
        f.write(repr(asdict(radon_cfg)) + "\n\n")

    for p in paths:
        gray = _read_gray(p)
        _v_proj, ang_proj, _dom_proj, _dbg_proj = get_chain_direction_projection_periodicity(
            gray, cfg=ch_cfg, proj_cfg=proj_cfg
        )
        _v_h, ang_h, _dom_h, _dbg_h = get_chain_direction_hough_peaks(gray, cfg=hough_cfg)
        _v_r, ang_r, _dom_r, _dbg_r = get_chain_direction_point_radon_peaks(gray, cfg=radon_cfg)

        # keep folder structure to make browsing easier
        rel = p.replace("\\", "/")
        # flatten: replace '/' with '__'
        rel_flat = rel.strip("/").replace("/", "__")
        out_path = os.path.join(out_dir, f"{rel_flat}_overlay.png")
        _draw_chain_overlay(
            gray,
            ang_proj_deg=float(ang_proj),
            ang_hough_deg=float(ang_h),
            ang_radon_deg=float(ang_r),
            out_path=out_path,
            title=rel,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
