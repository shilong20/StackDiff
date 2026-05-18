"""
【作用概述】从单层图像（仿真/实验，128x128）中自动提取 Re 原子二维像素坐标（点云）。
核心输入/输出：
- 输入：灰度图或 RGB 图（numpy 数组）；默认假设原子为“亮点”，坐标系为图像坐标（x-right/y-down）。
- 输出：点云 `points_px`（N×2，float32），并可选返回调试信息（阈值、面积阈值、轮廓数等）。

【关联说明】文件/模块：
- tools/pred_dadb/pipeline_single.py（单图 pipeline：调用本模块提原子点云）
- tools/pred_dadb/pipeline_bilayer_root.py（批量双层分类：默认也落盘 atoms.json）

【命令行用法】本文件不直接运行（由上述脚本调用）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class AtomDetectConfig:
    # 是否启用高通（实验图通常需要；仿真图可关闭）
    use_highpass: bool = True
    # 背景抑制的平滑尺度（像素）；值越大，越强调“亮点”
    bg_sigma_px: float = 6.0
    # 二值化阈值倍数：阈值=均值*bina_thre
    bina_thre: float = 1.8
    # 最小面积阈值（以 512x512 为基准）
    min_area_threshold: float = 100.0
    # 是否按图像尺寸缩放面积阈值（默认按 512 归一）
    auto_area_scale: bool = True
    # 面积阈值的参考尺寸
    area_scale_ref: int = 512
    # NMS 的最小距离（像素）
    min_distance_px: float = 5.0
    # 最多返回多少个原子点
    max_points: int = 512


def _to_gray_float01(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img)
    if x.ndim == 3 and x.shape[2] >= 3:
        # RGB -> gray
        x = x[..., :3].astype(np.float32)
        x = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    x = x.astype(np.float32)
    v0 = float(np.min(x))
    v1 = float(np.max(x))
    if v1 - v0 < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - v0) / (v1 - v0)).astype(np.float32)


def detect_atoms_from_image(
    img: np.ndarray,
    *,
    cfg: AtomDetectConfig = AtomDetectConfig(),
    return_debug: bool = False,
) -> Tuple[np.ndarray, Optional[Dict]]:
    """
    返回 (points_px, debug)：
      - points_px: (N,2) float32, [x,y] in pixel coords (y-down)
      - debug: 可选，包含 hp/阈值/面积阈值/轮廓数等
    """
    g = _to_gray_float01(img)

    # high-pass（可选）
    if bool(cfg.use_highpass):
        bg = ndimage.gaussian_filter(g, sigma=float(cfg.bg_sigma_px)).astype(np.float32)
        hp = (g - bg).astype(np.float32)
        hp = hp - float(hp.min())
        vmax = float(hp.max())
        if vmax > 1e-9:
            hp = (hp / vmax).astype(np.float32)
    else:
        bg = None
        hp = g.copy()

    mean_val = float(hp.mean())
    thr = float(mean_val * float(cfg.bina_thre))
    _, binary = cv2.threshold(hp, thr, 1.0, cv2.THRESH_BINARY)
    binary_u8 = (binary * 255.0).astype(np.uint8)
    contours, _ = cv2.findContours(binary_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n_contours = int(len(contours))

    # 面积阈值（按尺寸缩放）
    area_thr = float(cfg.min_area_threshold)
    if bool(cfg.auto_area_scale):
        ref = float(max(1, int(cfg.area_scale_ref)))
        scale = float(min(hp.shape[0], hp.shape[1])) / ref
        area_thr = float(area_thr * scale * scale)

    centers_xy = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= area_thr:
            continue
        M = cv2.moments(contour)
        if abs(float(M.get("m00", 0.0))) < 1e-9:
            continue
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
        centers_xy.append((cx, cy))

    if not centers_xy:
        debug = None
        if return_debug:
            debug = {
                "gray": g,
                "hp": hp,
                "bg": bg,
                "thr": float(thr),
                "area_thr": float(area_thr),
                "n_contours": int(n_contours),
                "peaks_xy": np.zeros((0, 2), dtype=np.float32),
            }
        return np.zeros((0, 2), dtype=np.float32), debug

    # greedy NMS by Euclidean distance（避免过密重复点）
    min_d2 = float(cfg.min_distance_px) ** 2
    keep_xy = []
    for (x, y) in centers_xy:
        if len(keep_xy) >= int(cfg.max_points):
            break
        ok = True
        for (kx, ky) in keep_xy:
            dx = float(x) - float(kx)
            dy = float(y) - float(ky)
            if dx * dx + dy * dy < min_d2:
                ok = False
                break
        if ok:
            keep_xy.append((float(x), float(y)))

    pts = np.asarray(keep_xy, dtype=np.float32).reshape(-1, 2)

    debug = None
    if return_debug:
        debug = {
            "gray": g,
            "hp": hp,
            "bg": bg,
            "peaks_xy": pts.copy(),
            "thr": float(thr),
            "area_thr": float(area_thr),
            "n_contours": int(n_contours),
        }
    return pts.astype(np.float32), debug


__all__ = ["AtomDetectConfig", "detect_atoms_from_image"]

