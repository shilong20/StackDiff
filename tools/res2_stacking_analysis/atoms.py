"""
Purpose: Detect Re atom coordinates from a single-layer ReS2 image. The main function accepts a grayscale/RGB image array and returns atom points in image coordinates plus optional diagnostics.
Related files: tools/res2_stacking_analysis/single_layer_lattice.py and tools/res2_stacking_analysis/classify_bilayers.py.
CLI usage: This module is imported by the da/db pipelines and is not intended to be executed directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class AtomDetectConfig:

    use_highpass: bool = True

    bg_sigma_px: float = 6.0

    bina_thre: float = 1.8

    min_area_threshold: float = 100.0

    auto_area_scale: bool = True

    area_scale_ref: int = 512

    min_distance_px: float = 5.0

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
    """Internal helper."""
    g = _to_gray_float01(img)


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
