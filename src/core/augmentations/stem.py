# -*- coding: utf-8 -*-
"""
【作用概述】提供 StackDiff 训练数据管线使用的共享 STEM 图像增强函数，包括几何变换、carbon 背景、裁剪、翻转、扫描/泊松/高斯噪声、显示归一化和张量缩放；函数按调用方输入返回增强结果。
【关联说明】关联文件：src/core/datasets/augment_dataset.py、src/tools/synthetic_materials/source_runner.py、configs/train/*.yml。
【命令行用法】本模块由训练和 source 图像生成相关模块导入，不作为命令行入口直接执行。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any

import cv2
import numpy as np
from PIL import Image


_DEF_BORDER = cv2.BORDER_REFLECT_101
_DEF_E = 1.602176634e-19
_SUPPORTED_DISABLES = {
    "elastic",
    "perspective",
    "carbon",
    "rotate",
    "crop",
    "flip",
    "noise",
    "display",
    "edge_mask",
}


def to_float01(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.asarray(image)
    else:
        arr = np.asarray(image)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
        arr = np.clip(arr, 0.0, 1.0) if arr.max() <= 1.5 else np.clip(arr / 255.0, 0.0, 1.0)
    return arr


def gaussian_field(h: int, w: int, sigma: float) -> np.ndarray:
    g = np.random.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
    return cv2.GaussianBlur(g, (0, 0), sigmaX=sigma)


def elastic_warp_cv(img: np.ndarray, alpha: float, sigma: float, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    h, w = img.shape[:2]
    dx = gaussian_field(h, w, sigma)
    dy = gaussian_field(h, w, sigma)
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    dx *= alpha
    dy *= alpha
    map_x = x + dx
    map_y = y + dy
    return cv2.remap(img, map_x, map_y, interpolation=interpolation, borderMode=_DEF_BORDER)


def perspective_warp_cv(img: np.ndarray, max_ratio: float, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    j = float(max_ratio)
    jitter = np.random.uniform(-1.0, 1.0, size=(4, 2)).astype(np.float32)
    jitter[:, 0] *= j * w
    jitter[:, 1] *= j * h
    dst = src + jitter
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=interpolation, borderMode=_DEF_BORDER)


def rotate_cv(img: np.ndarray, angle: float, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle), 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=interpolation, borderMode=_DEF_BORDER)


def distance_transform(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0.5).astype(np.uint8)
    return cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)


def add_carbon_background(img01: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    I = img01.astype(np.float32)
    H, W = I.shape
    cover = float(cfg.get("cover", 0.7))
    alpha = float(cfg.get("alpha", 0.4))
    sigma_mask = float(cfg.get("sigma_mask", 100))
    sigma_field = float(cfg.get("sigma_field", 100))
    coarse_down = max(1, int(cfg.get("coarse_down", 100)))
    focus_enable = bool(cfg.get("focus_enable", True))
    focus_center = cfg.get("focus_center", (0.5, 0.5))
    if isinstance(focus_center, (list, tuple)):
        cx, cy = float(focus_center[0]), float(focus_center[1])
    else:
        cx = cy = 0.5
    focus_radius_ratio = float(cfg.get("focus_radius_ratio", 0.5))
    focus_gain = float(cfg.get("focus_gain", 5.0))
    morph_kernel = int(cfg.get("morph_kernel", 11))
    morph_iter = int(cfg.get("morph_iter", 1))


    B = np.random.random((H, W)).astype(np.float32) - 0.5
    B = cv2.GaussianBlur(B, (0, 0), sigmaX=sigma_field)
    B = (B - B.min()) / max(B.max() - B.min(), 1e-8)


    Hc, Wc = max(1, H // coarse_down), max(1, W // coarse_down)
    M_pre = np.random.random((Hc, Wc)).astype(np.float32)
    M_pre = cv2.resize(M_pre, (W, H), interpolation=cv2.INTER_LINEAR)
    M_pre = cv2.GaussianBlur(M_pre, (0, 0), sigmaX=sigma_mask)

    if focus_enable:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        xx /= max(W - 1, 1)
        yy /= max(H - 1, 1)
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        rad = max(1e-3, focus_radius_ratio)
        focus = np.exp(-0.5 * (rr / rad) ** 2).astype(np.float32)
        M_pre = M_pre + focus_gain * focus

    M_pre = (M_pre - M_pre.min()) / max(M_pre.max() - M_pre.min(), 1e-8)
    thr = np.quantile(M_pre, 1.0 - cover)
    M_bin = (M_pre > thr).astype(np.uint8)
    k = morph_kernel + (morph_kernel % 2 == 0)
    k = max(3, int(k))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    M_bin = cv2.morphologyEx(M_bin, cv2.MORPH_CLOSE, kernel, iterations=max(1, morph_iter))
    M = cv2.GaussianBlur(M_bin.astype(np.float32), (0, 0), sigmaX=sigma_mask)
    return np.clip(I + alpha * M * B, 0.0, 1.0)


def choose_crop(dist_map: np.ndarray, side: int, safe_radius: int) -> Tuple[int, int]:
    H, W = dist_map.shape
    half = side // 2
    if H <= 2 * half or W <= 2 * half:
        raise RuntimeError("Crop side is too large for the augmentation mask.")
    allowed = dist_map > float(safe_radius)
    border = np.zeros_like(allowed, dtype=bool)
    border[half:H - half, half:W - half] = True
    cand = np.argwhere(allowed & border)
    if cand.size == 0:

        allowed = dist_map > float(side / 2.0)
        cand = np.argwhere(allowed & border)
        if cand.size == 0:
            raise RuntimeError("No safe crop center was found for the requested crop side.")
    cy, cx = cand[np.random.randint(len(cand))]
    return int(cy), int(cx)


def crop_and_resize(arr: np.ndarray, cy: int, cx: int, side: int, out_size: int = 128) -> np.ndarray:
    half = side // 2
    y1, y2 = cy - half, cy + half
    x1, x2 = cx - half, cx + half
    patch = arr[y1:y2, x1:x2]
    return cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def random_flip(arr: np.ndarray, h_prob: float = 0.5, v_prob: float = 0.5) -> np.ndarray:
    if np.random.rand() < h_prob:
        arr = cv2.flip(arr, 1)
    if np.random.rand() < v_prob:
        arr = cv2.flip(arr, 0)
    return arr


def add_scan_noise_from_original(
    image: np.ndarray,
    *,
    pixel_size_A_orig: float,
    k: float,
    width_orig_px: int | None = None,
    dwell_time_scan_s_orig: float | None = None,
    retrace_time_s: float = 0.0,
    maintain_row_time: bool = True,
    sigma_jitter_A: float = 0.2,
    line_freq_hz: float = 60.0,
    phase_x: float = 0.0,
    phase_y: float = math.pi / 2,
) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    h, w_new = img.shape
    if width_orig_px is None:
        width_orig_px = int(round(w_new * k))
    pixel_size_A_new = pixel_size_A_orig * k
    sigma_px_new = float(sigma_jitter_A) / float(pixel_size_A_new)
    if dwell_time_scan_s_orig is None:
        dwell_time_scan_s_orig = 2e-6
    t_dwell_new = dwell_time_scan_s_orig * (width_orig_px / w_new) if maintain_row_time else dwell_time_scan_s_orig
    row_time = w_new * t_dwell_new + retrace_time_s
    rows = np.arange(h, dtype=np.float64)
    t_i = rows * row_time
    gx = np.random.normal(0.0, 1.0, size=h)
    gy = np.random.normal(0.0, 1.0, size=h)
    delta_x = gx * sigma_px_new * np.sin(2.0 * math.pi * line_freq_hz * t_i + phase_x)
    delta_y = gy * sigma_px_new * np.sin(2.0 * math.pi * line_freq_hz * t_i + phase_y)
    Y, X = np.indices((h, w_new), dtype=np.float32)
    map_x = X - delta_y[:, None].astype(np.float32)
    map_y = Y - delta_x[:, None].astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=_DEF_BORDER)


def add_poisson_noise_from_original(
    image: np.ndarray,
    *,
    beam_current_A: float,
    dwell_time_s: float,
    k: float,
    preserve_scale: bool = True,
    preserve_areal_dose: bool = True,
) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    vmin, vmax = float(img.min()), float(img.max())
    scale = vmax - vmin if vmax > vmin else 1.0
    rel = (img - vmin) / scale
    n_orig = dwell_time_s * beam_current_A / _DEF_E
    n_new = n_orig * (k * k) if preserve_areal_dose else n_orig
    n_new = max(n_new, 1e-12)
    counts = np.random.poisson(rel * n_new)
    frac = counts / n_new
    return (frac * scale + vmin).astype(np.float32) if preserve_scale else frac.astype(np.float32)


def add_gaussian_noise(image: np.ndarray, *, sigma: float = 0.01, preserve_scale: bool = True) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    if preserve_scale:
        vmin, vmax = float(img.min()), float(img.max())
        scale = vmax - vmin if vmax > vmin else 1.0
        noise = np.random.normal(0.0, sigma * scale, size=img.shape).astype(np.float32)
        out = np.clip(img + noise, vmin, vmax)
        return out
    noise = np.random.normal(0.0, sigma, size=img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)


def adjust_display_post_noise(
    disp01: np.ndarray,
    *,
    gain: float | None = None,
    bias: float | None = None,
    gamma: float | None = None,
    rng=np.random,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if gain is None:
        gain = rng.uniform(0.9, 1.1)
    if bias is None:
        bias = rng.uniform(-0.03, 0.03)
    if gamma is None:
        gamma = rng.uniform(0.95, 1.05)
    x = np.clip(disp01 * float(gain) + float(bias), 0.0, 1.0)
    x = np.power(np.clip(x, 1e-7, 1.0), float(gamma))
    return np.clip(x, 0.0, 1.0), {"gain": float(gain), "bias": float(bias), "gamma": float(gamma)}


def apply_edge_mask(
    image: np.ndarray, *, prob: float = 0.0005, bg_min: float = 0.0, bg_max: float = 0.05
) -> np.ndarray:
    """Randomly replace irregular edge regions with low-intensity background."""
    img = np.asarray(image, dtype=np.float32)
    h, w = img.shape[:2]


    directions = []
    for d in ['left', 'right', 'top', 'bottom']:
        if np.random.random() < prob:
            directions.append(d)

    if not directions:
        return img


    combined_mask = np.zeros((h, w), dtype=bool)

    for direction in directions:

        if direction == 'left':
            y_coords = np.linspace(0.1, 0.9, 4) * h
            x_coords = np.random.uniform(0.15, 0.30, size=4) * w

        elif direction == 'right':
            y_coords = np.linspace(0.1, 0.9, 4) * h
            x_coords = w - np.random.uniform(0.15, 0.30, size=4) * w

        elif direction == 'top':
            x_coords = np.linspace(0.1, 0.9, 4) * w
            y_coords = np.random.uniform(0.15, 0.30, size=4) * h

        else:  # bottom
            x_coords = np.linspace(0.1, 0.9, 4) * w
            y_coords = h - np.random.uniform(0.15, 0.30, size=4) * h


        if direction in ['left', 'right']:
            coeffs = np.polyfit(y_coords, x_coords, 3)
            a, b, c, d = coeffs
            yy, xx = np.indices((h, w), dtype=np.float32)
            x_boundary = a * yy**3 + b * yy**2 + c * yy + d
            mask = xx < x_boundary if direction == 'left' else xx > x_boundary

        else:  # top or bottom
            coeffs = np.polyfit(x_coords, y_coords, 3)
            a, b, c, d = coeffs
            yy, xx = np.indices((h, w), dtype=np.float32)
            y_boundary = a * xx**3 + b * xx**2 + c * xx + d
            mask = yy < y_boundary if direction == 'top' else yy > y_boundary

        combined_mask |= mask


    masked_img = img.copy()
    bg_values = np.random.uniform(bg_min, bg_max, size=int(combined_mask.sum())).astype(np.float32)
    if img.ndim == 2:
        masked_img[combined_mask] = bg_values
    else:
        masked_img[combined_mask, :] = bg_values[:, None]

    return masked_img


def _uniform(a: float, b: float) -> float:
    return float(np.random.uniform(a, b))


def _maybe_sample(value: Any) -> Any:
    """Sample a scalar from a two-value range or return a fixed value."""
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(x, (int, float)) for x in value):
        lo, hi = float(value[0]), float(value[1])
        return _uniform(lo, hi)
    return value


@dataclass
class AugmentConfig:
    image_size: int = 128

    cfg: Dict[str, Any] = None
    disabled: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        cfg = self.cfg or {}
        disable_value = cfg.get("disable")
        if disable_value is None:
            disable_value = cfg.get("disabled")
        if isinstance(disable_value, (str, bytes)):
            candidates = [disable_value]
        elif isinstance(disable_value, (list, tuple, set)):
            candidates = list(disable_value)
        else:
            candidates = []
        cleaned = {str(name).strip().lower() for name in candidates if str(name).strip()}
        if "all" in cleaned:
            self.disabled = set(_SUPPORTED_DISABLES)
        else:
            self.disabled = {name for name in cleaned if name in _SUPPORTED_DISABLES}
        self.cfg = cfg

    def is_enabled(self, name: str) -> bool:
        return name.lower() not in self.disabled

    def sample_once(self) -> Dict[str, Any]:
        """Sample one concrete augmentation parameter set from the config."""
        c = self.cfg or {}
        out: Dict[str, Any] = {}

        # elastic
        el = c.get("elastic", {})
        elastic_enabled = self.is_enabled("elastic")
        out["elastic.enabled"] = elastic_enabled
        if elastic_enabled:
            out["elastic.alpha_ratio"] = _maybe_sample(el.get("alpha_ratio", [0.0, 0.2]))
            out["elastic.sigma_ratio"] = el.get("sigma_ratio", 0.05)
        else:
            out["elastic.alpha_ratio"] = 0.0
            out["elastic.sigma_ratio"] = 0.0

        # perspective
        ps = c.get("perspective", {})
        perspective_enabled = self.is_enabled("perspective")
        out["perspective.enabled"] = perspective_enabled
        if perspective_enabled:
            out["perspective.max_ratio"] = _maybe_sample(ps.get("max_ratio", [0.0, 0.015]))
        else:
            out["perspective.max_ratio"] = 0.0

        # carbon
        cb = c.get("carbon", {})
        prob_val = _maybe_sample(cb.get("prob", 0.01))
        prob_val = float(np.clip(prob_val, 0.0, 1.0))
        carbon_enabled = self.is_enabled("carbon")
        out["carbon.prob"] = prob_val if carbon_enabled else 0.0
        if carbon_enabled:
            out["carbon.enabled"] = (np.random.rand() < prob_val)
        else:
            out["carbon.enabled"] = False
        out["carbon.cover"] = _maybe_sample(cb.get("cover", [0.3, 0.7]))
        out["carbon.alpha"] = _maybe_sample(cb.get("alpha", [0.3, 0.5]))
        out["carbon.sigma_mask"] = cb.get("sigma_mask", 100)
        out["carbon.sigma_field"] = cb.get("sigma_field", 100)
        out["carbon.coarse_down"] = cb.get("coarse_down", 100)
        out["carbon.focus_enable"] = cb.get("focus_enable", True)

        fc = cb.get("focus_center", [0.1, 0.9])
        if isinstance(fc, (list, tuple)) and len(fc) == 2:
            out["carbon.focus_center"] = (_uniform(fc[0], fc[1]), _uniform(fc[0], fc[1]))
        else:
            out["carbon.focus_center"] = (0.5, 0.5)
        out["carbon.focus_radius_ratio"] = _maybe_sample(cb.get("focus_radius_ratio", [0.3, 0.7]))
        out["carbon.focus_gain"] = cb.get("focus_gain", 5.0)
        out["carbon.morph_kernel"] = cb.get("morph_kernel", 11)
        out["carbon.morph_iter"] = cb.get("morph_iter", 1)

        # rotate
        rt = c.get("rotate", {})
        rotate_enabled = self.is_enabled("rotate")
        out["rotate.enabled"] = rotate_enabled
        if rotate_enabled:
            out["rotate.angle"] = _maybe_sample(rt.get("range", [0.0, 360.0]))
        else:
            out["rotate.angle"] = 0.0

        # crop
        cr = c.get("crop", {})
        crop_enabled = self.is_enabled("crop")
        out["crop.enabled"] = crop_enabled
        if crop_enabled:
            side_rng = cr.get("side", [372, 744])
            if isinstance(side_rng, (list, tuple)) and len(side_rng) == 2:
                side = int(round(_uniform(float(side_rng[0]), float(side_rng[1]))))
            else:
                side = int(side_rng)
            if side % 2 == 1:
                side += 1
            side = max(2, side)
            out["crop.side"] = int(side)
            safe_radius = cr.get("safe_radius", 372)
            out["crop.safe_radius"] = None if safe_radius is None else int(safe_radius)
        else:
            out["crop.side"] = None
            out["crop.safe_radius"] = None

        # flip
        fp = c.get("flip", {})
        flip_enabled = self.is_enabled("flip")
        out["flip.enabled"] = flip_enabled
        if flip_enabled:
            out["flip.h_prob"] = float(fp.get("h_prob", 0.5))
            out["flip.v_prob"] = float(fp.get("v_prob", 0.5))
        else:
            out["flip.h_prob"] = 0.0
            out["flip.v_prob"] = 0.0

        # noise
        nz = c.get("noise", {})
        noise_enabled = self.is_enabled("noise")
        out["noise.enabled"] = noise_enabled
        out["noise.pixel_size_A_orig"] = float(nz.get("pixel_size_A_orig", 0.12))
        out["noise.beam_current_A"] = float(nz.get("beam_current_A", 30e-12))
        out["noise.dwell_time_s"] = _maybe_sample(nz.get("dwell_time_s", [1e-6, 3e-6]))
        out["noise.width_orig_px"] = int(nz.get("width_orig_px", 3289))
        out["noise.sigma_jitter_A"] = _maybe_sample(nz.get("sigma_jitter_A", [0.0, 0.2]))
        out["noise.line_freq_hz"] = float(nz.get("line_freq_hz", 60.0))
        out["noise.phase_x"] = float(nz.get("phase_x", 0.0))
        out["noise.phase_y"] = float(nz.get("phase_y", math.pi / 2))
        out["noise.gaussian_sigma"] = _maybe_sample(nz.get("gaussian_sigma", [0.0, 0.1]))
        if not noise_enabled:
            out["noise.gaussian_sigma"] = 0.0


        ds = c.get("display", {})
        display_enabled = self.is_enabled("display")
        out["display.enabled"] = display_enabled
        if display_enabled:
            out["display.gain"] = _maybe_sample(ds.get("gain", [0.6, 1.1]))
            out["display.bias"] = _maybe_sample(ds.get("bias", [-0.1, 0.1]))
            out["display.gamma"] = _maybe_sample(ds.get("gamma", [0.9, 1.2]))
        else:
            out["display.gain"] = 1.0
            out["display.bias"] = 0.0
            out["display.gamma"] = 1.0


        em = c.get("edge_mask", {})
        edge_mask_enabled = self.is_enabled("edge_mask")
        out["edge_mask.enabled"] = edge_mask_enabled
        if edge_mask_enabled:
            out["edge_mask.prob"] = float(em.get("prob", 0.0005))
            out["edge_mask.bg_min"] = float(em.get("bg_min", 0.0))
            out["edge_mask.bg_max"] = float(em.get("bg_max", 0.05))
        else:
            out["edge_mask.prob"] = 0.0
            out["edge_mask.bg_min"] = 0.0
            out["edge_mask.bg_max"] = 0.0

        return out


class StemAugmentor:
    def __init__(
        self,
        *,
        mask01: np.ndarray,
        dist_map: np.ndarray | None,
        aug_cfg: AugmentConfig,
    ):
        self.mask01 = to_float01(mask01)
        if dist_map is not None:
            self.dist_map0 = np.asarray(dist_map, dtype=np.float32)
        else:
            self.dist_map0 = distance_transform(self.mask01)
        self.aug_cfg = aug_cfg

    def __call__(self, img: Image.Image | np.ndarray) -> np.ndarray:
        """Apply the configured STEM augmentation pipeline and return a 1xHxW tensor array in [-1, 1]."""
        img01 = to_float01(img)
        H, W = img01.shape[:2]
        params = self.aug_cfg.sample_once()


        if params.get("elastic.enabled", True):
            alpha = float(params.get("elastic.alpha_ratio", 0.0)) * min(H, W)
            sigma = float(params.get("elastic.sigma_ratio", 0.0)) * min(H, W)
            if alpha > 0.0 and sigma > 0.0:
                img01 = elastic_warp_cv(img01, alpha=alpha, sigma=sigma)
        if params.get("perspective.enabled", True):
            max_ratio = float(params.get("perspective.max_ratio", 0.0) or 0.0)
            if max_ratio > 0.0:
                img01 = perspective_warp_cv(img01, max_ratio)


        if bool(params["carbon.enabled"]) is True:
            img01 = add_carbon_background(img01, {
                "cover": params["carbon.cover"],
                "alpha": params["carbon.alpha"],
                "sigma_mask": params["carbon.sigma_mask"],
                "sigma_field": params["carbon.sigma_field"],
                "coarse_down": params["carbon.coarse_down"],
                "focus_enable": params["carbon.focus_enable"],
                "focus_center": params["carbon.focus_center"],
                "focus_radius_ratio": params["carbon.focus_radius_ratio"],
                "focus_gain": params["carbon.focus_gain"],
                "morph_kernel": params["carbon.morph_kernel"],
                "morph_iter": params["carbon.morph_iter"],
            })


        rotate_enabled = params.get("rotate.enabled", True)
        angle = float(params.get("rotate.angle", 0.0)) % 360.0
        if rotate_enabled and angle != 0.0:
            img_rot = rotate_cv(img01, angle)
            dist_rot = rotate_cv(self.dist_map0, angle, interpolation=cv2.INTER_NEAREST)
        else:
            img_rot = img01
            dist_rot = self.dist_map0


        crop_enabled = params.get("crop.enabled", True)
        if crop_enabled:
            side_param = params.get("crop.side")
            side = int(side_param) if side_param is not None else min(H, W)
            side = max(2, side - (side % 2))
            safe_radius_param = params.get("crop.safe_radius")
            safe_radius = int(safe_radius_param) if safe_radius_param is not None else 372
            cy, cx = choose_crop(dist_rot, side, safe_radius)
            patch = crop_and_resize(img_rot, cy, cx, side, out_size=int(self.aug_cfg.image_size))
            effective_side = side
        else:
            patch = cv2.resize(
                img_rot,
                (int(self.aug_cfg.image_size), int(self.aug_cfg.image_size)),
                interpolation=cv2.INTER_LINEAR,
            )
            effective_side = min(H, W)


        if params.get("flip.enabled", True):
            patch = random_flip(patch, params.get("flip.h_prob", 0.0), params.get("flip.v_prob", 0.0))


        if params.get("edge_mask.enabled", True):
            edge_prob = float(params.get("edge_mask.prob", 0.0))
            if edge_prob > 0.0:
                patch = apply_edge_mask(
                    patch,
                    prob=edge_prob,
                    bg_min=params.get("edge_mask.bg_min", 0.0),
                    bg_max=params.get("edge_mask.bg_max", 0.05),
                )


        k = 2.0 * float(effective_side) / float(self.aug_cfg.image_size)
        if params.get("noise.enabled", True):
            patch = add_scan_noise_from_original(
                patch,
                pixel_size_A_orig=params["noise.pixel_size_A_orig"],
                k=k,
                width_orig_px=params["noise.width_orig_px"],
                dwell_time_scan_s_orig=params["noise.dwell_time_s"],
                sigma_jitter_A=params["noise.sigma_jitter_A"],
                line_freq_hz=params["noise.line_freq_hz"],
                phase_x=params["noise.phase_x"],
                phase_y=params["noise.phase_y"],
            )
            patch = add_poisson_noise_from_original(
                patch,
                beam_current_A=params["noise.beam_current_A"],
                dwell_time_s=params["noise.dwell_time_s"],
                k=k,
            )
            patch = add_gaussian_noise(
                patch, sigma=float(params.get("noise.gaussian_sigma", 0.0)), preserve_scale=True
            )


        if params.get("display.enabled", True):
            patch, _ = adjust_display_post_noise(
                patch,
                gain=params.get("display.gain", None),
                bias=params.get("display.bias", None),
                gamma=params.get("display.gamma", None),
            )


        patch = np.clip(patch, 0.0, 1.0)
        patch = (patch.astype(np.float32) * 2.0 - 1.0)[None, ...]  # 1xHxW
        return patch
