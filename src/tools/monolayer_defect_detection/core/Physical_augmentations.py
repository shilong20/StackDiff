# PhysicalAugmentations.py

import math
import cv2
import numpy as np


SEED = None
rng = np.random.default_rng(SEED)

_DEF_BORDER = cv2.BORDER_REFLECT101
_DEF_ELECTRON_CHARGE = 1.602176634e-19


def gaussian_kernel_size(sigma: float) -> int:
    k = int(max(3, round(4 * float(sigma))))
    if k % 2 == 0:
        k += 1
    return k



def elastic_deform_cv(img: np.ndarray, alpha_ratio: float, sigma_ratio: float, *, rng, interpolation=cv2.INTER_LINEAR, field=None):
    h, w = img.shape[:2]
    short = min(h, w)
    alpha = max(0.1, float(alpha_ratio) * short)
    sigma = max(1.0, float(sigma_ratio) * short)
    if field is None:
        k = gaussian_kernel_size(sigma)
        dx = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
        dy = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
        dx = cv2.GaussianBlur(dx, (k, k), sigmaX=sigma) * alpha
        dy = cv2.GaussianBlur(dy, (k, k), sigmaX=sigma) * alpha
        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = grid_x + dx
        map_y = grid_y + dy
        field = (map_x, map_y)
    else:
        map_x, map_y = field
    deformed = cv2.remap(img, map_x, map_y, interpolation=interpolation, borderMode=cv2.BORDER_CONSTANT)
    # print("dx max:", dx.max(), "dy max:", dy.max())
    return deformed, field



def perspective_warp_cv(img: np.ndarray, max_ratio: float, *, rng, interpolation=cv2.INTER_LINEAR, matrix=None):
    h, w = img.shape[:2]
    if matrix is None:
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        jitter = rng.uniform(-1.0, 1.0, size=(4, 2)).astype(np.float32)
        jitter[:, 0] *= float(max_ratio) * w
        jitter[:, 1] *= float(max_ratio) * h
        dst = src + jitter
        matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, matrix, (w, h), flags=interpolation, borderMode=cv2.BORDER_CONSTANT)
    return warped, matrix



def add_carbon_background(img: np.ndarray, cfg: dict, *, rng=rng) -> np.ndarray:
    I = img.astype(np.float32, copy=False)
    H, W = I.shape
    cover = float(cfg['cover'])
    alpha = float(cfg['alpha'])
    sigma_mask = float(cfg['sigma_mask'])
    sigma_field = float(cfg['sigma_field'])
    coarse_down = max(1, int(cfg['coarse_down']))
    focus_enable = bool(cfg['focus_enable'])
    focus_center = tuple(map(float, cfg['focus_center']))
    focus_radius_ratio = float(cfg['focus_radius_ratio'])
    focus_gain = float(cfg['focus_gain'])
    morph_kernel = int(cfg['morph_kernel'])
    morph_iter = int(cfg['morph_iter'])

    B = rng.random((H, W), dtype=np.float32) - 0.5
    B = cv2.GaussianBlur(B, (0, 0), sigmaX=sigma_field)
    B = (B - B.min()) / max(B.max() - B.min(), 1e-8)

    Hc = max(1, H // coarse_down)
    Wc = max(1, W // coarse_down)
    M_pre = rng.random((Hc, Wc), dtype=np.float32)
    M_pre = cv2.resize(M_pre, (W, H), interpolation=cv2.INTER_LINEAR)
    M_pre = cv2.GaussianBlur(M_pre, (0, 0), sigmaX=sigma_mask)

    if focus_enable:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        xx /= max(W - 1, 1)
        yy /= max(H - 1, 1)
        cx, cy = focus_center
        radius = max(1e-3, focus_radius_ratio)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        focus = np.exp(-0.5 * (dist / radius) ** 2).astype(np.float32)
        M_pre = M_pre + focus_gain * focus

    M_pre = (M_pre - M_pre.min()) / max(M_pre.max() - M_pre.min(), 1e-8)
    threshold = np.quantile(M_pre, 1.0 - cover)
    M_bin = (M_pre > threshold).astype(np.uint8)
    k = morph_kernel + (morph_kernel % 2 == 0)
    k = max(3, k)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    M_bin = cv2.morphologyEx(M_bin, cv2.MORPH_CLOSE, kernel, iterations=max(1, morph_iter))
    M = cv2.GaussianBlur(M_bin.astype(np.float32), (0, 0), sigmaX=sigma_mask)

    return np.clip(I + alpha * M * B, 0.0, 1.0)



def add_scan_noise_from_original(image: np.ndarray, *, pixel_size_A_orig, k, width_orig_px=None,
                                 dwell_time_scan_s_orig=None, retrace_time_s=0.0, maintain_row_time=True,
                                 sigma_jitter_A=0.2, line_freq_hz=60.0, phase_x=0.0, phase_y=math.pi / 2, rng=rng):
    img = np.asarray(image, dtype=np.float32)
    h, w_new = img.shape
    if width_orig_px is None:
        width_orig_px = int(round(w_new * k))
    if dwell_time_scan_s_orig is None:
        dwell_time_scan_s_orig = 2e-6
    pixel_size_A_new = float(pixel_size_A_orig) * k
    sigma_px_new = float(sigma_jitter_A) / max(pixel_size_A_new, 1e-8)
    t_dwell_new = dwell_time_scan_s_orig * (width_orig_px / w_new) if maintain_row_time else dwell_time_scan_s_orig
    row_time = w_new * t_dwell_new + float(retrace_time_s)
    rows = np.arange(h, dtype=np.float64)
    t_i = rows * row_time
    gx = rng.normal(0.0, 1.0, size=h)
    gy = rng.normal(0.0, 1.0, size=h)
    delta_x = gx * sigma_px_new * np.sin(2.0 * math.pi * float(line_freq_hz) * t_i + float(phase_x))
    delta_y = gy * sigma_px_new * np.sin(2.0 * math.pi * float(line_freq_hz) * t_i + float(phase_y))
    Y, X = np.indices((h, w_new), dtype=np.float32)
    map_x = X - delta_y[:, None].astype(np.float32)
    map_y = Y - delta_x[:, None].astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=_DEF_BORDER)



def add_poisson_noise_from_original(image: np.ndarray, *, beam_current_A, dwell_time_s, k,
                                   preserve_scale=True, preserve_areal_dose=True, rng=rng):
    img = np.asarray(image, dtype=np.float32)
    vmin, vmax = float(img.min()), float(img.max())
    scale = vmax - vmin if vmax > vmin else 1.0
    rel = (img - vmin) / scale if preserve_scale else img
    dwell_time_s = float(dwell_time_s)
    beam_current_A = float(beam_current_A)
    n_orig = dwell_time_s * beam_current_A / _DEF_ELECTRON_CHARGE
    n_new = n_orig * (k * k) if preserve_areal_dose else n_orig
    n_new = max(n_new, 1e-8)
    lam = np.clip(rel * n_new, 0.0, None)
    counts = rng.poisson(lam).astype(np.float32)
    frac = counts / n_new
    if preserve_scale:
        return (frac * scale + vmin).astype(np.float32)
    return np.clip(frac, 0.0, 1.0).astype(np.float32)



def add_gaussian_noise(image: np.ndarray, *, sigma=0.01, preserve_scale=True, rng=rng):
    img = np.asarray(image, dtype=np.float32)
    if preserve_scale:
        vmin, vmax = float(img.min()), float(img.max())
        scale = vmax - vmin if vmax > vmin else 1.0
        noise = rng.normal(0.0, float(sigma) * scale, size=img.shape).astype(np.float32)
        return np.clip(img + noise, vmin, vmax)
    noise = rng.normal(0.0, float(sigma), size=img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)


def adjust_display_post_noise(image: np.ndarray, *, gain=None, bias=None, gamma=None, rng=rng):
    if gain is None:
        gain = rng.uniform(0.9, 1.1)
    if bias is None:
        bias = rng.uniform(-0.03, 0.03)
    if gamma is None:
        gamma = rng.uniform(0.95, 1.05)
    x = np.clip(image * float(gain) + float(bias), 0.0, 1.0)
    x = np.power(np.clip(x, 1e-7, 1.0), float(gamma))
    return np.clip(x, 0.0, 1.0), {'gain': float(gain), 'bias': float(bias), 'gamma': float(gamma)}