# -*- coding: utf-8 -*-
"""
从 notebooks/data_augmentation_overview.ipynb 抽取的可复用数据增强与可视化工具函数。
仅包含函数与常量定义，不包含具体演示流程。
"""

import math
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

SEED = None  # 使用随机种子，每次运行结果不同
rng = np.random.default_rng(SEED)

_DEF_BORDER = cv2.BORDER_REFLECT101
_DEF_ELECTRON_CHARGE = 1.602176634e-19

def configure_cn_font():
    """为 Matplotlib 设置可显示中文的字体，避免出现方框。"""
    candidate_fonts = ['SimHei', 'Noto Sans CJK SC', 'Noto Sans CJK', 'Microsoft YaHei',
                      'PingFang SC', 'Source Han Sans CN', 'WenQuanYi Micro Hei', 'Droid Sans Fallback']
    for font_name in candidate_fonts:
        try:
            font_manager.fontManager.findfont(font_name, fallback_to_default=False)
        except ValueError:
            continue
        else:
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            print('使用 Matplotlib 字体:', font_name)
            return
    plt.rcParams['axes.unicode_minus'] = False
    print('未找到中文字体，继续使用默认字体（可能无法显示汉字）。')



def resolve_path(rel_path: str) -> str:
    here = Path.cwd()
    for candidate in (here / rel_path, here.parent / rel_path, here.parent.parent / rel_path):
        if candidate.exists():
            return str(candidate)
    return str((here / rel_path).resolve())


def to_float01(img) -> np.ndarray:
    arr = np.array(img, dtype=np.float32) if isinstance(img, Image.Image) else img.astype(np.float32, copy=False)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)


def gaussian_kernel_size(sigma: float) -> int:
    k = int(max(3, round(4 * float(sigma))))
    if k % 2 == 0:
        k += 1
    return k


def show_images(images, titles, cols=3, cmap='gray', figsize_per_col=4):
    total = len(images)
    rows = math.ceil(total / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per_col * cols, figsize_per_col * rows))
    axes = np.array(axes).reshape(rows, cols)
    for ax in axes.flat:
        ax.axis('off')
    for idx, (img, title) in enumerate(zip(images, titles)):
        ax = axes.flat[idx]
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap, vmin=0.0, vmax=1.0)
        else:
            ax.imshow(np.clip(img, 0.0, 1.0))
        ax.set_title(title, fontsize=11)
        ax.axis('off')
    for ax in axes.flat[total:]:
        ax.axis('off')
    plt.tight_layout()


def distance_transform(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask > 0.5).astype(np.uint8)
    return cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)


def elastic_deform_cv(img: np.ndarray, alpha_ratio: float, sigma_ratio: float, *, rng=rng, interpolation=cv2.INTER_LINEAR, field=None):
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
    deformed = cv2.remap(img, map_x, map_y, interpolation=interpolation, borderMode=_DEF_BORDER)
    return deformed, field


def perspective_warp_cv(img: np.ndarray, max_ratio: float, *, rng=rng, interpolation=cv2.INTER_LINEAR, matrix=None):
    h, w = img.shape[:2]
    if matrix is None:
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        jitter = rng.uniform(-1.0, 1.0, size=(4, 2)).astype(np.float32)
        jitter[:, 0] *= float(max_ratio) * w
        jitter[:, 1] *= float(max_ratio) * h
        dst = src + jitter
        matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, matrix, (w, h), flags=interpolation, borderMode=_DEF_BORDER)
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


def rotate_cv(img: np.ndarray, angle: float, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle), 1.0)
    return cv2.warpAffine(img, matrix, (w, h), flags=interpolation, borderMode=_DEF_BORDER)


def draw_crop_outline(img: np.ndarray, center, side: int) -> np.ndarray:
    cy, cx = center
    half = side // 2
    y1, y2 = max(0, cy - half), min(img.shape[0], cy + half)
    x1, x2 = max(0, cx - half), min(img.shape[1], cx + half)
    vis = np.repeat(img[..., None], 3, axis=-1).astype(np.float32)
    cv2.rectangle(vis, (x1, y1), (x2 - 1, y2 - 1), (1.0, 0.2, 0.2), thickness=2)
    return np.clip(vis, 0.0, 1.0)


def choose_crop(dist_map: np.ndarray, side: int, safe_radius: float, *, rng=rng):
    H, W = dist_map.shape
    half = side // 2
    if H <= 2 * half or W <= 2 * half:
        raise ValueError(f'图像尺寸 {H}x{W} 无法裁剪 side={side}')
    allowed = dist_map > float(safe_radius)
    mask_border = np.zeros_like(allowed, dtype=bool)
    mask_border[half:H - half, half:W - half] = True
    candidates = np.argwhere(allowed & mask_border)
    if candidates.size == 0:
        relaxed = dist_map > float(side / 2.0)
        candidates = np.argwhere(relaxed & mask_border)
        if candidates.size == 0:
            raise RuntimeError('未找到合法的裁剪中心，请检查掩膜或参数。')
    idx = int(rng.integers(0, len(candidates)))
    cy, cx = candidates[idx]
    return int(cy), int(cx)


def crop_and_resize(arr: np.ndarray, cy: int, cx: int, side: int, out_size: int = 128) -> np.ndarray:
    half = side // 2
    y1, y2 = cy - half, cy + half
    x1, x2 = cx - half, cx + half
    patch = arr[y1:y2, x1:x2]
    return cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def random_flip(arr: np.ndarray, h_prob: float = 0.5, v_prob: float = 0.5, *, rng=rng):
    result = arr.copy()
    applied = []
    if rng.random() < float(h_prob):
        result = cv2.flip(result, 1)
        applied.append('水平翻转')
    if rng.random() < float(v_prob):
        result = cv2.flip(result, 0)
        applied.append('垂直翻转')
    return result, applied


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


def apply_edge_mask(image: np.ndarray, *, prob=0.0005, bg_min=0.0, bg_max=0.05, rng=rng):
    """
    以指定概率在图像的多个方向组合施加三次函数形状的边缘遮罩。
    
    参数：
    - prob: 每个方向（上下左右）的独立应用概率，默认0.0005。
    - use_random_bg: 是否使用小随机值填充遮罩区域（模拟背景噪声），默认True。
    - mask_value: 当 use_random_bg=False 时使用的固定填充值，默认0.0。
    - rng: 随机数生成器。

    返回：
    - masked_img: 应用遮罩后的图像。
    - mask_info: 字典，包含遮罩信息（方向列表、各方向的采样点与系数）。
    
    改进：支持多方向组合遮罩（如左上角、右下角等）
    - 每个方向独立以 prob 概率采样
    - 多个方向的遮罩取并集
    - 左右遮罩：4个点沿竖直方向均匀分布，x坐标变化范围15%-30%
    - 上下遮罩：4个点沿水平方向均匀分布，y坐标变化范围15%-30%
    - 遮罩区域填充小随机值（0.0-0.05），后续噪声会叠加，模拟无原子但有背景信号的真实情况
    """
    img = np.asarray(image, dtype=np.float32)
    h, w = img.shape[:2]

    # 每个方向独立采样
    directions = []
    for d in ['left', 'right', 'top', 'bottom']:
        if rng.random() < prob:
            directions.append(d)

    if not directions:
        return img, {'directions': []}

    # 对每个方向生成遮罩并取并集
    combined_mask = np.zeros((h, w), dtype=bool)
    mask_details = {}

    for direction in directions:
        # 根据方向选择四个点并构建三次函数
        if direction == 'left':
            y_coords = np.linspace(0.1, 0.9, 4) * h
            x_coords = rng.uniform(0.15, 0.30, size=4) * w

        elif direction == 'right':
            y_coords = np.linspace(0.1, 0.9, 4) * h
            x_coords = w - rng.uniform(0.15, 0.30, size=4) * w

        elif direction == 'top':
            x_coords = np.linspace(0.1, 0.9, 4) * w
            y_coords = rng.uniform(0.15, 0.30, size=4) * h

        else:  # bottom
            x_coords = np.linspace(0.1, 0.9, 4) * w
            y_coords = h - rng.uniform(0.15, 0.30, size=4) * h

        # 拟合三次函数：ax^3 + bx^2 + cx + d
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

        combined_mask |= mask  # 并集
        mask_details[direction] = {
            'points': list(zip(x_coords, y_coords)),
            'coeffs': coeffs.tolist(),
        }

    # 应用组合遮罩：每个像素独立采样背景值
    masked_img = img.copy()
    bg_values = rng.uniform(bg_min, bg_max, size=int(combined_mask.sum())).astype(np.float32)
    if img.ndim == 2:
        masked_img[combined_mask] = bg_values
    else:
        masked_img[combined_mask, :] = bg_values[:, None]

    mask_info = {
        'directions': directions,
        'details': mask_details,
    }

    return masked_img, mask_info


