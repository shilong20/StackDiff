"""
【作用概述】通过结构张量（Structure Tensor / Second-moment Matrix）在单层实验图中估计 Re 链的主轴方向，
用于消除 da/db 约 60° 的系统性误选。
核心输入/输出：
- 输入：单层灰度图（H×W，uint8/float32 均可；默认假设 Re 原子为亮点、链条呈现各向异性纹理）。
- 输出：
  - `chain_unit_xy`：主轴单位向量 (vx,vy)（图像坐标 x-right/y-down；轴向，v 与 -v 等价）
  - `angle_deg`：主轴角度（度，[-180,180)，来自 atan2(vy,vx)）
  - `dominance`：方向直方图的主峰占比（0..1），越大表示“单一方向更占主导”，越小表示多模态/各向同性。
    说明：该指标用于门控（是否使用链方向约束），能避免“两个方向各占一半时取均值”的错误。

新增（更稳健的链方向仲裁）：
- 在结构张量得到 2~3 个候选主方向后，使用“投影周期测量法（Projection Periodicity Test）”做二次判别：
  - 将图像按候选角度旋转，使候选方向水平；
  - 对旋转后图像沿 x 方向求和得到 1D profile(y)；
  - 对 profile 做 1D 自相关，提取“行距/周期”（第一个显著主峰 lag）；
  - 选择“周期最大”的方向作为 Re 链方向（直观上：沿正确链方向看，链与链之间间隔最大）。
该方法主要用于解决结构张量在部分样本上误选到 ±60° 等等价周期方向的问题。

新增（点云法链方向估计）：
- 对部分仿真/低噪声数据，图像纹理各向异性不明显时，结构张量/投影法可能不稳定；
- 可改用 `detect_atoms_from_image` 提取的 Re 原子点云，在最近邻位移向量上做二阶矩（PCA）估计链轴方向，
  该方法不依赖图像灰度纹理，只依赖点云几何结构。

新增（Hough/Radon 风格的 top-k 峰选取）：
- 有些真实图会出现“两个看似合理的链方向（相差 ~60°）”，此时不应对角度做平均；
- `get_chain_direction_hough_peaks` 基于 Canny + HoughLines 检测直线方向，并在角度直方图里做 WTA（取最高峰），
  能更直接地实现“只选最强峰、忽略次峰”的策略（依赖 OpenCV `cv2`）。
- `get_chain_direction_point_radon_peaks` 基于“点集 Radon/Hough”思想：先把亮点原子提成点云，然后对每个法向角 θ 计算
  r = x cosθ + y sinθ 的 1D 直方图并用“峰值集中度”打分，最后取得分曲线的 top-k 峰并做 WTA。
  该实现更贴近你描述的“sinogram 上取 top-k 孤立峰”，且对“原子是圆形亮点、缺少连续边缘”的情况比 HoughLines 更稳。

实现要点：
- 对图像做 GaussianBlur（sigma）抑制噪声；
- 用 Sobel 得到梯度 gx,gy；
- 构造结构张量分量 Jxx=gx^2, Jyy=gy^2, Jxy=gx*gy，并做更大尺度的平滑；
- 在每个像素计算“局部主轴方向”并做方向直方图统计：
  - 方向是轴向（θ 与 θ+π 等价），因此直方图定义在 [0,π)；
  - 对直方图做轻微平滑后取 WTA（Winner-Takes-All）峰值作为全局链方向；
  - 可选用像素权重（trace*coherence）抑制背景与弱纹理区域。

【关联说明】文件/模块：tools/pred_dadb/vis_scripts/vis_chain_hough_on_synth.py（可视化/对比链方向估计）。
【命令行用法】本文件不直接运行（由批处理脚本调用）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage, signal
from scipy.spatial import cKDTree

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


@dataclass(frozen=True)
class ChainDirectionConfig:
    # 预平滑 sigma（像素）：越大越关注“长尺度链方向”，越小越受原子点局部影响
    sigma_px: float = 3.0
    # 结构张量平滑尺度倍数（相对 sigma_px）
    tensor_sigma_mul: float = 3.0
    # 方向直方图 bins（定义在 [0,π)）
    hist_bins: int = 60  # 3°/bin
    # 对直方图做 circular 平滑（单位：bin）
    hist_smooth_sigma_bins: float = 1.0
    # 只统计纹理强度位于该分位数以上的像素（trace percentile），抑制背景
    trace_percentile: float = 70.0
    # 只统计局部 coherence（discr/trace）高于该阈值的像素（抑制各向同性区域）
    min_local_coherence: float = 0.2


@dataclass(frozen=True)
class ProjectionPeriodicityConfig:
    # 从结构张量直方图中取多少个候选峰（再对每个峰生成 ±60° 备选）
    topk_candidates: int = 3
    # 候选峰之间的最小间隔（bin），避免取到同一峰附近的邻居
    min_sep_bins: int = 6

    # 旋转参数（ndimage.rotate）
    rotate_order: int = 1
    rotate_mode: str = "constant"
    rotate_cval: float = 0.0

    # 投影前的高通/背景抑制（像素 sigma；0 表示关闭）
    # 经验上对 HighT1 更稳：强调“亮点原子”，减少慢变背景对 profile 的干扰。
    hp_sigma_px: float = 6.0

    # 1D profile 与 autocorr 平滑
    profile_smooth_sigma: float = 1.0
    autocorr_smooth_sigma: float = 1.0

    # 自相关主峰搜索范围（像素 lag）
    # 该范围只用于“找第一个显著周期”，不要求精确；过宽会让噪声的远距峰干扰判别。
    min_lag_px: int = 4
    max_lag_px: int = 80

    # 峰值判定：prominence 相对阈值（相对于 autocorr[0]）
    min_peak_prom_rel: float = 0.05
    # 峰值间距（避免把噪声小峰当周期）
    peak_distance_px: int = 4

    # 当候选之间很接近时，用 score=period*peak_val 仲裁（更稳健），而非只看 period
    use_strength_weight: bool = True

    # 若候选之间区分度不足，则启用 0~180 的粗扫描作为兜底（更慢但更稳）
    enable_bruteforce_fallback: bool = True
    fallback_dom_threshold: float = 0.06
    fallback_step_deg: float = 2.0
    fallback_refine_deg: float = 2.0
    fallback_refine_step_deg: float = 0.5


@dataclass(frozen=True)
class HoughChainDirectionConfig:
    """
    基于 Canny + HoughLines 的链方向估计配置。

    说明：
    - 该方法的目标是“从线检测结果里取 top-k 峰”，而不是对角度做均值，从而避免 0°/60° 双峰时被均值拖偏。
    - HoughLines 的返回结果通常按累加器强度排序；我们只取前 topk_lines 并做角度直方图 WTA。
    """

    # 预平滑：抑制噪声但保留链纹理（像素 sigma）
    blur_sigma_px: float = 1.5

    # Canny 阈值：若为 None 则按图像中位数自适应
    canny_low: Optional[int] = None
    canny_high: Optional[int] = None
    canny_auto_sigma: float = 0.33  # 仅在 low/high 为 None 时使用

    # Hough 参数
    hough_rho: float = 1.0
    hough_theta_deg: float = 1.0
    hough_threshold: int = 60
    topk_lines: int = 200

    # 角度直方图 bin 宽度（度，定义在 [0,180)）
    angle_bin_deg: float = 2.0


@dataclass(frozen=True)
class PointRadonChainDirectionConfig:
    """
    点集 Radon/Hough 峰值法（推荐替代纯 HoughLines）：
    - 先用与原子提取一致的方式把亮点提成点云；
    - 对每个法向角 θ（[0,180)）计算 r = x cosθ + y sinθ 的 1D 直方图；
    - 用“集中度/peakiness”（例如 sum(hist^2)）作为该 θ 的得分；
    - 在得分曲线上取 top-k 峰（WTA 取最高峰）得到主方向。
    """

    # 角度扫描步长（度）；越小越精但越慢
    angle_step_deg: float = 0.5
    # r 直方图 bin 宽（像素）；通常 1px 即可
    r_bin_px: float = 1.0
    # 取 top-k 峰用于诊断/双峰情况
    topk_peaks: int = 5
    # 峰之间最小间隔（度），避免同一峰附近重复
    min_sep_deg: float = 8.0

    # 原子点提取配置（默认与 HighT1 实验图一致；仿真图可把 use_highpass 关掉）
    atom_use_highpass: bool = True
    atom_bg_sigma_px: float = 6.0
    atom_bina_thre: float = 1.8
    atom_min_area_threshold: float = 100.0
    atom_min_distance_px: float = 5.0
    atom_max_points: int = 512


@dataclass(frozen=True)
class ChainDirectionFromPointsConfig:
    # 每个点查询多少近邻（包含自身时为 k+1）
    knn_k: int = 8
    # 距离窗口：若提供 sideA_A，则用 bond_A 推算窗口；否则用最近邻距离中位数估计窗口
    bond_A: float = 2.82
    bond_tol_ratio: float = 0.25
    # 若未提供 sideA_A，则窗口以 d_med 为中心：[(1-tol)*d_med, (1+tol)*d_med]
    fallback_tol_ratio: float = 0.35
    # 距离权重（强调短键）
    weight_power: float = 1.0  # w = 1/(r^p)
    min_edges: int = 40


def _to_gray_float(img: np.ndarray) -> np.ndarray:
    x = np.asarray(img)
    if x.ndim == 3 and x.shape[2] >= 3:
        x = x[..., :3].astype(np.float32)
        x = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    x = x.astype(np.float32)
    return x


def get_chain_direction_hough_peaks(
    image: np.ndarray,
    *,
    cfg: HoughChainDirectionConfig = HoughChainDirectionConfig(),
) -> Tuple[np.ndarray, float, float, Dict]:
    """
    用 Canny 边缘 + HoughLines 估计链方向，输出格式与其它方法一致：
    (chain_unit_xy, angle_deg, dominance, debug)。

    角度定义：
    - 输出 angle_deg 为“轴向角”（[0,180)），表示链方向本身；v 与 -v 等价。
    - OpenCV HoughLines 返回 theta 为“法向角”，因此链方向角 = theta - 90°（再 mod 180）。
    """
    if cv2 is None:
        raise RuntimeError("cv2 not available; cannot use get_chain_direction_hough_peaks()")

    img = _to_gray_float(image)
    v0 = float(np.min(img))
    v1 = float(np.max(img))
    if v1 - v0 > 1e-9:
        img = (img - v0) / (v1 - v0)
    img = img.astype(np.float32)

    if float(cfg.blur_sigma_px) > 0.0:
        img = ndimage.gaussian_filter(img, sigma=float(cfg.blur_sigma_px)).astype(np.float32)

    # OpenCV expects uint8
    u8 = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)

    if cfg.canny_low is None or cfg.canny_high is None:
        med = float(np.median(u8))
        sig = float(cfg.canny_auto_sigma)
        low = int(max(0.0, (1.0 - sig) * med))
        high = int(min(255.0, (1.0 + sig) * med))
    else:
        low = int(cfg.canny_low)
        high = int(cfg.canny_high)
    if high <= low:
        high = min(255, low + 1)

    edges = cv2.Canny(u8, low, high)

    theta = np.deg2rad(float(cfg.hough_theta_deg))
    lines = cv2.HoughLines(edges, float(cfg.hough_rho), float(theta), int(cfg.hough_threshold))

    dbg: Dict = {
        "method": "hough_peaks",
        "canny_low": int(low),
        "canny_high": int(high),
        "hough_threshold": int(cfg.hough_threshold),
        "n_lines_raw": 0,
    }

    if lines is None or len(lines) == 0:
        # No line evidence; return a default with zero confidence.
        z = np.array([1.0, 0.0], dtype=np.float32)
        dbg["reason"] = "no_lines"
        return z, 0.0, 0.0, dbg

    lines = lines.reshape(-1, 2)
    dbg["n_lines_raw"] = int(lines.shape[0])
    n = int(min(int(cfg.topk_lines), lines.shape[0]))
    # HoughLines typically returns strongest first; treat top-k as peaks.
    lines = lines[:n]

    # Convert to axial chain direction angles in [0,180)
    theta_norm = lines[:, 1].astype(np.float64)  # normal angle (rad), [0,pi)
    ang_chain = (theta_norm - 0.5 * np.pi) % np.pi
    ang_deg = np.degrees(ang_chain).astype(np.float64)  # [0,180)

    bin_w = float(max(0.5, cfg.angle_bin_deg))
    nb = int(max(12, round(180.0 / bin_w)))
    edges_deg = np.linspace(0.0, 180.0, nb + 1, dtype=np.float64)
    hist, _ = np.histogram(ang_deg, bins=edges_deg)
    hist = hist.astype(np.float64)

    peak = int(np.argmax(hist))
    peak_count = float(hist[peak])
    total = float(np.sum(hist))
    dominance = 0.0 if total <= 0 else float(peak_count / total)
    theta_peak = float((edges_deg[peak] + edges_deg[peak + 1]) * 0.5)  # degrees, [0,180)

    rad = np.deg2rad(theta_peak)
    v = np.array([np.cos(rad), np.sin(rad)], dtype=np.float32)  # image coords x-right/y-down

    # secondary peak info (for diagnosing bimodality)
    hist2 = hist.copy()
    hist2[peak] = -1.0
    peak2 = int(np.argmax(hist2))
    peak2_count = float(hist[peak2])
    theta_peak2 = float((edges_deg[peak2] + edges_deg[peak2 + 1]) * 0.5)

    dbg.update(
        {
            "topk_lines_used": int(n),
            "angle_bin_deg": float(bin_w),
            "hist": hist.astype(np.float32),
            "peak_theta_deg": float(theta_peak),
            "peak_count": float(peak_count),
            "peak2_theta_deg": float(theta_peak2),
            "peak2_count": float(peak2_count),
            "dominance": float(dominance),
        }
    )
    return v, float(theta_peak), float(dominance), dbg


def get_chain_direction_point_radon_peaks(
    image: np.ndarray,
    *,
    cfg: PointRadonChainDirectionConfig = PointRadonChainDirectionConfig(),
) -> Tuple[np.ndarray, float, float, Dict]:
    """
    点集 Radon/Hough 峰值法估计链方向（轴向），输出格式与其它方法一致：
    (chain_unit_xy, angle_deg, dominance, debug)。

    说明：
    - 得分曲线的 θ 是“法向角”（与 Radon/Hough 定义一致）；最终链方向角 = θ - 90°（再 mod 180）。
    - 该方法与 `get_chain_direction_hough_peaks` 的区别是：不依赖边缘上的长直线，而是依赖“点列共线”的统计峰值，
      因此对原子呈圆形亮点、链条在图像中不连续的情况更稳。
    """
    # local import to avoid circular deps
    from tools.pred_dadb.atoms import AtomDetectConfig, detect_atoms_from_image  # noqa: WPS433

    atom_cfg = AtomDetectConfig(
        use_highpass=bool(cfg.atom_use_highpass),
        bg_sigma_px=float(cfg.atom_bg_sigma_px),
        bina_thre=float(cfg.atom_bina_thre),
        min_area_threshold=float(cfg.atom_min_area_threshold),
        min_distance_px=float(cfg.atom_min_distance_px),
        max_points=int(cfg.atom_max_points),
    )
    pts, dbg_atom = detect_atoms_from_image(image, cfg=atom_cfg, return_debug=True)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)

    dbg: Dict = {
        "method": "point_radon_peaks",
        "n_points": int(pts.shape[0]),
        "angle_step_deg": float(cfg.angle_step_deg),
        "r_bin_px": float(cfg.r_bin_px),
    }
    if dbg_atom is not None:
        dbg["atom_detect"] = {
            "thr": float(dbg_atom.get("thr", 0.0)),
            "area_thr": float(dbg_atom.get("area_thr", 0.0)),
            "n_contours": int(dbg_atom.get("n_contours", 0)),
        }

    if pts.shape[0] < 8:
        z = np.array([1.0, 0.0], dtype=np.float32)
        dbg["reason"] = "too_few_points"
        return z, 0.0, 0.0, dbg

    # center coordinates for numeric stability
    H, W = int(image.shape[0]), int(image.shape[1])
    cx = 0.5 * float(W - 1)
    cy = 0.5 * float(H - 1)
    xy = pts.copy()
    xy[:, 0] -= cx
    xy[:, 1] -= cy

    step = float(max(0.1, cfg.angle_step_deg))
    thetas_deg = np.arange(0.0, 180.0, step, dtype=np.float64)
    thetas = np.deg2rad(thetas_deg).astype(np.float64)

    # compute r hist peakiness score per theta
    bin_w = float(max(0.25, cfg.r_bin_px))
    # r range roughly within [-diag/2, diag/2]
    diag = float(np.hypot(W, H))
    r_max = 0.55 * diag
    n_bins = int(max(64, np.ceil((2.0 * r_max) / bin_w)))
    scores = np.zeros_like(thetas_deg, dtype=np.float64)

    x = xy[:, 0].astype(np.float64)
    y = xy[:, 1].astype(np.float64)
    for i, th in enumerate(thetas):
        ct = float(np.cos(th))
        st = float(np.sin(th))
        r = x * ct + y * st
        # histogram over fixed range for comparability
        hist, _ = np.histogram(r, bins=n_bins, range=(-r_max, r_max))
        hist = hist.astype(np.float64)
        # peakiness: larger when points concentrate on fewer r (i.e., stronger line families)
        scores[i] = float(np.sum(hist * hist))

    # find top-k local maxima with circular distance constraint
    def _circ_dist(a: int, b: int, n: int) -> int:
        d = abs(int(a) - int(b))
        return min(d, n - d)

    nT = int(scores.size)
    order = np.argsort(-scores)
    min_sep_bins = int(max(1, round(float(cfg.min_sep_deg) / step)))
    peaks = []
    for idx in order.tolist():
        ok = True
        for j in peaks:
            if _circ_dist(idx, j, nT) < min_sep_bins:
                ok = False
                break
        if ok:
            peaks.append(int(idx))
            if len(peaks) >= int(cfg.topk_peaks):
                break
    if not peaks:
        z = np.array([1.0, 0.0], dtype=np.float32)
        dbg["reason"] = "no_peaks"
        return z, 0.0, 0.0, dbg

    best = int(peaks[0])
    best_score = float(scores[best])
    second_score = float(scores[peaks[1]]) if len(peaks) > 1 else 0.0
    dominance = 0.0 if best_score <= 1e-12 else float(max(0.0, (best_score - second_score) / best_score))

    theta_norm_deg = float(thetas_deg[best])  # normal angle
    theta_chain_deg = (theta_norm_deg - 90.0) % 180.0  # chain axis angle
    rad = np.deg2rad(theta_chain_deg)
    v = np.array([np.cos(rad), np.sin(rad)], dtype=np.float32)

    dbg.update(
        {
            "theta_norm_deg": float(theta_norm_deg),
            "theta_chain_deg": float(theta_chain_deg),
            "dominance": float(dominance),
            "scores_deg": thetas_deg.astype(np.float32),
            "scores": scores.astype(np.float32),
            "peak_indices": [int(p) for p in peaks],
            "peak_theta_norm_deg": [float(thetas_deg[p]) for p in peaks],
            "peak_scores": [float(scores[p]) for p in peaks],
        }
    )
    return v, float(theta_chain_deg), float(dominance), dbg


def get_chain_direction_structure_tensor(
    image: np.ndarray,
    *,
    cfg: ChainDirectionConfig = ChainDirectionConfig(),
) -> Tuple[np.ndarray, float, float, Dict]:
    """
    返回 (chain_unit_xy, angle_deg, dominance, debug)。
    - chain_unit_xy: (2,) float32 in (vx,vy)，图像坐标（y-down）
    """
    img = _to_gray_float(image)
    # normalize to [0,1] (robust enough for these tiles)
    v0 = float(np.min(img))
    v1 = float(np.max(img))
    if v1 - v0 > 1e-9:
        img = (img - v0) / (v1 - v0)
    img = img.astype(np.float32)

    sigma = float(cfg.sigma_px)
    if sigma > 0.0:
        img_blur = ndimage.gaussian_filter(img, sigma=sigma).astype(np.float32)
    else:
        img_blur = img

    # gradients in image coords: axis=1 is x, axis=0 is y (y-down)
    gx = ndimage.sobel(img_blur, axis=1).astype(np.float64)
    gy = ndimage.sobel(img_blur, axis=0).astype(np.float64)

    Jxx = gx * gx
    Jyy = gy * gy
    Jxy = gx * gy

    sigma_t = max(0.0, float(cfg.tensor_sigma_mul) * max(sigma, 1e-6))
    if sigma_t > 0.0:
        Jxx_s = ndimage.gaussian_filter(Jxx, sigma=sigma_t).astype(np.float64)
        Jyy_s = ndimage.gaussian_filter(Jyy, sigma=sigma_t).astype(np.float64)
        Jxy_s = ndimage.gaussian_filter(Jxy, sigma=sigma_t).astype(np.float64)
    else:
        Jxx_s, Jyy_s, Jxy_s = Jxx, Jyy, Jxy

    # per-pixel coherence: discr/trace
    trace = (Jxx_s + Jyy_s).astype(np.float64)
    diff = (Jxx_s - Jyy_s).astype(np.float64)
    discr = np.sqrt(diff * diff + 4.0 * Jxy_s * Jxy_s).astype(np.float64)
    local_coh = (discr / np.maximum(trace, 1e-12)).astype(np.float64)  # (H,W), 0..1

    # tangent orientation (axis) in [0,pi):
    # angle_normal = 0.5*atan2(2Jxy, Jxx-Jyy); tangent = angle_normal + pi/2
    ang_norm = 0.5 * np.arctan2(2.0 * Jxy_s, diff)  # [-pi/2,pi/2]
    ang_tan = ang_norm + 0.5 * np.pi
    ang_tan = np.mod(ang_tan, np.pi)  # axial

    # histogram mask + weights
    tp = float(cfg.trace_percentile)
    tr_thr = float(np.percentile(trace, tp))
    m = (trace >= tr_thr) & (local_coh >= float(cfg.min_local_coherence))
    if not np.any(m):
        # fallback：只用 trace 阈值
        m = trace >= tr_thr
    if not np.any(m):
        # extreme fallback：全图
        m = np.ones_like(trace, dtype=bool)

    w = (trace * local_coh).astype(np.float64)
    w = np.where(m, w, 0.0)
    total_w = float(np.sum(w))

    nb = int(max(12, cfg.hist_bins))
    # bin index in [0,nb)
    idx = np.floor((ang_tan / np.pi) * nb).astype(np.int32)
    idx = np.clip(idx, 0, nb - 1)
    hist = np.zeros((nb,), dtype=np.float64)
    np.add.at(hist, idx.ravel(), w.ravel())

    # circular smooth
    hs = hist.copy()
    sig = float(cfg.hist_smooth_sigma_bins)
    if sig > 0.0:
        # wrap padding: 3*sigma on both sides
        pad = int(max(1, round(3.0 * sig)))
        x = np.concatenate([hs[-pad:], hs, hs[:pad]], axis=0)
        x = ndimage.gaussian_filter1d(x, sigma=sig).astype(np.float64)
        hs = x[pad : pad + nb]

    peak = int(np.argmax(hs))
    peak_w = float(hs[peak])
    dominance = 0.0 if total_w <= 0.0 else float(peak_w / total_w)

    # peak angle (bin center)
    theta = (float(peak) + 0.5) * (np.pi / float(nb))
    v = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    ang = float(np.degrees(theta))

    # also compute a global tensor for debug (old behavior)
    avg_Jxx = float(np.mean(Jxx_s))
    avg_Jyy = float(np.mean(Jyy_s))
    avg_Jxy = float(np.mean(Jxy_s))
    T = np.array([[avg_Jxx, avg_Jxy], [avg_Jxy, avg_Jyy]], dtype=np.float64)
    eigvals, _eigvecs = np.linalg.eigh(T)
    lam1 = float(eigvals[0])
    lam2 = float(eigvals[1])
    global_coh = float((lam2 - lam1) / max(lam2 + lam1, 1e-12))

    dbg = {
        "tensor": T.tolist(),
        "eigvals": [lam1, lam2],
        "global_coherence": global_coh,
        "dominance": dominance,
        "angle_deg": ang,
        "hist_bins": nb,
        "hist_raw": hist.astype(np.float32),
        "hist_smooth": hs.astype(np.float32),
        "trace_thr": float(tr_thr),
        "trace_percentile": float(tp),
        "min_local_coherence": float(cfg.min_local_coherence),
    }
    return v.astype(np.float32), ang, dominance, dbg


def _topk_circular_bins(h: np.ndarray, *, k: int, min_sep: int) -> np.ndarray:
    """在环形直方图 h 上取 top-k bin index，并保证任意两者的环形距离 >= min_sep。"""
    x = np.asarray(h, dtype=np.float64).reshape(-1)
    n = int(x.shape[0])
    order = np.argsort(-x)
    picked = []
    for idx in order:
        i = int(idx)
        ok = True
        for j in picked:
            d = abs(i - int(j))
            d = min(d, n - d)
            if d < int(min_sep):
                ok = False
                break
        if ok:
            picked.append(i)
            if len(picked) >= int(k):
                break
    return np.asarray(picked, dtype=np.int32)


def _period_from_profile_autocorr(
    profile: np.ndarray,
    *,
    cfg: ProjectionPeriodicityConfig,
) -> Tuple[int, float]:
    """
    从 1D profile 估计主周期（像素 lag）与主峰强度（归一化到 autocorr[0]）。
    返回 (period_lag, peak_strength)。失败则 (0,0)。
    """
    p = np.asarray(profile, dtype=np.float64).reshape(-1)
    if p.size < 16:
        return 0, 0.0

    if float(cfg.profile_smooth_sigma) > 0.0:
        p = ndimage.gaussian_filter1d(p, sigma=float(cfg.profile_smooth_sigma)).astype(np.float64)
    p = p - float(np.mean(p))

    # FFT 自相关（正半轴）
    n = int(p.size)
    F = np.fft.rfft(p)
    ac = np.fft.irfft(F * np.conj(F), n=n).astype(np.float64)
    z0 = float(ac[0])
    if z0 <= 1e-12:
        return 0, 0.0
    ac = (ac / z0).astype(np.float64)
    if float(cfg.autocorr_smooth_sigma) > 0.0:
        ac = ndimage.gaussian_filter1d(ac, sigma=float(cfg.autocorr_smooth_sigma)).astype(np.float64)

    lo = int(max(1, cfg.min_lag_px))
    hi = int(min(cfg.max_lag_px, n - 1))
    if hi <= lo + 1:
        return 0, 0.0

    seg = ac[lo:hi]
    prom = float(cfg.min_peak_prom_rel)
    # 在 seg 上找峰；prominence 在 seg 的局部尺度里定义，这里用相对阈值近似
    peaks, props = signal.find_peaks(
        seg,
        distance=int(max(1, cfg.peak_distance_px)),
        prominence=float(prom),
    )
    if peaks.size == 0:
        return 0, 0.0

    # 取“第一个显著峰”；若多个，要求高度接近主峰（避免误选很早的噪声峰）
    heights = seg[peaks]
    m = float(np.max(heights))
    good = peaks[heights >= 0.6 * max(m, 1e-9)]
    if good.size > 0:
        k = int(np.min(good))
    else:
        k = int(peaks[int(np.argmax(heights))])

    lag = int(lo + k)
    strength = float(ac[lag])
    return lag, strength


def get_chain_direction_projection_periodicity(
    image: np.ndarray,
    *,
    cfg: ChainDirectionConfig = ChainDirectionConfig(),
    proj_cfg: ProjectionPeriodicityConfig = ProjectionPeriodicityConfig(),
) -> Tuple[np.ndarray, float, float, Dict]:
    """
    更稳健的 Re 链方向估计：结构张量提供候选方向，投影周期测量法做仲裁。
    返回格式与 `get_chain_direction_structure_tensor` 一致：(chain_unit_xy, angle_deg, dominance, debug)。
    """
    v0, ang0, dom0, dbg0 = get_chain_direction_structure_tensor(image, cfg=cfg)

    hs = np.asarray(dbg0.get("hist_smooth"), dtype=np.float64).reshape(-1)
    nb = int(dbg0.get("hist_bins", hs.size if hs.size else 0))
    if hs.size == 0 or nb <= 0:
        return v0, float(ang0), float(dom0), {**dbg0, "arb_used": False, "arb_reason": "no_hist"}

    # 结构张量直方图的 top-k 候选（轴向角在 [0,pi)）
    top_bins = _topk_circular_bins(hs, k=int(proj_cfg.topk_candidates), min_sep=int(proj_cfg.min_sep_bins))
    cand_deg = []
    for b in top_bins.tolist():
        theta = (float(b) + 0.5) * (180.0 / float(nb))  # [0,180)
        cand_deg.append(theta)
        # 增加 ±60° 的等价候选，用投影周期来仲裁（ReS2 常见误差）
        cand_deg.append((theta + 60.0) % 180.0)
        cand_deg.append((theta + 120.0) % 180.0)

    # 去重（角度量化到 0.5°）
    uniq = sorted({int(round(t * 2.0)) for t in cand_deg})
    cand_deg = [u / 2.0 for u in uniq]

    img = _to_gray_float(image)
    # normalize to [0,1]
    vmin = float(np.min(img))
    vmax = float(np.max(img))
    if vmax - vmin > 1e-9:
        img = (img - vmin) / (vmax - vmin)
    img = img.astype(np.float32)
    if float(proj_cfg.hp_sigma_px) > 0.0:
        bg = ndimage.gaussian_filter(img, sigma=float(proj_cfg.hp_sigma_px)).astype(np.float32)
        hp = (img - bg).astype(np.float32)
        # 只保留正半轴，避免 profile 的均值漂移（原子是亮点）
        hp = np.clip(hp, 0.0, 1.0).astype(np.float32)
        img = hp

    best = None
    best_score = -1e30
    scores = []

    for theta in cand_deg:
        # rotate so that candidate axis becomes horizontal (x direction)
        rot = ndimage.rotate(
            img,
            angle=-float(theta),
            reshape=False,
            order=int(proj_cfg.rotate_order),
            mode=str(proj_cfg.rotate_mode),
            cval=float(proj_cfg.rotate_cval),
        ).astype(np.float32)
        profile = np.sum(rot, axis=1).astype(np.float64)  # along x -> profile(y)
        period, strength = _period_from_profile_autocorr(profile, cfg=proj_cfg)
        if period <= 0:
            sc = -1e9
        else:
            sc = float(period)
            if bool(proj_cfg.use_strength_weight):
                sc = sc * float(max(strength, 1e-6))

        scores.append((float(theta), int(period), float(strength), float(sc)))
        if sc > best_score:
            best_score = float(sc)
            best = (float(theta), int(period), float(strength))

    if best is None:
        return v0, float(ang0), float(dom0), {**dbg0, "arb_used": False, "arb_reason": "no_candidate"}

    # dominance：用最优与次优 score 的差值比例（0..1）；仅作为“可信度”门控的粗指标
    s_sorted = sorted(scores, key=lambda t: t[3], reverse=True)
    s1 = float(s_sorted[0][3]) if s_sorted else 0.0
    s2 = float(s_sorted[1][3]) if len(s_sorted) > 1 else 0.0
    dom = 0.0 if s1 <= 1e-9 else float(max(0.0, (s1 - s2) / s1))

    # 候选区分度不足时，做 0~180 的粗扫描兜底（先粗后细），避免结构张量在少数样本上错峰。
    used_fallback = False
    fallback_scores = None
    if bool(proj_cfg.enable_bruteforce_fallback) and float(dom) < float(proj_cfg.fallback_dom_threshold):
        used_fallback = True
        fallback_scores = []

        def _eval_theta(theta: float) -> Tuple[float, int, float, float]:
            rot = ndimage.rotate(
                img,
                angle=-float(theta),
                reshape=False,
                order=int(proj_cfg.rotate_order),
                mode=str(proj_cfg.rotate_mode),
                cval=float(proj_cfg.rotate_cval),
            ).astype(np.float32)
            profile = np.sum(rot, axis=1).astype(np.float64)
            period, strength = _period_from_profile_autocorr(profile, cfg=proj_cfg)
            if period <= 0:
                sc = -1e9
            else:
                sc = float(period)
                if bool(proj_cfg.use_strength_weight):
                    sc = sc * float(max(strength, 1e-6))
            return float(theta), int(period), float(strength), float(sc)

        step = float(max(0.5, proj_cfg.fallback_step_deg))
        thetas = np.arange(0.0, 180.0, step, dtype=np.float64)
        best_fb = None
        best_fb_score = -1e30
        for th in thetas.tolist():
            item = _eval_theta(float(th))
            fallback_scores.append(item)
            if float(item[3]) > best_fb_score:
                best_fb_score = float(item[3])
                best_fb = item

        # 细化：在最优附近 +/- refine_deg 用更小步长扫描
        if best_fb is not None and float(proj_cfg.fallback_refine_step_deg) < step:
            cth = float(best_fb[0])
            r = float(max(0.0, proj_cfg.fallback_refine_deg))
            step2 = float(max(0.25, proj_cfg.fallback_refine_step_deg))
            thetas2 = np.arange(cth - r, cth + r + 1e-9, step2, dtype=np.float64)
            for th in thetas2.tolist():
                # wrap to [0,180)
                t = float(th) % 180.0
                item = _eval_theta(t)
                fallback_scores.append(item)
                if float(item[3]) > best_fb_score:
                    best_fb_score = float(item[3])
                    best_fb = item

        if best_fb is not None and float(best_fb[3]) > best_score + 1e-6:
            best = (float(best_fb[0]), int(best_fb[1]), float(best_fb[2]))
            best_score = float(best_fb[3])

    theta = float(best[0])
    rad = np.deg2rad(theta)
    v = np.array([np.cos(rad), np.sin(rad)], dtype=np.float32)  # image coords (x-right/y-down)
    # 输出的“可信度”用于下游是否启用链方向约束：
    # - dominance_tensor：结构张量直方图的主峰占比（在某些样本上更稳定）
    # - dominance_proj：投影周期仲裁的区分度（best-vs-second）
    # 为减少“链方向本身正确但 dominance_proj 过小导致下游完全不使用”，这里取二者最大值。
    dominance_out = float(max(dom, dom0))

    return v, float(theta), float(dominance_out), {
        **dbg0,
        "arb_used": True,
        "arb_theta_deg": float(theta),
        "arb_period_px": int(best[1]),
        "arb_strength": float(best[2]),
        "arb_score": float(best_score),
        "arb_scores": scores,
        "arb_fallback_used": bool(used_fallback),
        "arb_fallback_best": None if not used_fallback else {"theta_deg": float(best[0]), "period_px": int(best[1]), "strength": float(best[2]), "score": float(best_score)},
        "arb_hp_sigma_px": float(proj_cfg.hp_sigma_px),
        "dominance_proj": float(dom),
        "dominance_tensor": float(dom0),
        "dominance_out": float(dominance_out),
    }


def get_chain_direction_from_points_knn(
    points_px: np.ndarray,
    *,
    size_px: int = 128,
    sideA_A: Optional[float] = None,
    cfg: ChainDirectionFromPointsConfig = ChainDirectionFromPointsConfig(),
) -> Tuple[np.ndarray, float, float, Dict]:
    """
    用点云最近邻位移的二阶矩（PCA）估计链方向（轴向）。
    返回 (chain_unit_xy, angle_deg, dominance, debug)：
    - chain_unit_xy: (2,) float32，图像坐标 x-right/y-down
    - angle_deg: [0,180)（轴向）
    - dominance: 用“最大特征值占比”作为各向异性强度（0..1）
    """
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 8:
        z = np.array([1.0, 0.0], dtype=np.float32)
        return z, 0.0, 0.0, {"reason": "too_few_points", "n_points": int(pts.shape[0])}

    tree = cKDTree(pts)

    # 距离窗口：优先用物理尺度推算
    if sideA_A is not None and float(sideA_A) > 1e-6:
        bond_px = float(cfg.bond_A) / float(sideA_A) * float(size_px)
        tol = float(cfg.bond_tol_ratio)
        r_lo = bond_px * (1.0 - tol)
        r_hi = bond_px * (1.0 + tol)
        win_mode = "bond_from_sideA"
    else:
        # 用最近邻距离的中位数作为尺度
        d, _ = tree.query(pts, k=2)
        # d[:,0]=0, d[:,1]=nn
        d_med = float(np.median(d[:, 1]))
        tol = float(cfg.fallback_tol_ratio)
        r_lo = d_med * (1.0 - tol)
        r_hi = d_med * (1.0 + tol)
        win_mode = "bond_from_nn_median"

    k = int(max(4, cfg.knn_k)) + 1
    dists, idxs = tree.query(pts, k=min(k, pts.shape[0]))

    vecs = []
    ws = []
    pwr = float(cfg.weight_power)
    for i in range(pts.shape[0]):
        for j in range(1, idxs.shape[1]):  # skip self
            r = float(dists[i, j])
            if r < r_lo or r > r_hi:
                continue
            nb = int(idxs[i, j])
            dv = (pts[nb] - pts[i]).astype(np.float64)
            if r > 1e-9:
                w = 1.0 / (r**pwr) if pwr > 0 else 1.0
            else:
                w = 1.0
            vecs.append(dv)
            ws.append(w)

    if len(vecs) < int(cfg.min_edges):
        z = np.array([1.0, 0.0], dtype=np.float32)
        return z, 0.0, 0.0, {
            "reason": "too_few_edges",
            "n_points": int(pts.shape[0]),
            "n_edges": int(len(vecs)),
            "r_lo": float(r_lo),
            "r_hi": float(r_hi),
            "window_mode": win_mode,
        }

    V = np.asarray(vecs, dtype=np.float64)
    W = np.asarray(ws, dtype=np.float64)
    # 二阶矩：M = sum w * dv dv^T
    M = (V.T * W) @ V
    # eigenvalues ascending
    evals, evecs = np.linalg.eigh(M)
    lam1 = float(evals[0])
    lam2 = float(evals[1])
    v = evecs[:, 1].astype(np.float64)  # major axis
    nv = float(np.hypot(v[0], v[1]))
    if nv < 1e-12:
        z = np.array([1.0, 0.0], dtype=np.float32)
        return z, 0.0, 0.0, {"reason": "degenerate", "n_edges": int(len(vecs))}
    v = v / nv
    # 轴向：限制到 [0,180)
    theta = float(np.degrees(np.arctan2(v[1], v[0]))) % 180.0
    # dominance：各向异性（0..1）
    dom = 0.0 if (lam1 + lam2) <= 1e-12 else float((lam2 - lam1) / (lam2 + lam1))
    return v.astype(np.float32), float(theta), float(dom), {
        "window_mode": win_mode,
        "r_lo": float(r_lo),
        "r_hi": float(r_hi),
        "n_edges": int(len(vecs)),
        "eigvals": [lam1, lam2],
        "dominance": float(dom),
    }
