#!/usr/bin/env python3
"""
【作用概述】评估 generate_sample 的分解结果，支持三种任务：
1) shift_pbc：从两张分解单层图中提取原子中心点，做粗到细的平移搜索估计 Pred shift；随后在二维周期矢量 (V1/V2) 下将误差折叠为最小像（wrap-around）误差，避免跨晶胞边界时出现“真实很小但数值很大”的误差。
2) twist：仅估计两层之间的相对扭角（rotation），不求解层间滑移；适用于仅关注扭角的 rotation 数据集（如 results_root 下样本目录名形如 ReS2_r0.01，表示 GT 扭角为 0.01°）。
3) twist_labels：从 `MOIRE_SAVE_LABELS` 导出的 `.npz` 中直接读取两层的 GT 原子坐标（点云），在“最理想输入”的前提下评估扭角估计误差（不依赖分解结果）。
三种任务都会在运行时逐条追加写入 CSV，并支持断点重续（默认跳过 CSV 中已存在的 stem）。
增强：当提供 `--separate_yml` 时，本脚本可在评估前先调用 `src/main.py` 完成分解（读取并快照该 yml 到内存，避免并发实验时 yml 被修改影响本次运行）；评估结束后会对误差偏大的样本自动进行多 seed 重试（默认 4 次），并将“误差最小”的分解结果覆盖回 `results_root`，最终输出仅保留最佳结果的 CSV。
补充：twist 任务的周期折叠参数 `twist_period_deg` 可从 `generate_sample/*.json` 读取（键：twist_period_deg），也可用命令行 `--twist_period_deg` 覆写。
补充：wrap-around 误差折叠不再假设 Pred/GT 的差值已接近 0（即不再依赖很小的枚举 k_range）；而是先在 (V1,V2) 基下估计一个“中心整数 (kx0,ky0)”并在其附近做小范围搜索，避免 multi_origin 搜索导致 Pred 落在“相隔多个晶胞”的等价解时误差被误判为 ~1 个晶格常数。
补充：ReS2 的默认中心提取参数会先做轻微高斯平滑并过滤更小的连通域，用于抑制高噪声分解层中被打散的颗粒噪声被误识别为原子中心。
补充：MoTe2 的 slip 误差规约会额外允许 (V1+V2)/2 半晶格等价偏移；该材料的金属子格存在这一等价位移，不加入时会把正确解误报为约 3.6 Å 的系统性大误差。
【关联说明】文件/模块：generate_sample/*.json（shift_pbc 的周期矢量来源：period.t1_A / period.t2_A；若缺失则尝试使用材料默认值或从结构推断）；generate_sample/batch_runner.py（rotation_range 与 r 命名规则；以及 rotate/flip 下周期矢量变换）；data/experiments/*/result（分解输出目录结构）。
【命令行用法】(shift_pbc) python tools/evaluate_generate_sample_wraparound_pbc.py --task shift_pbc --results_root data/experiments/MoS2/result --batch_config generate_sample/MoS2.json
（参数：--results_root=分解结果根目录；--out_csv=默认 <results_root>/result.csv；--resume/--no_resume=默认跳过已写入样本；--limit=仅评估前 N 个样本；--stems=仅评估指定 stem；--debug_vis_dir=导出中心提取可视化）
(twist) python tools/evaluate_generate_sample_wraparound_pbc.py --task twist --results_root data/experiments/ReS2_rotation/result --batch_config generate_sample/ReS2.json
（参数：--out_csv=默认 <results_root>/result_twist.csv；--batch_config=可提供 twist_period_deg；--twist_period_deg=可覆写；--twist_* 为扭角估计算法超参数）
(twist_labels) python tools/evaluate_generate_sample_wraparound_pbc.py --task twist_labels --results_root data/experiments/TaS2_rotation_labels --batch_config generate_sample/TaS2.json
（参数：--out_csv=默认 <results_root>/result_twist_labels.csv；--batch_config/--twist_period_deg 同 twist）
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree
from ase.io import read as ase_read
import yaml

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

# 允许以“python tools/xxx.py”方式运行时也能 import 同级脚本
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import tools.evaluate_generate_sample_wraparound as base  # type: ignore
except ModuleNotFoundError:
    # 兼容：若用户已清理旧版 eval（tools/evaluate_generate_sample_wraparound.py），
    # 则在本脚本内提供最小实现，确保 evaluate_generate_sample_wraparound_pbc.py 可独立运行。
    FILENAME_RE = re.compile(
        r"^(?P<material>[A-Za-z0-9]+)"
        r"_r(?P<r>-?\d+(?:\.\d+)?)"
        r"_gtx(?P<gtx>-?\d+(?:\.\d+)?)"
        r"_gty(?P<gty>-?\d+(?:\.\d+)?)"
        r"_sideA(?P<sideA>-?\d+(?:\.\d+)?)"
        r"(?:_v1x(?P<v1x>-?\d+(?:\.\d+)?)_v1y(?P<v1y>-?\d+(?:\.\d+)?)"
        r"_v2x(?P<v2x>-?\d+(?:\.\d+)?)_v2y(?P<v2y>-?\d+(?:\.\d+)?))?"
        r"\.png$"
    )

    @dataclass(frozen=True)
    class SampleResult:
        stem: str
        gtx: float
        gty: float
        sideA: float
        shift_pred_128_px: Tuple[float, float]
        shift_gt_128_px: Tuple[float, float]
        err_A: float
        kx: int
        ky: int
        swapped: bool
        centers0: int
        centers1: int

    def _to_gray01(img_u8: np.ndarray) -> np.ndarray:
        return (img_u8.astype(np.float32) / 255.0).clip(0.0, 1.0)

    def find_centers(
        img_gray01: np.ndarray,
        *,
        min_area_threshold: int = 50,
        bina_thre: float = 1.5,
    ) -> list[tuple[int, int]]:
        import cv2

        mean_val = float(img_gray01.mean())
        if mean_val <= 0.0:
            return []
        _, binary = cv2.threshold(img_gray01, mean_val * float(bina_thre), 1, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(np.uint8(binary), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers: list[tuple[int, int]] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= float(min_area_threshold):
                continue
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))
        return centers

    def parse_gt_from_name(name: str) -> Optional[tuple[float, float, float, float, Optional[np.ndarray], Optional[np.ndarray]]]:
        m = FILENAME_RE.match(name)
        if not m:
            return None
        r = float(m.group("r"))
        gtx = float(m.group("gtx"))
        gty = float(m.group("gty"))
        sideA = float(m.group("sideA"))
        if m.group("v1x") is not None:
            v1 = np.array([float(m.group("v1x")), float(m.group("v1y"))], dtype=np.float64)
            v2 = np.array([float(m.group("v2x")), float(m.group("v2y"))], dtype=np.float64)
        else:
            v1 = None
            v2 = None
        return r, gtx, gty, sideA, v1, v2

    def iter_bilayer_pngs(bilayer_dir: Path) -> Iterable[Path]:
        for p in sorted(bilayer_dir.glob("*.png")):
            if parse_gt_from_name(p.name) is not None:
                yield p

    def load_aug_meta_from_labels(labels_dir: Path, stem: str) -> Optional[dict]:
        npz_path = labels_dir / f"{stem}.npz"
        if not npz_path.is_file():
            return None
        try:
            data = np.load(npz_path, allow_pickle=True)
            return json.loads(str(data["meta"]))
        except Exception:
            return None

    def transform_period_vecs_by_meta(T1: np.ndarray, T2: np.ndarray, *, meta: dict) -> tuple[np.ndarray, np.ndarray]:
        v1 = np.asarray(T1, dtype=np.float64).copy()
        v2 = np.asarray(T2, dtype=np.float64).copy()
        rotate_enabled = bool(meta.get("rotate_enabled", True))
        angle_deg = float(meta.get("angle_deg", 0.0))
        if rotate_enabled and angle_deg != 0.0:
            ang = math.radians(angle_deg)
            c = math.cos(ang)
            s = math.sin(ang)
            R = np.array([[c, -s], [s, c]], dtype=np.float64)
            v1 = (R @ v1.reshape(2, 1)).reshape(2)
            v2 = (R @ v2.reshape(2, 1)).reshape(2)
        if bool(meta.get("flip_h", False)):
            v1[0] = -v1[0]
            v2[0] = -v2[0]
        if bool(meta.get("flip_v", False)):
            v1[1] = -v1[1]
            v2[1] = -v2[1]
        return v1, v2

    def infer_period_vecs_from_centers(_centers: np.ndarray, *, expected_len_px: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("infer_period_vecs_from_centers 未实现，使用 batch_config/labels 的周期矢量即可")

    def infer_period_vecs_from_structure(structure_path: Path) -> tuple[np.ndarray, np.ndarray]:
        # 兼容多材料：用 ASE 读取 Extended XYZ，并在分数坐标系下统计候选平移。
        atoms = ase_read(str(structure_path))
        syms = atoms.get_chemical_symbols()
        if not syms:
            raise ValueError(f"空结构：{structure_path}")

        # 选择计数最少的元素作为 lattice basis（在 MX2 中通常是金属层）
        counts: dict[str, int] = {}
        for s in syms:
            counts[s] = counts.get(s, 0) + 1
        basis = min(counts.items(), key=lambda kv: kv[1])[0]
        pts = np.asarray([a.position[:2] for a in atoms if a.symbol == basis], dtype=np.float64)
        if len(pts) < 10:
            raise ValueError(f"{basis} 原子点过少，无法推断周期：{structure_path}")

        cell = atoms.get_cell().array
        A2 = np.asarray(cell[:2, :2], dtype=np.float64)
        det = float(np.linalg.det(A2))
        if abs(det) < 1e-12:
            raise ValueError(f"二维晶胞矩阵奇异，无法推断周期：{structure_path}")
        A2_inv_T = np.linalg.inv(A2).T
        frac = pts @ A2_inv_T
        frac = frac - np.floor(frac)

        # 统计候选位移（Å）：对 ReS2 避开键长尺度，其它材料更宽松
        r_min, r_max = (4.0, 8.5) if basis == "Re" else (1.5, 9.5)
        rng = np.random.RandomState(0)
        pick = rng.choice(len(frac), size=min(len(frac), 260), replace=False)
        anchors = frac[pick]
        disps: list[np.ndarray] = []
        for p in anchors:
            d = frac - p
            d -= np.round(d)
            cart = d @ A2.T
            r = np.hypot(cart[:, 0], cart[:, 1])
            m = (r > r_min) & (r < r_max)
            if np.any(m):
                disps.append(cart[m])
        if not disps:
            raise ValueError(f"未能提取候选位移：{structure_path}")
        disps_arr = np.vstack(disps)

        step = 0.01
        binned = np.round(disps_arr / step) * step

        def _canon(v: np.ndarray) -> tuple[float, float]:
            x, y = float(v[0]), float(v[1])
            if abs(x) > 1e-9:
                if x < 0:
                    x, y = -x, -y
            else:
                if y < 0:
                    x, y = -x, -y
            return (x, y)

        from collections import Counter

        counter: Counter[tuple[float, float]] = Counter(_canon(v) for v in binned)
        items = sorted(counter.items(), key=lambda kv: (-kv[1], float(np.hypot(kv[0][0], kv[0][1]))))
        if len(items) < 2:
            raise ValueError(f"候选周期矢量不足：{structure_path}")

        v1 = np.array(items[0][0], dtype=np.float64)

        def _pick_v2(min_count_ratio: float) -> Optional[np.ndarray]:
            ref_cnt = float(items[0][1])
            best: Optional[np.ndarray] = None
            best_len = float("inf")
            best_cnt = -1
            for (x, y), cnt in items[1:200]:
                if ref_cnt > 0 and float(cnt) < ref_cnt * min_count_ratio:
                    continue
                cand = np.array([x, y], dtype=np.float64)
                cross = abs(float(v1[0] * cand[1] - v1[1] * cand[0]))
                if cross < 0.5:
                    continue
                clen = float(np.hypot(cand[0], cand[1]))
                if clen < best_len - 1e-9 or (abs(clen - best_len) <= 1e-9 and cnt > best_cnt):
                    best = cand
                    best_len = clen
                    best_cnt = int(cnt)
            return best

        v2 = _pick_v2(0.20) or _pick_v2(0.05)
        if v2 is None:
            raise ValueError(f"未能找到第二周期矢量：{structure_path}")
        if abs(float(v2[1])) < abs(float(v1[1])):
            v1, v2 = v2, v1
        return v1, v2

    def wraparound_min_err(delta_A: np.ndarray, *, T1: np.ndarray, T2: np.ndarray, k_range: int = 1) -> tuple[float, int, int]:
        k_range = int(max(0, k_range))
        best = None
        best_k = (0, 0)
        d = np.asarray(delta_A, dtype=np.float64).reshape(2)
        T1 = np.asarray(T1, dtype=np.float64).reshape(2)
        T2 = np.asarray(T2, dtype=np.float64).reshape(2)
        for kx in range(-k_range, k_range + 1):
            for ky in range(-k_range, k_range + 1):
                dd = d - float(kx) * T1 - float(ky) * T2
                n = float(np.linalg.norm(dd))
                if best is None or n < best:
                    best = n
                    best_k = (kx, ky)
        assert best is not None
        return float(best), int(best_k[0]), int(best_k[1])

    base = types.SimpleNamespace(
        SampleResult=SampleResult,
        _to_gray01=_to_gray01,
        find_centers=find_centers,
        parse_gt_from_name=parse_gt_from_name,
        iter_bilayer_pngs=iter_bilayer_pngs,
        load_aug_meta_from_labels=load_aug_meta_from_labels,
        transform_period_vecs_by_meta=transform_period_vecs_by_meta,
        infer_period_vecs_from_centers=infer_period_vecs_from_centers,
        infer_period_vecs_from_structure=infer_period_vecs_from_structure,
        wraparound_min_err=wraparound_min_err,
    )


_CSV_HEADER = [
    "stem",
    "gtx",
    "gty",
    "sideA",
    "predx_128",
    "predy_128",
    "err_A",
    "kx",
    "ky",
    "swapped",
    "centers0",
    "centers1",
]

_TWIST_STEM_RE = re.compile(r"^(?P<material>[A-Za-z0-9]+)_r(?P<twist>-?\d+(?:\.\d+)?)$")

_CSV_HEADER_TWIST = [
    "stem",
    "gt_r_deg_raw",
    "gt_twist_deg",
    "pred_twist_deg",
    "abs_err_deg",
    "score",
    "twist_period_deg",
    "thr_rel",
    "min_area",
    "k_neighbors",
    "r_max_px",
    "trim_ratio",
    "coarse_step_deg",
    "centers0",
    "centers1",
    "vecs0",
    "vecs1",
]


_DEFAULT_CENTER_PARAMS_BY_MATERIAL: dict[str, dict[str, float | str]] = {
    # 说明：
    # - contour：使用 base.find_centers（均值阈值 + 外轮廓面积）
    # - cc_relmax：对灰度图做模糊后，用 (thr_rel * max) 阈值取连通域，再取质心；适合在“包含轻元素弱峰”的图中仅保留强峰（例如 MoS2 只取 Mo）
    # ReS2 噪声分解结果中，某一层可能被高频噪声打散；轻微平滑 + 更大的连通域面积阈值能显著减少噪声碎点伪检。
    "res2": {"center_method": "contour", "bina_thre": 1.5, "min_area": 120.0, "blur_sigma": 0.6},
    # MoS2/TaS2/MoTe2：默认优先只取强峰（通常是金属层）。min_area 设小一些以避免“对比度偏低时全部过滤掉”的 nan。
    "mos2": {"center_method": "cc_relmax", "thr_rel": 0.75, "min_area": 20.0, "blur_sigma": 1.2},
    "tas2": {"center_method": "cc_relmax", "thr_rel": 0.75, "min_area": 20.0, "blur_sigma": 1.2},
    "mote2": {"center_method": "cc_relmax", "thr_rel": 0.75, "min_area": 20.0, "blur_sigma": 1.2},
}

_FRACTIONAL_EQUIV_OFFSETS_BY_MATERIAL: dict[str, tuple[tuple[float, float], ...]] = {
    # 1T'/distorted MoTe2 has two equivalent metal sublattices separated by
    # half of the generated slip basis; treating that offset as distinct
    # produces a false error mode around |(V1+V2)/2| ~= 3.6 A.
    "mote2": ((0.0, 0.0), (0.5, 0.5), (-0.5, -0.5)),
}


def _material_key_from_stem(stem: str) -> str:
    # 文件名规则是 <material>_r...，material 里不含下划线（见正则）。
    return stem.split("_", 1)[0].strip().lower()


def _apply_default_center_params(
    *,
    material_key: str,
    center_method: Optional[str],
    bina_thre: Optional[float],
    min_area_threshold: Optional[int],
    blur_sigma: Optional[float],
    thr_rel: Optional[float],
) -> tuple[str, float, int, float, float]:
    base_cfg = _DEFAULT_CENTER_PARAMS_BY_MATERIAL.get(material_key, _DEFAULT_CENTER_PARAMS_BY_MATERIAL["res2"])
    method = str(center_method or base_cfg.get("center_method", "contour"))
    bina = float(bina_thre if bina_thre is not None else float(base_cfg.get("bina_thre", 1.5)))  # type: ignore[arg-type]
    min_area = int(min_area_threshold if min_area_threshold is not None else int(float(base_cfg.get("min_area", 50.0))))  # type: ignore[arg-type]
    sigma = float(blur_sigma if blur_sigma is not None else float(base_cfg.get("blur_sigma", 0.0)))  # type: ignore[arg-type]
    rel = float(thr_rel if thr_rel is not None else float(base_cfg.get("thr_rel", 0.75)))  # type: ignore[arg-type]
    return method, bina, min_area, sigma, rel


def _find_centers_cc_relmax(
    img_gray01_for_mask: np.ndarray,
    *,
    thr_rel: float,
    min_area_threshold: int,
    img_gray01_for_weight: Optional[np.ndarray] = None,
) -> list[tuple[float, float]]:
    if cv2 is None:
        raise RuntimeError("cv2 不可用，无法使用 cc_relmax 提取中心点。")
    img_mask = np.asarray(img_gray01_for_mask, dtype=np.float32)
    img_w = img_mask if img_gray01_for_weight is None else np.asarray(img_gray01_for_weight, dtype=np.float32)
    mx = float(img_mask.max())
    if mx <= 0.0 or not np.isfinite(mx):
        return []
    thr = float(thr_rel) * mx
    mask = (img_mask >= thr).astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return []
    centers: list[tuple[float, float]] = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(min_area_threshold):
            continue
        # 用强度加权的质心做亚像素细化（有助于 MoS2 这类“亮斑+拖尾”的情况）
        ys, xs = np.where(labels == i)
        if xs.size == 0:
            continue
        w = img_w[ys, xs].astype(np.float64)
        s = float(w.sum())
        if not np.isfinite(s) or s <= 1e-12:
            cx, cy = centroids[i]
            centers.append((float(cx), float(cy)))
            continue
        cx = float((xs.astype(np.float64) * w).sum() / s)
        cy = float((ys.astype(np.float64) * w).sum() / s)
        centers.append((float(cx), float(cy)))
    return centers


def _render_centers_debug(
    *,
    out_dir: Path,
    stem: str,
    tag: str,
    img01: np.ndarray,
    centers: list[tuple[float, float]],
    method: str,
    meta: dict,
) -> None:
    if cv2 is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    img_u8 = (np.clip(img01, 0.0, 1.0) * 255.0).astype(np.uint8)
    cv2.imwrite(str(out_dir / f"{stem}_{tag}_img.png"), img_u8)

    # 额外导出“二值中间结果”，方便人工检查阈值是否把轻元素/噪声也吃进来了。
    m = str(method).strip().lower()
    if m in {"cc_relmax", "relmax"}:
        thr_rel = float(meta.get("center_thr_rel", 0.75))
        mx = float(img01.max())
        thr = thr_rel * mx if mx > 0 else float("inf")
        mask = (img01 >= thr).astype(np.uint8) * 255
        cv2.imwrite(str(out_dir / f"{stem}_{tag}_mask.png"), mask)
    elif m in {"contour"}:
        bina_thre = float(meta.get("bina_thre", 1.5))
        mean_val = float(img01.mean())
        thr = mean_val * bina_thre if mean_val > 0 else float("inf")
        mask = (img01 >= thr).astype(np.uint8) * 255
        cv2.imwrite(str(out_dir / f"{stem}_{tag}_mask.png"), mask)

    rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    for (x, y) in centers:
        cv2.circle(rgb, (int(round(float(x))), int(round(float(y)))), 4, (0, 0, 255), 1, lineType=cv2.LINE_AA)
    cv2.putText(
        rgb,
        f"{stem} {tag} n={len(centers)} method={method}",
        (10, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 128, 0),
        1,
        lineType=cv2.LINE_AA,
    )
    cv2.imwrite(str(out_dir / f"{stem}_{tag}_overlay.png"), rgb)

    # 仅点云（不叠加灰度），用于更清晰地看中心提取数量/分布。
    pts = np.zeros_like(rgb)
    for (x, y) in centers:
        cv2.circle(pts, (int(round(float(x))), int(round(float(y)))), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    cv2.imwrite(str(out_dir / f"{stem}_{tag}_points.png"), pts)
    (out_dir / f"{stem}_{tag}_centers.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_done_stems_from_csv(out_csv: Path) -> set[str]:
    """
    读取 out_csv 中已写入的 stem 集合，用于断点重续。

    兼容策略：
    - 空行/格式不完整行会被跳过；
    - 若首行是 header（stem,...）则跳过；
    - 若 CSV 不存在则返回空集合。
    """
    if not out_csv.is_file():
        return set()
    done: set[str] = set()
    with out_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            stem = str(row[0]).strip()
            if not stem:
                continue
            if stem == "stem":
                continue
            done.add(stem)
    return done


def _to_int_xy(v: np.ndarray) -> tuple[int, int]:
    return (int(round(float(v[0]))), int(round(float(v[1]))))


def _parse_twist_deg_from_stem(stem: str) -> Optional[float]:
    """
    从样本目录名解析 GT 扭角（degree）。

    约定：stem 形如 <material>_r0.01，表示 GT 扭角为 0.01°（允许负号，但最终会按 twist_period 折叠）。
    """
    m = _TWIST_STEM_RE.match(str(stem).strip())
    if not m:
        return None
    try:
        return float(m.group("twist"))
    except Exception:
        return None


def _twist_fold_deg(angle_deg: float, *, period_deg: float) -> float:
    """
    将角度折叠到 [0, period/2]。

    例如 period=180 时：112.03 -> 67.97；179.6 -> 0.4。
    """
    p = float(period_deg)
    if not np.isfinite(p) or p <= 0.0:
        raise ValueError(f"非法 period_deg={period_deg}")
    a = float(angle_deg) % p
    return float(min(a, p - a))


def _load_gray01(path: Path) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("cv2 不可用，无法读取/处理 PNG。")
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return (img.astype(np.float32) / 255.0).clip(0.0, 1.0)


def _find_centers_cc_relmax_weighted(
    img_gray01: np.ndarray,
    *,
    thr_rel: float,
    min_area_threshold: int,
    blur_sigma: float = 0.0,
) -> np.ndarray:
    """
    相对 max 阈值 + 连通域 + 强度加权质心（亚像素）中心提取。

    返回：N×2 的 (x,y) 浮点坐标。
    """
    if cv2 is None:
        raise RuntimeError("cv2 不可用，无法提取中心点。")
    img = np.asarray(img_gray01, dtype=np.float32)
    if float(blur_sigma) > 1e-9:
        img = cv2.GaussianBlur(img, (0, 0), float(blur_sigma))
    mx = float(img.max())
    if not np.isfinite(mx) or mx <= 0.0:
        return np.zeros((0, 2), dtype=np.float64)
    thr = float(thr_rel) * mx
    mask = (img >= thr).astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return np.zeros((0, 2), dtype=np.float64)
    pts: list[tuple[float, float]] = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < int(min_area_threshold):
            continue
        ys, xs = np.where(labels == i)
        if xs.size == 0:
            continue
        w = img[ys, xs].astype(np.float64)
        s = float(w.sum())
        if not np.isfinite(s) or s <= 1e-12:
            cx, cy = centroids[i]
            pts.append((float(cx), float(cy)))
            continue
        cx = float((xs.astype(np.float64) * w).sum() / s)
        cy = float((ys.astype(np.float64) * w).sum() / s)
        pts.append((cx, cy))
    if not pts:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def _displacement_vectors_from_centers(
    centers_xy: np.ndarray,
    *,
    k_neighbors: int,
    r_max_px: float,
) -> np.ndarray:
    """
    从中心点构造“邻接位移向量”集合（用于估计旋转；对平移天然不敏感）。

    策略：对每个点取 kNN（排除自身），保留距离 <= r_max_px 的向量，并补全对称向量 ±v（保证 180° 周期性）。
    """
    pts = np.asarray(centers_xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 3:
        return np.zeros((0, 2), dtype=np.float64)
    k = int(max(1, k_neighbors))
    r_max = float(r_max_px)
    if not np.isfinite(r_max) or r_max <= 0.0:
        raise ValueError(f"非法 r_max_px={r_max_px}")
    tree = KDTree(pts.astype(np.float64))
    dists, idxs = tree.query(pts, k=min(k + 1, pts.shape[0]))
    vecs: list[np.ndarray] = []
    for i in range(pts.shape[0]):
        for j in range(1, idxs.shape[1]):
            dist = float(dists[i, j])
            if not np.isfinite(dist) or dist <= 1e-6 or dist > r_max:
                continue
            vecs.append((pts[int(idxs[i, j])] - pts[i]).reshape(1, 2))
    if not vecs:
        return np.zeros((0, 2), dtype=np.float64)
    V = np.vstack(vecs).astype(np.float64)
    return np.vstack([V, -V]).astype(np.float64)


def _rotate_vecs_deg(vecs_xy: np.ndarray, angle_deg: float) -> np.ndarray:
    a = math.radians(float(angle_deg))
    c = math.cos(a)
    s = math.sin(a)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return np.asarray(vecs_xy, dtype=np.float64) @ R.T


def _trimmed_mean_nn_distance(tree_ref: KDTree, pts_mov: np.ndarray, *, trim_ratio: float, min_keep: int = 120) -> float:
    dists, _ = tree_ref.query(np.asarray(pts_mov, dtype=np.float64))
    d = np.sort(np.asarray(dists, dtype=np.float64).ravel())
    if d.size == 0:
        return float("inf")
    tr = float(max(0.0, min(0.49, trim_ratio)))
    keep = int(max(min_keep, math.floor(float(d.size) * (1.0 - tr))))
    keep = int(min(keep, d.size))
    if keep <= 0:
        return float(np.mean(d))
    return float(np.mean(d[:keep]))


def _estimate_rotation_deg_from_vecs_global_search(
    vecs_ref: np.ndarray,
    vecs_mov: np.ndarray,
    *,
    coarse_step_deg: float,
    trim_ratio: float,
) -> tuple[float, float]:
    """
    用“位移向量集合”的最近邻距离来估计相对旋转（不依赖 PBC 周期向量，也不输出平移）。

    返回：(best_deg_in_[0,180), best_score)。
    """
    ref = np.asarray(vecs_ref, dtype=np.float64).reshape(-1, 2)
    mov = np.asarray(vecs_mov, dtype=np.float64).reshape(-1, 2)
    if ref.shape[0] < 50 or mov.shape[0] < 50:
        raise ValueError(f"位移向量过少：ref={ref.shape[0]} mov={mov.shape[0]}")
    step0 = float(coarse_step_deg)
    if not np.isfinite(step0) or step0 <= 0.0:
        raise ValueError(f"非法 coarse_step_deg={coarse_step_deg}")

    tree_ref = KDTree(ref)
    best_deg = 0.0
    best_score = float("inf")

    # 粗搜索：[0,180)
    deg = 0.0
    while deg < 180.0 - 1e-12:
        s = _trimmed_mean_nn_distance(tree_ref, _rotate_vecs_deg(mov, deg), trim_ratio=float(trim_ratio))
        if s < best_score:
            best_score = s
            best_deg = deg
        deg += step0

    # 逐级细化（经验：比直接做到 0.005° 更稳）
    for step, radius in [(0.2, 2.0), (0.02, 0.4), (0.005, 0.1)]:
        start = best_deg - radius
        end = best_deg + radius
        deg = start
        while deg <= end + 1e-12:
            d = float(deg) % 180.0
            s = _trimmed_mean_nn_distance(tree_ref, _rotate_vecs_deg(mov, d), trim_ratio=float(trim_ratio))
            if s < best_score:
                best_score = s
                best_deg = d
            deg += step

    return float(best_deg % 180.0), float(best_score)


def _find_twist_layer_pngs(sample_dir: Path, stem: str) -> tuple[Path, Path]:
    """
    twist 任务默认取 <stem>_0.png 与 <stem>_1.png。
    若不存在则尝试在 sample_dir 下按 *_0.png / *_1.png 兜底匹配。
    """
    p0 = sample_dir / f"{stem}_0.png"
    p1 = sample_dir / f"{stem}_1.png"
    if p0.is_file() and p1.is_file():
        return p0, p1
    cand0 = sorted([p for p in sample_dir.glob("*_0.png") if p.is_file()])
    cand1 = sorted([p for p in sample_dir.glob("*_1.png") if p.is_file()])
    if len(cand0) == 1 and len(cand1) == 1:
        return cand0[0], cand1[0]
    raise FileNotFoundError(f"未找到 layer PNG：{sample_dir}（期望 {p0.name}/{p1.name} 或唯一的 *_0.png/*_1.png）")


def build_periodic_ref_tree(
    ref_pts: np.ndarray,
    *,
    V1_px_512: np.ndarray,
    V2_px_512: np.ndarray,
    k_range: int = 1,
) -> KDTree:
    """
    将参考点按二维周期矢量 (V1/V2) 做有限范围的平移复制，构造 KDTree。

    目的：mov_pts 平移后即使跨越了晶胞边界，也能在 ref 的周期复制中找到相邻匹配点，
    避免使用非周期距离导致的“少数点距离巨大 -> mean 失真”。
    """
    k_range = int(max(0, k_range))
    if ref_pts.size == 0:
        return KDTree(ref_pts.reshape(0, 2))

    shifts: list[np.ndarray] = []
    for kx in range(-k_range, k_range + 1):
        for ky in range(-k_range, k_range + 1):
            shifts.append((float(kx) * V1_px_512 + float(ky) * V2_px_512).reshape(1, 2))
    ref_aug = np.vstack([ref_pts + s for s in shifts]).astype(np.float32)
    return KDTree(ref_aug)


def mean_nn_distance_periodic(tree_ref: KDTree, mov_pts: np.ndarray) -> float:
    if mov_pts.size == 0:
        return float("inf")
    distances, _ = tree_ref.query(mov_pts)
    return float(np.mean(distances))


def _trimmed_mean(values: np.ndarray, *, trim_ratio: float) -> float:
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return float("inf")
    trim_ratio = float(max(0.0, min(0.49, trim_ratio)))
    if trim_ratio <= 0.0:
        return float(np.mean(v))
    v = np.sort(v)
    k = int(math.floor(float(v.size) * trim_ratio))
    if v.size - 2 * k <= 0:
        return float(np.mean(v))
    return float(np.mean(v[k : (v.size - k)]))


def mean_nn_distance_robust(tree_ref: KDTree, mov_pts: np.ndarray, *, trim_ratio: float = 0.20) -> float:
    """
    非周期最近邻距离的鲁棒评分：使用 trimmed mean 降低少数边界/缺点导致的大距离影响。

    说明：这里不使用“周期复制”的 KDTree 来做对齐评分，因为允许每个点自由匹配到任意晶胞副本会让评分对平移不敏感，
    在规则晶格（例如 MoS2）上容易退化到“几乎任何平移都很小”的局部最优（常见为 0 位移）。
    """
    if mov_pts.size == 0:
        return float("inf")
    distances, _ = tree_ref.query(mov_pts)
    return _trimmed_mean(np.asarray(distances, dtype=np.float64), trim_ratio=float(trim_ratio))


def estimate_shift_translation_multiscale_periodic(
    tree_ref: KDTree,
    centers_mov: np.ndarray,
    *,
    origin: tuple[int, int] = (0, 0),
    max_shift: int,
    coarse_step: int = 4,
) -> Optional[np.ndarray]:
    """
    粗到细的整像素平移搜索（周期最近邻评分版本）：
    - 使用周期 KDTree 计算均值最近邻距离作为 score；
    - 返回“需要施加到 centers_mov 上”的平移 (dx,dy)。
    """
    if centers_mov.size == 0:
        return None

    def _search(dx0: int, dy0: int, step: int, radius: int) -> Tuple[Tuple[int, int], float]:
        best_shift: Optional[Tuple[int, int]] = None
        best_score: Optional[float] = None
        for dx in range(dx0 - radius, dx0 + radius + 1, step):
            for dy in range(dy0 - radius, dy0 + radius + 1, step):
                shifted = centers_mov + np.array([dx, dy], dtype=np.float32)
                score = mean_nn_distance_periodic(tree_ref, shifted)
                if best_score is None or score < best_score:
                    best_score = float(score)
                    best_shift = (dx, dy)
        assert best_shift is not None and best_score is not None
        return best_shift, best_score

    coarse_step = int(max(1, coarse_step))
    max_shift = int(max(1, max_shift))

    ox, oy = int(origin[0]), int(origin[1])
    best_xy, _ = _search(ox, oy, coarse_step, max_shift)
    fine_radius = max(2, coarse_step)
    best_xy, _ = _search(int(best_xy[0]), int(best_xy[1]), 1, fine_radius)
    return np.array(best_xy, dtype=np.float32)


def estimate_shift_translation_multiscale_robust(
    tree_ref: KDTree,
    centers_mov: np.ndarray,
    *,
    origin: tuple[int, int] = (0, 0),
    max_shift: int,
    coarse_step: int = 4,
    trim_ratio: float = 0.20,
) -> Optional[np.ndarray]:
    """
    粗到细的整像素平移搜索（鲁棒非周期最近邻评分版本）：
    - tree_ref 仅包含 ref 点，不做周期复制；
    - 使用 trimmed mean 的最近邻距离作为 score；
    - 返回“需要施加到 centers_mov 上”的平移 (dx,dy)。
    """
    if centers_mov.size == 0:
        return None

    def _search(dx0: int, dy0: int, step: int, radius: int) -> Tuple[Tuple[int, int], float]:
        best_shift: Optional[Tuple[int, int]] = None
        best_score: Optional[float] = None
        for dx in range(dx0 - radius, dx0 + radius + 1, step):
            for dy in range(dy0 - radius, dy0 + radius + 1, step):
                shifted = centers_mov + np.array([dx, dy], dtype=np.float32)
                score = mean_nn_distance_robust(tree_ref, shifted, trim_ratio=float(trim_ratio))
                if best_score is None or score < best_score:
                    best_score = float(score)
                    best_shift = (dx, dy)
        assert best_shift is not None and best_score is not None
        return best_shift, best_score

    coarse_step = int(max(1, coarse_step))
    max_shift = int(max(1, max_shift))

    ox, oy = int(origin[0]), int(origin[1])
    best_xy, _ = _search(ox, oy, coarse_step, max_shift)
    fine_radius = max(2, coarse_step)
    best_xy, _ = _search(int(best_xy[0]), int(best_xy[1]), 1, fine_radius)
    return np.array(best_xy, dtype=np.float32)


def load_period_vecs_from_batch_config(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    material = str(data.get("material", "")).strip().lower()

    # 内置默认：与 generate_sample/batch_runner.py 的默认值保持一致（用于 period.t1/t2 缺失时避免直接推断失败）
    defaults: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
        "res2": ((6.42, 0.0), (3.15, -5.71)),
        "mos2": ((3.18, 0.0), (1.59, -2.75)),
        "tas2": ((3.33, 0.0), (1.66, -2.88)),
        "mote2": ((3.49, 0.0), (0.0, 6.37)),
    }
    period = data.get("period") or {}
    t1 = period.get("t1_A")
    t2 = period.get("t2_A")
    if isinstance(t1, (list, tuple)) and isinstance(t2, (list, tuple)) and len(t1) == 2 and len(t2) == 2:
        return np.array([float(t1[0]), float(t1[1])], dtype=np.float64), np.array([float(t2[0]), float(t2[1])], dtype=np.float64)

    if material in defaults:
        (t1x, t1y), (t2x, t2y) = defaults[material]
        return np.array([float(t1x), float(t1y)], dtype=np.float64), np.array([float(t2x), float(t2y)], dtype=np.float64)

    structure_path = Path(data.get("structure_path", "generate_sample/one_layer.xyz"))
    if not structure_path.is_absolute():
        structure_path = (path.parent / structure_path).resolve()
    return base.infer_period_vecs_from_structure(structure_path)


def load_twist_period_deg_from_batch_config(path: Optional[Path]) -> Optional[float]:
    """
    从 generate_sample/*.json 读取 twist_period_deg（degree）。

    说明：shift_pbc 依赖二维周期矢量 T1/T2；twist 依赖旋转对称性的周期折叠参数。
    """
    if path is None:
        return None
    p = Path(path).resolve()
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    v = data.get("twist_period_deg", None)
    try:
        f = float(v)
    except Exception:
        return None
    if not np.isfinite(f) or f <= 0.0:
        return None
    return float(f)


def _resolve_twist_period_deg(args) -> float:
    """
    优先级：
    1) 显式传入 --twist_period_deg
    2) batch_config.json 里的 twist_period_deg
    3) 默认 180（与历史行为兼容）
    """
    if getattr(args, "twist_period_deg", None) is not None:
        try:
            v = float(args.twist_period_deg)
            if np.isfinite(v) and v > 0.0:
                return float(v)
        except Exception:
            pass
    bc = getattr(args, "batch_config", None)
    period = load_twist_period_deg_from_batch_config(Path(bc)) if bc else None
    return float(period) if period is not None else 180.0


def _twist_csv_is_compatible(out_csv: Path, *, period_deg: float) -> bool:
    """
    用于断点重续：当 out_csv 中的 twist_period_deg 与当前评估使用的 period 不一致时，
    旧 CSV 不可用于 resume（否则会“跳过但数值基于旧折叠周期”，导致误差看起来异常）。
    """
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        return True
    try:
        with out_csv.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            if not r.fieldnames or "twist_period_deg" not in set(r.fieldnames):
                return True
            for row in r:
                v = row.get("twist_period_deg", "")
                try:
                    pv = float(v)
                except Exception:
                    continue
                if np.isfinite(pv) and abs(float(pv) - float(period_deg)) > 1e-6:
                    return False
    except Exception:
        return True
    return True


def iter_bilayer_pngs(bilayer_dir: Path) -> Iterable[Path]:
    yield from base.iter_bilayer_pngs(bilayer_dir)


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ensure_trash_run_dir(*, prefix: str) -> Path:
    trash = _ensure_dir((_REPO_ROOT / "trash").resolve())
    run_dir = trash / f"{prefix}_{_now_tag()}_{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _move_to_trash(path: Path, *, trash_run_dir: Path, note: str) -> Path:
    """
    按项目约定：不直接删除目录/文件，统一移动到 trash/ 下。
    """
    path = path.resolve()
    if not path.exists():
        return path
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", note).strip("_") or "moved"
    dst = trash_run_dir / safe_name
    if dst.exists():
        dst = trash_run_dir / f"{safe_name}_{int(time.time())}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(dst))
    return dst


def _atomic_replace_file_with_backup(*, src: Path, dst: Path, backup_dir: Optional[Path]) -> None:
    """
    原子替换单个文件，并（可选）将旧文件副本备份到 backup_dir。

    设计目标：
    - 避免“先删除/移动旧文件再写新文件”导致的中断窗口；
    - 避免对 results_root 做破坏性清空，提升断点重续稳定性。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if backup_dir is not None and dst.exists() and dst.is_file():
        backup_dir.mkdir(parents=True, exist_ok=True)
        b = backup_dir / dst.name
        if b.exists():
            b = backup_dir / f"{dst.stem}_{int(time.time() * 1e6)}{dst.suffix}"
        shutil.copy2(str(dst), str(b))

    tmp = dst.parent / f".{dst.name}.tmp_{os.getpid()}_{int(time.time() * 1e6)}"
    shutil.copy2(str(src), str(tmp))
    os.replace(str(tmp), str(dst))


def _atomic_sync_sample_outputs(
    *,
    src_dir: Path,
    dst_dir: Path,
    stem: str,
    backup_root: Optional[Path],
) -> None:
    """
    将 src_dir 下属于该 stem 的输出文件同步到 dst_dir：
    - 只同步常见输出（png/csv，且文件名前缀匹配 stem）；
    - 采用“备份 + 原子替换”策略，避免中断造成 dst_dir 变空。
    """
    if not src_dir.is_dir():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = (backup_root / stem) if backup_root is not None else None

    for p in sorted(src_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".png", ".csv"}:
            continue
        if not p.name.startswith(f"{stem}_"):
            continue
        _atomic_replace_file_with_backup(src=p, dst=dst_dir / p.name, backup_dir=backup_dir)


def _list_input_pngs(input_dir: Path) -> list[Path]:
    """
    从输入目录中列出“原始双层图像”PNG（仅扫描一级目录，避免把 result/ 下的分解图也当输入）。
    """
    if not input_dir.is_dir():
        return []
    pngs = []
    for p in sorted(input_dir.glob("*.png")):
        name = p.name
        if name.endswith(("_0.png", "_1.png", "_conbine.png", "_original.png", "_combine.png")):
            continue
        pngs.append(p)
    return pngs


def _load_yaml_snapshot(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"separate_yml 不是有效的 YAML 映射：{path}")
    return data


def _dump_yaml_to_path(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    path.write_text(text, encoding="utf-8")


def _prepare_separate_cfg(
    base_cfg: dict,
    *,
    input_dir: Path,
    output_dir: Path,
    selected_files: Optional[list[str]],
) -> dict:
    cfg = copy.deepcopy(base_cfg)
    paths = cfg.get("paths") if isinstance(cfg.get("paths"), dict) else {}
    paths["input"] = str(input_dir)
    paths["output"] = str(output_dir)
    cfg["paths"] = paths

    if selected_files is not None:
        proc = cfg.get("processing") if isinstance(cfg.get("processing"), dict) else {}
        proc["selected_files"] = list(selected_files)
        cfg["processing"] = proc
    return cfg


def _override_superposition_params(cfg: dict, *, adaptive_superposition: Optional[bool], fixed_superposition_k: Optional[float]) -> dict:
    """
    覆写分解配置中的叠加相关参数（用于 NaN 样本的自动修复重跑）。

    兼容不同 YAML 结构：若缺少 separation 节点会自动创建。
    """
    cfg2 = copy.deepcopy(cfg)
    sep = cfg2.get("separation")
    if not isinstance(sep, dict):
        sep = {}
    if adaptive_superposition is not None:
        sep["adaptive_superposition"] = bool(adaptive_superposition)
    if fixed_superposition_k is not None:
        sep["fixed_superposition_k"] = float(fixed_superposition_k)
    cfg2["separation"] = sep
    return cfg2


def _is_nan_number(x) -> bool:
    try:
        v = float(x)
    except Exception:
        return True
    return math.isnan(v)


def _run_decompose(
    *,
    separate_cfg: dict,
    cfg_path: Path,
    seed: int,
    verbose: str,
) -> None:
    _dump_yaml_to_path(separate_cfg, cfg_path)
    cmd = [
        sys.executable,
        str((_REPO_ROOT / "src" / "main.py").resolve()),
        "--config",
        str(cfg_path),
        "--seed",
        str(int(seed)),
        "--verbose",
        str(verbose),
    ]
    subprocess.run(cmd, check=True)


def _eval_one_twist(
    *,
    stem: str,
    results_root: Path,
    debug_vis_dir: Optional[Path],
    twist_period_deg: float,
    thr_rel: float,
    thr_rel_candidates: Optional[str],
    min_area: int,
    blur_sigma: float,
    k_neighbors: int,
    r_max_px: float,
    trim_ratio: float,
    coarse_step_deg: float,
) -> Optional[dict]:
    gt_r = _parse_twist_deg_from_stem(stem)
    if gt_r is None:
        return None
    sample_dir = results_root / stem
    try:
        p0, p1 = _find_twist_layer_pngs(sample_dir, stem)
    except Exception:
        return None

    img0 = _load_gray01(p0)
    img1 = _load_gray01(p1)

    if thr_rel_candidates:
        thr_candidates = []
        for s in str(thr_rel_candidates).split(","):
            s = s.strip()
            if not s:
                continue
            try:
                thr_candidates.append(float(s))
            except Exception:
                pass
        if not thr_candidates:
            thr_candidates = [float(thr_rel)]
    else:
        thr_candidates = [float(thr_rel)]

    best = None
    best_payload = None
    for thr in thr_candidates:
        try:
            c0 = _find_centers_cc_relmax_weighted(
                img0,
                thr_rel=float(thr),
                min_area_threshold=int(min_area),
                blur_sigma=float(blur_sigma),
            )
            c1 = _find_centers_cc_relmax_weighted(
                img1,
                thr_rel=float(thr),
                min_area_threshold=int(min_area),
                blur_sigma=float(blur_sigma),
            )
            v0 = _displacement_vectors_from_centers(
                c0,
                k_neighbors=int(k_neighbors),
                r_max_px=float(r_max_px),
            )
            v1 = _displacement_vectors_from_centers(
                c1,
                k_neighbors=int(k_neighbors),
                r_max_px=float(r_max_px),
            )
            rot_deg_raw, score = _estimate_rotation_deg_from_vecs_global_search(
                v0,
                v1,
                coarse_step_deg=float(coarse_step_deg),
                trim_ratio=float(trim_ratio),
            )
        except Exception:
            continue
        if best is None or score < best[0]:
            best = (float(score), float(rot_deg_raw), float(thr))
            best_payload = (c0, c1, v0, v1)

    if best is None or best_payload is None:
        return None

    score, rot_deg_raw, thr_used = best
    centers0, centers1, vecs0, vecs1 = best_payload

    if debug_vis_dir is not None:
        meta0 = {
            "stem": stem,
            "task": "twist",
            "tag": "0",
            "thr_rel": float(thr_used),
            "min_area": int(min_area),
            "blur_sigma": float(blur_sigma),
        }
        meta1 = dict(meta0)
        meta1["tag"] = "1"
        _render_centers_debug(
            out_dir=debug_vis_dir,
            stem=stem,
            tag="0",
            img01=img0,
            centers=[(float(x), float(y)) for (x, y) in centers0.tolist()],
            method="cc_relmax_twist",
            meta=meta0,
        )
        _render_centers_debug(
            out_dir=debug_vis_dir,
            stem=stem,
            tag="1",
            img01=img1,
            centers=[(float(x), float(y)) for (x, y) in centers1.tolist()],
            method="cc_relmax_twist",
            meta=meta1,
        )

    twist_period = float(twist_period_deg)
    gt_twist = _twist_fold_deg(gt_r, period_deg=twist_period)
    pred_twist = _twist_fold_deg(rot_deg_raw, period_deg=twist_period)
    abs_err = float(abs(pred_twist - gt_twist))

    return {
        "stem": stem,
        "gt_r_deg_raw": float(gt_r),
        "gt_twist_deg": float(gt_twist),
        "pred_twist_deg": float(pred_twist),
        "abs_err_deg": float(abs_err),
        "score": float(score),
        "twist_period_deg": float(twist_period),
        "thr_rel": float(thr_used),
        "min_area": int(min_area),
        "k_neighbors": int(k_neighbors),
        "r_max_px": float(r_max_px),
        "trim_ratio": float(trim_ratio),
        "coarse_step_deg": float(coarse_step_deg),
        "centers0": int(centers0.shape[0]),
        "centers1": int(centers1.shape[0]),
        "vecs0": int(vecs0.shape[0]),
        "vecs1": int(vecs1.shape[0]),
    }


def _eval_one_twist_from_labels_npz(
    *,
    stem: str,
    labels_root: Path,
    debug_vis_dir: Optional[Path],
    twist_period_deg: float,
    k_neighbors: int,
    r_max_px: float,
    trim_ratio: float,
    coarse_step_deg: float,
) -> Optional[dict]:
    """
    twist_labels：从 labels_root/<stem>.npz 读取两层点云直接评估扭角。

    说明：
    - labels 文件由 generate_sample/batch_runner.py 在设置 MOIRE_SAVE_LABELS=1 时导出；
    - 当前只使用 layer1_xy128/layer2_xy128（与图像同为 128×128 坐标系），不依赖分解结果与阈值超参。
    """
    gt_r = _parse_twist_deg_from_stem(stem)
    if gt_r is None:
        return None
    npz_path = labels_root / f"{stem}.npz"
    if not npz_path.is_file():
        return None
    try:
        data = np.load(npz_path, allow_pickle=True)
        c0 = np.asarray(data["layer1_xy128"], dtype=np.float64).reshape(-1, 2)
        c1 = np.asarray(data["layer2_xy128"], dtype=np.float64).reshape(-1, 2)
    except Exception:
        return None

    try:
        v0 = _displacement_vectors_from_centers(c0, k_neighbors=int(k_neighbors), r_max_px=float(r_max_px))
        v1 = _displacement_vectors_from_centers(c1, k_neighbors=int(k_neighbors), r_max_px=float(r_max_px))
        rot_deg_raw, score = _estimate_rotation_deg_from_vecs_global_search(
            v0,
            v1,
            coarse_step_deg=float(coarse_step_deg),
            trim_ratio=float(trim_ratio),
        )
    except Exception:
        return None

    if debug_vis_dir is not None:
        img01 = np.zeros((128, 128), dtype=np.float32)
        meta0 = {"stem": stem, "task": "twist_labels", "tag": "0", "source": "labels_npz"}
        meta1 = dict(meta0)
        meta1["tag"] = "1"
        _render_centers_debug(
            out_dir=debug_vis_dir,
            stem=stem,
            tag="labels0",
            img01=img01,
            centers=[(float(x), float(y)) for (x, y) in c0.tolist()],
            method="labels_npz",
            meta=meta0,
        )
        _render_centers_debug(
            out_dir=debug_vis_dir,
            stem=stem,
            tag="labels1",
            img01=img01,
            centers=[(float(x), float(y)) for (x, y) in c1.tolist()],
            method="labels_npz",
            meta=meta1,
        )

    twist_period = float(twist_period_deg)
    gt_twist = _twist_fold_deg(gt_r, period_deg=twist_period)
    pred_twist = _twist_fold_deg(rot_deg_raw, period_deg=twist_period)
    abs_err = float(abs(pred_twist - gt_twist))

    return {
        "stem": stem,
        "gt_r_deg_raw": float(gt_r),
        "gt_twist_deg": float(gt_twist),
        "pred_twist_deg": float(pred_twist),
        "abs_err_deg": float(abs_err),
        "score": float(score),
        "twist_period_deg": float(twist_period),
        "thr_rel": float("nan"),
        "min_area": 0,
        "k_neighbors": int(k_neighbors),
        "r_max_px": float(r_max_px),
        "trim_ratio": float(trim_ratio),
        "coarse_step_deg": float(coarse_step_deg),
        "centers0": int(c0.shape[0]),
        "centers1": int(c1.shape[0]),
        "vecs0": int(v0.shape[0]),
        "vecs1": int(v1.shape[0]),
    }


def _write_shift_csv(out_csv: Path, results: list[base.SampleResult]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_HEADER)
        for res in results:
            writer.writerow(
                [
                    res.stem,
                    res.gtx,
                    res.gty,
                    res.sideA,
                    res.shift_pred_128_px[0],
                    res.shift_pred_128_px[1],
                    res.err_A,
                    res.kx,
                    res.ky,
                    int(res.swapped),
                    res.centers0,
                    res.centers1,
                ]
            )


def _write_twist_csv(out_csv: Path, rows: list[dict]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_HEADER_TWIST)
        for row in rows:
            writer.writerow([row.get(k) for k in _CSV_HEADER_TWIST])


def _load_shift_csv_best(out_csv: Path) -> dict[str, base.SampleResult]:
    """
    读取 shift_pbc CSV（可能存在重复 stem 行），返回“最后一次出现”为准的 best dict。
    说明：该 dict 用于断点重续与一体化流程的多阶段状态承载。
    """
    best: dict[str, base.SampleResult] = {}
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        return best
    with out_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            stem = str(row.get("stem", "")).strip()
            if not stem:
                continue
            try:
                gtx = float(row.get("gtx", "nan"))
                gty = float(row.get("gty", "nan"))
                sideA = float(row.get("sideA", "nan"))
                predx = float(row.get("predx_128", "nan"))
                predy = float(row.get("predy_128", "nan"))
                err_A = float(row.get("err_A", "nan"))
                kx = int(float(row.get("kx", "0")))
                ky = int(float(row.get("ky", "0")))
                swapped = bool(int(float(row.get("swapped", "0"))))
                centers0 = int(float(row.get("centers0", "0")))
                centers1 = int(float(row.get("centers1", "0")))
            except Exception:
                continue
            best[stem] = base.SampleResult(
                stem=stem,
                gtx=gtx,
                gty=gty,
                sideA=sideA,
                shift_pred_128_px=(predx, predy),
                shift_gt_128_px=(gtx, gty),
                err_A=err_A,
                kx=kx,
                ky=ky,
                swapped=swapped,
                centers0=centers0,
                centers1=centers1,
            )
    return best


def _load_twist_csv_best(out_csv: Path) -> dict[str, dict]:
    """
    读取 twist CSV（可能存在重复 stem 行），返回“最后一次出现”为准的 best dict。
    """
    best: dict[str, dict] = {}
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        return best
    with out_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            stem = str(row.get("stem", "")).strip()
            if not stem:
                continue
            best[stem] = dict(row)
    return best


def _has_shift_outputs(results_root: Path, stem: str) -> bool:
    d = results_root / stem
    if not d.is_dir():
        return False
    return (d / f"{stem}_0.png").is_file() and (d / f"{stem}_1.png").is_file()


def _flush_shift_best_csv(out_csv: Path, best_by_stem: dict[str, base.SampleResult]) -> None:
    # 保证输出干净：每次刷新都重写为“每 stem 一行”的最优结果
    ordered = sorted(best_by_stem.values(), key=lambda r: r.stem)
    _write_shift_csv(out_csv, ordered)


def _flush_twist_best_csv(out_csv: Path, best_by_stem: dict[str, dict]) -> None:
    rows = list(best_by_stem.values())
    rows.sort(key=lambda r: str(r.get("stem", "")))
    _write_twist_csv(out_csv, rows)


def _run_pipeline_decompose_eval_retry(args, *, results_root: Path, out_csv: Path, debug_vis_dir: Optional[Path]) -> int:
    """
    新工作流：
    1) 从 separate_yml 快照配置到内存，并强制将 input/output 指向 results_root.parent / results_root；
    2) 调用 src/main.py 完成分解；
    3) 评估并写入 CSV；
    4) 对超过阈值的样本做多 seed 重试（默认 4 次），以最小误差的分解结果覆盖回 results_root，并重写 CSV。
    """
    task = str(args.task)
    if not args.separate_yml:
        raise ValueError("internal: separate_yml is required")

    separate_yml = Path(args.separate_yml).resolve()
    base_cfg = _load_yaml_snapshot(separate_yml)

    input_dir = results_root.parent.resolve()
    pngs = _list_input_pngs(input_dir)
    if not pngs:
        raise SystemExit(f"未找到输入 PNG：{input_dir}/*.png（请确认 results_root 的父目录是双层图输出目录）")

    if args.stems:
        stems = list(args.stems)
    else:
        stems = [p.stem for p in pngs]
    if args.limit is not None:
        stems = list(stems)[: int(args.limit)]
    if not stems:
        raise SystemExit("stems 为空，无法运行。")

    # 防呆：shift_pbc 必须能从文件名解析出 gtx/gty/sideA（slip 数据集命名）。
    # 若用户误把 rotation 数据集（如 MoS2_r0.14.png）当 shift_pbc 运行，会导致全部样本无法评估 -> 全量进入重试，反复分解 1000 张图，耗时极长。
    if task == "shift_pbc":
        probe = stems[: min(50, len(stems))]
        ok = 0
        for s in probe:
            if base.parse_gt_from_name(f"{s}.png") is not None:
                ok += 1
        if ok == 0:
            example = probe[0] if probe else "<none>"
            raise SystemExit(
                "当前 task=shift_pbc 需要输入文件名包含 _gtx/_gty/_sideA/_v1/_v2 等 slip 命名字段，"
                f"但抽样检查 {len(probe)} 张均无法解析（例如：{example}.png）。\n"
                "这通常表示你正在处理 rotation 数据集（命名如 <Mat>_r0.14.png），请改用：\n"
                "  python tools/evaluate_generate_sample_wraparound_pbc.py --task twist --results_root <.../rotation/result>\n"
                "（如需一体化分解+评估，可额外传 --separate_yml）。"
            )

    trash_run_dir = _ensure_trash_run_dir(prefix=f"eval_autopipeline_{task}")
    # 保存 yml 快照（防止并发实验时 yml 被修改）
    shutil.copy2(str(separate_yml), str(trash_run_dir / "separate_yml_original.yml"))

    # 断点重续：从 out_csv 恢复已完成的评估结果（以及当前 best）
    resume_enabled = bool(getattr(args, "resume", True))
    if task == "shift_pbc":
        best_by_stem_shift = _load_shift_csv_best(out_csv) if resume_enabled else {}
        done_stems = set(best_by_stem_shift.keys())
    else:
        best_by_stem_twist = _load_twist_csv_best(out_csv) if resume_enabled else {}
        done_stems = set(best_by_stem_twist.keys())

    # baseline：只对“缺少分解输出”的 stem 做分解；默认不清空 results_root（大规模实验必须支持断点重续）
    overwrite_backup_dir = _ensure_dir(trash_run_dir / "baseline_overwrite_backup")
    stems_need_decompose: list[str] = []
    for stem in stems:
        if stem in done_stems and _has_shift_outputs(results_root, stem) and task == "shift_pbc":
            continue
        if task == "twist":
            # twist 同样需要 0/1 两层图：复用 _find_twist_layer_pngs 的逻辑
            sample_dir = results_root / stem
            try:
                _find_twist_layer_pngs(sample_dir, stem)
                has_out = True
            except Exception:
                has_out = False
            if stem in done_stems and has_out:
                continue
            if not has_out:
                stems_need_decompose.append(stem)
            continue
        # shift_pbc
        if not _has_shift_outputs(results_root, stem):
            stems_need_decompose.append(stem)

    # baseline 分解（一次加载权重，分解所有缺失样本）
    baseline_seed = int(getattr(args, "separate_seed", 1234))
    separate_verbose = str(getattr(args, "separate_verbose", "info"))
    if stems_need_decompose:
        baseline_cfg = _prepare_separate_cfg(
            base_cfg,
            input_dir=input_dir,
            output_dir=results_root,
            selected_files=[f"{s}.png" for s in stems_need_decompose],
        )
        baseline_cfg_path = trash_run_dir / "separate_cfg_snapshot.yml"
        _run_decompose(separate_cfg=baseline_cfg, cfg_path=baseline_cfg_path, seed=baseline_seed, verbose=separate_verbose)

    # ---------------- eval baseline ----------------
    if task == "shift_pbc":
        T1, T2 = load_period_vecs_from_batch_config(Path(args.batch_config))
        best_by_stem: dict[str, base.SampleResult] = _load_shift_csv_best(out_csv) if resume_enabled else {}
        # baseline 评估：仅评估尚未写入 CSV 的 stem（或缺少分解输出导致之前无法评估）
        updated = 0
        for idx, stem in enumerate(stems, start=1):
            if (
                resume_enabled
                and stem in best_by_stem
                and _has_shift_outputs(results_root, stem)
                and not math.isnan(float(best_by_stem[stem].err_A))
            ):
                continue
            if not _has_shift_outputs(results_root, stem):
                continue
            gt = base.parse_gt_from_name(f"{stem}.png")
            if gt is None:
                continue
            _r, _gtx, _gty, sideA, _v1, _v2 = gt
            if args.max_shift_px_512 is not None:
                max_shift_px_512 = int(args.max_shift_px_512)
            else:
                A_per_px_512 = float(sideA) / 512.0
                max_shift_px_512 = int(math.ceil(float(args.max_shift_A) / max(A_per_px_512, 1e-9)))

            material_key = _material_key_from_stem(stem)
            center_method, bina_thre, min_area, blur_sigma, center_thr_rel = _apply_default_center_params(
                material_key=material_key,
                center_method=None if args.center_method is None else str(args.center_method),
                bina_thre=None if args.bina_thre is None else float(args.bina_thre),
                min_area_threshold=None if args.min_area is None else int(args.min_area),
                blur_sigma=None if args.blur_sigma is None else float(args.blur_sigma),
                thr_rel=None if args.center_thr_rel is None else float(args.center_thr_rel),
            )

            res = evaluate_one_pbc(
                stem=stem,
                results_root=results_root,
                T1=T1,
                T2=T2,
                up_factor=int(args.up_factor),
                max_shift_px_512=int(max_shift_px_512),
                max_shift_A=float(args.max_shift_A),
                coarse_step=int(args.coarse_step),
                min_area_threshold=int(min_area),
                bina_thre=float(bina_thre),
                blur_sigma=float(blur_sigma),
                center_method=str(center_method),
                center_thr_rel=float(center_thr_rel),
                debug_vis_dir=debug_vis_dir,
                k_range=int(args.k_range),
                pbc_k_range=int(args.pbc_k_range),
                multi_origin=bool(args.multi_origin),
            )
            if res is not None:
                best_by_stem[stem] = res
                updated += 1
                # 逐步刷新 CSV，支持断点重续（避免长跑中断造成整体回滚）
                if updated <= 3 or updated % 20 == 0:
                    _flush_shift_best_csv(out_csv, best_by_stem)

        # ---------------- nan 修复：叠加系数改为 0.5 再分解/评估 ----------------
        # 说明：少数样本会出现某一层几乎为空，导致 centers<3 -> err_A=NaN；此时优先尝试关闭自适应叠加并固定 k=0.5 重跑分解。
        nanfix_enabled = bool(getattr(args, "nanfix_superposition", True))
        nanfix_k = float(getattr(args, "nanfix_superposition_k", 0.5))
        nan_stems = [s for s, r in best_by_stem.items() if r is not None and math.isnan(float(r.err_A))]
        if nanfix_enabled and nan_stems:
            nanfix_root = _ensure_dir(trash_run_dir / "nanfix_superposition_k")
            out_nanfix = (nanfix_root / f"k_{nanfix_k:g}" / "result").resolve()
            if out_nanfix.exists():
                _move_to_trash(out_nanfix, trash_run_dir=nanfix_root, note=f"k_{nanfix_k:g}_overwrite")
            out_nanfix.mkdir(parents=True, exist_ok=True)

            nanfix_cfg = _override_superposition_params(
                base_cfg,
                adaptive_superposition=False,
                fixed_superposition_k=float(nanfix_k),
            )
            nanfix_cfg = _prepare_separate_cfg(
                nanfix_cfg,
                input_dir=input_dir,
                output_dir=out_nanfix,
                selected_files=[f"{s}.png" for s in nan_stems],
            )
            nanfix_cfg_path = nanfix_root / f"k_{nanfix_k:g}" / "separate_cfg_snapshot.yml"
            _run_decompose(separate_cfg=nanfix_cfg, cfg_path=nanfix_cfg_path, seed=baseline_seed, verbose=separate_verbose)

            # 评估并回填：只要从 NaN 变成非 NaN 就覆盖（用原子替换文件避免目录变空）
            for stem in nan_stems:
                gt = base.parse_gt_from_name(f"{stem}.png")
                if gt is None:
                    continue
                _r, _gtx, _gty, sideA, _v1, _v2 = gt
                if args.max_shift_px_512 is not None:
                    max_shift_px_512 = int(args.max_shift_px_512)
                else:
                    A_per_px_512 = float(sideA) / 512.0
                    max_shift_px_512 = int(math.ceil(float(args.max_shift_A) / max(A_per_px_512, 1e-9)))

                material_key = _material_key_from_stem(stem)
                center_method, bina_thre, min_area, blur_sigma, center_thr_rel = _apply_default_center_params(
                    material_key=material_key,
                    center_method=None if args.center_method is None else str(args.center_method),
                    bina_thre=None if args.bina_thre is None else float(args.bina_thre),
                    min_area_threshold=None if args.min_area is None else int(args.min_area),
                    blur_sigma=None if args.blur_sigma is None else float(args.blur_sigma),
                    thr_rel=None if args.center_thr_rel is None else float(args.center_thr_rel),
                )
                nanfix_res = evaluate_one_pbc(
                    stem=stem,
                    results_root=out_nanfix,
                    T1=T1,
                    T2=T2,
                    up_factor=int(args.up_factor),
                    max_shift_px_512=int(max_shift_px_512),
                    max_shift_A=float(args.max_shift_A),
                    coarse_step=int(args.coarse_step),
                    min_area_threshold=int(min_area),
                    bina_thre=float(bina_thre),
                    blur_sigma=float(blur_sigma),
                    center_method=str(center_method),
                    center_thr_rel=float(center_thr_rel),
                    debug_vis_dir=None,
                    k_range=int(args.k_range),
                    pbc_k_range=int(args.pbc_k_range),
                    multi_origin=bool(args.multi_origin),
                )
                if nanfix_res is None or math.isnan(float(nanfix_res.err_A)):
                    continue

                # 当前 best 必为 NaN：因此只要 nanfix 得到非 NaN 就更新
                src_dir = out_nanfix / stem
                if src_dir.is_dir():
                    _atomic_sync_sample_outputs(
                        src_dir=src_dir,
                        dst_dir=results_root / stem,
                        stem=stem,
                        backup_root=_ensure_dir(trash_run_dir / "nanfix_replaced_original"),
                    )
                best_by_stem[stem] = nanfix_res
                _flush_shift_best_csv(out_csv, best_by_stem)

        # ---------------- retry bad ----------------
        retry_thr_A = float(getattr(args, "retry_thr_A", 0.1))
        retry_runs = int(getattr(args, "retry_runs", 4))
        retry_seed_stride = int(getattr(args, "retry_seed_stride", 1))

        retry_root = _ensure_dir(trash_run_dir / "retries")
        replaced_root = _ensure_dir(trash_run_dir / "replaced_original")
        retry_overwrite_backup = _ensure_dir(trash_run_dir / "retry_overwrite_backup")

        # bad stems：err_A 为 NaN 或 > 阈值（或 baseline 根本没评估出结果/输出缺失）
        bad_stems: list[str] = []
        for stem in stems:
            cur = best_by_stem.get(stem)
            if cur is None:
                bad_stems.append(stem)
                continue
            e = float(cur.err_A)
            if math.isnan(e) or e > retry_thr_A:
                bad_stems.append(stem)

        # 批量重试：每个 seed 只启动一次分解（避免每张图重复加载权重）
        for i in range(retry_runs):
            if not bad_stems:
                break
            seed = baseline_seed + (i + 1) * retry_seed_stride
            out_retry = (retry_root / f"seed_{seed:04d}" / "result").resolve()
            if out_retry.exists():
                _move_to_trash(out_retry, trash_run_dir=retry_overwrite_backup, note=f"seed_{seed:04d}")
            out_retry.mkdir(parents=True, exist_ok=True)

            retry_cfg = _prepare_separate_cfg(
                base_cfg,
                input_dir=input_dir,
                output_dir=out_retry,
                selected_files=[f"{s}.png" for s in bad_stems],
            )
            retry_cfg_path = retry_root / f"seed_{seed:04d}" / "separate_cfg_snapshot.yml"
            _run_decompose(separate_cfg=retry_cfg, cfg_path=retry_cfg_path, seed=int(seed), verbose=separate_verbose)

            # eval：逐 stem 更新最优
            for stem in bad_stems:
                gt = base.parse_gt_from_name(f"{stem}.png")
                if gt is None:
                    continue
                _r, _gtx, _gty, sideA, _v1, _v2 = gt
                if args.max_shift_px_512 is not None:
                    max_shift_px_512 = int(args.max_shift_px_512)
                else:
                    A_per_px_512 = float(sideA) / 512.0
                    max_shift_px_512 = int(math.ceil(float(args.max_shift_A) / max(A_per_px_512, 1e-9)))

                material_key = _material_key_from_stem(stem)
                center_method, bina_thre, min_area, blur_sigma, center_thr_rel = _apply_default_center_params(
                    material_key=material_key,
                    center_method=None if args.center_method is None else str(args.center_method),
                    bina_thre=None if args.bina_thre is None else float(args.bina_thre),
                    min_area_threshold=None if args.min_area is None else int(args.min_area),
                    blur_sigma=None if args.blur_sigma is None else float(args.blur_sigma),
                    thr_rel=None if args.center_thr_rel is None else float(args.center_thr_rel),
                )
                retry_res = evaluate_one_pbc(
                    stem=stem,
                    results_root=out_retry,
                    T1=T1,
                    T2=T2,
                    up_factor=int(args.up_factor),
                    max_shift_px_512=int(max_shift_px_512),
                    max_shift_A=float(args.max_shift_A),
                    coarse_step=int(args.coarse_step),
                    min_area_threshold=int(min_area),
                    bina_thre=float(bina_thre),
                    blur_sigma=float(blur_sigma),
                    center_method=str(center_method),
                    center_thr_rel=float(center_thr_rel),
                    debug_vis_dir=None,
                    k_range=int(args.k_range),
                    pbc_k_range=int(args.pbc_k_range),
                    multi_origin=bool(args.multi_origin),
                )
                if retry_res is None:
                    continue
                if math.isnan(float(retry_res.err_A)):
                    continue

                cur_best = best_by_stem.get(stem)
                if cur_best is None or math.isnan(float(cur_best.err_A)) or float(retry_res.err_A) < float(cur_best.err_A):
                    # 覆盖回 results_root：只在确实更优时替换（原子替换文件，避免中断窗口导致样本目录变空）
                    src_dir = out_retry / stem
                    if src_dir.is_dir():
                        _atomic_sync_sample_outputs(
                            src_dir=src_dir,
                            dst_dir=results_root / stem,
                            stem=stem,
                            backup_root=replaced_root,
                        )
                    best_by_stem[stem] = retry_res
                    _flush_shift_best_csv(out_csv, best_by_stem)

            # 更新待重试集合：已达标的不再继续重试
            remaining: list[str] = []
            for stem in bad_stems:
                cur = best_by_stem.get(stem)
                if cur is None:
                    remaining.append(stem)
                    continue
                e = float(cur.err_A)
                if math.isnan(e) or e > retry_thr_A:
                    remaining.append(stem)
            bad_stems = remaining

        _flush_shift_best_csv(out_csv, best_by_stem)
        return 0

    # ---------------- twist ----------------
    if task == "twist":
        # twist_period：优先命令行，其次 batch_config.json，其次默认 180
        args.twist_period_deg = _resolve_twist_period_deg(args)

        # 若 period 不一致，则旧 CSV 不可用于 resume：备份后重算
        if resume_enabled and not _twist_csv_is_compatible(out_csv, period_deg=float(args.twist_period_deg)):
            backup_dir = _ensure_trash_run_dir(prefix="eval_twist_incompatible_csv")
            _move_to_trash(out_csv, trash_run_dir=backup_dir, note=out_csv.name)
            resume_enabled = False

        best_by_stem: dict[str, dict] = _load_twist_csv_best(out_csv) if resume_enabled else {}
        updated = 0
        for stem in stems:
            if resume_enabled and stem in best_by_stem and not _is_nan_number(best_by_stem[stem].get("abs_err_deg", float("nan"))):
                sample_dir = results_root / stem
                try:
                    _find_twist_layer_pngs(sample_dir, stem)
                    continue
                except Exception:
                    pass
            sample_dir = results_root / stem
            try:
                _find_twist_layer_pngs(sample_dir, stem)
            except Exception:
                continue

            row = _eval_one_twist(
                stem=stem,
                results_root=results_root,
                debug_vis_dir=debug_vis_dir,
                twist_period_deg=float(args.twist_period_deg),
                thr_rel=float(args.twist_thr_rel),
                thr_rel_candidates=None if args.twist_thr_rel_candidates is None else str(args.twist_thr_rel_candidates),
                min_area=int(args.twist_min_area),
                blur_sigma=float(args.twist_blur_sigma),
                k_neighbors=int(args.twist_k_neighbors),
                r_max_px=float(args.twist_r_max_px),
                trim_ratio=float(args.twist_trim_ratio),
                coarse_step_deg=float(args.twist_coarse_step_deg),
            )
            if row is not None:
                best_by_stem[stem] = row
                updated += 1
                if updated <= 3 or updated % 50 == 0:
                    _flush_twist_best_csv(out_csv, best_by_stem)

        # ---------------- twist nan 修复：仅对 abs_err_deg=NaN 的样本，用固定叠加系数 k=0.5 重跑分解 ----------------
        nanfix_enabled = bool(getattr(args, "nanfix_superposition", True))
        nanfix_k = float(getattr(args, "nanfix_superposition_k", 0.5))
        nan_stems = [s for s, row in best_by_stem.items() if _is_nan_number(row.get("abs_err_deg", float("nan")))]
        if nanfix_enabled and nan_stems:
            nanfix_root = _ensure_dir(trash_run_dir / "nanfix_superposition_k")
            out_nanfix = (nanfix_root / f"k_{nanfix_k:g}" / "result").resolve()
            if out_nanfix.exists():
                _move_to_trash(out_nanfix, trash_run_dir=nanfix_root, note=f"k_{nanfix_k:g}_overwrite")
            out_nanfix.mkdir(parents=True, exist_ok=True)

            nanfix_cfg = _override_superposition_params(
                base_cfg,
                adaptive_superposition=False,
                fixed_superposition_k=float(nanfix_k),
            )
            nanfix_cfg = _prepare_separate_cfg(
                nanfix_cfg,
                input_dir=input_dir,
                output_dir=out_nanfix,
                selected_files=[f"{s}.png" for s in nan_stems],
            )
            nanfix_cfg_path = nanfix_root / f"k_{nanfix_k:g}" / "separate_cfg_snapshot.yml"
            _run_decompose(separate_cfg=nanfix_cfg, cfg_path=nanfix_cfg_path, seed=baseline_seed, verbose=separate_verbose)

            for stem in nan_stems:
                row = _eval_one_twist(
                    stem=stem,
                    results_root=out_nanfix,
                    debug_vis_dir=None,
                    twist_period_deg=float(args.twist_period_deg),
                    thr_rel=float(args.twist_thr_rel),
                    thr_rel_candidates=None if args.twist_thr_rel_candidates is None else str(args.twist_thr_rel_candidates),
                    min_area=int(args.twist_min_area),
                    blur_sigma=float(args.twist_blur_sigma),
                    k_neighbors=int(args.twist_k_neighbors),
                    r_max_px=float(args.twist_r_max_px),
                    trim_ratio=float(args.twist_trim_ratio),
                    coarse_step_deg=float(args.twist_coarse_step_deg),
                )
                if row is None:
                    continue
                if _is_nan_number(row.get("abs_err_deg", float("nan"))):
                    continue

                src_dir = out_nanfix / stem
                if src_dir.is_dir():
                    _atomic_sync_sample_outputs(
                        src_dir=src_dir,
                        dst_dir=results_root / stem,
                        stem=stem,
                        backup_root=_ensure_dir(trash_run_dir / "nanfix_replaced_original"),
                    )
                best_by_stem[stem] = row
                _flush_twist_best_csv(out_csv, best_by_stem)

        retry_thr_deg = float(getattr(args, "retry_thr_deg", 0.5))
        retry_runs = int(getattr(args, "retry_runs", 4))
        retry_seed_stride = int(getattr(args, "retry_seed_stride", 1))
        retry_root = _ensure_dir(trash_run_dir / "retries")
        replaced_root = _ensure_dir(trash_run_dir / "replaced_original")
        retry_overwrite_backup = _ensure_dir(trash_run_dir / "retry_overwrite_backup")

        bad_stems: list[str] = []
        for stem in stems:
            base_row = best_by_stem.get(stem)
            if base_row is None:
                bad_stems.append(stem)
                continue
            try:
                base_err = float(base_row.get("abs_err_deg", float("nan")))
            except Exception:
                base_err = float("nan")
            if math.isnan(base_err) or base_err > retry_thr_deg:
                bad_stems.append(stem)

        for i in range(retry_runs):
            if not bad_stems:
                break
            seed = baseline_seed + (i + 1) * retry_seed_stride
            out_retry = (retry_root / f"seed_{seed:04d}" / "result").resolve()
            if out_retry.exists():
                _move_to_trash(out_retry, trash_run_dir=retry_overwrite_backup, note=f"seed_{seed:04d}")
            out_retry.mkdir(parents=True, exist_ok=True)

            retry_cfg = _prepare_separate_cfg(
                base_cfg,
                input_dir=input_dir,
                output_dir=out_retry,
                selected_files=[f"{s}.png" for s in bad_stems],
            )
            retry_cfg_path = retry_root / f"seed_{seed:04d}" / "separate_cfg_snapshot.yml"
            _run_decompose(separate_cfg=retry_cfg, cfg_path=retry_cfg_path, seed=int(seed), verbose=separate_verbose)

            for stem in bad_stems:
                row = _eval_one_twist(
                    stem=stem,
                    results_root=out_retry,
                    debug_vis_dir=None,
                    twist_period_deg=float(args.twist_period_deg),
                    thr_rel=float(args.twist_thr_rel),
                    thr_rel_candidates=None if args.twist_thr_rel_candidates is None else str(args.twist_thr_rel_candidates),
                    min_area=int(args.twist_min_area),
                    blur_sigma=float(args.twist_blur_sigma),
                    k_neighbors=int(args.twist_k_neighbors),
                    r_max_px=float(args.twist_r_max_px),
                    trim_ratio=float(args.twist_trim_ratio),
                    coarse_step_deg=float(args.twist_coarse_step_deg),
                )
                if row is None:
                    continue
                cur_err = float(row.get("abs_err_deg", float("nan")))
                if math.isnan(cur_err):
                    continue

                best_row = best_by_stem.get(stem)
                try:
                    best_err = float(best_row.get("abs_err_deg", float("nan"))) if best_row is not None else float("nan")
                except Exception:
                    best_err = float("nan")
                if best_row is None or math.isnan(best_err) or cur_err < best_err:
                    src_dir = out_retry / stem
                    if src_dir.is_dir():
                        _atomic_sync_sample_outputs(
                            src_dir=src_dir,
                            dst_dir=results_root / stem,
                            stem=stem,
                            backup_root=replaced_root,
                        )
                    best_by_stem[stem] = row
                    _flush_twist_best_csv(out_csv, best_by_stem)

            remaining: list[str] = []
            for stem in bad_stems:
                cur = best_by_stem.get(stem)
                if cur is None:
                    remaining.append(stem)
                    continue
                try:
                    e = float(cur.get("abs_err_deg", float("nan")))
                except Exception:
                    e = float("nan")
                if math.isnan(e) or e > retry_thr_deg:
                    remaining.append(stem)
            bad_stems = remaining

        _flush_twist_best_csv(out_csv, best_by_stem)
        return 0

    raise ValueError(f"未知 task={task}")


def wraparound_min_err_around(
    delta_A: np.ndarray,
    *,
    T1: np.ndarray,
    T2: np.ndarray,
    k_range: int = 2,
) -> tuple[float, int, int]:
    """
    在二维周期矢量 (T1,T2) 下对 delta_A 做最小像折叠，返回 (err_A, kx, ky)。

    与简单的 [-k_range, +k_range] 枚举不同，这里会：
    1) 先求解 delta_A 在 (T1,T2) 基下的系数 c；
    2) 取 k0 = round(c) 作为“中心整数”；
    3) 仅在 k0 的邻域内做小范围枚举（范围由 k_range 控制）。

    这样即使 delta_A 由于 multi_origin 平移搜索带来“跨多个晶胞”的等价解，
    也能用较小的 k_range 找到正确的最小像。
    """
    k_range = int(max(0, k_range))
    d = np.asarray(delta_A, dtype=np.float64).reshape(2)
    T1 = np.asarray(T1, dtype=np.float64).reshape(2)
    T2 = np.asarray(T2, dtype=np.float64).reshape(2)

    B = np.stack([T1, T2], axis=1).astype(np.float64)  # 2x2, columns
    try:
        cond = float(np.linalg.cond(B))
        if not np.isfinite(cond) or cond > 1e10:
            c = (np.linalg.pinv(B) @ d.reshape(2, 1)).reshape(2)
        else:
            c = np.linalg.solve(B, d.reshape(2, 1)).reshape(2)
    except Exception:
        c = (np.linalg.pinv(B) @ d.reshape(2, 1)).reshape(2)

    k0 = np.round(c).astype(np.int64)
    kx0 = int(k0[0])
    ky0 = int(k0[1])

    best: Optional[float] = None
    best_k = (0, 0)
    for kx in range(kx0 - k_range, kx0 + k_range + 1):
        for ky in range(ky0 - k_range, ky0 + k_range + 1):
            dd = d - float(kx) * T1 - float(ky) * T2
            n = float(np.linalg.norm(dd))
            if best is None or n < best:
                best = n
                best_k = (int(kx), int(ky))
    assert best is not None
    return float(best), int(best_k[0]), int(best_k[1])


def wraparound_min_err_with_fractional_offsets(
    delta_A: np.ndarray,
    *,
    T1: np.ndarray,
    T2: np.ndarray,
    k_range: int = 2,
    fractional_offsets: tuple[tuple[float, float], ...] = ((0.0, 0.0),),
) -> tuple[float, int, int]:
    """
    在整数晶胞折叠之外，额外允许材料特定的分数晶格等价偏移。

    返回值仍保持旧 CSV 语义 `(err_A, kx, ky)`：`kx/ky` 记录最优解的整数晶胞
    部分；分数偏移仅用于计算最小等价误差，不写入旧版 CSV。
    """
    offsets = tuple(fractional_offsets) or ((0.0, 0.0),)
    best: Optional[tuple[float, int, int]] = None
    d = np.asarray(delta_A, dtype=np.float64).reshape(2)
    T1_arr = np.asarray(T1, dtype=np.float64).reshape(2)
    T2_arr = np.asarray(T2, dtype=np.float64).reshape(2)

    B = np.stack([T1_arr, T2_arr], axis=1).astype(np.float64)
    try:
        cond = float(np.linalg.cond(B))
        if not np.isfinite(cond) or cond > 1e10:
            coeff = (np.linalg.pinv(B) @ d.reshape(2, 1)).reshape(2)
        else:
            coeff = np.linalg.solve(B, d.reshape(2, 1)).reshape(2)
    except Exception:
        coeff = (np.linalg.pinv(B) @ d.reshape(2, 1)).reshape(2)

    span = int(max(0, k_range))
    for hx, hy in offsets:
        k0 = np.round(coeff - np.array([float(hx), float(hy)], dtype=np.float64)).astype(np.int64)
        kx0 = int(k0[0])
        ky0 = int(k0[1])
        for kx in range(kx0 - span, kx0 + span + 1):
            for ky in range(ky0 - span, ky0 + span + 1):
                dd = d - (float(kx) + float(hx)) * T1_arr - (float(ky) + float(hy)) * T2_arr
                err = float(np.linalg.norm(dd))
                if best is None or err < best[0]:
                    best = (err, int(kx), int(ky))

    assert best is not None
    return float(best[0]), int(best[1]), int(best[2])


def evaluate_one_pbc(
    *,
    stem: str,
    results_root: Path,
    T1: np.ndarray,
    T2: np.ndarray,
    up_factor: int,
    max_shift_px_512: int,
    max_shift_A: float,
    coarse_step: int,
    min_area_threshold: int,
    bina_thre: float,
    blur_sigma: float,
    center_method: str,
    center_thr_rel: float,
    debug_vis_dir: Optional[Path],
    k_range: int,
    pbc_k_range: int,
    multi_origin: bool,
) -> Optional[base.SampleResult]:
    gt = base.parse_gt_from_name(f"{stem}.png")
    if gt is None:
        return None
    _r, gtx, gty, sideA, v1_name, v2_name = gt
    sample_dir = results_root / stem
    img0_path = sample_dir / f"{stem}_0.png"
    img1_path = sample_dir / f"{stem}_1.png"
    if not img0_path.is_file() or not img1_path.is_file():
        return None
    if cv2 is None:
        raise RuntimeError("cv2 不可用，无法读取/处理分解图像。")

    img0 = cv2.imread(str(img0_path), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(str(img1_path), cv2.IMREAD_GRAYSCALE)
    if img0 is None or img1 is None:
        return None
    if img0.shape != img1.shape:
        return None

    H, W = img0.shape
    target_size = (int(W * up_factor), int(H * up_factor))
    img0_up = cv2.resize(img0, target_size, interpolation=cv2.INTER_CUBIC)
    img1_up = cv2.resize(img1, target_size, interpolation=cv2.INTER_CUBIC)

    img0_raw = base._to_gray01(img0_up)
    img1_raw = base._to_gray01(img1_up)

    img0_f = img0_raw
    img1_f = img1_raw
    if blur_sigma > 0.0:
        img0_f = cv2.GaussianBlur(img0_raw, (0, 0), sigmaX=float(blur_sigma))
        img1_f = cv2.GaussianBlur(img1_raw, (0, 0), sigmaX=float(blur_sigma))

    if str(center_method).strip().lower() in {"cc_relmax", "relmax"}:
        centers0 = _find_centers_cc_relmax(
            img0_f,
            thr_rel=float(center_thr_rel),
            min_area_threshold=int(min_area_threshold),
            img_gray01_for_weight=img0_raw,
        )
        centers1 = _find_centers_cc_relmax(
            img1_f,
            thr_rel=float(center_thr_rel),
            min_area_threshold=int(min_area_threshold),
            img_gray01_for_weight=img1_raw,
        )
    else:
        centers0 = base.find_centers(img0_f, min_area_threshold=min_area_threshold, bina_thre=bina_thre)
        centers1 = base.find_centers(img1_f, min_area_threshold=min_area_threshold, bina_thre=bina_thre)

    if debug_vis_dir is not None:
        meta0 = {
            "stem": stem,
            "tag": "0",
            "center_method": str(center_method),
            "bina_thre": float(bina_thre),
            "min_area": int(min_area_threshold),
            "blur_sigma": float(blur_sigma),
            "center_thr_rel": float(center_thr_rel),
        }
        meta1 = dict(meta0)
        meta1["tag"] = "1"
        _render_centers_debug(out_dir=debug_vis_dir, stem=stem, tag="0", img01=img0_f, centers=centers0, method=str(center_method), meta=meta0)
        _render_centers_debug(out_dir=debug_vis_dir, stem=stem, tag="1", img01=img1_f, centers=centers1, method=str(center_method), meta=meta1)
    if len(centers0) < 3 or len(centers1) < 3:
        return base.SampleResult(
            stem=stem,
            gtx=gtx,
            gty=gty,
            sideA=sideA,
            shift_pred_128_px=(float("nan"), float("nan")),
            shift_gt_128_px=(float(gtx), float(gty)),
            err_A=float("nan"),
            kx=0,
            ky=0,
            swapped=False,
            centers0=len(centers0),
            centers1=len(centers1),
        )

    centers0_arr = np.asarray(centers0, dtype=np.float32)
    centers1_arr = np.asarray(centers1, dtype=np.float32)

    A_per_px_512 = float(sideA) / float(H * up_factor)

    # 周期矢量（与原版一致：优先从文件名读取 v1/v2，其次 labels meta，再其次推断）
    labels_dir = (results_root.parent / f"{results_root.parent.name}_labels").resolve()
    meta = base.load_aug_meta_from_labels(labels_dir, stem) if labels_dir.is_dir() else None
    if v1_name is not None and v2_name is not None:
        V1_A, V2_A = v1_name, v2_name
        V1_px_512 = V1_A / max(A_per_px_512, 1e-9)
        V2_px_512 = V2_A / max(A_per_px_512, 1e-9)
    elif meta is not None:
        V1_A, V2_A = base.transform_period_vecs_by_meta(T1, T2, meta=meta)
        V1_px_512 = V1_A / max(A_per_px_512, 1e-9)
        V2_px_512 = V2_A / max(A_per_px_512, 1e-9)
    else:
        exp1_px = float(np.linalg.norm(T1)) / max(A_per_px_512, 1e-9)
        exp2_px = float(np.linalg.norm(T2)) / max(A_per_px_512, 1e-9)
        try:
            V1_px_512, V2_px_512 = base.infer_period_vecs_from_centers(centers0_arr, expected_len_px=(exp1_px, exp2_px))
        except Exception:
            V1_px_512 = (T1 / max(A_per_px_512, 1e-9)).astype(np.float64)
            V2_px_512 = (T2 / max(A_per_px_512, 1e-9)).astype(np.float64)
        V1_A = V1_px_512 * float(A_per_px_512)
        V2_A = V2_px_512 * float(A_per_px_512)

    # origins：
    # - 单起点 (0,0) 足以覆盖一个晶胞内的“最小像”平移（搭配后续 wrap-around / pairmin 误差折叠），速度更快；
    # - 多起点可一定程度缓解噪声/缺点导致的局部误匹配，但计算量约为 origins 数量的倍数。
    if multi_origin:
        origins: list[tuple[int, int]] = [
            (0, 0),
            _to_int_xy(V1_px_512),
            _to_int_xy(-V1_px_512),
            _to_int_xy(V2_px_512),
            _to_int_xy(-V2_px_512),
            _to_int_xy(V1_px_512 + V2_px_512),
            _to_int_xy(V1_px_512 - V2_px_512),
            _to_int_xy(-V1_px_512 + V2_px_512),
            _to_int_xy(-V1_px_512 - V2_px_512),
        ]
    else:
        origins = [(0, 0)]

    # 对齐评分：使用非周期 KDTree + trimmed mean（更适合规则晶格；避免“周期复制评分”在 MoS2 上退化到 0 位移局部最优）
    tree_ref = KDTree(centers0_arr)
    material_key = _material_key_from_stem(stem)
    align_trim_ratio = 0.25 if material_key in {"mos2", "tas2", "mote2"} else 0.20

    # 判定 GT 方向（是否需要取反）：用鲁棒评分比较 ±GT
    shift_gt_128_raw = np.array([gtx, gty], dtype=np.float64)
    shift_gt_512_raw = shift_gt_128_raw * float(up_factor)
    score_minus = mean_nn_distance_robust(
        tree_ref,
        centers1_arr + (-shift_gt_512_raw).astype(np.float32).reshape(1, 2),
        trim_ratio=float(align_trim_ratio),
    )
    score_plus = mean_nn_distance_robust(
        tree_ref,
        centers1_arr + (+shift_gt_512_raw).astype(np.float32).reshape(1, 2),
        trim_ratio=float(align_trim_ratio),
    )
    if score_minus <= score_plus:
        shift_gt_512 = shift_gt_512_raw
        swapped = False
    else:
        shift_gt_512 = -shift_gt_512_raw
        swapped = True

    gt_A = shift_gt_512 * A_per_px_512

    # 估计 pred：鲁棒评分 +（可选）多起点搜索
    best_t_align: Optional[np.ndarray] = None
    best_score: Optional[float] = None
    for origin in origins:
        t = estimate_shift_translation_multiscale_robust(
            tree_ref,
            centers1_arr,
            origin=origin,
            max_shift=max_shift_px_512,
            coarse_step=coarse_step,
            trim_ratio=float(align_trim_ratio),
        )
        if t is None:
            continue
        score = mean_nn_distance_robust(tree_ref, centers1_arr + t.reshape(1, 2), trim_ratio=float(align_trim_ratio))
        if best_score is None or score < best_score:
            best_score = float(score)
            best_t_align = t

    if best_t_align is None:
        return None

    shift_pred_512 = (-best_t_align.astype(np.float64))

    pred_A = shift_pred_512 * A_per_px_512
    delta_A = pred_A - gt_A
    equiv_offsets = _FRACTIONAL_EQUIV_OFFSETS_BY_MATERIAL.get(material_key, ((0.0, 0.0),))
    if len(equiv_offsets) > 1:
        err_A, kx, ky = wraparound_min_err_with_fractional_offsets(
            delta_A,
            T1=V1_A,
            T2=V2_A,
            k_range=k_range,
            fractional_offsets=equiv_offsets,
        )
    else:
        err_A, kx, ky = wraparound_min_err_around(delta_A, T1=V1_A, T2=V2_A, k_range=k_range)

    shift_pred_128 = (shift_pred_512 / float(up_factor)).astype(np.float64)
    return base.SampleResult(
        stem=stem,
        gtx=gtx,
        gty=gty,
        sideA=sideA,
        shift_pred_128_px=(float(shift_pred_128[0]), float(shift_pred_128[1])),
        shift_gt_128_px=(float(shift_gt_512[0] / float(up_factor)), float(shift_gt_512[1] / float(up_factor))),
        err_A=float(err_A),
        kx=int(kx),
        ky=int(ky),
        swapped=bool(swapped),
        centers0=len(centers0),
        centers1=len(centers1),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="评估 generate_sample 分解结果（支持 shift_pbc / twist / twist_labels）")
    ap.add_argument(
        "--task",
        choices=["shift_pbc", "twist", "twist_labels"],
        default="shift_pbc",
        help=(
            "评估任务：shift_pbc=评估滑移（wrap-around + PBC）；"
            "twist=仅评估扭角（不求解滑移，不依赖周期矢量）；"
            "twist_labels=从 labels(npz) 点云直接评估扭角（理想输入，不依赖分解）。"
        ),
    )
    ap.add_argument(
        "--results_root",
        required=True,
        help=(
            "结果根目录：\n"
            "- task=shift_pbc/twist：其下每个子文件夹都是一个样本（子文件夹名即 stem）\n"
            "- task=twist_labels：该目录下应包含 <stem>.npz（由 MOIRE_SAVE_LABELS 导出）"
        ),
    )
    ap.add_argument(
        "--out_csv",
        default=None,
        help=(
            "输出 CSV 路径（shift_pbc 默认 <results_root>/result.csv；"
            "twist 默认 <results_root>/result_twist.csv；"
            "twist_labels 默认 <results_root>/result_twist_labels.csv）"
        ),
    )
    ap.add_argument("--stems", nargs="*", default=None, help="仅评估指定 stem（不传则评估 results_root 下全部子文件夹）")
    ap.add_argument("--limit", type=int, default=None, help="仅评估前 N 个样本（用于小批量测试；默认不限制）")
    ap.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="断点重续：若 out_csv 已存在则跳过已写入 stem（默认开启）",
    )
    ap.add_argument(
        "--no_resume",
        dest="resume",
        action="store_false",
        help="关闭断点重续：即使 out_csv 已存在也重新计算并追加写入（可能产生重复行）",
    )
    ap.add_argument(
        "--debug_vis_dir",
        default=None,
        help="导出中心提取可视化（叠加圆点）到该目录；用于人工检查中心提取是否合理。",
    )

    # 可选：分解 + 评估一体化（yml 配置仅快照一次到内存，避免并发修改影响）
    ap.add_argument(
        "--separate_yml",
        default=None,
        help=(
            "可选：提供分解 YAML（configs/separate/*.yml）。若提供，将先调用 src/main.py 在 results_root 下生成分解结果，"
            "随后再执行 eval；并对超过阈值的样本自动做多 seed 重试，最终仅保留最佳结果。"
            "注意：本模式假定双层输入 PNG 在 results_root 的父目录下（即 results_root.parent）。"
        ),
    )
    ap.add_argument("--separate_seed", type=int, default=1234, help="分解阶段的基础随机种子（默认 1234）")
    ap.add_argument(
        "--separate_verbose",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="分解阶段的日志级别（传给 src/main.py --verbose）",
    )
    ap.add_argument("--retry_runs", type=int, default=4, help="对 bad 样本的重试次数（不同 seed，默认 4）")
    ap.add_argument("--retry_seed_stride", type=int, default=1, help="重试 seed 的步长：seed_i = base + (i+1)*stride（默认 1）")
    ap.add_argument("--retry_thr_A", type=float, default=0.1, help="task=shift_pbc 时的 bad 阈值（Å，默认 0.1）")
    ap.add_argument("--retry_thr_deg", type=float, default=0.5, help="task=twist 时的 bad 阈值（度，默认 0.5）")
    ap.add_argument(
        "--nanfix_superposition",
        action="store_true",
        default=True,
        help="仅一体化模式生效：若评估出现 err_A=NaN，则尝试关闭自适应叠加并固定叠加系数重跑分解（默认开启）。",
    )
    ap.add_argument(
        "--no_nanfix_superposition",
        dest="nanfix_superposition",
        action="store_false",
        help="关闭 NaN 叠加系数修复重跑。",
    )
    ap.add_argument(
        "--nanfix_superposition_k",
        type=float,
        default=0.5,
        help="NaN 修复重跑时使用的固定叠加系数 k（默认 0.5）。",
    )

    # shift_pbc 参数（保留原行为；twist 也可选用其中的 twist_period_deg）
    ap.add_argument(
        "--batch_config",
        default="generate_sample/batch_config.json",
        help="generate_sample/*.json 路径。shift_pbc 用于读取二维周期矢量；twist 可选读取 twist_period_deg（若 json 中包含该键）。",
    )
    ap.add_argument("--up_factor", type=int, default=4, help="上采样倍数（默认 4：128->512）")
    ap.add_argument("--max_shift_px_512", type=int, default=None, help="512 坐标系下的搜索半径（像素）")
    ap.add_argument("--max_shift_A", type=float, default=4.0, help="物理搜索半径（Å，默认 4，与原版一致）")
    ap.add_argument("--coarse_step", type=int, default=4, help="粗搜索步长（默认 4）")
    ap.add_argument("--min_area", type=int, default=None, help="中心提取面积阈值（默认按 material 自动选择）")
    ap.add_argument("--bina_thre", type=float, default=None, help="contour 方法的阈值倍数（默认按 material 自动选择）")
    ap.add_argument("--blur_sigma", type=float, default=None, help="中心提取前的高斯模糊 sigma（默认按 material 自动选择）")
    ap.add_argument(
        "--center_method",
        default=None,
        help="原子中心提取方法：contour 或 cc_relmax（默认按 material 自动选择）",
    )
    ap.add_argument(
        "--center_thr_rel",
        type=float,
        default=None,
        help="cc_relmax 的相对阈值：thr = center_thr_rel * max（默认按 material 自动选择）",
    )
    ap.add_argument("--k_range", type=int, default=2, help="wrap-around 枚举范围（默认 2：kx,ky∈[-2,2]）")
    ap.add_argument("--pbc_k_range", type=int, default=2, help="（兼容保留/当前未使用）旧版用于构造周期 KDTree 的复制范围。")
    ap.add_argument("--multi_origin", dest="multi_origin", action="store_true", help="启用多起点平移搜索（默认开启）")
    ap.add_argument("--no_multi_origin", dest="multi_origin", action="store_false", help="禁用多起点：仅 origin=(0,0)")
    ap.set_defaults(multi_origin=True)

    # twist 参数（仅在 task=twist 生效）
    ap.add_argument(
        "--twist_period_deg",
        type=float,
        default=None,
        help="扭角折叠周期（度）。默认不传时：若 batch_config 含 twist_period_deg 则使用之，否则默认为 180。",
    )
    ap.add_argument("--twist_thr_rel", type=float, default=0.55, help="中心提取阈值：thr = thr_rel * max（默认 0.55）")
    ap.add_argument(
        "--twist_thr_rel_candidates",
        type=str,
        default=None,
        help="可选：用逗号分隔的一组 thr_rel 候选（如 0.5,0.55,0.6,0.65）。若提供，将逐个尝试并按 score 选最优。",
    )
    ap.add_argument("--twist_min_area", type=int, default=5, help="中心提取连通域最小面积（像素，默认 5）")
    ap.add_argument("--twist_blur_sigma", type=float, default=0.0, help="中心提取前高斯模糊 sigma（默认 0）")
    ap.add_argument("--twist_k_neighbors", type=int, default=8, help="构造位移向量时的 kNN 邻居数（默认 8）")
    ap.add_argument("--twist_r_max_px", type=float, default=40.0, help="构造位移向量时的最大邻居距离阈值（像素，默认 40）")
    ap.add_argument("--twist_trim_ratio", type=float, default=0.25, help="最近邻距离评分的 trimmed mean 比例（默认 0.25）")
    ap.add_argument("--twist_coarse_step_deg", type=float, default=1.0, help="粗搜索角度步长（度，默认 1.0）")

    args = ap.parse_args()

    task = str(args.task).strip().lower()
    results_root = Path(args.results_root).resolve()

    out_csv = (
        Path(args.out_csv).resolve()
        if args.out_csv
        else (
            results_root
            / ("result_twist_labels.csv" if task == "twist_labels" else ("result_twist.csv" if task == "twist" else "result.csv"))
        ).resolve()
    )
    debug_vis_dir = Path(args.debug_vis_dir).resolve() if args.debug_vis_dir else None
    if debug_vis_dir is not None:
        debug_vis_dir.mkdir(parents=True, exist_ok=True)

    if task == "twist_labels" and args.separate_yml is not None:
        raise SystemExit("task=twist_labels 不支持 --separate_yml（labels 评估不依赖分解）。")

    # 一体化模式：先分解，再评估与自动重试（最终重写 out_csv）
    if args.separate_yml is not None:
        results_root.mkdir(parents=True, exist_ok=True)
        return _run_pipeline_decompose_eval_retry(args, results_root=results_root, out_csv=out_csv, debug_vis_dir=debug_vis_dir)

    if not results_root.is_dir():
        raise SystemExit(f"results_root 不存在或不是目录：{results_root}")

    if args.stems:
        stems = list(args.stems)
    else:
        if task == "twist_labels":
            stems = sorted([p.stem for p in results_root.glob("*.npz") if _parse_twist_deg_from_stem(p.stem) is not None])
        else:
            stems = sorted([p.name for p in results_root.iterdir() if p.is_dir() and not p.name.startswith("debug")])
    if args.limit is not None:
        stems = list(stems)[: int(args.limit)]
    if not stems:
        raise SystemExit(f"未找到任何样本：{results_root}")

    # ---------------- twist_labels（labels 点云直接评估）----------------
    if task == "twist_labels":
        args.twist_period_deg = _resolve_twist_period_deg(args)

        if bool(args.resume) and not _twist_csv_is_compatible(out_csv, period_deg=float(args.twist_period_deg)):
            backup_dir = _ensure_trash_run_dir(prefix="eval_twist_labels_incompatible_csv")
            _move_to_trash(out_csv, trash_run_dir=backup_dir, note=out_csv.name)
            args.resume = False

        total = len(stems)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        done_stems = load_done_stems_from_csv(out_csv) if bool(args.resume) else set()
        if done_stems:
            print(f"[RESUME] 已存在 {len(done_stems)} 条记录，将跳过这些 stem")

        need_header = (not out_csv.exists()) or out_csv.stat().st_size == 0
        mode = "a" if out_csv.exists() else "w"
        processed = 0
        skipped = 0
        written = 0
        errs_deg: list[float] = []

        with out_csv.open(mode, encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if need_header:
                writer.writerow(_CSV_HEADER_TWIST)
                f.flush()

            for idx, stem in enumerate(stems, start=1):
                if stem in done_stems:
                    skipped += 1
                    if skipped <= 5 or skipped % 200 == 0:
                        print(f"[SKIP] {idx}/{total} {stem}（已在 CSV 中）")
                    continue
                processed += 1

                row = _eval_one_twist_from_labels_npz(
                    stem=stem,
                    labels_root=results_root,
                    debug_vis_dir=debug_vis_dir,
                    twist_period_deg=float(args.twist_period_deg),
                    k_neighbors=int(args.twist_k_neighbors),
                    r_max_px=float(args.twist_r_max_px),
                    trim_ratio=float(args.twist_trim_ratio),
                    coarse_step_deg=float(args.twist_coarse_step_deg),
                )
                if row is None:
                    print(f"[WARN] 跳过：{stem}（labels npz 缺失或扭角估计失败）")
                    continue

                writer.writerow([row.get(k) for k in _CSV_HEADER_TWIST])
                f.flush()
                written += 1
                done_stems.add(stem)

                abs_err = float(row.get("abs_err_deg", float("nan")))
                if np.isfinite(abs_err):
                    errs_deg.append(abs_err)

                print(
                    f"[OK] {idx}/{total} {stem}: gt_twist={float(row['gt_twist_deg']):.3f}° "
                    f"pred_twist={float(row['pred_twist_deg']):.3f}° abs_err={abs_err:.3f}° score={float(row['score']):.5f} "
                    f"centers=({int(row['centers0'])},{int(row['centers1'])})"
                )

        if errs_deg:
            e = np.asarray(errs_deg, dtype=np.float64)
            print(
                f"\nSummary(twist_labels): n={len(e)} mean={e.mean():.4f}° median={np.median(e):.4f}° max={e.max():.4f}° "
                f"<=0.5°: {(e <= 0.5).sum()}/{len(e)}"
            )
        print(f"Processed={processed} skipped={skipped} written={written} total_samples={total}")
        print(f"CSV: {out_csv}")
        return 0

    # ---------------- twist ----------------
    if task == "twist":
        # twist_period：优先命令行，其次 batch_config.json，其次默认 180
        args.twist_period_deg = _resolve_twist_period_deg(args)

        # 若 period 不一致，则旧 CSV 不可用于 resume：备份后重算
        if bool(args.resume) and not _twist_csv_is_compatible(out_csv, period_deg=float(args.twist_period_deg)):
            backup_dir = _ensure_trash_run_dir(prefix="eval_twist_incompatible_csv")
            _move_to_trash(out_csv, trash_run_dir=backup_dir, note=out_csv.name)
            args.resume = False

        total = len(stems)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        done_stems = load_done_stems_from_csv(out_csv) if bool(args.resume) else set()
        if done_stems:
            print(f"[RESUME] 已存在 {len(done_stems)} 条记录，将跳过这些 stem")

        need_header = (not out_csv.exists()) or out_csv.stat().st_size == 0
        mode = "a" if out_csv.exists() else "w"
        processed = 0
        skipped = 0
        written = 0
        errs_deg: list[float] = []

        with out_csv.open(mode, encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if need_header:
                writer.writerow(_CSV_HEADER_TWIST)
                f.flush()

            for idx, stem in enumerate(stems, start=1):
                gt_r = _parse_twist_deg_from_stem(stem)
                if gt_r is None:
                    print(f"[WARN] 跳过（命名不匹配，无法解析 r）：{stem}")
                    continue
                processed += 1

                sample_dir = results_root / stem
                try:
                    p0, p1 = _find_twist_layer_pngs(sample_dir, stem)
                except Exception as e:
                    print(f"[WARN] 跳过：{stem}（{e}）")
                    continue
                if stem in done_stems:
                    skipped += 1
                    if skipped <= 5 or skipped % 200 == 0:
                        print(f"[SKIP] {idx}/{total} {stem}（已在 CSV 中且输出存在）")
                    continue

                img0 = _load_gray01(p0)
                img1 = _load_gray01(p1)

                # 按 score 选择最优 thr_rel（默认只尝试 twist_thr_rel）
                if args.twist_thr_rel_candidates:
                    thr_candidates = []
                    for s in str(args.twist_thr_rel_candidates).split(","):
                        s = s.strip()
                        if not s:
                            continue
                        try:
                            thr_candidates.append(float(s))
                        except Exception:
                            pass
                    if not thr_candidates:
                        thr_candidates = [float(args.twist_thr_rel)]
                else:
                    thr_candidates = [float(args.twist_thr_rel)]

                best = None
                best_payload = None
                for thr in thr_candidates:
                    try:
                        c0 = _find_centers_cc_relmax_weighted(
                            img0,
                            thr_rel=float(thr),
                            min_area_threshold=int(args.twist_min_area),
                            blur_sigma=float(args.twist_blur_sigma),
                        )
                        c1 = _find_centers_cc_relmax_weighted(
                            img1,
                            thr_rel=float(thr),
                            min_area_threshold=int(args.twist_min_area),
                            blur_sigma=float(args.twist_blur_sigma),
                        )
                        v0 = _displacement_vectors_from_centers(
                            c0,
                            k_neighbors=int(args.twist_k_neighbors),
                            r_max_px=float(args.twist_r_max_px),
                        )
                        v1 = _displacement_vectors_from_centers(
                            c1,
                            k_neighbors=int(args.twist_k_neighbors),
                            r_max_px=float(args.twist_r_max_px),
                        )
                        rot_deg_raw, score = _estimate_rotation_deg_from_vecs_global_search(
                            v0,
                            v1,
                            coarse_step_deg=float(args.twist_coarse_step_deg),
                            trim_ratio=float(args.twist_trim_ratio),
                        )
                    except Exception:
                        continue
                    if best is None or score < best[0]:
                        best = (float(score), float(rot_deg_raw), float(thr))
                        best_payload = (c0, c1, v0, v1)

                if best is None or best_payload is None:
                    print(f"[WARN] 跳过：{stem}（扭角估计失败：所有 thr_rel 候选均失败）")
                    continue

                score, rot_deg_raw, thr_used = best
                centers0, centers1, vecs0, vecs1 = best_payload

                if debug_vis_dir is not None:
                    meta0 = {
                        "stem": stem,
                        "task": "twist",
                        "tag": "0",
                        "thr_rel": float(thr_used),
                        "min_area": int(args.twist_min_area),
                        "blur_sigma": float(args.twist_blur_sigma),
                    }
                    meta1 = dict(meta0)
                    meta1["tag"] = "1"
                    _render_centers_debug(
                        out_dir=debug_vis_dir,
                        stem=stem,
                        tag="0",
                        img01=img0,
                        centers=[(float(x), float(y)) for (x, y) in centers0.tolist()],
                        method="cc_relmax_twist",
                        meta=meta0,
                    )
                    _render_centers_debug(
                        out_dir=debug_vis_dir,
                        stem=stem,
                        tag="1",
                        img01=img1,
                        centers=[(float(x), float(y)) for (x, y) in centers1.tolist()],
                        method="cc_relmax_twist",
                        meta=meta1,
                    )

                twist_period = float(args.twist_period_deg)
                gt_twist = _twist_fold_deg(gt_r, period_deg=twist_period)
                pred_twist = _twist_fold_deg(rot_deg_raw, period_deg=twist_period)
                abs_err = float(abs(pred_twist - gt_twist))

                writer.writerow(
                    [
                        stem,
                        float(gt_r),
                        float(gt_twist),
                        float(pred_twist),
                        float(abs_err),
                        float(score),
                        float(twist_period),
                        float(thr_used),
                        int(args.twist_min_area),
                        int(args.twist_k_neighbors),
                        float(args.twist_r_max_px),
                        float(args.twist_trim_ratio),
                        float(args.twist_coarse_step_deg),
                        int(centers0.shape[0]),
                        int(centers1.shape[0]),
                        int(vecs0.shape[0]),
                        int(vecs1.shape[0]),
                    ]
                )
                f.flush()
                written += 1
                done_stems.add(stem)
                errs_deg.append(abs_err)

                print(
                    f"[OK] {idx}/{total} {stem}: gt_twist={gt_twist:.3f}° pred_twist={pred_twist:.3f}° "
                    f"abs_err={abs_err:.3f}° score={score:.5f} centers=({centers0.shape[0]},{centers1.shape[0]})"
                )

        if errs_deg:
            e = np.asarray(errs_deg, dtype=np.float64)
            print(
                f"\nSummary(twist): n={len(e)} mean={e.mean():.4f}° median={np.median(e):.4f}° max={e.max():.4f}° "
                f"<=0.1°: {(e <= 0.1).sum()}/{len(e)}"
            )
        print(f"Processed={processed} skipped={skipped} written={written} total_samples={total}")
        print(f"CSV: {out_csv}")
        return 0

    # ---------------- shift_pbc（原逻辑）----------------
    batch_config = Path(args.batch_config).resolve()
    T1, T2 = load_period_vecs_from_batch_config(batch_config)
    if float(np.linalg.norm(T1)) <= 0.0 or float(np.linalg.norm(T2)) <= 0.0:
        raise SystemExit(f"无效周期矢量：T1={T1}, T2={T2}（来自 {batch_config} 推断）")
    print(f"[PERIOD] T1={T1} Å, |T1|={float(np.linalg.norm(T1)):.3f} Å")
    print(f"[PERIOD] T2={T2} Å, |T2|={float(np.linalg.norm(T2)):.3f} Å")

    errs: list[float] = []
    total = len(stems)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done_stems = load_done_stems_from_csv(out_csv) if bool(args.resume) else set()
    if done_stems:
        print(f"[RESUME] 已存在 {len(done_stems)} 条记录，将跳过这些 stem")

    need_header = (not out_csv.exists()) or out_csv.stat().st_size == 0
    mode = "a" if out_csv.exists() else "w"
    processed = 0
    skipped = 0
    written = 0

    with out_csv.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(_CSV_HEADER)
            f.flush()

        for idx, stem in enumerate(stems, start=1):
            gt = base.parse_gt_from_name(f"{stem}.png")
            if gt is None:
                print(f"[WARN] 跳过（命名不匹配）：{stem}")
                continue
            _r, _gtx, _gty, sideA, _v1, _v2 = gt
            processed += 1

            if stem in done_stems and _has_shift_outputs(results_root, stem):
                skipped += 1
                if skipped <= 5 or skipped % 200 == 0:
                    print(f"[SKIP] {idx}/{total} {stem}（已在 CSV 中且输出存在）")
                continue

            if args.max_shift_px_512 is not None:
                max_shift_px_512 = int(args.max_shift_px_512)
            else:
                max_shift_A = float(args.max_shift_A)
                A_per_px_512 = float(sideA) / 512.0
                max_shift_px_512 = int(math.ceil(max_shift_A / max(A_per_px_512, 1e-9)))

            material_key = _material_key_from_stem(stem)
            center_method, bina_thre, min_area, blur_sigma, center_thr_rel = _apply_default_center_params(
                material_key=material_key,
                center_method=None if args.center_method is None else str(args.center_method),
                bina_thre=None if args.bina_thre is None else float(args.bina_thre),
                min_area_threshold=None if args.min_area is None else int(args.min_area),
                blur_sigma=None if args.blur_sigma is None else float(args.blur_sigma),
                thr_rel=None if args.center_thr_rel is None else float(args.center_thr_rel),
            )

            res = evaluate_one_pbc(
                stem=stem,
                results_root=results_root,
                T1=T1,
                T2=T2,
                up_factor=int(args.up_factor),
                max_shift_px_512=max_shift_px_512,
                max_shift_A=float(args.max_shift_A),
                coarse_step=int(args.coarse_step),
                min_area_threshold=int(min_area),
                bina_thre=float(bina_thre),
                blur_sigma=float(blur_sigma),
                center_method=str(center_method),
                center_thr_rel=float(center_thr_rel),
                debug_vis_dir=debug_vis_dir,
                k_range=int(args.k_range),
                pbc_k_range=int(args.pbc_k_range),
                multi_origin=bool(args.multi_origin),
            )
            if res is None:
                print(f"[WARN] 跳过：{stem}")
                continue

            writer.writerow(
                [
                    res.stem,
                    res.gtx,
                    res.gty,
                    res.sideA,
                    res.shift_pred_128_px[0],
                    res.shift_pred_128_px[1],
                    res.err_A,
                    res.kx,
                    res.ky,
                    int(res.swapped),
                    res.centers0,
                    res.centers1,
                ]
            )
            f.flush()
            written += 1
            done_stems.add(res.stem)

            if not math.isnan(res.err_A):
                errs.append(float(res.err_A))
            print(
                f"[OK] {idx}/{total} {res.stem}: err_A={res.err_A:.4f} Å, "
                f"pred128=({res.shift_pred_128_px[0]:.3f},{res.shift_pred_128_px[1]:.3f}) "
                f"gt128=({res.shift_gt_128_px[0]:.3f},{res.shift_gt_128_px[1]:.3f}) "
                f"k=({res.kx},{res.ky}) swapped={int(res.swapped)} centers=({res.centers0},{res.centers1})"
            )

    if errs:
        errs_np = np.asarray(errs, dtype=np.float64)
        print(
            f"\nSummary: n={len(errs)} mean={errs_np.mean():.4f} Å "
            f"median={np.median(errs_np):.4f} Å max={errs_np.max():.4f} Å "
            f"<=0.1Å: {(errs_np <= 0.1).sum()}/{len(errs_np)}"
        )
    print(f"Processed={processed} skipped={skipped} written={written} total_pngs={total}")
    print(f"CSV: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
