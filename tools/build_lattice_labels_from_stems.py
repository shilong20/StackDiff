#!/usr/bin/env python3
"""
【作用概述】从样本文件名（stem）解析 GT 滑移 `gtx/gty`、物理尺寸 `sideA` 以及二维周期矢量 `v1/v2`，
并在 128×128 坐标系下构造“基元原子（通常为金属层）”的二维布拉菲点阵点云，导出为与
`generate_sample/batch_runner.py` 的 `MOIRE_SAVE_LABELS=1` 同构的 `labels/*.npz` 文件。
该脚本用于在历史数据未保存 labels 时，快速得到可对齐的 GT 点云（允许整体平移不确定），以便与分解后的中心点做对比诊断。

【关联说明】文件/模块：generate_sample/batch_runner.py（labels npz 字段定义）；tools/debug_decomposition_vs_labels.py（读取 labels 进行点云对比）。

【命令行用法】python tools/build_lattice_labels_from_stems.py --png_dir data/experiments/MoS2_slip_worst10_redecomp --out_dir data/experiments/MoS2_slip_worst10_redecomp_labels
（参数：--pad=点云生成边界留白（像素，128 坐标系）；--origin=center 表示强制一个点落在 (64,64) 以减少边界缺失）。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


_STEM_RE = re.compile(
    r"^(?P<material>[A-Za-z0-9]+)"
    r"_r(?P<r>-?\d+(?:\.\d+)?)"
    r"_gtx(?P<gtx>-?\d+(?:\.\d+)?)"
    r"_gty(?P<gty>-?\d+(?:\.\d+)?)"
    r"_sideA(?P<sideA>-?\d+(?:\.\d+)?)"
    r"(?:_v1x(?P<v1x>-?\d+(?:\.\d+)?)_v1y(?P<v1y>-?\d+(?:\.\d+)?)"
    r"_v2x(?P<v2x>-?\d+(?:\.\d+)?)_v2y(?P<v2y>-?\d+(?:\.\d+)?))?$"
)


def _parse_stem(stem: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    m = _STEM_RE.match(stem)
    if not m:
        raise ValueError(f"无法解析 stem：{stem}")
    gtx = float(m.group("gtx"))
    gty = float(m.group("gty"))
    sideA = float(m.group("sideA"))
    if m.group("v1x") is None:
        raise ValueError(f"stem 缺少 v1/v2：{stem}")
    v1A = np.array([float(m.group("v1x")), float(m.group("v1y"))], dtype=np.float32)
    v2A = np.array([float(m.group("v2x")), float(m.group("v2y"))], dtype=np.float32)
    gt_shift128 = np.array([gtx, gty], dtype=np.float32)
    return gt_shift128, v1A, v2A, sideA


def _build_lattice_points_128(
    *,
    v1A: np.ndarray,
    v2A: np.ndarray,
    sideA: float,
    origin_mode: str,
    pad_px: float,
) -> np.ndarray:
    """
    用 (v1,v2) 构造 2D 点阵（单位：px@128），并裁剪到 [-pad, 128+pad]。
    注意：这里不包含多原子基（只构造一个 Bravais 点阵），适用于 MoS2/TaS2/MoTe2 的金属层点云分析。
    """
    A_per_px_128 = float(sideA) / 128.0
    v1_px = (np.asarray(v1A, dtype=np.float64) / max(A_per_px_128, 1e-12)).reshape(2)
    v2_px = (np.asarray(v2A, dtype=np.float64) / max(A_per_px_128, 1e-12)).reshape(2)

    # 选一个简单 origin：保证至少有一个点在图中央附近，减少边界点缺失带来的对齐不稳定。
    origin_mode = str(origin_mode).strip().lower()
    if origin_mode in {"center", "c"}:
        origin = np.array([64.0, 64.0], dtype=np.float64)
    elif origin_mode in {"zero", "0"}:
        origin = np.array([0.0, 0.0], dtype=np.float64)
    else:
        raise ValueError(f"非法 origin_mode={origin_mode!r}（可选：center/zero）")

    # 估算需要的 i/j 范围：按最短向量长度取一个保守上界，再留一些冗余。
    l1 = float(np.hypot(v1_px[0], v1_px[1]))
    l2 = float(np.hypot(v2_px[0], v2_px[1]))
    lmin = max(1.0, min(l1, l2))
    n = int(np.ceil((128.0 + 2.0 * float(pad_px)) / lmin)) + 6

    ii, jj = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1), indexing="ij")
    pts = origin.reshape(1, 1, 2) + ii[..., None] * v1_px.reshape(1, 1, 2) + jj[..., None] * v2_px.reshape(1, 1, 2)
    pts = pts.reshape(-1, 2).astype(np.float32)

    lo = -float(pad_px)
    hi = 128.0 + float(pad_px)
    m = (pts[:, 0] >= lo) & (pts[:, 0] <= hi) & (pts[:, 1] >= lo) & (pts[:, 1] <= hi)
    pts = pts[m]

    # 轻度去重：将坐标量化到 1e-3 px 以避免浮点导致的重复点。
    if pts.size == 0:
        return pts.reshape(0, 2)
    q = np.round(pts.astype(np.float64) * 1000.0).astype(np.int64)
    _, uniq_idx = np.unique(q, axis=0, return_index=True)
    return pts[np.sort(uniq_idx)].astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="从 stem 构造简化 lattice labels（用于无 labels 数据集的点云诊断）")
    ap.add_argument("--png_dir", required=True, help="包含 bilayer PNG 的目录（文件名即 stem.png）")
    ap.add_argument("--out_dir", required=True, help="输出 labels 目录（写入 <stem>.npz）")
    ap.add_argument("--pad", type=float, default=1.0, help="点云生成边界留白（像素，128 坐标系；默认 1）")
    ap.add_argument("--origin", default="center", help="点阵 origin 策略：center 或 zero（默认 center）")
    args = ap.parse_args()

    png_dir = Path(args.png_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pngs = sorted([p for p in png_dir.glob("*.png") if p.is_file()])
    if not pngs:
        raise SystemExit(f"未找到 PNG：{png_dir}")

    wrote = 0
    for p in pngs:
        stem = p.stem
        try:
            gt_shift128, v1A, v2A, sideA = _parse_stem(stem)
        except Exception as e:
            print(f"[WARN] skip {stem}: {e}")
            continue

        layer1_xy128 = _build_lattice_points_128(v1A=v1A, v2A=v2A, sideA=sideA, origin_mode=str(args.origin), pad_px=float(args.pad))
        layer2_xy128 = (layer1_xy128 + gt_shift128.reshape(1, 2)).astype(np.float32)

        meta = {
            "source": "stem_lattice",
            "note": "该 labels 由 stem 的 v1/v2 + gt_shift 构造，整体相位(平移)不唯一；对比时需允许全局平移对齐。",
            "origin_mode": str(args.origin),
            "pad_px_128": float(args.pad),
        }
        np.savez_compressed(
            out_dir / f"{stem}.npz",
            layer1_xy128=layer1_xy128.astype(np.float32),
            layer2_xy128=layer2_xy128.astype(np.float32),
            gt_shift128=gt_shift128.astype(np.float32),
            sideA=float(sideA),
            v1A=v1A.astype(np.float32),
            v2A=v2A.astype(np.float32),
            meta=json.dumps(meta, ensure_ascii=False),
        )
        wrote += 1

    print(f"[OK] wrote={wrote} out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

