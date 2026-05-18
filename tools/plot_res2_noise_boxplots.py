#!/usr/bin/env python3
"""
【作用概述】读取 ReS2 噪声鲁棒性实验的评估 CSV，按采样步数分别汇总 `err_A`（可视作当前这批结果里的 delta d 误差量），并输出“横轴为 PSNR、纵轴为误差”的四联箱线图；绘图前会对每个“步数 × PSNR”分组做 IQR 去异常值处理，同时可选导出过滤统计表。
【关联说明】文件/模块：data/experiments/ReS2_noise_*_defect0/result_step*/result_shift_pbc.csv（输入评估结果）；tools/evaluate_generate_sample_wraparound_pbc.py（生成 `err_A` 列）；data/experiments/plots/（默认输出目录）。
【命令行用法】python tools/plot_res2_noise_boxplots.py --experiments_root data/experiments --output data/experiments/plots/res2_noise_errA_boxplot_by_step_psnr_no_outliers.png（参数：--steps=要展示的步数列表；--psnrs=PSNR 档位列表；--column=纵轴列名，默认 err_A；--summary_csv=导出每组过滤统计信息）。
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class GroupStats:
    """单个“步数 × PSNR”分组的原始/过滤统计。"""

    step: int
    psnr: int
    n_raw: int
    n_filtered: int
    q1: float
    q3: float
    iqr: float
    lo: float
    hi: float
    mean_filtered: float
    median_filtered: float


def parse_int_list(raw: str) -> List[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("列表不能为空。")
    return vals


def resolve_csv_path(experiments_root: Path, psnr: int, step: int) -> Path:
    base = experiments_root / "noise" / f"ReS2_noise_{psnr}_defect0"
    candidates = [
        base / f"result_step{step}" / "result_shift_pbc.csv",
    ]
    if step == 200:
        # 兼容未重命名前的老目录布局。
        candidates.append(base / "result" / "result.csv")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"未找到 PSNR={psnr}, step={step} 的结果 CSV。已尝试: {candidates}")


def load_numeric_column(csv_path: Path, column: str) -> np.ndarray:
    vals: List[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise KeyError(f"{csv_path} 中不存在列 {column!r}，现有列: {reader.fieldnames}")
        for row in reader:
            raw = (row.get(column) or "").strip()
            if not raw:
                continue
            value = float(raw)
            if math.isnan(value) or math.isinf(value):
                continue
            vals.append(value)
    if not vals:
        raise ValueError(f"{csv_path} 的列 {column!r} 没有可用数值。")
    return np.asarray(vals, dtype=np.float64)


def iqr_filter(values: np.ndarray) -> tuple[np.ndarray, float, float, float, float, float]:
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    iqr = float(q3 - q1)
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    kept = values[(values >= lo) & (values <= hi)]
    if kept.size == 0:
        kept = values.copy()
    return kept, q1, q3, iqr, lo, hi


def write_summary_csv(summary_csv: Path, rows: Sequence[GroupStats]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "PSNR",
                "n_raw",
                "n_filtered",
                "q1",
                "q3",
                "iqr",
                "iqr_lo",
                "iqr_hi",
                "mean_filtered",
                "median_filtered",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.step,
                    row.psnr,
                    row.n_raw,
                    row.n_filtered,
                    row.q1,
                    row.q3,
                    row.iqr,
                    row.lo,
                    row.hi,
                    row.mean_filtered,
                    row.median_filtered,
                ]
            )


def iter_group_data(
    experiments_root: Path,
    steps: Sequence[int],
    psnrs: Sequence[int],
    column: str,
) -> tuple[dict[int, List[np.ndarray]], List[GroupStats]]:
    grouped: dict[int, List[np.ndarray]] = {}
    stats_rows: List[GroupStats] = []
    for step in steps:
        series: List[np.ndarray] = []
        for psnr in psnrs:
            csv_path = resolve_csv_path(experiments_root, psnr=psnr, step=step)
            raw_vals = load_numeric_column(csv_path, column=column)
            kept, q1, q3, iqr, lo, hi = iqr_filter(raw_vals)
            series.append(kept)
            stats_rows.append(
                GroupStats(
                    step=step,
                    psnr=psnr,
                    n_raw=int(raw_vals.size),
                    n_filtered=int(kept.size),
                    q1=q1,
                    q3=q3,
                    iqr=iqr,
                    lo=lo,
                    hi=hi,
                    mean_filtered=float(np.mean(kept)),
                    median_filtered=float(np.median(kept)),
                )
            )
        grouped[step] = series
    return grouped, stats_rows


def make_plot(
    grouped: dict[int, List[np.ndarray]],
    steps: Sequence[int],
    psnrs: Sequence[int],
    output_path: Path,
    ylabel: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes_arr = np.asarray(axes).reshape(-1)
    positions = np.arange(1, len(psnrs) + 1, dtype=np.float64)
    colors = ["#C44E52", "#4C72B0", "#55A868", "#8172B2", "#CCB974"]

    for ax, step in zip(axes_arr, steps):
        data = grouped[step]
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 1.6},
            whiskerprops={"color": "#555555", "linewidth": 1.2},
            capprops={"color": "#555555", "linewidth": 1.2},
            boxprops={"edgecolor": "#555555", "linewidth": 1.2},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_title(f"Step {step}")
        ax.set_xticks(positions)
        ax.set_xticklabels([str(p) for p in psnrs])
        ax.set_xlabel("PSNR (dB)")
        ax.grid(axis="y", linestyle="--", alpha=0.35)

    axes_arr[0].set_ylabel(ylabel)
    axes_arr[2].set_ylabel(ylabel)
    fig.suptitle("ReS2 Noise Robustness: Boxplots by PSNR and Sampling Steps (Outliers Removed)", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="为 ReS2 噪声实验绘制按步数拆分的 PSNR 箱线图（IQR 去异常值）")
    ap.add_argument("--experiments_root", default="data/experiments", help="实验根目录（默认 data/experiments）")
    ap.add_argument("--steps", default="25,50,100,200", help="要绘制的步数列表，逗号分隔（默认 25,50,100,200）")
    ap.add_argument("--psnrs", default="12,15,18,22,26,30", help="PSNR 档位列表，逗号分隔（默认 12,15,18,22,26,30）")
    ap.add_argument("--column", default="err_A", help="纵轴数据列名（默认 err_A）")
    ap.add_argument(
        "--output",
        default="data/experiments/plots/res2_noise_errA_boxplot_by_step_psnr_no_outliers.png",
        help="输出图片路径",
    )
    ap.add_argument(
        "--summary_csv",
        default="data/experiments/plots/res2_noise_errA_boxplot_by_step_psnr_no_outliers_summary.csv",
        help="输出每组 IQR 过滤统计表路径",
    )
    ap.add_argument("--ylabel", default="delta d / err_A (A)", help="纵轴标题")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    experiments_root = Path(args.experiments_root).resolve()
    output_path = Path(args.output).resolve()
    summary_csv = Path(args.summary_csv).resolve()
    steps = parse_int_list(str(args.steps))
    psnrs = parse_int_list(str(args.psnrs))

    if len(steps) != 4:
        raise SystemExit(f"当前脚本固定输出 4 个子图，请传入 4 个步数。收到: {steps}")

    grouped, stats_rows = iter_group_data(experiments_root, steps=steps, psnrs=psnrs, column=str(args.column))
    make_plot(grouped, steps=steps, psnrs=psnrs, output_path=output_path, ylabel=str(args.ylabel))
    write_summary_csv(summary_csv, stats_rows)

    print(f"[OK] 图已保存到: {output_path}")
    print(f"[OK] 统计表已保存到: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
