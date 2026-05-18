#!/usr/bin/env python3
"""
【作用概述】通用鲁棒性实验箱线图工具：从多个"参数档位"的评估 CSV 中读取误差列，绘制"横轴=参数档位、纵轴=误差"的箱线图（IQR 去异常值）。
支持 dopant、carbon 和其他自定义参数扫描实验。
【关联说明】文件/模块：data/experiments/ReS2_dopant*/result_step*/result_shift_pbc.csv；
data/experiments/ReS2_carbon_alpha*/result_step*/result_shift_pbc.csv；
tools/evaluate_generate_sample_wraparound_pbc.py（生成 err_A 列）。
【命令行用法】
  dopant 实验：
    python tools/plot_robustness_boxplots.py --mode dopant --experiments_root data/experiments --step 200
  carbon 实验：
    python tools/plot_robustness_boxplots.py --mode carbon --experiments_root data/experiments --step 200
  自定义：
    python tools/plot_robustness_boxplots.py --mode custom --dir_pattern "ReS2_foo_{param}" --params "1,2,3" ...
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class GroupStats:
    label: str
    n_raw: int
    n_filtered: int
    q1: float
    q3: float
    iqr: float
    lo: float
    hi: float
    mean_filtered: float
    median_filtered: float


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
        writer.writerow(["label", "n_raw", "n_filtered", "q1", "q3", "iqr",
                         "iqr_lo", "iqr_hi", "mean_filtered", "median_filtered"])
        for row in rows:
            writer.writerow([row.label, row.n_raw, row.n_filtered, row.q1, row.q3,
                             row.iqr, row.lo, row.hi, row.mean_filtered, row.median_filtered])


# ----- 预定义模式的目录名 / 标签映射 -----

def _dopant_dir_and_labels(params: List[str]) -> list[tuple[str, str]]:
    return [(f"ReS2_dopant{p}", f"{float(p)*100:.0f}%") for p in params]


def _carbon_dir_and_labels(params: List[str]) -> list[tuple[str, str]]:
    return [(f"ReS2_carbon_alpha{p}", f"α={p.replace('p','.')}") for p in params]


MODE_DEFAULTS = {
    "dopant": {
        "params": "0.05,0.10,0.15,0.20",
        "resolver": _dopant_dir_and_labels,
        "xlabel": "Dopant Rate (Re→Mo)",
        "title": "ReS2 Dopant Robustness",
    },
    "carbon": {
        "params": "0p1,0p2,0p3,0p5,0p7",
        "resolver": _carbon_dir_and_labels,
        "xlabel": "Carbon Contamination (α)",
        "title": "ReS2 Carbon Contamination Robustness",
    },
}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="通用鲁棒性实验箱线图（IQR 去异常值）")
    ap.add_argument("--mode", choices=["dopant", "carbon", "custom"], required=True,
                    help="实验模式：dopant / carbon / custom")
    ap.add_argument("--experiments_root", default="data/experiments", help="实验根目录")
    ap.add_argument("--step", type=int, default=200, help="采样步数（默认 200）")
    ap.add_argument("--params", default=None, help="参数列表（逗号分隔），覆盖模式默认值")
    ap.add_argument("--dir_pattern", default=None,
                    help="custom 模式下目录名模板，用 {param} 作占位符（例如 ReS2_foo_{param}）")
    ap.add_argument("--column", default="err_A", help="纵轴数据列名（默认 err_A）")
    ap.add_argument("--ylabel", default="delta d / err_A (Å)", help="纵轴标题")
    ap.add_argument("--xlabel", default=None, help="横轴标题（覆盖模式默认值）")
    ap.add_argument("--title", default=None, help="图标题（覆盖模式默认值）")
    ap.add_argument("--output", default=None, help="输出图片路径（默认自动生成）")
    ap.add_argument("--summary_csv", default=None, help="输出统计表路径（默认自动生成）")
    return ap


def main() -> int:
    args = build_argparser().parse_args()
    experiments_root = Path(args.experiments_root).resolve()
    step = int(args.step)

    if args.mode in MODE_DEFAULTS:
        defaults = MODE_DEFAULTS[args.mode]
        params_raw = args.params or defaults["params"]
        params = [p.strip() for p in params_raw.split(",") if p.strip()]
        dir_labels = defaults["resolver"](params)
        xlabel = args.xlabel or defaults["xlabel"]
        title = args.title or defaults["title"]
    elif args.mode == "custom":
        if not args.params or not args.dir_pattern:
            raise SystemExit("custom 模式需要同时提供 --params 和 --dir_pattern。")
        params = [p.strip() for p in args.params.split(",") if p.strip()]
        dir_labels = [(args.dir_pattern.replace("{param}", p), p) for p in params]
        xlabel = args.xlabel or "Parameter"
        title = args.title or "Robustness Boxplot"
    else:
        raise SystemExit(f"未知 mode: {args.mode}")

    plots_dir = experiments_root / "plots"
    output_path = Path(args.output).resolve() if args.output else plots_dir / f"robustness_{args.mode}_step{step}.png"
    summary_csv = Path(args.summary_csv).resolve() if args.summary_csv else plots_dir / f"robustness_{args.mode}_step{step}_summary.csv"

    data_list: List[np.ndarray] = []
    labels: List[str] = []
    stats_rows: List[GroupStats] = []

    for dirname, label in dir_labels:
        csv_path = experiments_root / dirname / f"result_step{step}" / "result_shift_pbc.csv"
        raw_vals = load_numeric_column(csv_path, column=str(args.column))
        kept, q1, q3, iqr, lo, hi = iqr_filter(raw_vals)
        data_list.append(kept)
        labels.append(label)
        stats_rows.append(GroupStats(
            label=label, n_raw=int(raw_vals.size), n_filtered=int(kept.size),
            q1=q1, q3=q3, iqr=iqr, lo=lo, hi=hi,
            mean_filtered=float(np.mean(kept)), median_filtered=float(np.median(kept)),
        ))

    # 绘图
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD", "#DD8452"]
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(1, len(data_list) + 1, dtype=np.float64)
    bp = ax.boxplot(
        data_list, positions=positions, widths=0.5, patch_artist=True, showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.6},
        whiskerprops={"color": "#555555", "linewidth": 1.2},
        capprops={"color": "#555555", "linewidth": 1.2},
        boxprops={"edgecolor": "#555555", "linewidth": 1.2},
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.75)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(str(args.ylabel))
    ax.set_title(f"{title} (Step {step}, Outliers Removed)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    write_summary_csv(summary_csv, stats_rows)
    print(f"[OK] 图已保存到: {output_path}")
    print(f"[OK] 统计表已保存到: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
