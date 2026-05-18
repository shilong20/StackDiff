#!/usr/bin/env python3
"""
【作用概述】解析训练输出的 `progress.csv`，绘制 loss/mse/vb 等曲线并保存图片，用于快速检查训练稳定性与断点续训影响。
【关联说明】文件/模块：src/core/sample/progress.csv 或训练输出目录中的 progress.csv；与 generate_sample 无强耦合。
【命令行用法】python tools/analyze_training_loss.py --log <path/to/progress.csv> --out loss_analysis.png（参数：--breakpoint=断点 step 可选）。

分析 ReS2 训练日志中 loss 的变化趋势

本脚本分析扩散模型训练过程中的损失函数变化：
- loss: 主要损失函数 = MSE + VB (如果学习方差)
- mse: 均方误差损失，衡量模型预测与目标的差异 
- vb: 变分下界损失，用于学习方差参数
- loss_q0-q3: 按时间步分位数统计的损失
- 支持分析断点续训的影响
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

def _empty_data_dict() -> Dict[str, List]:
    return {
        'step': [],  # 改为单数以匹配日志字段
        'loss': [],
        'mse': [],
        'vb': [],
        'loss_q0': [],
        'loss_q1': [],
        'loss_q2': [],
        'loss_q3': [],
        'grad_norm': [],
        'param_norm': []
    }


def parse_progress_csv(csv_file: Path) -> Dict[str, np.ndarray]:
    """从 progress.csv 读取训练曲线，保持数值精度"""
    data = _empty_data_dict()

    with csv_file.open('r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for metric in data.keys():
                raw = row.get(metric)
                if raw is None or raw == "":
                    data[metric].append(np.nan)
                    continue

                if metric == 'step':
                    data[metric].append(int(float(raw)))
                else:
                    try:
                        data[metric].append(float(raw))
                    except ValueError:
                        data[metric].append(np.nan)

    for key in data:
        data[key] = np.array(data[key])

    return data
def load_training_data(log_file: Path) -> Dict[str, np.ndarray]:
    """加载 progress.csv 训练数据"""
    if log_file.suffix != '.csv':
        raise ValueError(f'当前仅支持解析 progress.csv，请提供 CSV 文件 (收到: {log_file.name})')

    if not log_file.exists():
        raise FileNotFoundError(f'找不到指定的 CSV 文件: {log_file}')

    return parse_progress_csv(log_file)


def plot_loss_curves(data: Dict[str, np.ndarray], breakpoint: int = None, save_path: str = 'loss_analysis.png'):
    """绘制训练损失曲线"""
    steps = data['step']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ReS2 Training Loss Analysis', fontsize=16)

    # 1. 主要损失曲线
    ax1 = axes[0, 0]
    ax1.plot(steps, data['loss'], 'b-', label='Training Loss', alpha=0.7, linewidth=1)

    # 如果指定了断点，绘制断点线
    if breakpoint is not None:
        before_break = steps <= breakpoint
        after_break = steps > breakpoint
        ax1.axvline(x=breakpoint, color='green', linestyle='--', alpha=0.8, label=f'{breakpoint} step breakpoint')
    else:
        # 如果没有断点，可以选择性地标记一些关键点
        pass

    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Total Loss (MSE + VB)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. MSE vs VB损失
    ax2 = axes[0, 1]
    if not np.all(np.isnan(data['mse'])):
        ax2.plot(steps, data['mse'], 'b-', label='MSE Loss', alpha=0.7, linewidth=1)
    if not np.all(np.isnan(data['vb'])):
        ax2.plot(steps, data['vb'], 'r-', label='VB Loss', alpha=0.7, linewidth=1)
    if breakpoint is not None:
        ax2.axvline(x=breakpoint, color='green', linestyle='--', alpha=0.8)
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Loss Value')
    ax2.set_title('MSE vs VB Loss Decomposition')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # 3. 分位数损失
    ax3 = axes[1, 0]
    colors = ['purple', 'orange', 'green', 'red']
    for i, q in enumerate(['q0', 'q1', 'q2', 'q3']):
        loss_key = f'loss_{q}'
        if not np.all(np.isnan(data[loss_key])):
            ax3.plot(steps, data[loss_key], color=colors[i], label=f'Loss {q}', alpha=0.7, linewidth=1)
    if breakpoint is not None:
        ax3.axvline(x=breakpoint, color='green', linestyle='--', alpha=0.8)
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Loss Value')
    ax3.set_title('Timestep Quantile Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # 4. 梯度和参数范数
    ax4 = axes[1, 1]
    if not np.all(np.isnan(data['grad_norm'])):
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(steps, data['grad_norm'], 'b-', label='Gradient Norm', alpha=0.7, linewidth=1)
        if not np.all(np.isnan(data['param_norm'])):
            line2 = ax4_twin.plot(steps, data['param_norm'], 'r-', label='Parameter Norm', alpha=0.7, linewidth=1)
        if breakpoint is not None:
            ax4.axvline(x=breakpoint, color='green', linestyle='--', alpha=0.8)
        ax4.set_xlabel('Training Steps')
        ax4.set_ylabel('Gradient Norm', color='b')
        ax4_twin.set_ylabel('Parameter Norm', color='r')
        ax4.set_title('Gradient and Parameter Norms')

        # 合并图例
        lines = line1
        if not np.all(np.isnan(data['param_norm'])):
            lines += line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper left')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def analyze_discontinuity(data: Dict[str, np.ndarray], breakpoint: int = None):
    """分析断点续训的影响"""
    # 只有当用户明确指定断点时才进行分析
    if breakpoint is None:
        print("未指定断点续训位置，跳过断点分析")
        print("如需分析断点影响，请使用 --breakpoint 参数指定断点步数")
        return

    steps = data['step']

    # 找到断点前后的数据点
    before_idx = np.where(steps < breakpoint)[0]
    after_idx = np.where(steps >= breakpoint)[0]

    if len(before_idx) == 0 or len(after_idx) == 0:
        print(f"未找到断点{breakpoint}前后的数据，无法分析断点续训影响")
        return

    print(f"=== 断点续训影响分析 ({breakpoint}步) ===")
    print()

    # 分析主要指标的跳跃
    metrics_to_analyze = ['loss', 'mse', 'vb']

    for metric in metrics_to_analyze:
        if np.all(np.isnan(data[metric])):
            continue

        # 断点前最后几个值的平均值
        before_values = data[metric][before_idx]
        after_values = data[metric][after_idx]

        valid_before = before_values[~np.isnan(before_values)]
        valid_after = after_values[~np.isnan(after_values)]

        if len(valid_before) == 0 or len(valid_after) == 0:
            continue

        # 取断点前后各5个点的平均值来减少噪声影响
        before_avg = np.mean(valid_before[-5:]) if len(valid_before) >= 5 else valid_before[-1]
        after_avg = np.mean(valid_after[:5]) if len(valid_after) >= 5 else valid_after[0]

        jump = after_avg - before_avg
        relative_change = (jump / before_avg) * 100 if before_avg != 0 else 0

        print(f"{metric.upper()}:")
        print(f"  断点前平均值: {before_avg:.6f}")
        print(f"  续训后平均值: {after_avg:.6f}")
        print(f"  绝对跳跃: {jump:.6f}")
        print(f"  相对变化: {relative_change:+.2f}%")
        print()

def detailed_statistics(data: Dict[str, np.ndarray], breakpoint: int = None):
    """详细统计分析"""
    steps = data['step']

    print(f"=== 详细统计分析 ===")
    print(f"总数据点: {len(steps)}")
    print(f"训练步数范围: {steps.min()} - {steps.max()}")
    print()

    # 如果有断点，按阶段分析；否则整体分析
    if breakpoint is not None:
        # 有断点的情况
        before_break = steps < breakpoint
        after_break = steps >= breakpoint

        phases = [
            (before_break, f'阶段1 ({steps.min()}-{breakpoint})'),
            (after_break, f'阶段2 ({breakpoint}-{steps.max()})')
        ]
    else:
        # 没有断点的情况，整体分析
        all_mask = np.ones(len(steps), dtype=bool)
        phases = [
            (all_mask, f'整体训练 ({steps.min()}-{steps.max()})')
        ]

    for mask, name in phases:
        if not np.any(mask):
            continue

        print(f"{name}:")
        print(f"  数据点数: {np.sum(mask)}")

        for metric in ['loss', 'mse', 'vb']:
            values = data[metric][mask]
            valid_values = values[~np.isnan(values)]

            if len(valid_values) == 0:
                continue

            print(f"  {metric.upper()}:")
            print(f"    范围: {valid_values.min():.6f} - {valid_values.max():.6f}")
            print(f"    平均值: {valid_values.mean():.6f}")
            print(f"    标准差: {valid_values.std():.6f}")
            print(f"    最终值: {valid_values[-1]:.6f}")
        print()

def main():
    parser = argparse.ArgumentParser(description='分析ReS2训练日志（仅支持 progress.csv）')
    parser.add_argument('--log_file', '-l', default='models/checkpoints/ReS2/progress.csv',
                       help='训练日志 CSV 文件路径（默认指向 progress.csv）')
    parser.add_argument('--breakpoint', '-b', type=int, default=None,
                       help='手动指定断点续训的步数位置（如果不指定，将尝试自动检测）')
    parser.add_argument('--output', '-o', default='outputs/loss_analysis.png',
                       help='输出图片路径')
    parser.add_argument('--no-plot', action='store_true',
                       help='跳过绘图，只进行数值分析')
    
    args = parser.parse_args()
    
    print("正在解析训练日志...")
    data = load_training_data(Path(args.log_file))
    
    print("正在进行统计分析...")
    detailed_statistics(data, args.breakpoint)
    
    print("正在分析断点续训影响...")
    analyze_discontinuity(data, args.breakpoint)
    
    if not args.no_plot:
        print("正在生成可视化图表...")
        plot_loss_curves(data, args.breakpoint, args.output)
        print(f"图表已保存至: {args.output}")

if __name__ == "__main__":
    main()
