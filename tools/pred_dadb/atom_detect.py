"""
【作用概述】
兼容模块：提供 `AtomDetectConfig` 与 `detect_atoms_from_image`，用于旧代码路径的平滑迁移。
当前实现直接重定向到 `tools/pred_dadb/atoms.py`（现行 pipeline 使用的原子检测实现）。

核心输入/输出：
- 输入：图像数组（numpy），以及 AtomDetectConfig（是否高通、阈值倍数、最小面积等）。
- 输出：原子点云 (N,2) 像素坐标；可选 debug 字典。

【关联说明】文件/模块：
- tools/pred_dadb/atoms.py（实际实现：AtomDetectConfig / detect_atoms_from_image）
- tools/pred_dadb/vis_scripts/detect_atoms_batch.py（命令行批处理 + overlay 可视化）
- tools/pred_dadb/pipeline_single.py（单图 pipeline：默认直接 import atoms.py）

【命令行用法】本文件不直接运行；如需批处理可视化请使用：
python tools/pred_dadb/vis_scripts/detect_atoms_batch.py --root data/HighT1 --num 20 --seed 0
"""

from __future__ import annotations

from tools.pred_dadb.atoms import AtomDetectConfig, detect_atoms_from_image

__all__ = ["AtomDetectConfig", "detect_atoms_from_image"]

