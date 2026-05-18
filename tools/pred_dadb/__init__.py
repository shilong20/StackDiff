"""
【作用概述】pred_dadb 工具包：在单层 ReS2 图像中提取 Re 原子点云/四元环质心，并输出当前流程定义下的 (da,db) 与原点（像素坐标系）。
【关联说明】文件/模块：
- tools/pred_dadb/atoms.py（原子点云提取：detect_atoms_from_image）
- tools/pred_dadb/cycles.py（四元环提取：seed + 模板吸附 + 平移生长）
- tools/pred_dadb/centroid_chain.py（质心点云 -> Re 链方向；并提供 da/db 的推导工具）
- tools/pred_dadb/pipeline_single.py（单张图：原子->四元环/质心->Re链->da/db+原点；库形式）
- tools/pred_dadb/pipeline_bilayer_root.py（批量双层分解文件夹分类：slip/twist/flip_*；并可选输出 atoms.json）
【命令行用法】本文件不直接运行（作为包初始化与说明）。
"""
