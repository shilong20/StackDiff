"""
【作用概述】读取材料 JSON 配置并调用 batch_runner 批量生成双层 STEM 仿真 PNG；默认不保留中间文件，支持可选导出 labels（npz + 对齐可视化）。
【关联说明】文件/模块：generate_sample/batch_runner.py；generate_sample/{ReS2,MoS2,MoTe2,TaS2}.json；generate_sample/{ReS2,MoS2,MoTe2,TaS2}.xyz；generate_sample/incostem；generate_sample/mask/*.png。
【命令行用法】python generate_sample/Batch_generate.py --config generate_sample/ReS2.json
（参数：--config=配置路径；环境变量：`MOIRE_SAVE_LABELS=1` 导出 labels；`MOIRE_SAVE_MONOLAYER=1` 额外导出单层图 *_0/*_1，便于后处理测试）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from batch_runner import load_config, run_batch


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="批量生成：直接输出 PNG（扭角可选，无中间文件落地）")
    ap.add_argument("--config", default=str(here / "ReS2.json"), help="材料 JSON 配置文件路径")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    run_batch(cfg)
    print(f"完成，PNG 输出目录：{cfg.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
