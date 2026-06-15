"""
【作用概述】命令行入口，用于生成训练前的单层/source STEM 仿真大图；读取 source JSON 后调用 source_runner，并在 data/training_source/<material>/ 写出 PNG、mask、manifest 和配置快照。
【关联说明】关联文件：src/tools/synthetic_materials/source_runner.py、src/tools/synthetic_materials/configs/*.json、configs/train/*.yml、data/training_source/*/mask.png。
【命令行用法】python src/tools/synthetic_materials/generate_source.py --config src/tools/synthetic_materials/configs/ReS2.json（参数：--config 选择材料 source 生成 JSON）
"""

from __future__ import annotations

import argparse
from pathlib import Path

from source_runner import load_source_config, run_source_batch


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Generate pre-training source STEM simulation PNGs.")
    ap.add_argument(
        "--config",
        default=str(here / "configs" / "ReS2.json"),
        help="Path to a source-generation JSON configuration.",
    )
    args = ap.parse_args()

    config_path = Path(args.config)
    cfg = load_source_config(config_path)
    run_source_batch(cfg, source_config_path=config_path)
    print(f"Done. Source PNG output directory: {cfg.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
