"""
Purpose: Command-line entry point for generating pre-training source STEM simulation images. It reads a source-generation JSON config, calls the source runner, and writes source PNGs, mask.png, manifest.jsonl, and a config snapshot under data/training_source/<material>. This command does not apply training-time crop/noise/display augmentation; those transformations are applied by the training data loader.
Related files: src/tools/synthetic_materials/source_runner.py, src/tools/synthetic_materials/source_configs/*.json, configs/train/*.yml, and data/training_source/*/mask.png.
CLI usage: python src/tools/synthetic_materials/generate_source.py --config src/tools/synthetic_materials/source_configs/ReS2.json (arguments: --config selects a material source-generation JSON).
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
        default=str(here / "source_configs" / "ReS2.json"),
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
