"""
Generate synthetic bilayer STEM images from a material JSON configuration.

Related files: src/tools/synthetic_materials/runner.py,
src/tools/synthetic_materials/configs/*.json, src/tools/synthetic_materials/structures/*.xyz,
src/tools/synthetic_materials/masks/*.png, and the external
src/tools/synthetic_materials/incostem executable.

Command-line usage:
    python src/tools/synthetic_materials/generate.py --config src/tools/synthetic_materials/configs/ReS2.json

Set MOIRE_SAVE_LABELS=1 to export per-image label npz files. Set
MOIRE_SAVE_MONOLAYER=1 to additionally export per-layer images for debugging.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from runner import load_config, run_batch


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Generate synthetic bilayer STEM PNGs.")
    ap.add_argument(
        "--config",
        default=str(here / "configs" / "ReS2.json"),
        help="Path to a synthetic material JSON configuration.",
    )
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    run_batch(cfg)
    print(f"Done. PNG output directory: {cfg.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
