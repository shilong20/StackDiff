"""
Generate synthetic ReS2 bilayer STEM images from a JSON configuration.

Related files: tools/synthetic_res2/runner.py, tools/synthetic_res2/config.json,
tools/synthetic_res2/structures/ReS2.xyz, tools/synthetic_res2/masks/ReS2.png,
and the external tools/synthetic_res2/incostem executable.

Command-line usage:
    python tools/synthetic_res2/generate.py --config tools/synthetic_res2/config.json

Set MOIRE_SAVE_LABELS=1 to export per-image label npz files. Set
MOIRE_SAVE_MONOLAYER=1 to additionally export per-layer images for debugging.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from runner import load_config, run_batch


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Generate synthetic ReS2 bilayer STEM PNGs.")
    ap.add_argument("--config", default=str(here / "config.json"), help="Path to the synthetic ReS2 JSON configuration.")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    run_batch(cfg)
    print(f"Done. PNG output directory: {cfg.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
