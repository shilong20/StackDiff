# StackDiff

StackDiff is a diffusion-based toolkit for ReS2 multilayer STEM image separation and interlayer analysis. This release is intentionally compact: it keeps the ReS2 inference pipeline, a small ReS2 example set, ReS2 lattice/interlayer analysis utilities, and an optional ReS2 synthetic STEM generator.

Training scripts, large datasets, full experiment logs, non-ReS2 benchmark configs, and model checkpoint files are not included in the public repository.

## Contents

- `src/main.py`: ReS2 multilayer separation entry point.
- `configs/separate/ReS2.yml`: default ReS2 separation configuration.
- `data/examples/ReS2/`: small demo multilayer inputs.
- `tools/pred_dadb/`: ReS2 atom detection, lattice-vector estimation, and bilayer classification.
- `tools/analyze_interlayer_res2.py`: slip and twist analysis built on `tools/pred_dadb`.
- `tools/synthetic_res2/`: optional ReS2 synthetic bilayer STEM generator.

## Installation

```bash
pip install -r requirements.txt
```

The separation pipeline expects a ReS2 checkpoint at:

```text
models/checkpoints/ReS2/ema_0.9999_200000.pt
```

Checkpoint files are not tracked by Git. After the public checkpoint is uploaded, fill the URL in `scripts/download_weights.py` or place the file manually at the path above.

## ReS2 Layer Separation

Run the bundled ReS2 example:

```bash
python src/main.py --config configs/separate/ReS2.yml
```

By default, inputs are read from:

```text
data/examples/ReS2/
```

and outputs are written to:

```text
outputs/separation/ReS2/
```

For your own images, either place them under `data/examples/ReS2/` or edit `paths.default_input` and `paths.default_output` in `configs/separate/ReS2.yml`.

Expected outputs for each input image include:

- `_original.png`: the processed input image.
- `_0.png`: separated layer 0.
- `_1.png`: separated layer 1.
- `_combine.png`: the reconstructed superposition.

If your input filename ends with a physical field-of-view suffix such as `sample-7.1x3.1.png`, the auto-crop logic can infer a grid and crop/resize plan from that size. Otherwise, the pipeline falls back to the configured default grid settings.

## ReS2 da/db and Interlayer Analysis

If you already have per-layer images organized as one folder per bilayer sample, with files named `*_0.png` and `*_1.png`, run:

```bash
python tools/pred_dadb/pipeline_bilayer_root.py \
  --root path/to/bilayer_folders \
  --out_csv outputs/res2_pred_dadb.csv
```

This detects Re atom positions, estimates the ReS2 lattice vectors `(da, db)` for each layer, and classifies each sample as `slip`, `twist`, `flip_slip`, `flip_twist`, or `unknown`.

To compute physical interlayer quantities from the same folder tree:

```bash
python tools/analyze_interlayer_res2.py \
  --root path/to/bilayer_folders \
  --out_csv outputs/res2_interlayer.csv \
  --pred_csv outputs/res2_pred_dadb.csv \
  --pred_atoms_dir outputs/res2_pred_dadb_atoms
```

For slip-like samples, the analyzer reports pixel and physical shifts and their projection in the `(da, db)` basis. For twist-like samples, it reports a refined twist angle.

## Optional Synthetic ReS2 Generator

The optional generator lives under `tools/synthetic_res2/`.

It depends on the external `incostem` executable from the official computem/temsim project:

https://sourceforge.net/projects/computem/files/

`incostem` is not distributed with StackDiff. Place a local executable at:

```text
tools/synthetic_res2/incostem
```

or edit `tools/synthetic_res2/config.json` and set `incostem_path` to an absolute path.

Generate synthetic ReS2 bilayer PNGs:

```bash
python tools/synthetic_res2/generate.py --config tools/synthetic_res2/config.json
```

To also export ground-truth label files:

```bash
MOIRE_SAVE_LABELS=1 python tools/synthetic_res2/generate.py --config tools/synthetic_res2/config.json
```

## Repository Hygiene

Before publishing or tagging a release, run:

```bash
python scripts/check_release.py
```

The check guards against accidentally committing model weights, large archives, local paths, cache folders, and other non-release artifacts.

## License

This project is released under the MIT License. See `LICENSE` for details.

## Acknowledgements

StackDiff builds on ideas and code patterns from diffusion inverse-problem methods and OpenAI guided-diffusion. External simulator tools such as computem/temsim are separate projects and are not vendored in this repository.
