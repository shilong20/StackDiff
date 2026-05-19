# StackDiff

StackDiff is a diffusion-based toolkit for multilayer and material-stacking STEM image separation. This public release includes runnable configs and small smoke-test examples for the four paper materials: ReS2, MoS2, MoTe2, and TaS2.

Training datasets, full evaluation sets, experiment logs, and model checkpoint files are not included in Git. Each material is expected to use one released EMA checkpoint.

## Contents

- `src/main.py`: YAML-driven multilayer separation entry point.
- `configs/separate/`: public separation configs for ReS2, MoS2, MoTe2, and TaS2.
- `configs/sample/`: public unconditional sampling configs for the same four materials.
- `configs/train/`: training templates for locally prepared source images.
- `data/examples/<material>/`: small demo multilayer inputs named `0.png` through `4.png`.
- `tools/synthetic_materials/`: optional synthetic STEM generator assets for the four materials.
- `tools/res2_stacking_analysis/`: ReS2-specific stacking, slip, and twist analysis example.

## Installation

```bash
pip install -r requirements.txt
```

Place released checkpoints at:

```text
models/checkpoints/ReS2/ema_0.9999_200000.pt
models/checkpoints/MoS2/ema_0.9999_200000.pt
models/checkpoints/MoTe2/ema_0.9999_200000.pt
models/checkpoints/TaS2/ema_0.9999_200000.pt
```

After public checkpoint URLs are available, `scripts/download_weights.py` can be filled and used as:

```bash
python scripts/download_weights.py --material all
```

## Layer Separation

Run one of the bundled examples:

```bash
python src/main.py --config configs/separate/ReS2.yml
python src/main.py --config configs/separate/MoS2.yml
python src/main.py --config configs/separate/MoTe2.yml
python src/main.py --config configs/separate/TaS2.yml
```

By default, inputs are read from `data/examples/<material>/` and outputs are written to `outputs/separation/<material>/`.

Expected outputs for each input image include:

- `_original.png`: processed input image.
- `_0.png`: separated layer 0.
- `_1.png`: separated layer 1.
- `_combine.png`: reconstructed superposition.

For your own images, edit `paths.default_input` and `paths.default_output` in the relevant YAML file. If an input filename ends with a physical field-of-view suffix such as `sample-7.1x3.1.png`, auto-crop can infer a grid and crop/resize plan from that size when enabled in the config.

## Sampling And Training Templates

Generate unconditional samples from a released checkpoint:

```bash
python src/core/scripts/image_sample.py --config configs/sample/ReS2.yml
python src/core/scripts/image_sample.py --config configs/sample/MoS2.yml
python src/core/scripts/image_sample.py --config configs/sample/MoTe2.yml
python src/core/scripts/image_sample.py --config configs/sample/TaS2.yml
```

Training configs are templates for locally prepared source images. Before training, update the source-image and mask paths in `configs/train/<material>.yml`.

```bash
python src/core/scripts/image_train.py --config configs/train/ReS2.yml
```

## Synthetic Material Generator

The optional generator lives under `tools/synthetic_materials/` and includes JSON/XYZ/mask assets for ReS2, MoS2, MoTe2, and TaS2.

It depends on the external `incostem` executable from the official computem/temsim project:

https://sourceforge.net/projects/computem/files/

`incostem` is not distributed with StackDiff. Place a local executable at:

```text
tools/synthetic_materials/incostem
```

or edit the material JSON and set `incostem_path` to an absolute path.

Generate synthetic bilayer PNGs:

```bash
python tools/synthetic_materials/generate.py --config tools/synthetic_materials/configs/ReS2.json
python tools/synthetic_materials/generate.py --config tools/synthetic_materials/configs/MoS2.json
python tools/synthetic_materials/generate.py --config tools/synthetic_materials/configs/MoTe2.json
python tools/synthetic_materials/generate.py --config tools/synthetic_materials/configs/TaS2.json
```

To also export ground-truth label files:

```bash
MOIRE_SAVE_LABELS=1 python tools/synthetic_materials/generate.py --config tools/synthetic_materials/configs/ReS2.json
```

## ReS2 Stacking Analysis Example

The stacking-analysis tools are ReS2-specific examples. They demonstrate atom detection, per-layer lattice vector estimation, slip da/db projection, and twist/flip classification for separated ReS2 bilayers. They are not presented as material-general analysis tools.

If you have one folder per bilayer sample with files named `*_0.png` and `*_1.png`, run:

```bash
python tools/res2_stacking_analysis/classify_bilayers.py \
  --root path/to/bilayer_folders \
  --out_csv outputs/res2_stacking.csv
```

To compute interlayer quantities from the same folder tree:

```bash
python tools/res2_stacking_analysis/analyze_interlayer.py \
  --root path/to/bilayer_folders \
  --out_csv outputs/res2_interlayer.csv \
  --stacking_csv outputs/res2_stacking.csv \
  --stacking_atoms_dir outputs/res2_stacking_atoms
```

For slip-like samples, the analyzer reports pixel/physical shifts and their projection in the `(da, db)` basis. For twist-like samples, it reports a refined twist angle.

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
