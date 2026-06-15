# Training Source STEM Generator

This directory contains optional source-image STEM simulation helpers for StackDiff training. It supports the public materials: ReS2, MoS2, MoTe2, 1T-TaS2, 1H-TaS2, and CrBr3.

The repository includes source-image generator code, material source JSON files, monolayer structure files, and safe-crop masks. The external `incostem` executable is not distributed with StackDiff.

This public release exposes only the single-layer/source image generation path used before training. Separation examples under `data/examples/<material>/` are static demo inputs; their generation process is not part of this public toolchain.

## External Simulator

Download or compile `incostem` from the official computem/temsim project page:

https://sourceforge.net/projects/computem/files/

Place the executable here:

```bash
cp /path/to/incostem src/tools/synthetic_materials/incostem
chmod +x src/tools/synthetic_materials/incostem
```

Alternatively, edit a JSON file under `configs/` and set `incostem_path` to an absolute path. The local executable path is ignored by Git.

## Files

- `generate_source.py`: command-line entry point for pre-training source STEM images.
- `source_runner.py`: source-image structure perturbation, simulator invocation, mask copying, and manifest writing.
- `configs/*.json`: source-image generation configurations. They record the legacy source-image sizes and simulator-level source variations.
- `structures/*.xyz`: monolayer structure files.
- `masks/*.png`: safe-crop masks.

## Usage

Generate source STEM images for training:

```bash
python src/tools/synthetic_materials/generate_source.py --config src/tools/synthetic_materials/configs/ReS2.json
python src/tools/synthetic_materials/generate_source.py --config src/tools/synthetic_materials/configs/MoS2.json
python src/tools/synthetic_materials/generate_source.py --config src/tools/synthetic_materials/configs/MoTe2.json
python src/tools/synthetic_materials/generate_source.py --config src/tools/synthetic_materials/configs/1T-TaS2.json
python src/tools/synthetic_materials/generate_source.py --config src/tools/synthetic_materials/configs/1H-TaS2.json
python src/tools/synthetic_materials/generate_source.py --config src/tools/synthetic_materials/configs/CrBr3.json
```

Source-image generation writes PNG files, `mask.png`, `manifest.jsonl`, and `source_config_snapshot.json` under `data/training_source/<material>/`. This step only applies simulator-level variations such as thermal displacement and defects. It does not apply training-time crop, rotation, scan-noise, display, or edge-mask augmentation.

Source configs use weighted sampling for these simulator-level variations. The public defaults sample thermal displacement as `0.00/0.01/0.02/0.04 A = 15/35/35/15%` and defect presets as `clean/low/medium/high = 50/30/15/5%`. For multi-element materials, each defect preset sets the relevant element defect rates together.

The generated source images are intended for `configs/train/<material>.yml`, where the training augmentation pipeline crops and perturbs them during training.
