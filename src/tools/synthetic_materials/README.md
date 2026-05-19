# Synthetic Material STEM Generator

This directory contains the optional synthetic bilayer STEM generator used for StackDiff examples and training-data preparation. It supports the four public paper materials: ReS2, MoS2, MoTe2, and TaS2.

The repository includes generator code, material JSON files, monolayer structure files, and safe-crop masks. The external `incostem` executable is not distributed with StackDiff.

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

- `generate.py`: command-line entry point.
- `runner.py`: structure construction, simulator invocation, augmentation, and PNG writing.
- Shared augmentation utilities are imported from `src/core/augmentations/stem.py`.
- `configs/*.json`: material generation configurations.
- `structures/*.xyz`: monolayer structure files.
- `masks/*.png`: safe-crop masks.

## Usage

```bash
python src/tools/synthetic_materials/generate.py --config src/tools/synthetic_materials/configs/ReS2.json
python src/tools/synthetic_materials/generate.py --config src/tools/synthetic_materials/configs/MoS2.json
python src/tools/synthetic_materials/generate.py --config src/tools/synthetic_materials/configs/MoTe2.json
python src/tools/synthetic_materials/generate.py --config src/tools/synthetic_materials/configs/TaS2.json
```

By default, generated PNG files are written to `outputs/generated/<material>_rotation`.

To also export ground-truth label files:

```bash
MOIRE_SAVE_LABELS=1 python src/tools/synthetic_materials/generate.py --config src/tools/synthetic_materials/configs/ReS2.json
```
