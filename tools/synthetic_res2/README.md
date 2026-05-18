# ReS2 Synthetic STEM Generator

This directory contains the optional ReS2 synthetic bilayer STEM image generator used for examples and sanity checks. It is not required for running layer separation on your own images.

The repository includes only the generator code, one ReS2 structure file, one mask, and a JSON configuration. The external `incostem` executable is not distributed with StackDiff.

## External Simulator

Download or compile `incostem` from the official computem/temsim project page:

https://sourceforge.net/projects/computem/files/

Place the executable here:

```bash
cp /path/to/incostem tools/synthetic_res2/incostem
chmod +x tools/synthetic_res2/incostem
```

Alternatively, edit `config.json` and set `incostem_path` to an absolute path. The local executable path is ignored by Git.

## Files

- `generate.py`: command-line entry point.
- `runner.py`: structure construction, simulator invocation, augmentation, and PNG writing.
- `augment.py`: lightweight image augmentation utilities used by `runner.py`.
- `config.json`: default ReS2 generation configuration.
- `structures/ReS2.xyz`: monolayer ReS2 structure.
- `masks/ReS2.png`: safe-crop mask.

## Usage

```bash
python tools/synthetic_res2/generate.py --config tools/synthetic_res2/config.json
```

By default, generated PNG files are written to `outputs/generated/ReS2_rotation`.

To also export ground-truth label files:

```bash
MOIRE_SAVE_LABELS=1 python tools/synthetic_res2/generate.py --config tools/synthetic_res2/config.json
```
