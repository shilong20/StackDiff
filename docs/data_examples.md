# Example Data

This repository includes only lightweight examples under `data/examples/`.
Large training datasets, full experiment folders, model outputs, and paper
result archives are not tracked in Git.

## Structure

- `data/examples/training/<material>/`: a few representative training-style
  STEM images and masks where available.
- `data/examples/<material>/`: a few bilayer slip examples for
  smoke tests and documentation.
- `generate_sample/*.xyz`: material structure files required by the synthetic
  generation pipeline.

The public release focuses on the four materials evaluated in the paper:
ReS2, MoS2, MoTe2, and TaS2. Additional exploratory configs, if present under
`docs/archived_configs/untested/`, are not part of the reported benchmark.

## External Tools

The `generate_sample` pipeline expects an external `incostem` executable for
STEM simulation. The binary and upstream source are not distributed here.
Install or compile it separately, then place the executable at
`generate_sample/incostem` when running generation locally.
