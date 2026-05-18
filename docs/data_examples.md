# Example Data

This repository includes only lightweight examples under `data/examples/`.
Large training datasets, full experiment folders, model outputs, and paper
result archives are not tracked in Git.

## Structure

- `data/examples/training/<material>/`: a few representative training-style
  STEM images and masks where available.
- `data/examples/slip_psnr24/<material>/`: a few bilayer slip examples for
  smoke tests and documentation.
- `generate_sample/*.xyz`: material structure files required by the synthetic
  generation pipeline.

## External Tools

The `generate_sample` pipeline expects an external `incostem` executable for
STEM simulation. The binary and upstream source are not distributed here.
Install or compile it separately, then place the executable at
`generate_sample/incostem` when running generation locally.
