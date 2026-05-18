# Example Data

This directory contains small examples copied from the development workspace
for release smoke tests. It is not the full training or evaluation dataset.

- `training/`: representative training-style images and masks for the four paper materials.
- `slip_psnr24/`: small bilayer slip examples for ReS2, MoS2, MoTe2, and TaS2.
  ReS2 examples come from a PSNR=24 noise-vs-GT set; the other materials use
  existing slip example folders because PSNR=24 variants were not available locally.
