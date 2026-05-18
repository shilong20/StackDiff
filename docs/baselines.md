# Baseline Reproduction Notes

Baseline source trees are intentionally not vendored in this repository.

## PGDGAN

- Upstream repository: https://github.com/shahviraj/pgdgan
- The paper comparison uses PGDGAN as an external baseline with the same input
  examples and downstream evaluation scripts where applicable.
- Keep any local modifications as a patch or command note rather than copying
  the full repository into this release.

## DCGAN-tensorflow

- Upstream repository: https://github.com/carpedm20/DCGAN-tensorflow
- The paper comparison uses DCGAN-tensorflow as an external GAN prior baseline.
- TensorFlow 1.x environments are separate from the main StackDiff PyTorch
  environment and should be reproduced from the upstream baseline instructions.
