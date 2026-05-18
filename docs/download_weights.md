# Model Checkpoints

The Git repository does not contain model checkpoints. The public release uses
one recommended EMA checkpoint per material, and configs point to these paths:

| Material | Checkpoint path |
| --- | --- |
| ReS2 | `models/checkpoints/ReS2/ema_0.9999_200000.pt` |
| MoS2 | `models/checkpoints/MoS2/ema_0.9999_200000.pt` |
| MoTe2 | `models/checkpoints/MoTe2/ema_0.9999_200000.pt` |
| TaS2 | `models/checkpoints/TaS2/ema_0.9999_200000.pt` |
| WS2 | `models/checkpoints/WS2/ema_0.9999_150000.pt` |
| CrI3 | `models/checkpoints/CrI3/ema_0.9999_200000.pt` |

Recommended hosting:

- Hugging Face Hub for day-to-day checkpoint download.
- Zenodo for archival DOI after the paper/release metadata is finalized.

After uploading the files, update this document with the final URLs and
SHA256 checksums.
