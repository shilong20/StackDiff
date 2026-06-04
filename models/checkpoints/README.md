# Model Checkpoints

Model checkpoint files are not tracked in Git.

The public configs expect one released EMA checkpoint per material:

```text
models/checkpoints/ReS2/ema_0.9999_200000.pt
models/checkpoints/MoS2/ema_0.9999_200000.pt
models/checkpoints/MoTe2/ema_0.9999_200000.pt
models/checkpoints/1T-TaS2/ema_0.9999_200000.pt
models/checkpoints/1H-TaS2/ema_0.9999_200000.pt
models/checkpoints/CrBr3/ema_0.9999_200000.pt
```

After uploading public checkpoints, fill `scripts/download_weights.py`; until
then, place the files manually at the paths above.
