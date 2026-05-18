# Model Checkpoints

Model checkpoint files are not tracked in Git.

The public ReS2 inference configuration expects:

```text
models/checkpoints/ReS2/ema_0.9999_200000.pt
```

Place the released ReS2 EMA checkpoint at that path before running
`python src/main.py --config configs/separate/ReS2.yml`.
