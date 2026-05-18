#!/usr/bin/env python3
"""为 carbon_samples 和 noise_samples 选择“横跨各组相同”的 top-3 样本。

对每个 stem 计算 4 个组中 err_A 的平均值，挑选平均误差最小的 3 个样本，
并复制 (layer0/layer1/combined/input) 到每个组的 sample{i}_err{avg:.4f}A/ 下。
"""
from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CARBON_GROUPS = [
    ("alpha0.2", REPO / "data/experiments/carbon/ReS2_carbon_alpha0p2/result_step200"),
    ("alpha0.4", REPO / "data/experiments/carbon/ReS2_carbon_alpha0p4/result_step200"),
    ("alpha0.6", REPO / "data/experiments/carbon/ReS2_carbon_alpha0p6/result_step200"),
    ("alpha0.8", REPO / "data/experiments/carbon/ReS2_carbon_alpha0p8/result_step200"),
]
NOISE_GROUPS = [
    ("PSNR12", REPO / "data/experiments/noise/ReS2_noise_12_defect0/result_step200"),
    ("PSNR15", REPO / "data/experiments/noise/ReS2_noise_15_defect0/result_step200"),
    ("PSNR18", REPO / "data/experiments/noise/ReS2_noise_18_defect0/result_step200"),
    ("PSNR22", REPO / "data/experiments/noise/ReS2_noise_22_defect0/result_step200"),
    ("PSNR26", REPO / "data/experiments/noise/ReS2_noise_26_defect0/result_step200"),
    ("PSNR30", REPO / "data/experiments/noise/ReS2_noise_30_defect0/result_step200"),
]

FILE_MAP = {
    "layer0.png": "_0.png",
    "layer1.png": "_1.png",
    "combined.png": "_conbine.png",
    "input.png": "_original.png",
}


def load_errs(result_dir: Path) -> dict[str, float]:
    csv_path = result_dir / "result_shift_pbc.csv"
    out: dict[str, float] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                out[row["stem"]] = float(row["err_A"])
            except (KeyError, ValueError):
                continue
    return out


def pick_top(groups: list[tuple[str, Path]], k: int = 3) -> list[tuple[str, float, dict[str, float]]]:
    per_group = {name: load_errs(d) for name, d in groups}
    common = set.intersection(*(set(v.keys()) for v in per_group.values()))
    rows: list[tuple[str, float, dict[str, float]]] = []
    for stem in common:
        errs = {g: per_group[g][stem] for g in per_group}
        vals = list(errs.values())
        if any(math.isnan(v) or math.isinf(v) for v in vals):
            continue
        avg = sum(vals) / len(vals)
        rows.append((stem, avg, errs))
    rows.sort(key=lambda x: x[1])
    return rows[:k]


def copy_samples(groups: list[tuple[str, Path]], selections: list[tuple[str, float, dict[str, float]]], out_root: Path, dry_run: bool = False) -> None:
    for group_name, src_dir in groups:
        group_out = out_root / group_name
        if group_out.exists():
            for child in group_out.iterdir():
                if child.is_dir() and child.name.startswith("sample"):
                    if not dry_run:
                        shutil.rmtree(child)
                    else:
                        print(f"  [dry] rm -rf {child}")
        if not dry_run:
            group_out.mkdir(parents=True, exist_ok=True)
        for idx, (stem, avg, errs) in enumerate(selections, start=1):
            dst_dir = group_out / f"sample{idx}_err{avg:.4f}A"
            src_stem_dir = src_dir / stem
            if not src_stem_dir.is_dir():
                print(f"  [warn] missing {src_stem_dir}")
                continue
            if not dry_run:
                dst_dir.mkdir(parents=True, exist_ok=True)
            for out_name, suffix in FILE_MAP.items():
                src = src_stem_dir / f"{stem}{suffix}"
                dst = dst_dir / out_name
                if not src.is_file():
                    print(f"  [warn] missing {src}")
                    continue
                if dry_run:
                    print(f"  [dry] cp {src} -> {dst}")
                else:
                    shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    for label, groups, out_root in [
        ("CARBON", CARBON_GROUPS, REPO / "carbon_samples"),
        ("NOISE", NOISE_GROUPS, REPO / "noise_samples"),
    ]:
        print(f"\n=== {label} ===")
        selections = pick_top(groups, k=args.k)
        for i, (stem, avg, errs) in enumerate(selections, 1):
            parts = " ".join(f"{g}={errs[g]:.4f}" for g in errs)
            print(f"  sample{i} avg={avg:.4f}A  stem={stem}\n    {parts}")
        copy_samples(groups, selections, out_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
