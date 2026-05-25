"""
Purpose: Generate pre-training source STEM simulation images from monolayer structures. The runner builds one source structure per image, samples weighted thermal-displacement and defect-preset settings, calls the external incostem executable, copies the matching mask, and writes PNG images plus a JSONL manifest. Training-time crop/noise/display augmentations are handled later by configs/train/*.yml and are not applied here.
Related files: src/tools/synthetic_materials/generate_source.py, src/tools/synthetic_materials/source_configs/*.json, src/tools/synthetic_materials/structures/*.xyz, data/training_source/*/mask.png, and configs/train/*.yml.
CLI usage: This module is imported by src/tools/synthetic_materials/generate_source.py and is not intended to be executed directly.
"""

from __future__ import annotations

import json
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read
from PIL import Image

try:
    from runner import ROOT_DIR, _as_path, _incostem_xyz_lines, _run_incostem
except ModuleNotFoundError:
    from .runner import ROOT_DIR, _as_path, _incostem_xyz_lines, _run_incostem


@dataclass(frozen=True)
class ImageSize:
    width: int
    height: int


@dataclass(frozen=True)
class WeightedFloat:
    value: float
    weight: float


@dataclass(frozen=True)
class DefectPreset:
    name: str
    rates: dict[str, float]
    weight: float


@dataclass(frozen=True)
class SourceConfig:
    material: str
    num: int
    seed: int | None
    structure_path: Path
    incostem_path: Path
    incostem_exclude_symbols: list[str]
    output_dir: Path
    mask_path: Path
    image_size: ImageSize
    defocus_A: list[float]
    source_size_A: list[float]
    thermal_displacement_A: list[WeightedFloat]
    defect_presets: list[DefectPreset]
    metadata: dict[str, Any]


def _float_list(value: Any, *, default: list[float]) -> list[float]:
    if value is None:
        return list(default)
    if isinstance(value, (int, float)):
        return [float(value)]
    if not isinstance(value, list):
        raise ValueError("Expected a number or a list of numbers.")
    values = [float(v) for v in value]
    return values if values else list(default)


def _load_weighted_floats(value: Any, *, default: list[float]) -> list[WeightedFloat]:
    if value is None:
        return [WeightedFloat(float(v), 1.0) for v in default]
    if isinstance(value, (int, float)):
        return [WeightedFloat(float(value), 1.0)]
    if not isinstance(value, list):
        raise ValueError("Expected a number or a list of weighted number entries.")
    out: list[WeightedFloat] = []
    for item in value:
        if isinstance(item, (int, float)):
            out.append(WeightedFloat(float(item), 1.0))
            continue
        if isinstance(item, dict):
            out.append(WeightedFloat(float(item["value"]), float(item.get("weight", 1.0))))
            continue
        raise ValueError("Weighted number entries must be numbers or objects with value and weight.")
    if not out:
        out = [WeightedFloat(float(v), 1.0) for v in default]
    for item in out:
        if item.weight < 0.0:
            raise ValueError("Weights must be non-negative.")
    if sum(item.weight for item in out) <= 0.0:
        raise ValueError("At least one weighted number entry must have positive weight.")
    return out


def _load_image_size(data: dict[str, Any]) -> ImageSize:
    raw = data.get("image_size", {})
    if isinstance(raw, int):
        width = height = raw
    elif isinstance(raw, dict):
        width = int(raw["width"])
        height = int(raw["height"])
    else:
        raise ValueError("image_size must be an integer or an object with width and height.")
    if width <= 0 or height <= 0:
        raise ValueError("image_size width and height must be positive.")
    return ImageSize(width=width, height=height)


def _load_defect_presets(data: dict[str, Any]) -> list[DefectPreset]:
    raw = data.get("defect_presets")
    if raw is None:
        legacy = data.get("defects", [])
        if not legacy:
            return [DefectPreset(name="clean", rates={}, weight=1.0)]
        if not isinstance(legacy, list):
            raise ValueError("defects must be a list.")
        rates = {}
        for item in legacy:
            if not isinstance(item, dict):
                raise ValueError("Each defect entry must be an object.")
            symbol = str(item.get("symbol", "")).strip()
            if not symbol:
                raise ValueError("Each defect entry must define symbol.")
            values = _float_list(item.get("rates", item.get("rate")), default=[0.0])
            rates[symbol] = float(values[0])
        return [DefectPreset(name="legacy", rates=rates, weight=1.0)]
    if not isinstance(raw, list):
        raise ValueError("defect_presets must be a list.")
    out: list[DefectPreset] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each defect preset must be an object.")
        rates_raw = item.get("rates", {})
        if not isinstance(rates_raw, dict):
            raise ValueError("Each defect preset must define a rates object.")
        rates = {str(k).strip(): float(v) for k, v in rates_raw.items() if str(k).strip()}
        weight = float(item.get("weight", 1.0))
        if weight < 0.0:
            raise ValueError("Defect preset weights must be non-negative.")
        name = str(item.get("name", "")).strip() or ",".join(f"{k}{v:g}" for k, v in sorted(rates.items())) or "clean"
        out.append(DefectPreset(name=name, rates=rates, weight=weight))
    if not out:
        out = [DefectPreset(name="clean", rates={}, weight=1.0)]
    if sum(item.weight for item in out) <= 0.0:
        raise ValueError("At least one defect preset must have positive weight.")
    return out


def load_source_config(path: Path) -> SourceConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    base_dir = path.resolve().parent
    simulator = data.get("simulator", {})
    if not isinstance(simulator, dict):
        raise ValueError("simulator must be an object.")

    incostem_exclude_symbols = data.get("incostem_exclude_symbols", [])
    if not isinstance(incostem_exclude_symbols, list):
        raise ValueError("incostem_exclude_symbols must be a list.")

    return SourceConfig(
        material=str(data.get("material", "ReS2")),
        num=int(data.get("num", 1)),
        seed=None if data.get("seed") is None else int(data["seed"]),
        structure_path=_as_path(base_dir, data["structure_path"]),
        incostem_path=_as_path(base_dir, data["incostem_path"]),
        incostem_exclude_symbols=[str(s).strip() for s in incostem_exclude_symbols if str(s).strip()],
        output_dir=_as_path(ROOT_DIR, data["output_dir"]),
        mask_path=_as_path(ROOT_DIR, data["mask_path"]),
        image_size=_load_image_size(data),
        defocus_A=_float_list(simulator.get("defocus_A"), default=[35.0]),
        source_size_A=_float_list(simulator.get("source_size_A"), default=[0.6]),
        thermal_displacement_A=_load_weighted_floats(simulator.get("thermal_displacement_A"), default=[0.0]),
        defect_presets=_load_defect_presets(data),
        metadata=data.get("metadata", {}) if isinstance(data.get("metadata", {}), dict) else {},
    )


def _write_param(
    *,
    xyz_path: Path,
    tif_path: Path,
    image_size: ImageSize,
    defocus_A: float,
    source_size_A: float,
) -> str:
    return (
        f"{xyz_path}\n"
        "1 1 1\n"
        f"{tif_path}\n"
        f"{image_size.width} {image_size.height}\n"
        f"300 0 0 {defocus_A} 21.3\n"
        "39 200\n"
        "C12a 0 C12b 0 C21a 0 C21b 0 C23a 0 C23b 0 END\n"
        f"{source_size_A}\n"
        "0 \n"
        "n\n"
        "-1\n"
    )


def _choose(values: list[float]) -> float:
    return float(random.choice(values))


def _choose_weighted(items):
    total = float(sum(float(item.weight) for item in items))
    if total <= 0.0:
        raise ValueError("Weighted choices require a positive total weight.")
    mark = random.random() * total
    acc = 0.0
    for item in items:
        acc += float(item.weight)
        if mark <= acc:
            return item
    return items[-1]


def _apply_thermal_displacement(atoms, sigma_A: float) -> None:
    if sigma_A <= 0.0:
        return
    atoms.positions[:, :3] += np.random.normal(0.0, float(sigma_A), size=atoms.positions[:, :3].shape)


def _apply_defects(atoms, selected_rates: dict[str, float]) -> None:
    remove: list[int] = []
    for idx, atom in enumerate(atoms):
        rate = float(selected_rates.get(atom.symbol, 0.0))
        if rate > 0.0 and random.random() < rate:
            remove.append(idx)
    for idx in sorted(remove, reverse=True):
        atoms.pop(idx)


def _copy_mask(cfg: SourceConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    dst = cfg.output_dir / "mask.png"
    if not cfg.mask_path.exists():
        raise FileNotFoundError(f"Mask file does not exist: {cfg.mask_path}")
    if dst.exists() and cfg.mask_path.resolve() == dst.resolve():
        return
    shutil.copy2(cfg.mask_path, dst)


def _write_snapshot(cfg: SourceConfig, source_config_path: Path) -> None:
    snapshot = {
        "source_config": str(source_config_path),
        "material": cfg.material,
        "num": cfg.num,
        "image_size": {"width": cfg.image_size.width, "height": cfg.image_size.height},
        "mask_path": "mask.png",
        "metadata": cfg.metadata,
    }
    (cfg.output_dir / "source_config_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_source_batch(cfg: SourceConfig, *, source_config_path: Path) -> None:
    if cfg.num <= 0:
        raise ValueError("num must be positive.")
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
    if not cfg.structure_path.exists():
        raise FileNotFoundError(f"Structure file does not exist: {cfg.structure_path}")
    if not cfg.incostem_path.exists():
        raise FileNotFoundError(
            f"incostem executable does not exist: {cfg.incostem_path}. "
            "Download or compile it from the official computem/temsim project, or set incostem_path in the JSON config."
        )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    _copy_mask(cfg)
    _write_snapshot(cfg, source_config_path)

    manifest_path = cfg.output_dir / "manifest.jsonl"
    base_atoms = read(str(cfg.structure_path))
    exclude_symbols = {s.strip() for s in cfg.incostem_exclude_symbols if s.strip()}

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for idx in range(cfg.num):
            atoms = base_atoms.copy()
            thermal_choice = _choose_weighted(cfg.thermal_displacement_A)
            defect_choice = _choose_weighted(cfg.defect_presets)
            thermal_A = float(thermal_choice.value)
            defocus_A = _choose(cfg.defocus_A)
            source_size_A = _choose(cfg.source_size_A)
            defect_rates = dict(defect_choice.rates)

            _apply_thermal_displacement(atoms, thermal_A)
            _apply_defects(atoms, defect_rates)

            stem = f"{cfg.material}_source_{idx:06d}"
            out_png = cfg.output_dir / f"{stem}.png"
            if out_png.exists():
                continue

            with tempfile.TemporaryDirectory(prefix="stackdiff_source_incostem_") as td:
                td_path = Path(td)
                xyz_path = td_path / f"{stem}.xyz"
                tif_path = td_path / f"{stem}.tif"
                xyz_path.write_text(
                    _incostem_xyz_lines(atoms, exclude_symbols=exclude_symbols, no_show_s=False),
                    encoding="utf-8",
                )
                param = _write_param(
                    xyz_path=xyz_path,
                    tif_path=tif_path,
                    image_size=cfg.image_size,
                    defocus_A=defocus_A,
                    source_size_A=source_size_A,
                )
                _run_incostem(cfg.incostem_path, param)
                if not tif_path.exists():
                    raise RuntimeError(f"incostem did not create expected output: {tif_path}")
                with Image.open(tif_path) as im:
                    im.convert("L").save(out_png, format="PNG")

            rec = {
                "file": out_png.name,
                "material": cfg.material,
                "image_size": {"width": cfg.image_size.width, "height": cfg.image_size.height},
                "thermal_displacement_A": thermal_A,
                "thermal_displacement_weight": float(thermal_choice.weight),
                "defocus_A": defocus_A,
                "source_size_A": source_size_A,
                "defect_preset": defect_choice.name,
                "defect_preset_weight": float(defect_choice.weight),
                "defect_rates": defect_rates,
            }
            manifest.write(json.dumps(rec, ensure_ascii=False) + "\n")
            manifest.flush()
