"""
Purpose: Backend for the optional multi-material synthetic STEM generator. It builds bilayer structures, calls the external incostem executable, applies shared StackDiff augmentation, and writes 128 x 128 PNG images plus optional labels.
Related files: src/tools/synthetic_materials/generate.py, src/tools/synthetic_materials/configs/*.json, src/tools/synthetic_materials/structures/*.xyz, src/tools/synthetic_materials/masks/*.png, and src/core/augmentations/stem.py.
CLI usage: Use python src/tools/synthetic_materials/generate.py --config src/tools/synthetic_materials/configs/ReS2.json.
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
from ase.data import atomic_numbers
from ase.io import read
from PIL import Image





DEFAULT_IMAGE_SIZE = 1024
TARGET_SIZE = 128

ROOT_DIR = Path(__file__).resolve().parents[3]
SYS_SRC = ROOT_DIR / "src"
if str(SYS_SRC) not in sys.path:
    sys.path.append(str(SYS_SRC))

_DEFAULT_BASIS_SYMBOL_BY_MATERIAL: dict[str, str] = {
    "res2": "Re",
    "mos2": "Mo",
    "mote2": "Mo",
    "tas2": "Ta",
}

_DEFAULT_PERIOD_T1T2_A: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "res2": ((6.42, 0.0), (3.15, -5.71)),
    "mos2": ((3.18, 0.0), (1.59, -2.75)),
    "mote2": ((3.49, 0.0), (0.0, 6.37)),
    "tas2": ((3.33, 0.0), (1.66, -2.88)),
}

from core.augmentations.stem import (  # noqa: E402
    AugmentConfig,
    add_carbon_background,
    add_gaussian_noise,
    add_poisson_noise_from_original,
    add_scan_noise_from_original,
    adjust_display_post_noise,
    apply_edge_mask,
    crop_and_resize,
    distance_transform,
    elastic_warp_cv,
    perspective_warp_cv,
    rotate_cv,
    choose_crop,
    to_float01,
)


@dataclass(frozen=True)
class RangeSpec:
    start: float
    stop: float
    step: float
    num: Optional[int] = None

    def values(self) -> list[float]:
        if self.step < 0:
            raise ValueError('Invalid StackDiff configuration or runtime parameter.')
        if self.step == 0:
            count = self.num if self.num is not None else 2
            if count <= 0:
                raise ValueError('Invalid StackDiff configuration or runtime parameter.')
            if count == 1:
                return [self.start]
            if self.start == self.stop:
                return [self.start]
            delta = (self.stop - self.start) / (count - 1)
            return [self.start + i * delta for i in range(count)]

        n = int(math.floor((self.stop - self.start) / self.step + 1e-9))
        vals = [self.start + i * self.step for i in range(n + 1)]

        return [v for v in vals if v <= self.stop + self.step / 2]


@dataclass(frozen=True)
class BatchConfig:
    material: str
    num: Optional[int]
    pipeline_mode: str
    shift_basis: str  # "cartesian" | "t1t2"
    period_mode: str  # "fixed" | "infer"
    period_t1_A: Optional[tuple[float, float]]
    period_t2_A: Optional[tuple[float, float]]
    lattice_basis_symbol: Optional[str]
    incostem_exclude_symbols: list[str]
    structure_path: Path
    incostem_path: Path
    output_dir: Path
    mask_path: Path
    augment: dict
    x_range: RangeSpec
    y_range: RangeSpec
    rotation_range: Optional[RangeSpec]
    defect_rates_re: list[float]
    dopant_rates_re: list[float]
    dopant_target_symbol: str
    seed: Optional[int]


def _wrap_delta_rect(dx: np.ndarray, ax: float) -> np.ndarray:
    """Internal helper."""
    return dx - np.round(dx / max(ax, 1e-12)) * ax


def _infer_period_vecs_from_atoms(atoms, *, basis_symbol: str) -> tuple[np.ndarray, np.ndarray]:
    """Internal helper."""
    cell = atoms.get_cell().array
    A2 = np.asarray(cell[:2, :2], dtype=np.float64)
    det = float(np.linalg.det(A2))
    if abs(det) < 1e-12:
        raise RuntimeError('Invalid StackDiff configuration or runtime parameter.')

    pts = np.asarray([a.position[:2] for a in atoms if a.symbol == basis_symbol], dtype=np.float64)
    if pts.shape[0] < 10:
        raise RuntimeError('Invalid StackDiff configuration or runtime parameter.')


    A2_inv_T = np.linalg.inv(A2).T
    frac = pts @ A2_inv_T
    frac = frac - np.floor(frac)  # [0,1)


    if basis_symbol == "Re":
        r_min, r_max = 4.0, 8.5
    else:
        r_min, r_max = 1.5, 9.5

    rng = np.random.RandomState(0)
    pick = rng.choice(len(frac), size=min(len(frac), 260), replace=False)
    anchors = frac[pick]

    disps: list[np.ndarray] = []
    for p in anchors:
        d = frac - p
        d -= np.round(d)
        cart = d @ A2.T
        r = np.hypot(cart[:, 0], cart[:, 1])
        m = (r > r_min) & (r < r_max)
        if np.any(m):
            disps.append(cart[m])
    if not disps:
        raise RuntimeError('Invalid StackDiff configuration or runtime parameter.')
    disps_arr = np.vstack(disps)


    step = 0.01  # Å
    binned = np.round(disps_arr / step) * step

    def _canon(v: np.ndarray) -> tuple[float, float]:
        x, y = float(v[0]), float(v[1])
        if abs(x) > 1e-9:
            if x < 0:
                x, y = -x, -y
        else:
            if y < 0:
                x, y = -x, -y
        return (x, y)

    counter: Counter[tuple[float, float]] = Counter()
    for v in binned:
        counter[_canon(v)] += 1

    items = sorted(counter.items(), key=lambda kv: (-kv[1], float(np.hypot(kv[0][0], kv[0][1]))))
    if len(items) < 2:
        raise RuntimeError('Invalid StackDiff configuration or runtime parameter.')

    v1 = np.array(items[0][0], dtype=np.float64)

    def _pick_v2(min_count_ratio: float) -> Optional[np.ndarray]:
        ref_cnt = float(items[0][1])
        best: Optional[np.ndarray] = None
        best_len = float("inf")
        best_cnt = -1
        for (x, y), cnt in items[1:200]:
            if ref_cnt > 0 and float(cnt) < ref_cnt * min_count_ratio:
                continue
            cand = np.array([x, y], dtype=np.float64)
            cross = abs(float(v1[0] * cand[1] - v1[1] * cand[0]))
            if cross < 0.5:
                continue
            clen = float(np.hypot(cand[0], cand[1]))
            if clen < best_len - 1e-9 or (abs(clen - best_len) <= 1e-9 and cnt > best_cnt):
                best = cand
                best_len = clen
                best_cnt = int(cnt)
        return best

    v2 = _pick_v2(0.20)
    if v2 is None:
        v2 = _pick_v2(0.05)
    if v2 is None:
        raise RuntimeError('Invalid StackDiff configuration or runtime parameter.')


    if abs(float(v2[1])) < abs(float(v1[1])):
        v1, v2 = v2, v1
    return v1, v2


def _as_path(base_dir: Path, value: str) -> Path:
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def load_config(path: Path) -> BatchConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    base_dir = path.resolve().parent
    rotation_range = None
    if "rotation_range" in data and data["rotation_range"] is not None:
        rr = data["rotation_range"]
        rotation_range = RangeSpec(
            float(rr["start"]),
            float(rr["stop"]),
            float(rr.get("step", 0.0)),
            None if rr.get("num") is None else int(rr["num"]),
        )

    pipeline_mode = str(data.get("pipeline_mode", "final")).strip().lower()
    if pipeline_mode in {"final128", "final_128"}:
        pipeline_mode = "final"
    if pipeline_mode in {"debug1024", "debug_1024"}:
        pipeline_mode = "debug"
    if pipeline_mode not in {"final", "debug"}:
        raise ValueError('Invalid StackDiff configuration or runtime parameter.')

    augment = data.get("augment")
    if not isinstance(augment, dict):
        raise ValueError('Invalid StackDiff configuration or runtime parameter.')

    shift_basis = str(data.get("shift_basis", "cartesian")).strip().lower()
    if shift_basis in {"xy", "cart", "cartesian"}:
        shift_basis = "cartesian"
    if shift_basis in {"t1t2", "lattice", "frac", "fractional"}:
        shift_basis = "t1t2"
    if shift_basis not in {"cartesian", "t1t2"}:
        raise ValueError('Invalid StackDiff configuration or runtime parameter.')

    period = data.get("period", {}) if isinstance(data.get("period", {}), dict) else {}
    period_mode = str(period.get("mode", data.get("period_mode", "fixed"))).strip().lower()
    if period_mode in {"auto", "infer", "inferred"}:
        period_mode = "infer"
    if period_mode in {"fixed", "const", "constant", "preset"}:
        period_mode = "fixed"
    if period_mode not in {"fixed", "infer"}:
        raise ValueError('Invalid StackDiff configuration or runtime parameter.')

    def _parse_vec2(val) -> Optional[tuple[float, float]]:
        if val is None:
            return None
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return (float(val[0]), float(val[1]))
        raise ValueError('Invalid StackDiff configuration or runtime parameter.')

    period_t1_A = _parse_vec2(period.get("t1_A", period.get("t1")))
    period_t2_A = _parse_vec2(period.get("t2_A", period.get("t2")))

    lattice_basis_symbol = data.get("lattice_basis_symbol")
    if lattice_basis_symbol is not None:
        lattice_basis_symbol = str(lattice_basis_symbol).strip()
        if not lattice_basis_symbol:
            lattice_basis_symbol = None

    incostem_exclude_symbols = data.get("incostem_exclude_symbols")
    if incostem_exclude_symbols is None:
        incostem_exclude_symbols_list: list[str] = ["S"]
    else:
        if not isinstance(incostem_exclude_symbols, list):
            raise ValueError('Invalid StackDiff configuration or runtime parameter.')
        incostem_exclude_symbols_list = [str(s).strip() for s in incostem_exclude_symbols if str(s).strip()]

    defect_rates = data.get("defect_rates_re")
    if defect_rates is None:
        defect_rates = data.get("defect_rates_basis", data.get("defect_rates", data.get("defect_rate", [0.0])))
    if not isinstance(defect_rates, list):
        raise ValueError('Invalid StackDiff configuration or runtime parameter.')

    dopant_rates = data.get("dopant_rates_re", [0.0])
    if not isinstance(dopant_rates, list):
        raise ValueError('Invalid StackDiff configuration or runtime parameter.')
    dopant_target = str(data.get("dopant_target_symbol", "Mo")).strip()

    return BatchConfig(
        material=str(data.get("material", "ReS2")),
        num=None if data.get("num") is None else int(data["num"]),
        pipeline_mode=pipeline_mode,
        shift_basis=shift_basis,
        period_mode=period_mode,
        period_t1_A=period_t1_A,
        period_t2_A=period_t2_A,
        lattice_basis_symbol=lattice_basis_symbol,
        incostem_exclude_symbols=incostem_exclude_symbols_list,
        structure_path=_as_path(base_dir, data["structure_path"]),
        incostem_path=_as_path(base_dir, data["incostem_path"]),

        output_dir=_as_path(ROOT_DIR, data["output_dir"]),
        mask_path=_as_path(ROOT_DIR, str(data.get("mask_path", "src/tools/synthetic_materials/masks/ReS2.png"))),
        augment=augment,
        x_range=RangeSpec(
            float(data["x_range"]["start"]),
            float(data["x_range"]["stop"]),
            float(data["x_range"].get("step", 0.0)),
            None if data["x_range"].get("num") is None else int(data["x_range"]["num"]),
        ),
        y_range=RangeSpec(
            float(data["y_range"]["start"]),
            float(data["y_range"]["stop"]),
            float(data["y_range"].get("step", 0.0)),
            None if data["y_range"].get("num") is None else int(data["y_range"]["num"]),
        ),
        rotation_range=rotation_range,
        defect_rates_re=[float(x) for x in defect_rates],
        dopant_rates_re=[float(x) for x in dopant_rates],
        dopant_target_symbol=dopant_target,
        seed=None if data.get("seed") is None else int(data["seed"]),
    )


def _incostem_xyz_lines(
    atoms,
    *,
    exclude_symbols: Optional[set[str]] = None,
    no_show_s: bool = True,
) -> str:

    exclude_symbols = set() if exclude_symbols is None else set(exclude_symbols)
    if no_show_s:
        exclude_symbols.add("S")
    symbols = atoms.get_chemical_symbols()
    counts: dict[str, int] = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    header = "".join([f"{k}{v}" for k, v in counts.items()]) + f"\t{len(atoms)}\n"

    lengths = atoms.get_cell().lengths()
    header += f"{float(lengths[0])}\t{float(lengths[1])}\t{float(lengths[2])}\n"


    lines = []
    for atom in atoms:
        if atom.symbol in exclude_symbols:
            continue
        try:
            znum = int(atomic_numbers[atom.symbol])
        except Exception as e:
            raise ValueError('Invalid StackDiff configuration or runtime parameter.') from e
        x, y, z = atom.position
        lines.append(f"{znum}\t{x}\t{y}\t{z}\t1\t0\t")
    return header + "\n".join(lines) + "\n-1\n"


def _rotate_vec_px(vec_px: np.ndarray, angle_deg: float) -> np.ndarray:
    """Internal helper."""
    theta = math.radians(float(angle_deg))
    c = float(math.cos(theta))
    s = float(math.sin(theta))
    dx, dy = float(vec_px[0]), float(vec_px[1])
    return np.array([c * dx + s * dy, -s * dx + c * dy], dtype=np.float32)


def _augment_and_track_gt(
    img: Image.Image,
    *,
    dist_map0: np.ndarray,
    aug_cfg: AugmentConfig,
    cell_xy_A: tuple[float, float],
    shift_xy_A: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """Internal helper."""
    img01 = to_float01(img.convert("L"))
    H, W = img01.shape[:2]
    params = aug_cfg.sample_once()
    verbose = os.environ.get("MOIRE_GENERATE_VERBOSE", "").strip() not in {"", "0", "false", "False"}




    if bool(params.get("elastic.enabled", False)) or bool(params.get("perspective.enabled", False)):
        raise ValueError(
            'Invalid StackDiff configuration or runtime parameter.'
            'Invalid StackDiff configuration or runtime parameter.'
        )


    if params.get("elastic.enabled", True):
        alpha = float(params.get("elastic.alpha_ratio", 0.0)) * float(min(H, W))
        sigma = float(params.get("elastic.sigma_ratio", 0.0)) * float(min(H, W))
        if alpha > 0.0 and sigma > 0.0:
            img01 = elastic_warp_cv(img01, alpha=alpha, sigma=sigma)
    if params.get("perspective.enabled", True):
        max_ratio = float(params.get("perspective.max_ratio", 0.0) or 0.0)
        if max_ratio > 0.0:
            img01 = perspective_warp_cv(img01, max_ratio)


    if bool(params.get("carbon.enabled", False)) is True:
        img01 = add_carbon_background(
            img01,
            {
                "cover": params["carbon.cover"],
                "alpha": params["carbon.alpha"],
                "sigma_mask": params["carbon.sigma_mask"],
                "sigma_field": params["carbon.sigma_field"],
                "coarse_down": params["carbon.coarse_down"],
                "focus_enable": params["carbon.focus_enable"],
                "focus_center": params["carbon.focus_center"],
                "focus_radius_ratio": params["carbon.focus_radius_ratio"],
                "focus_gain": params["carbon.focus_gain"],
                "morph_kernel": params["carbon.morph_kernel"],
                "morph_iter": params["carbon.morph_iter"],
            },
        )


    Lx_A, Ly_A = float(cell_xy_A[0]), float(cell_xy_A[1])
    if Lx_A <= 0.0 or Ly_A <= 0.0:
        raise ValueError('Invalid StackDiff configuration or runtime parameter.')
    A_per_px_x = Lx_A / float(W)
    A_per_px_y = Ly_A / float(H)

    A_per_px = 0.5 * (A_per_px_x + A_per_px_y)
    shift_px = np.array(
        [float(shift_xy_A[0]) / A_per_px_x, float(shift_xy_A[1]) / A_per_px_y], dtype=np.float32
    )


    rotate_enabled = params.get("rotate.enabled", True)
    angle = float(params.get("rotate.angle", 0.0)) % 360.0
    if rotate_enabled and angle != 0.0:
        img_rot = rotate_cv(img01, angle)
        dist_rot = rotate_cv(dist_map0, angle, interpolation=cv2.INTER_NEAREST)
        shift_px = _rotate_vec_px(shift_px, angle)
    else:
        img_rot = img01
        dist_rot = dist_map0


    crop_enabled = params.get("crop.enabled", True)
    if not crop_enabled:
        raise ValueError('Invalid StackDiff configuration or runtime parameter.')
    side_param = params.get("crop.side")
    side = int(side_param) if side_param is not None else min(H, W)
    side = max(2, side - (side % 2))
    safe_radius_param = params.get("crop.safe_radius")
    safe_radius = int(safe_radius_param) if safe_radius_param is not None else 372
    cy, cx = choose_crop(dist_rot, side, safe_radius)
    patch = crop_and_resize(img_rot, cy, cx, side, out_size=int(aug_cfg.image_size))
    effective_side = int(side)

    scale_final = float(aug_cfg.image_size) / float(effective_side)
    shift_128 = (shift_px * float(scale_final)).astype(np.float32)
    sideA = float(effective_side) * float(A_per_px)


    do_h = False
    do_v = False
    if params.get("flip.enabled", True):
        h_prob = float(params.get("flip.h_prob", 0.0))
        v_prob = float(params.get("flip.v_prob", 0.0))
        do_h = bool(np.random.rand() < h_prob)
        do_v = bool(np.random.rand() < v_prob)
        if do_h:
            patch = cv2.flip(patch, 1)
            shift_128[0] = -shift_128[0]
        if do_v:
            patch = cv2.flip(patch, 0)
            shift_128[1] = -shift_128[1]


    if params.get("edge_mask.enabled", True):
        edge_prob = float(params.get("edge_mask.prob", 0.0))
        if edge_prob > 0.0:
            patch = apply_edge_mask(
                patch,
                prob=edge_prob,
                bg_min=float(params.get("edge_mask.bg_min", 0.0)),
                bg_max=float(params.get("edge_mask.bg_max", 0.05)),
            )


    k = 2.0 * float(effective_side) / float(aug_cfg.image_size)
    if params.get("noise.enabled", True):
        noise_cfg = (aug_cfg.cfg or {}).get("noise", {}) if isinstance(getattr(aug_cfg, "cfg", None), dict) else {}
        noise_mode = str(noise_cfg.get("mode", "full")).strip().lower()
        if noise_mode not in {"full", "gaussian_only"}:
            raise ValueError('Invalid StackDiff configuration or runtime parameter.')

        if noise_mode == "gaussian_only":
            sigma_val = noise_cfg.get("gaussian_sigma", params.get("noise.gaussian_sigma", 0.0))

            if isinstance(sigma_val, (list, tuple)) and len(sigma_val) == 2:
                raise ValueError(
                    'Invalid StackDiff configuration or runtime parameter.'
                    'Invalid StackDiff configuration or runtime parameter.'
                )
            patch = add_gaussian_noise(patch, sigma=float(sigma_val or 0.0), preserve_scale=True)
        else:
            patch = add_scan_noise_from_original(
                patch,
                pixel_size_A_orig=float(params["noise.pixel_size_A_orig"]),
                k=k,
                width_orig_px=int(params["noise.width_orig_px"]),
                dwell_time_scan_s_orig=float(params["noise.dwell_time_s"]),
                sigma_jitter_A=float(params["noise.sigma_jitter_A"]),
                line_freq_hz=float(params["noise.line_freq_hz"]),
                phase_x=float(params["noise.phase_x"]),
                phase_y=float(params["noise.phase_y"]),
            )
            patch = add_poisson_noise_from_original(
                patch,
                beam_current_A=float(params["noise.beam_current_A"]),
                dwell_time_s=float(params["noise.dwell_time_s"]),
                k=k,
            )
            patch = add_gaussian_noise(patch, sigma=float(params.get("noise.gaussian_sigma", 0.0)), preserve_scale=True)


    if params.get("display.enabled", True):
        patch, _ = adjust_display_post_noise(
            patch,
            gain=params.get("display.gain", None),
            bias=params.get("display.bias", None),
            gamma=params.get("display.gamma", None),
        )

    patch = np.clip(patch, 0.0, 1.0)
    patch_n11 = (patch.astype(np.float32) * 2.0 - 1.0)[None, ...]  # 1xHxW
    meta = {
        "H": int(H),
        "W": int(W),
        "rotate_enabled": bool(rotate_enabled),
        "angle_deg": float(angle),
        "crop_side": int(effective_side),
        "crop_cy": int(cy),
        "crop_cx": int(cx),
        "flip_h": bool(do_h),
        "flip_v": bool(do_v),
        "scale_final": float(scale_final),
        "A_per_px_x": float(A_per_px_x),
        "A_per_px_y": float(A_per_px_y),
    }
    if verbose:
        print(
            "[AUG] "
            f"angle={angle:.3f}deg flip_h={int(do_h)} flip_v={int(do_v)} "
            f"crop_side={effective_side} shift_A=({shift_xy_A[0]:.4f},{shift_xy_A[1]:.4f}) "
            f"shift128=({float(shift_128[0]):.3f},{float(shift_128[1]):.3f}) sideA={sideA:.3f}"
        )
    return patch_n11, shift_128.astype(np.float32), float(sideA), meta


def _run_incostem(incostem_path: Path, param_lines: str) -> str:
    p = subprocess.Popen(
        [str(incostem_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert p.stdin is not None
    p.stdin.write(param_lines)
    p.stdin.flush()
    out = p.communicate()[0]
    if p.returncode != 0:
        raise RuntimeError('Invalid StackDiff configuration or runtime parameter.')
    return out


def _write_param(
    *,
    xyz_path: Path,
    tif_path: Path,
    image_size: int,
    defocus: float = 35.0,
    source_size: float = 0.6,
) -> str:

    return (
        f"{xyz_path}\n"
        "1 1 1\n"
        f"{tif_path}\n"
        f"{image_size} {image_size}\n"
        f"300 0 0 {defocus} 21.3\n"
        "39 200\n"
        "C12a 0 C12b 0 C21a 0 C21b 0 C23a 0 C23b 0 END\n"
        f"{source_size}\n"
        "0 \n"
        "n\n"
        "-1\n"
    )


def _iter_values(cfg: BatchConfig) -> Iterable[tuple[float, float, float, float, float]]:
    xs = cfg.x_range.values()
    ys = cfg.y_range.values()
    rs = [0.0]
    if cfg.rotation_range is not None:
        rs = cfg.rotation_range.values()

    for defect_rate in cfg.defect_rates_re:
        for dopant_rate in cfg.dopant_rates_re:
            for r in rs:
                for x in xs:
                    for y in ys:
                        yield defect_rate, dopant_rate, r, x, y


def run_batch(cfg: BatchConfig) -> None:
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    if len(cfg.defect_rates_re) != 1:
        raise ValueError('Invalid StackDiff configuration or runtime parameter.')
    if len(cfg.dopant_rates_re) != 1:
        raise ValueError('Invalid StackDiff configuration or runtime parameter.')

    structure_path = cfg.structure_path
    incostem_path = cfg.incostem_path
    output_dir = cfg.output_dir
    mask_path = cfg.mask_path

    if not structure_path.exists():
        raise FileNotFoundError('Invalid StackDiff configuration or runtime parameter.')
    if not incostem_path.exists():
        raise FileNotFoundError(
            'Invalid StackDiff configuration or runtime parameter.'
            'Invalid StackDiff configuration or runtime parameter.'
            'Invalid StackDiff configuration or runtime parameter.'
            'Invalid StackDiff configuration or runtime parameter.'
        )
    if not mask_path.exists():
        raise FileNotFoundError('Invalid StackDiff configuration or runtime parameter.')

    output_dir.mkdir(parents=True, exist_ok=True)

    debug_dir: Optional[Path] = None
    if cfg.pipeline_mode == "debug":
        debug_dir = output_dir.parent / f"{output_dir.name}_debug1024"
        debug_dir.mkdir(parents=True, exist_ok=True)

    mask_img = Image.open(mask_path).convert("L")
    mask01 = to_float01(mask_img)
    dist_map0 = distance_transform(mask01)
    aug_cfg = AugmentConfig(image_size=TARGET_SIZE, cfg=cfg.augment)

    base_atoms = read(str(structure_path))
    center = base_atoms.get_cell().sum(axis=0) / 2.0
    cell_lengths = base_atoms.get_cell().lengths()
    cell_xy_A = (float(cell_lengths[0]), float(cell_lengths[1]))

    material_key = cfg.material.strip().lower()
    basis_symbol = cfg.lattice_basis_symbol or _DEFAULT_BASIS_SYMBOL_BY_MATERIAL.get(material_key)
    if basis_symbol is None:

        counts = Counter(base_atoms.get_chemical_symbols())
        basis_symbol = min(counts.items(), key=lambda kv: kv[1])[0]

    exclude_symbols = {s.strip() for s in cfg.incostem_exclude_symbols if s.strip()}


    if cfg.period_mode == "infer":
        T1_A_base, T2_A_base = _infer_period_vecs_from_atoms(base_atoms, basis_symbol=basis_symbol)
    else:
        if cfg.period_t1_A is not None and cfg.period_t2_A is not None:
            T1_A_base = np.array(cfg.period_t1_A, dtype=np.float64)
            T2_A_base = np.array(cfg.period_t2_A, dtype=np.float64)
        else:
            defaults = _DEFAULT_PERIOD_T1T2_A.get(material_key)
            if defaults is None:
                raise ValueError(
                    'Invalid StackDiff configuration or runtime parameter.'
                    'Invalid StackDiff configuration or runtime parameter.'
                )
            (t1x, t1y), (t2x, t2y) = defaults
            T1_A_base = np.array([t1x, t1y], dtype=np.float64)
            T2_A_base = np.array([t2x, t2y], dtype=np.float64)

    def _fmt(v: float) -> str:

        if abs(v) < 5e-4:
            v = 0.0
        s = f"{v:.3f}".rstrip("0").rstrip(".")
        return s if s else "0"

    def _fmt2(v: float) -> str:

        if abs(v) < 5e-4:
            v = 0.0
        return f"{float(v):.2f}"

    def _sample_uniform(spec: RangeSpec) -> float:
        if spec.start == spec.stop:
            return spec.start
        lo = min(spec.start, spec.stop)
        hi = max(spec.start, spec.stop)
        return random.uniform(lo, hi)

    def _resolve_shift_xy_A(v0: float, v1: float) -> tuple[float, float]:
        """Internal helper."""
        if cfg.shift_basis == "cartesian":
            return float(v0), float(v1)

        a = float(v0)
        b = float(v1)
        s = a * T1_A_base + b * T2_A_base
        return float(s[0]), float(s[1])

    def _choose_defect_rate() -> float:
        return cfg.defect_rates_re[0]

    def _choose_dopant_rate() -> float:
        return cfg.dopant_rates_re[0]

    def _generate_one(defect_rate: float, dopant_rate: float, r: float, x: float, y: float) -> Optional[Path]:

        r = float(r) if cfg.rotation_range is not None else 0.0


        sx_A, sy_A = _resolve_shift_xy_A(float(x), float(y))
        layer1 = base_atoms.copy()
        layer2 = base_atoms.copy()
        layer2.positions[:, 2] += 7.0
        layer2.positions[:, 0] += sx_A
        layer2.positions[:, 1] += sy_A
        layer2.positions[:, 2] += random.uniform(-0.5, 0.5)




        if cfg.rotation_range is not None and abs(float(r)) > 1e-8:
            layer2.rotate(r, "z", center)

            cell = layer2.get_cell()
            x_max, y_max = cell[0][0], cell[1][1]
            remove = []
            for i, atom in enumerate(layer2):
                ax, ay = atom.position[0], atom.position[1]
                if ax < 0 or ax > x_max or ay < 0 or ay > y_max:
                    remove.append(i)
            for idx in sorted(remove, reverse=True):
                layer2.pop(idx)


        def _introduce(layer):
            remove = []
            for i, atom in enumerate(layer):
                if atom.symbol == basis_symbol and random.random() < defect_rate:
                    remove.append(i)
            for idx in sorted(remove, reverse=True):
                layer.pop(idx)

        _introduce(layer1)
        _introduce(layer2)


        if dopant_rate > 0.0:
            target_sym = cfg.dopant_target_symbol
            merged = layer1 + layer2
            for atom in merged:
                if atom.symbol == basis_symbol and random.random() < dopant_rate:
                    atom.symbol = target_sym
            layer1 = merged[: len(layer1)]
            layer2 = merged[len(layer1):]

        atoms = layer1 + layer2


        with tempfile.TemporaryDirectory(prefix="moire_incostem_") as td:
            td_path = Path(td)
            xyz_path = td_path / f"{cfg.material}.xyz"
            tif_path = td_path / "out.tif"


            xyz_path.write_text(
                _incostem_xyz_lines(atoms, exclude_symbols=exclude_symbols, no_show_s=False),
                encoding="utf-8",
            )
            param = _write_param(xyz_path=xyz_path, tif_path=tif_path, image_size=DEFAULT_IMAGE_SIZE)
            _run_incostem(incostem_path, param)

            if not tif_path.exists():
                raise RuntimeError('Invalid StackDiff configuration or runtime parameter.')

            with Image.open(tif_path) as im:
                im = im.convert("L")

                aug, shift_128, sideA, meta = _augment_and_track_gt(
                    im,
                    dist_map0=dist_map0,
                    aug_cfg=aug_cfg,
                    cell_xy_A=cell_xy_A,
                    shift_xy_A=(float(sx_A), float(sy_A)),
                )
                img01 = (np.clip(aug[0], -1.0, 1.0) + 1.0) * 0.5
                img_u8 = (np.clip(img01, 0.0, 1.0) * 255.0).astype(np.uint8)
                gtx = float(shift_128[0])
                gty = float(shift_128[1])


                v1A = T1_A_base.copy()
                v2A = T2_A_base.copy()
                angle_deg = float(meta.get("angle_deg", 0.0) or 0.0) % 360.0
                if bool(meta.get("rotate_enabled", True)) and angle_deg != 0.0:
                    v1A = _rotate_vec_px(v1A.astype(np.float32), angle_deg).astype(np.float64)
                    v2A = _rotate_vec_px(v2A.astype(np.float32), angle_deg).astype(np.float64)
                if bool(meta.get("flip_h", False)):
                    v1A[0] = -v1A[0]
                    v2A[0] = -v2A[0]
                if bool(meta.get("flip_v", False)):
                    v1A[1] = -v1A[1]
                    v2A[1] = -v2A[1]




                if cfg.rotation_range is not None and abs(float(r)) > 1e-8:
                    out_png = output_dir / f"{cfg.material}_r{_fmt2(r)}.png"
                else:
                    out_png = output_dir / (
                        f"{cfg.material}_r{_fmt(r)}_gtx{_fmt(gtx)}_gty{_fmt(gty)}_sideA{_fmt(sideA)}"
                        f"_v1x{_fmt(float(v1A[0]))}_v1y{_fmt(float(v1A[1]))}"
                        f"_v2x{_fmt(float(v2A[0]))}_v2y{_fmt(float(v2A[1]))}.png"
                    )

                if out_png.exists():
                    return None

                if debug_dir is not None:
                    debug_path = debug_dir / out_png.name
                    im.save(debug_path, format="PNG")

                save_labels = os.environ.get("MOIRE_SAVE_LABELS", "").strip() not in {"", "0", "false", "False"}
                if save_labels:
                    label_dir = debug_dir if debug_dir is not None else (output_dir.parent / f"{output_dir.name}_labels")
                    label_dir.mkdir(parents=True, exist_ok=True)

                    def _save_gt_align_vis(
                        out_path: Path,
                        *,
                        layer1_xy128: np.ndarray,
                        layer2_xy128: np.ndarray,
                        gt_shift128: tuple[float, float],
                        size: int,
                    ) -> None:
                        """Internal helper."""


                        img = np.zeros((size, size, 3), dtype=np.uint16)

                        def _plot_points(pts: np.ndarray, color: tuple[int, int, int]) -> None:
                            if pts.size == 0:
                                return
                            for x, y in pts:
                                ix = int(round(float(x)))
                                iy = int(round(float(y)))
                                if ix < 0 or ix >= size or iy < 0 or iy >= size:
                                    continue

                                img[iy, ix] += np.array(color, dtype=np.uint16)
                                if ix - 1 >= 0:
                                    img[iy, ix - 1] += np.array(color, dtype=np.uint16)
                                if ix + 1 < size:
                                    img[iy, ix + 1] += np.array(color, dtype=np.uint16)
                                if iy - 1 >= 0:
                                    img[iy - 1, ix] += np.array(color, dtype=np.uint16)
                                if iy + 1 < size:
                                    img[iy + 1, ix] += np.array(color, dtype=np.uint16)

                        dx, dy = float(gt_shift128[0]), float(gt_shift128[1])
                        layer2_aligned = layer2_xy128.copy()
                        if layer2_aligned.size:
                            layer2_aligned[:, 0] -= dx
                            layer2_aligned[:, 1] -= dy

                        _plot_points(layer1_xy128, (80, 160, 255))     # blue-ish
                        _plot_points(layer2_aligned, (255, 170, 60))  # orange-ish
                        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
                        Image.fromarray(img_u8, mode="RGB").save(out_path, format="PNG")

                    def _layer_basis_xy128(layer) -> np.ndarray:

                        xy_A = np.array(
                            [[float(a.position[0]), float(a.position[1])] for a in layer if a.symbol == basis_symbol],
                            dtype=np.float32,
                        )
                        if xy_A.size == 0:
                            return xy_A.reshape(0, 2)
                        pts = np.empty_like(xy_A, dtype=np.float32)
                        pts[:, 0] = xy_A[:, 0] / float(meta["A_per_px_x"])
                        pts[:, 1] = xy_A[:, 1] / float(meta["A_per_px_y"])

                        angle_deg = float(meta["angle_deg"])
                        if bool(meta["rotate_enabled"]) and angle_deg != 0.0:
                            M = cv2.getRotationMatrix2D((float(meta["W"]) / 2.0, float(meta["H"]) / 2.0), angle_deg, 1.0)
                            pts = (pts @ M[:, :2].T + M[:, 2]).astype(np.float32)

                        side = int(meta["crop_side"])
                        half = side // 2
                        x1 = int(meta["crop_cx"]) - half
                        y1 = int(meta["crop_cy"]) - half
                        pts[:, 0] = (pts[:, 0] - float(x1)) * float(meta["scale_final"])
                        pts[:, 1] = (pts[:, 1] - float(y1)) * float(meta["scale_final"])

                        if bool(meta["flip_h"]):
                            pts[:, 0] = float(aug_cfg.image_size - 1) - pts[:, 0]
                        if bool(meta["flip_v"]):
                            pts[:, 1] = float(aug_cfg.image_size - 1) - pts[:, 1]
                        return pts.astype(np.float32)

                    layer1_xy128 = _layer_basis_xy128(layer1)
                    layer2_xy128 = _layer_basis_xy128(layer2)
                    labels_path = label_dir / f"{out_png.stem}.npz"
                    np.savez_compressed(
                        labels_path,
                        layer1_xy128=layer1_xy128,
                        layer2_xy128=layer2_xy128,
                        gt_shift128=np.array([gtx, gty], dtype=np.float32),
                        sideA=float(sideA),
                        v1A=v1A.astype(np.float32),
                        v2A=v2A.astype(np.float32),
                        meta=json.dumps(meta, ensure_ascii=False),
                    )
                    vis_path = label_dir / f"{out_png.stem}_gt_align.png"
                    _save_gt_align_vis(
                        vis_path,
                        layer1_xy128=layer1_xy128,
                        layer2_xy128=layer2_xy128,
                        gt_shift128=(gtx, gty),
                        size=int(aug_cfg.image_size),
                    )

                save_monolayer = os.environ.get("MOIRE_SAVE_MONOLAYER", "").strip() not in {"", "0", "false", "False"}
                if save_monolayer:
                    mono_root = output_dir.parent / f"{output_dir.name}_monolayer"
                    mono_root.mkdir(parents=True, exist_ok=True)


                    side_nm = float(sideA) / 10.0
                    sample_dir = mono_root / f"{out_png.stem}_{side_nm:.2f}x{side_nm:.2f}"
                    sample_dir.mkdir(parents=True, exist_ok=True)

                    def _apply_geom_from_meta(im_in: Image.Image) -> np.ndarray:
                        """Internal helper."""
                        img01_m = to_float01(im_in.convert("L"))
                        # rotate
                        if bool(meta.get("rotate_enabled", True)) and float(meta.get("angle_deg", 0.0)) % 360.0 != 0.0:
                            img01_m = rotate_cv(img01_m, float(meta["angle_deg"]))
                        # crop+resize
                        patch_m = crop_and_resize(
                            img01_m,
                            int(meta["crop_cy"]),
                            int(meta["crop_cx"]),
                            int(meta["crop_side"]),
                            out_size=int(aug_cfg.image_size),
                        )
                        # flip
                        if bool(meta.get("flip_h", False)):
                            patch_m = cv2.flip(patch_m, 1)
                        if bool(meta.get("flip_v", False)):
                            patch_m = cv2.flip(patch_m, 0)
                        patch_m = np.clip(patch_m, 0.0, 1.0)
                        return (patch_m * 255.0).astype(np.uint8)


                    xyz1 = td_path / f"{cfg.material}_layer1.xyz"
                    xyz2 = td_path / f"{cfg.material}_layer2.xyz"
                    tif1 = td_path / "out_layer1.tif"
                    tif2 = td_path / "out_layer2.tif"
                    xyz1.write_text(
                        _incostem_xyz_lines(layer1, exclude_symbols=exclude_symbols, no_show_s=False),
                        encoding="utf-8",
                    )
                    xyz2.write_text(
                        _incostem_xyz_lines(layer2, exclude_symbols=exclude_symbols, no_show_s=False),
                        encoding="utf-8",
                    )
                    _run_incostem(incostem_path, _write_param(xyz_path=xyz1, tif_path=tif1, image_size=DEFAULT_IMAGE_SIZE))
                    _run_incostem(incostem_path, _write_param(xyz_path=xyz2, tif_path=tif2, image_size=DEFAULT_IMAGE_SIZE))

                    if tif1.exists() and tif2.exists():
                        with Image.open(tif1) as im1:
                            u0 = _apply_geom_from_meta(im1)
                        with Image.open(tif2) as im2:
                            u1 = _apply_geom_from_meta(im2)
                        Image.fromarray(u0).save(sample_dir / f"{out_png.stem}_0.png", format="PNG")
                        Image.fromarray(u1).save(sample_dir / f"{out_png.stem}_1.png", format="PNG")

                Image.fromarray(img_u8).save(out_png, format="PNG")
                return out_png

    if cfg.num is None:
        for defect_rate, dopant_rate, r, x, y in _iter_values(cfg):
            out = _generate_one(defect_rate=float(defect_rate), dopant_rate=float(dopant_rate), r=float(r), x=float(x), y=float(y))
            if out is None:
                continue
        return


    if cfg.num <= 0:
        raise ValueError('Invalid StackDiff configuration or runtime parameter.')
    target_env = os.environ.get("MOIRE_TARGET_STEMS", "").strip()
    target_stems = {s.strip() for s in target_env.split(",") if s.strip()} if target_env else set()
    found_targets: set[str] = set()
    generated = 0
    attempts = 0
    max_attempts = int(max(100, cfg.num * 5000))
    while generated < cfg.num:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                'Invalid StackDiff configuration or runtime parameter.'
                'Invalid StackDiff configuration or runtime parameter.'
            )

        defect_rate = _choose_defect_rate()
        dopant_rate = _choose_dopant_rate()
        x = _sample_uniform(cfg.x_range)
        y = _sample_uniform(cfg.y_range)
        r = 0.0 if cfg.rotation_range is None else _sample_uniform(cfg.rotation_range)
        out = _generate_one(defect_rate=defect_rate, dopant_rate=dopant_rate, r=r, x=x, y=y)
        if out is None:
            continue
        generated += 1
        if target_stems:
            if out.stem in target_stems:
                found_targets.add(out.stem)
                print(f"[TARGET] found {len(found_targets)}/{len(target_stems)}: {out.stem}")
            if found_targets >= target_stems:
                print(f"[TARGET] all targets generated, stop early: {len(found_targets)}/{len(target_stems)}")
                return
