#!/usr/bin/env python3
"""
Purpose: Command-line entry point for StackDiff layer separation from a YAML configuration. It reads input STEM images, loads the configured material checkpoint, runs diffusion-based separation, and writes layer outputs to disk.
Related files: configs/separate/*.yml, src/core/datasets/__init__.py, src/core/guided_diffusion/diffusion.py, and src/core/utils/image_preprocessor.py.
CLI usage: python src/main.py --config configs/separate/ReS2.yml (arguments: --config selects the YAML file; --verbose sets logging level).
"""

import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import re
import math
from typing import Dict, Optional, Tuple
from PIL import Image
import torch
import numpy as np
from pathlib import Path
import fnmatch


sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
from guided_diffusion.diffusion import Diffusion
from datasets import IMG_EXTS
from utils.image_preprocessor import ImagePlan, build_preprocess_plan

def parse_args():
    parser = argparse.ArgumentParser(description="StackDiff layer separation from a YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")

    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--verbose", type=str, default="info",
                        choices=['debug', 'info', 'warning', 'error'],
                        help="Logging level")
    return parser.parse_args()

def load_config_from_path(config_path: str):
    """Internal helper."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def parse_materials_from_name(material_name: str):
    """Internal helper."""
    return [seg for seg in material_name.split('_') if seg]

def finalize_config(config: dict) -> dict:
    """Internal helper."""
    paths = config.get('paths', {})
    if 'input' not in paths:
        paths['input'] = paths.get('default_input')
    if 'output' not in paths:
        paths['output'] = paths.get('default_output')
    config['paths'] = paths
    return config

def dict2namespace(config):
    """Internal helper."""
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def setup_logging(verbose):
    """Internal helper."""
    level = getattr(logging, verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('Invalid StackDiff configuration or runtime parameter.')

    logger = logging.getLogger()
    logger.handlers.clear()

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(level)

    return logger

SIZE_PATTERN = re.compile(r"-\s*([0-9]+(?:\.[0-9]+)?)\s*[xX×]\s*([0-9]+(?:\.[0-9]+)?)")
GRID_UNIT_TOLERANCE = 1e-3

def parse_image_size_from_filename(filename: str) -> Optional[Tuple[float, float]]:
    """Internal helper."""
    match = SIZE_PATTERN.search(os.path.basename(filename))
    if not match:
        return None
    try:
        width_nm = float(match.group(1))
        height_nm = float(match.group(2))
        if width_nm <= 0 or height_nm <= 0:
            return None
        return width_nm, height_nm
    except (TypeError, ValueError):
        return None

def _grid_candidate_score(loss_area: float, unit_nm: float, row: int, col: int, unit_range: Tuple[float, float]) -> Tuple[float, float, float]:
    """Internal helper."""
    mid_unit = (unit_range[0] + unit_range[1]) * 0.5
    return (
        loss_area,
        abs(unit_nm - mid_unit),
        abs(row - col),
    )

def calculate_optimal_grid(
    height_nm: float,
    width_nm: float,
    unit_range: Tuple[float, float] = (2.4, 4.8),
    tol: float = GRID_UNIT_TOLERANCE,
) -> Optional[Dict[str, float]]:
    """Internal helper."""
    min_unit, max_unit = unit_range
    if height_nm <= 0 or width_nm <= 0 or min_unit <= 0 or max_unit <= 0:
        return None
    if min_unit > max_unit:
        min_unit, max_unit = max_unit, min_unit

    height_nm = float(height_nm)
    width_nm = float(width_nm)

    row_min = max(1, math.ceil(height_nm / max_unit))
    row_max = max(row_min, int(math.floor(height_nm / min_unit))) if min_unit > 0 else row_min
    col_min = max(1, math.ceil(width_nm / max_unit))
    col_max = max(col_min, int(math.floor(width_nm / min_unit))) if min_unit > 0 else col_min

    row_candidates = range(row_min, row_max + 1) if row_max >= row_min else range(1, 2)
    col_candidates = range(col_min, col_max + 1) if col_max >= col_min else range(1, 2)

    best_valid = None
    best_valid_score = None
    fallback = None
    fallback_score = None

    total_area = height_nm * width_nm

    for row in row_candidates:
        for col in col_candidates:
            if row <= 0 or col <= 0:
                continue
            unit_nm = min(height_nm / row, width_nm / col)
            if unit_nm <= 0:
                continue
            crop_height_nm = row * unit_nm
            crop_width_nm = col * unit_nm
            kept_area = crop_height_nm * crop_width_nm
            loss_area = max(0.0, total_area - kept_area)

            candidate = {
                "row": row,
                "col": col,
                "unit_nm": unit_nm,
                "crop_height_nm": crop_height_nm,
                "crop_width_nm": crop_width_nm,
                "loss_nm2": loss_area,
            }

            in_range = (unit_nm >= min_unit - tol) and (unit_nm <= max_unit + tol)
            score = _grid_candidate_score(loss_area, unit_nm, row, col, (min_unit, max_unit))

            if in_range:
                if best_valid is None or score < best_valid_score:
                    best_valid = candidate
                    best_valid_score = score
            else:
                if fallback is None or score < fallback_score:
                    fallback = candidate
                    fallback_score = score

    if best_valid:
        return best_valid

    if fallback:

        clipped_unit = min(max(fallback["unit_nm"], min_unit), max_unit)
        clipped_unit = min(clipped_unit, height_nm / fallback["row"], width_nm / fallback["col"])
        clipped_unit = max(clipped_unit, 0.0)
        crop_height_nm = fallback["row"] * clipped_unit
        crop_width_nm = fallback["col"] * clipped_unit
        kept_area = crop_height_nm * crop_width_nm
        loss_area = max(0.0, total_area - kept_area)
        fallback.update(
            {
                "unit_nm": clipped_unit,
                "crop_height_nm": crop_height_nm,
                "crop_width_nm": crop_width_nm,
                "loss_nm2": loss_area,
                "adjusted": True,
            }
        )
        return fallback

    return None

def align_physical_with_pixels(
    width_nm: float,
    height_nm: float,
    width_px: int,
    height_px: int,
) -> Tuple[float, float, bool]:
    """Internal helper."""
    if width_px <= 0 or height_px <= 0 or width_nm <= 0 or height_nm <= 0:
        return height_nm, width_nm, False

    ratio_px = width_px / height_px
    if ratio_px <= 0:
        return height_nm, width_nm, False

    ratio_nm = width_nm / height_nm if height_nm != 0 else float('inf')
    if ratio_nm <= 0 or math.isinf(ratio_nm):
        return height_nm, width_nm, False

    inv_ratio_nm = 1.0 / ratio_nm if ratio_nm != 0 else float('inf')
    if abs(ratio_px - ratio_nm) <= abs(ratio_px - inv_ratio_nm):
        return height_nm, width_nm, False
    return width_nm, height_nm, True

def prepare_output_dir(output_path):
    """Internal helper."""

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)



    return output_path

def prepare_input_dir(input_path):
    """Internal helper."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")


    if os.path.isfile(input_path):

        file_dir = os.path.dirname(input_path)
        temp_input_dir = os.path.join(file_dir, "temp_input")
        class_dir = temp_input_dir


        os.makedirs(class_dir, exist_ok=True)


        filename = os.path.basename(input_path)
        dst = os.path.join(class_dir, filename)
        shutil.copy2(input_path, dst)

        print(f"Single-file input; created temporary directory: {temp_input_dir}")
        return temp_input_dir


    if os.path.isdir(input_path):

        return input_path

    raise ValueError(f"Input path is neither a file nor a directory: {input_path}")

def list_input_images(root: str) -> Dict[str, str]:
    """Internal helper."""
    return list_input_images_excluding(root, exclude_dir_patterns=None)


def list_input_images_excluding(root: str, *, exclude_dir_patterns: Optional[list[str]]) -> Dict[str, str]:
    """Internal helper."""
    image_map: Dict[str, str] = {}
    exclude_dir_patterns = list(exclude_dir_patterns or [])
    for current_root, dirs, files in os.walk(root):
        if exclude_dir_patterns:
            dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, pat) for pat in exclude_dir_patterns)]
        rel_dir = os.path.relpath(current_root, root)
        if rel_dir in ("", "."):
            rel_dir = ""
        for fname in files:
            if not fname.lower().endswith(IMG_EXTS):
                continue
            full_path = os.path.join(current_root, fname)
            rel_path = os.path.join(rel_dir, fname) if rel_dir else fname
            rel_path = rel_path.replace("\\", "/")
            image_map[rel_path] = full_path
    return image_map


def _derive_input_excludes(input_path: str, output_path: str, *, config_excludes: Optional[list[str]]) -> list[str]:
    """Internal helper."""
    excludes = list(config_excludes) if config_excludes is not None else ["result", "result_*", "step_*"]
    try:
        in_root = Path(input_path).resolve()
        out_root = Path(output_path).resolve()
        rel = out_root.relative_to(in_root)
        if rel.parts:
            top = rel.parts[0]
            if top and top not in excludes:
                excludes.append(top)
    except Exception:
        pass

    seen = set()
    dedup = []
    for x in excludes:
        if not x or x in seen:
            continue
        seen.add(x)
        dedup.append(x)
    return dedup

def main():

    args = parse_args()


    logger = setup_logging(args.verbose)

    try:

        logger.info('Invalid StackDiff configuration or runtime parameter.')
        config = load_config_from_path(args.config)
        config = finalize_config(config)


        runtime_cfg = config.get('runtime', {}) or {}
        gpu_value = runtime_cfg.get('gpu', runtime_cfg.get('device', -1))
        try:
            selected_gpu = int(gpu_value) if gpu_value is not None else -1
        except (TypeError, ValueError):
            selected_gpu = -1
        if selected_gpu is not None and selected_gpu >= 0:
            os.environ["CUDA_DEVICE_ORDER"] = os.environ.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
            logger.info(
                'Invalid StackDiff configuration or runtime parameter.'
            )


        config_ns = dict2namespace(config)


        input_path = prepare_input_dir(config['paths']['input'])
        image_folder = prepare_output_dir(config['paths']['output'])



        proc_cfg = config.get('processing', {})
        input_excludes_cfg = proc_cfg.get("input_exclude_dirs", None)
        if input_excludes_cfg is not None and not isinstance(input_excludes_cfg, (list, tuple)):
            input_excludes_cfg = None
        input_exclude_dirs = _derive_input_excludes(input_path, image_folder, config_excludes=list(input_excludes_cfg) if input_excludes_cfg else None)


        selected_files = proc_cfg.get('selected_files')
        if not selected_files:
            selected_files = config.get('selected_files')
        selected_lookup = set()
        if selected_files:
            for name in selected_files:
                if not name:
                    continue
                candidates = set()
                candidates.add(name)
                normalized = name.replace("\\", "/")
                candidates.add(normalized)
                for candidate in list(candidates):
                    stem = os.path.splitext(candidate)[0]
                    candidates.add(stem)
                    base = os.path.basename(candidate)
                    candidates.add(base)
                    base_stem = os.path.splitext(base)[0]
                    candidates.add(base_stem)
                selected_lookup.update(c for c in candidates if c)
        if selected_files:
            logger.info(f"Selected files: {selected_files}")

        default_row = proc_cfg.get('row', 1) or 1
        default_col = proc_cfg.get('col', 1) or 1
        auto_crop_cfg = proc_cfg.get('auto_crop', {}) or {}
        auto_crop_enabled = bool(auto_crop_cfg.get('enabled', False))

        unit_range_cfg = auto_crop_cfg.get('unit_size_range', [2.4, 4.8])
        if isinstance(unit_range_cfg, (list, tuple)) and len(unit_range_cfg) == 2:
            try:
                unit_min = float(unit_range_cfg[0])
                unit_max = float(unit_range_cfg[1])
            except (TypeError, ValueError):
                unit_min, unit_max = 2.4, 4.8
        else:
            unit_min, unit_max = 2.4, 4.8
        unit_range_tuple = (unit_min, unit_max)
        tile_size_px = int(proc_cfg.get('tile_size_px', 128) or 128)
        if tile_size_px <= 0:
            tile_size_px = 128
        sliding_window_enabled = proc_cfg.get('sliding_window')
        if sliding_window_enabled is None:
            sliding_window_enabled = True
        if sliding_window_enabled:
            window_stride_px = max(1, tile_size_px // 2)
        else:
            window_stride_px = tile_size_px
        stride_ratio_effective = window_stride_px / tile_size_px if tile_size_px else 1.0
        overlap_ratio_effective = max(0.0, 1.0 - stride_ratio_effective)
        sliding_window_effective = sliding_window_enabled and window_stride_px < tile_size_px
        image_crop_plans: Dict[str, ImagePlan] = {}

        auto_plan_count = 0
        auto_plan_records = []
        if auto_crop_enabled:
            image_files = list_input_images_excluding(input_path, exclude_dir_patterns=input_exclude_dirs)
            if not image_files:
                logger.warning('Invalid StackDiff configuration or runtime parameter.')
            for rel_path, file_path in image_files.items():
                rel_no_ext = os.path.splitext(rel_path)[0]
                base_name = os.path.basename(rel_path)
                base_no_ext = os.path.splitext(base_name)[0]
                if selected_lookup:
                    if not any(key in selected_lookup for key in (rel_path, rel_no_ext, base_name, base_no_ext)):
                        continue
                size_tuple = parse_image_size_from_filename(base_name)
                if not size_tuple:
                    logger.warning(f"Filename has no physical-size tag; skipping auto-crop: {rel_path}")
                    continue
                width_nm_raw, height_nm_raw = size_tuple
                try:
                    with Image.open(file_path) as img:
                        width_px, height_px = img.size
                except Exception as exc:
                    logger.warning(f"Failed to read image size; skipping {rel_path}: {exc}")
                    continue

                height_nm_aligned, width_nm_aligned, swapped = align_physical_with_pixels(
                    width_nm_raw,
                    height_nm_raw,
                    width_px,
                    height_px,
                )

                grid_solution = calculate_optimal_grid(
                    height_nm_aligned,
                    width_nm_aligned,
                    unit_range=unit_range_tuple,
                )
                if grid_solution is None:
                    logger.warning(f"No feasible row/column grid found; skipping auto-crop: {rel_path}")
                    continue

                plan = build_preprocess_plan(
                    image_width_px=width_px,
                    image_height_px=height_px,
                    physical_width_nm=width_nm_aligned,
                    physical_height_nm=height_nm_aligned,
                    grid_solution=grid_solution,
                    tile_size_px=tile_size_px,
                    swapped_axes=swapped,
                )
                if plan is None:
                    logger.warning(f"Failed to build crop plan: {rel_path}")
                    continue

                image_crop_plans[rel_path] = plan
                image_crop_plans[rel_no_ext] = plan
                image_crop_plans.setdefault(base_name, plan)
                image_crop_plans.setdefault(base_no_ext, plan)
                auto_plan_count += 1
                auto_plan_records.append((rel_path, plan, swapped, grid_solution.get("adjusted", False)))


                logger.debug(
                    "AUTO-CROP %s -> row=%d col=%d unit_nm=%.3f loss=%.4f nm^2%s",
                    rel_path,
                    plan.row,
                    plan.col,
                    plan.unit_nm,
                    grid_solution.get("loss_nm2", -1.0),
                    " [axis swapped]" if swapped else "",
                )
            logger.info(f"Auto-crop enabled: generated {auto_plan_count} crop plans")
            logger.info(
                "Auto-crop settings: unit range[%.2f, %.2f] nm，tile=%d px",
                unit_min,
                unit_max,
                tile_size_px,
            )
            if auto_plan_records:
                logger.info("Auto-crop details:")
                for rel_name, plan, swapped, adjusted in sorted(auto_plan_records, key=lambda x: x[0]):
                    suffix = []
                    if swapped:
                        suffix.append("axis swapped")
                    if adjusted:
                        suffix.append("unit adjusted")
                    suffix_str = f" ({', '.join(suffix)})" if suffix else ""
                    stride_px = max(1, min(window_stride_px, tile_size_px))
                    effective_rows = max(1, math.floor(max(0, plan.target_size[1] - tile_size_px) / stride_px) + 1)
                    effective_cols = max(1, math.floor(max(0, plan.target_size[0] - tile_size_px) / stride_px) + 1)
                    stride_label = f"stride={stride_ratio_effective:.2f}×tile, overlap={overlap_ratio_effective:.0%}"
                    logger.info(
                        "  - %s -> row=%d col=%d (Sliding window=%dx%d, %s) unit=%.3f nm crop loss=%.4f nm²%s",
                        rel_name,
                        plan.row,
                        plan.col,
                        effective_rows,
                        effective_cols,
                        stride_label,
                        plan.unit_nm,
                        plan.meta.get("loss_nm2", 0.0) or 0.0,
                        suffix_str,
                    )
            elif not image_files:
                logger.info("Auto-crop produced no plans; falling back to the default row/col settings")

        run_args = argparse.Namespace(
            config=None,
            seed=args.seed,
            exp=config['paths']['output'],
            deg=config['separation']['method'],
            path_y=input_path,
            sigma_y=config['separation']['sigma_y'],
            row=default_row,
            col=default_col,
            selected_files=selected_files,
            N=config['separation']['num_layers'],
            eta=config['separation']['eta'],
            simplified=config['separation']['simplified'],
            adaptive_superposition=config['separation'].get('adaptive_superposition', False),
            fixed_superposition_k=config['separation'].get('fixed_superposition_k', None),
            residual_debug=config['separation'].get('residual_debug', False),
            image_folder=image_folder,
            deg_scale=4.0,
            deblur_sigma=config['separation'].get('deblur_sigma', 10.0),
            verbose=args.verbose,
            subset_start=-1,
            subset_end=-1,
            noise_type="gaussian",
            add_noise=False,
            auto_crop_enabled=auto_crop_enabled,
            image_crop_plans=image_crop_plans,
            tile_size_px=tile_size_px,
            unit_range=unit_range_tuple,
            window_stride_px=window_stride_px,
            input_exclude_dirs=input_exclude_dirs,
        )


        def get_model_config(material):
            """Internal helper."""
            base_config = {
                'image_size': 128,
                'in_channels': 1,
                'out_channels': 1,
                'num_channels': 256,
                'num_heads': 4,
                'num_res_blocks': 2,
                'attention_resolutions': "32,16,8",
                'dropout': 0.0,
                'resamp_with_conv': True,
                'learn_sigma': True,
                'use_scale_shift_norm': True,
                'use_fp16': True,
                'resblock_updown': True,
                'num_heads_upsample': -1,
                'var_type': 'fixedsmall',
                'num_head_channels': -1,
                'class_cond': False,
                'use_new_attention_order': False
            }


            if material in ['ReS2', 'MoS2', 'MoTe2', 'TaS2']:
                base_config['type'] = 'STEM_separate'
            elif material == 'Mixed':
                base_config['type'] = 'STEM_separate_mixed'

            return base_config


        materials = parse_materials_from_name(config.get('material', 'ReS2'))

        material_keywords = {'ReS2', 'MoS2', 'MoTe2', 'TaS2'}
        actual_materials = [m for m in materials if m in material_keywords]



        if len(actual_materials) == 0:

            primary_material = 'ReS2'
        elif len(actual_materials) == 1:
            primary_material = actual_materials[0]
        else:
            primary_material = 'Mixed'

        model_config = get_model_config(primary_material)
        config_ns.model = argparse.Namespace(**model_config)

        config_ns.data = argparse.Namespace(
            dataset='STEM',
            image_size=128,
            channels=1,
            logit_transform=False,
            uniform_dequantization=False,
            gaussian_dequantization=False,
            random_flip=False,
            rescaled=True,
            num_workers=32,
            subset_1k=True,
            out_of_dist=False
        )

        config_ns.diffusion = argparse.Namespace(
            beta_schedule='linear',
            beta_start=0.0001,
            beta_end=0.02,
            num_diffusion_timesteps=1000
        )

        # Time-travel back (RePaint-style) controls
        st_cfg = config.get('sampling', {}) or {}
        tt_cfg = (st_cfg.get('time_travel') or {}) if isinstance(st_cfg.get('time_travel'), dict) else {}
        tt_enable = bool(tt_cfg.get('enable', False))
        tt_length = int(tt_cfg.get('travel_length', 10)) if tt_enable else 1
        tt_repeat = int(tt_cfg.get('travel_repeat', 2)) if tt_enable else 1
        config_ns.time_travel = argparse.Namespace(
            T_sampling=config['sampling']['T_sampling'],
            travel_length=tt_length,
            travel_repeat=tt_repeat
        )
        if tt_enable:
            time_travel_msg = f"time-travel back enabled（length={tt_length}, repeat={tt_repeat})"
        else:
            time_travel_msg = "time-travel back disabled"

        config_ns.sampling = argparse.Namespace(
            batch_size=config['sampling']['batch_size']
        )


        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        device_desc = str(device)
        if torch.cuda.is_available():
            try:
                torch.cuda.set_device(0 if selected_gpu is not None and selected_gpu >= 0 else torch.cuda.current_device())
                cur_idx = torch.cuda.current_device()
                device_desc = f"cuda:{cur_idx} ({torch.cuda.get_device_name(cur_idx)})"
            except Exception:
                device_desc = str(device)
        else:
            device_desc = str(device)
        config_ns.device = device


        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True


        fixed_k = config['separation'].get('fixed_superposition_k', None)
        adaptive_superposition = config['separation'].get('adaptive_superposition', False)
        if fixed_k is not None:
            k_summary = f"fixed k = {fixed_k}（overrides adaptive mode）"
        elif adaptive_superposition:
            k_summary = "adaptive k"
        else:
            k_summary = 'Invalid StackDiff configuration or runtime parameter.'
        summary_lines = [
            "Run summary:",
            f"  Material: {config.get('material')} ({', '.join(materials)})",
            f"  Description: {config.get('description','')}",
            f"  Input: {input_path}",
            f"  Output: {config['paths']['output']}",
            f"  Number of layers: {config['separation']['num_layers']}",
            f"  Sampling steps: {config['sampling']['T_sampling']} | eta={config['separation']['eta']}",
            f"  GPU: {device_desc}",
            f"  Sliding window: {'enabled' if sliding_window_effective else 'disabled'} (stride={stride_ratio_effective:.2f}×tile, overlap={overlap_ratio_effective:.0%})",
            f"  {time_travel_msg}",
            f"  Image-size strategy: {'auto' if auto_crop_enabled else f'{default_row}x{default_col}'}",
            f"  Noise sigma_y: {config['separation']['sigma_y']}",
            f"  Superposition mode: {k_summary}",
        ]
        for line in summary_lines:
            logger.info(line)


        if 'model_checkpoints' in config.get('paths', {}):

            logger.info(f"Detected multiple model checkpoints: {len(config['paths']['model_checkpoints'])} ")


        logger.info('Invalid StackDiff configuration or runtime parameter.')
        runner = Diffusion(run_args, config_ns)
        runner.sample(config['separation']['simplified'])

        logger.info(f"Layer separation complete. Results saved to: {image_folder}")

    except Exception as e:
        logger.error(f"Runtime error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
