"""
Purpose: Build and apply crop/resize preprocessing plans for input STEM images before tiled inference. The helpers return ImagePlan records and transformed PIL images without changing source files.
Related files: src/main.py and src/core/datasets/__init__.py.
CLI usage: This module is imported by src/main.py and is not intended to be executed directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from PIL import Image


@dataclass(frozen=True)
class ImagePlan:
    """Internal helper."""
    row: int
    col: int
    unit_nm: float
    crop_box: Tuple[int, int, int, int]
    target_size: Tuple[int, int]
    meta: Dict[str, Any]


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def _center_crop_box(
    width_px: int,
    height_px: int,
    crop_width_px: int,
    crop_height_px: int,
) -> Tuple[int, int, int, int]:
    """Internal helper."""
    crop_width_px = _clamp(crop_width_px, 1, width_px)
    crop_height_px = _clamp(crop_height_px, 1, height_px)

    left = max(0, (width_px - crop_width_px) // 2)
    top = max(0, (height_px - crop_height_px) // 2)
    right = min(width_px, left + crop_width_px)
    bottom = min(height_px, top + crop_height_px)


    actual_width = right - left
    actual_height = bottom - top
    if actual_width < crop_width_px and right == width_px:
        left = max(0, width_px - crop_width_px)
        right = width_px
    if actual_height < crop_height_px and bottom == height_px:
        top = max(0, height_px - crop_height_px)
        bottom = height_px

    return left, top, right, bottom


def build_preprocess_plan(
    image_width_px: int,
    image_height_px: int,
    physical_width_nm: float,
    physical_height_nm: float,
    grid_solution: Dict[str, float],
    tile_size_px: int = 128,
    swapped_axes: bool = False,
) -> Optional[ImagePlan]:
    """Internal helper."""
    row = int(grid_solution.get("row", 0))
    col = int(grid_solution.get("col", 0))
    unit_nm = float(grid_solution.get("unit_nm", 0))
    crop_width_nm = float(grid_solution.get("crop_width_nm", 0))
    crop_height_nm = float(grid_solution.get("crop_height_nm", 0))

    if row <= 0 or col <= 0 or tile_size_px <= 0:
        return None
    if physical_width_nm <= 0 or physical_height_nm <= 0:
        return None

    scale_w = image_width_px / physical_width_nm
    scale_h = image_height_px / physical_height_nm

    crop_width_px = int(round(crop_width_nm * scale_w))
    crop_height_px = int(round(crop_height_nm * scale_h))

    target_width = col * tile_size_px
    target_height = row * tile_size_px

    crop_box = _center_crop_box(
        image_width_px,
        image_height_px,
        crop_width_px,
        crop_height_px,
    )

    metadata = {
        "unit_nm": unit_nm,
        "crop_width_nm": crop_width_nm,
        "crop_height_nm": crop_height_nm,
        "loss_nm2": grid_solution.get("loss_nm2"),
        "nm_per_px_width": physical_width_nm / image_width_px if image_width_px else None,
        "nm_per_px_height": physical_height_nm / image_height_px if image_height_px else None,
        "swapped_axes": swapped_axes,
    }

    return ImagePlan(
        row=row,
        col=col,
        unit_nm=unit_nm,
        crop_box=crop_box,
        target_size=(target_width, target_height),
        meta=metadata,
    )


def apply_preprocess_plan(image: Image.Image, plan: Optional[ImagePlan]) -> Image.Image:
    """Internal helper."""
    if plan is None:
        return image

    left, top, right, bottom = plan.crop_box
    processed = image.crop((left, top, right, bottom))
    processed = processed.resize(plan.target_size, Image.BICUBIC)
    return processed
