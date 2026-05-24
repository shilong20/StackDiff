# -*- coding: utf-8 -*-
"""
Purpose: Provide an IterableDataset that samples local source STEM images, applies shared StackDiff augmentation, and yields tensors for training. Optional preview images may be written when save_samples.enable is true.
Related files: configs/train/*.yml, src/core/scripts/image_train.py, and src/core/augmentations/stem.py.
CLI usage: This module is imported by src/core/scripts/image_train.py and is not intended to be executed directly.
"""

from __future__ import annotations

import os
import random
import time
import uuid
import sys
from typing import Iterator, List, Tuple, Dict, Any

from PIL import Image
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.augmentations.stem import StemAugmentor, AugmentConfig, to_float01, distance_transform


def worker_init_fn(worker_id: int):
    """Initialize worker-local random seeds and thread limits."""
    import os


    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


    try:
        import cv2
        cv2.setNumThreads(0)  # 0 means disable OpenCV's internal threading
    except Exception:
        pass


    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _list_image_files_recursively(data_dir: str) -> List[str]:
    results: List[str] = []
    for root, dirs, files in os.walk(data_dir):
        for name in files:
            lower = name.lower()
            if lower == "mask.png":
                continue
            if any(lower.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff"]):
                results.append(os.path.join(root, name))
    results.sort()
    return results


class TrainingAugmentIterableDataset(IterableDataset):
    def __init__(
        self,
        *,
        data_root: str,
        mask_path: str,
        mask_cache: str | None,
        image_size: int,
        augment_cfg: Dict[str, Any],
        save_samples_cfg: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.mask_path = mask_path
        self.mask_cache = mask_cache
        self.image_size = int(image_size)
        self.augment_cfg = augment_cfg or {}
        self._paths: List[str] | None = None
        self._augmentor = None
        self._mask01 = None
        self._dist_map = None
        self.save_samples_cfg = save_samples_cfg or {}
        self._save_enabled = bool(self.save_samples_cfg.get("enable", False))
        self._save_dir = self.save_samples_cfg.get("dir", "")
        self._save_format = self.save_samples_cfg.get("format", "png").lower()
        self._save_counter = 0
        self._save_limit = int(self.save_samples_cfg.get("max_per_worker", -1))
        self._save_seed = int(self.save_samples_cfg.get("seed", 0))
        if self._save_seed:
            random.seed(self._save_seed + os.getpid())

    def _ensure_save_dir(self) -> None:
        if self._save_enabled and self._save_dir:
            os.makedirs(self._save_dir, exist_ok=True)

    def _load_mask_and_cache(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._mask01 is not None and self._dist_map is not None:
            return self._mask01, self._dist_map
        if self.mask_cache and os.path.exists(self.mask_cache):
            cache = np.load(self.mask_cache)
            mask01 = cache.get("mask")
            dist_map = cache.get("dist")
            if mask01 is None or dist_map is None:
                raise RuntimeError("Mask cache is missing required mask or distance-transform arrays.")
            self._mask01 = mask01.astype(np.float32)
            self._dist_map = dist_map.astype(np.float32)
            return self._mask01, self._dist_map

        mask_img = Image.open(self.mask_path).convert('L')
        mask01 = to_float01(mask_img)
        dist_map = distance_transform(mask01)
        if self.mask_cache:
            tmp_path = f"{self.mask_cache}.tmp.{os.getpid()}.npz"
            np.savez_compressed(tmp_path, mask=mask01, dist=dist_map)
            try:
                os.replace(tmp_path, self.mask_cache)
            except OSError:

                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        self._mask01 = mask01.astype(np.float32)
        self._dist_map = dist_map.astype(np.float32)
        return self._mask01, self._dist_map

    def _init_worker_state(self) -> None:
        if self._paths is None:
            if not os.path.isdir(self.data_root):
                raise RuntimeError(
                    f"Training source image directory does not exist: {self.data_root} "
                    f"(abs={os.path.abspath(self.data_root)})"
                )
            self._paths = _list_image_files_recursively(self.data_root)
            if not self._paths:
                raise RuntimeError(f"No training images were found under {self.data_root}.")
        if self._augmentor is None:
            mask01, dist_map = self._load_mask_and_cache()
            self._augmentor = StemAugmentor(
                mask01=mask01,
                dist_map=dist_map,
                aug_cfg=AugmentConfig(image_size=self.image_size, cfg=self.augment_cfg),
            )
        self._ensure_save_dir()

    def _sample_path(self, rng: random.Random) -> str:
        assert self._paths is not None and len(self._paths) > 0
        return rng.choice(self._paths)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, Dict[str, Any]]]:
        worker = get_worker_info()
        rng = random.Random()
        rng.seed(os.getpid() + (worker.id if worker is not None else 0))

        self._init_worker_state()

        worker_tag = f"w{worker.id:02d}" if worker is not None else "w00"

        while True:
            img_path = self._sample_path(rng)
            try:
                img = Image.open(img_path).convert('L')
                patch = self._augmentor(img)  # 1xHxW, float32, [-1,1]
                # Convert numpy array to torch tensor
                patch_tensor = torch.from_numpy(patch).float()
                if self._save_enabled and self._save_dir:
                    self._save_patch(patch, img_path, worker_tag)
                yield patch_tensor, {}
            except (TypeError, ValueError, RuntimeError):
                raise
            except Exception:

                continue

    def _save_patch(self, patch: np.ndarray, origin_path: str, worker_tag: str) -> None:
        # patch: 1xHxW, [-1,1]
        arr = np.asarray(patch[0], dtype=np.float32)
        arr = np.clip((arr + 1.0) * 0.5, 0.0, 1.0)
        arr255 = (arr * 255.0).astype(np.uint8)
        img = Image.fromarray(arr255, mode='L')
        base = os.path.splitext(os.path.basename(origin_path))[0]
        ts = int(time.time() * 1000)
        unique = uuid.uuid4().hex[:8]
        idx = self._save_counter
        if self._save_limit > 0 and idx >= self._save_limit:
            return
        self._save_counter += 1
        filename = f"{worker_tag}_{ts}_{idx:06d}_{unique}_{base}.{self._save_format}"
        save_path = os.path.join(self._save_dir, filename)
        try:
            img.save(save_path)
        except Exception:
            pass


def load_training_data(
    *,
    data_root: str,
    mask_path: str,
    batch_size: int,
    image_size: int,
    augment_cfg: Dict[str, Any],
    mask_cache: str | None = None,
    save_samples_cfg: Dict[str, Any] | None = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
):
    """Build an infinite training data iterator from local source images."""
    dataset = TrainingAugmentIterableDataset(
        data_root=data_root,
        mask_path=mask_path,
        mask_cache=mask_cache,
        image_size=image_size,
        augment_cfg=augment_cfg,
        save_samples_cfg=save_samples_cfg,
    )
    use_workers = max(0, int(num_workers))
    prefetch = int(prefetch_factor) if use_workers > 0 else None
    persistent = bool(persistent_workers) if use_workers > 0 else False
    use_pin_memory = bool(pin_memory)

    if use_workers > 0:
        try:
            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            lock = ctx.Lock()
            lock.acquire()
            lock.release()
        except Exception:
            use_workers = 0
            prefetch = None
            persistent = False
            use_pin_memory = False

    def _build_loader(n_workers: int, use_pin: bool):
        prefetch_local = int(prefetch_factor) if n_workers > 0 else None
        persistent_local = bool(persistent_workers) if n_workers > 0 else False
        worker_init = worker_init_fn if n_workers > 0 else None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=n_workers,
            pin_memory=use_pin,
            prefetch_factor=prefetch_local,
            persistent_workers=persistent_local,
            worker_init_fn=worker_init,
            drop_last=True,
        )

    loader = _build_loader(use_workers, use_pin_memory)
    while True:
        try:
            for batch in loader:
                yield batch
        except (PermissionError, OSError):
            loader = _build_loader(0, False)
