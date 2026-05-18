"""
Purpose: Provide dataset loading helpers for StackDiff inference. The module enumerates image files, applies optional preprocessing plans, and returns tensors plus relative image identifiers to the sampler.
Related files: src/main.py, src/core/utils/image_preprocessor.py, and configs/separate/ReS2.yml.
CLI usage: This module is imported by src/main.py and is not intended to be executed directly.
"""

import os
import torch
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image
import fnmatch

from torchvision import transforms
from torch.utils.data import Dataset

from utils.image_preprocessor import apply_preprocess_plan

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

class FlatImageDataset(Dataset):
    """Internal helper."""

    def __init__(
        self,
        root,
        transform=None,
        selected_files=None,
        image_plans=None,
        fallback_size=None,
        tile_size_px=128,
        exclude_dir_patterns=None,
    ):
        self.root = root
        self.transform = transform
        self.image_plans = image_plans or {}
        self.fallback_size = fallback_size
        self.tile_size_px = tile_size_px
        self.exclude_dir_patterns = list(exclude_dir_patterns or [])

        all_candidates = []
        lookup = {}
        path_to_rel = {}

        def _normalize_key(key: str) -> str:
            return key.replace("\\", "/")

        def _add_lookup_entry(key: str, path: str):
            if not key:
                return

            if path not in lookup.setdefault(key, []):
                lookup[key].append(path)
            normalized = _normalize_key(key)
            if normalized != key:
                if path not in lookup.setdefault(normalized, []):
                    lookup[normalized].append(path)

        for current_root, dirs, files in os.walk(root):
            if self.exclude_dir_patterns:
                dirs[:] = [
                    d
                    for d in dirs
                    if not any(fnmatch.fnmatch(d, pat) for pat in self.exclude_dir_patterns)
                ]
            rel_dir = os.path.relpath(current_root, root)
            if rel_dir in ("", "."):
                rel_dir = ""
            for fname in files:
                if not fname.lower().endswith(IMG_EXTS):
                    continue
                full_path = os.path.join(current_root, fname)
                rel_path = os.path.join(rel_dir, fname) if rel_dir else fname
                rel_path = _normalize_key(rel_path)
                base = os.path.basename(rel_path)
                stem = os.path.splitext(base)[0]
                rel_stem = os.path.splitext(rel_path)[0]

                all_candidates.append((full_path, rel_path, base))
                path_to_rel[full_path] = rel_path

                _add_lookup_entry(rel_path, full_path)
                _add_lookup_entry(rel_stem, full_path)
                _add_lookup_entry(base, full_path)
                _add_lookup_entry(stem, full_path)

        if not all_candidates:
            raise FileNotFoundError('Invalid StackDiff configuration or runtime parameter.')

        if selected_files:
            self.samples = []
            for selected in selected_files:
                candidates = lookup.get(selected) or lookup.get(os.path.splitext(selected)[0])
                if not candidates:
                    alt = _normalize_key(selected)
                    candidates = lookup.get(alt) or lookup.get(os.path.splitext(alt)[0])
                if not candidates:
                    print('Invalid StackDiff configuration or runtime parameter.')
                    continue
                for path in candidates:
                    rel_path = path_to_rel.get(path)
                    base = os.path.basename(path)
                    if rel_path is None:
                        rel_path = _normalize_key(base)
                    self.samples.append((path, rel_path, base))
            if not self.samples:
                raise FileNotFoundError('Invalid StackDiff configuration or runtime parameter.')
        else:

            self.samples = sorted(all_candidates, key=lambda x: x[1])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, rel_path, basename = self.samples[index]
        with Image.open(path) as img:
            img = img.convert('L')

            candidates = [
                rel_path,
                os.path.splitext(rel_path)[0],
                basename,
                os.path.splitext(basename)[0],
            ]
            plan = None
            for key in candidates:
                if key in self.image_plans:
                    plan = self.image_plans[key]
                    break
            if plan:
                img = apply_preprocess_plan(img, plan)
            if not plan and self.fallback_size:
                row, col = self.fallback_size
                if row > 0 and col > 0:
                    target_size = (col * self.tile_size_px, row * self.tile_size_px)
                    img = img.resize(target_size, Image.BICUBIC)

            if self.transform:
                tensor = self.transform(img)
            else:
                tensor = transforms.ToTensor()(img)

        cls = 0
        return tensor, cls, rel_path


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )

def center_crop_arr(pil_image, image_size = 256):
    # Imported from openai/guided-diffusion
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

def get_dataset(args, config):
    if config.data.dataset == 'STEM':
        selected_files = getattr(args, 'selected_files', None)
        plans = getattr(args, 'image_crop_plans', {}) or {}
        fallback_size = (getattr(args, 'row', 1), getattr(args, 'col', 1))
        tile_size_px = getattr(args, 'tile_size_px', getattr(config.data, 'image_size', 128))
        exclude_dir_patterns = getattr(args, 'input_exclude_dirs', None)
        base_transform = transforms.ToTensor()
        dataset = FlatImageDataset(
            args.path_y,
            transform=base_transform,
            selected_files=selected_files,
            image_plans=plans,
            fallback_size=fallback_size,
            tile_size_px=tile_size_px,
            exclude_dir_patterns=exclude_dir_patterns,
        )
        test_dataset = dataset
        return dataset, test_dataset


    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )
    else:
        tran_transform = transforms.Compose(
            [
                transforms.Resize(config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        test_transform = transforms.Compose(
            [transforms.Resize(config.data.image_size), transforms.ToTensor()]
        )

    dataset, test_dataset = None, None

    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)
