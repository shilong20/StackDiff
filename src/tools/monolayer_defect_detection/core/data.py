import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .Physical_augmentations import (
    elastic_deform_cv,
    perspective_warp_cv,
    add_carbon_background,
    add_scan_noise_from_original,
    add_poisson_noise_from_original,
    add_gaussian_noise,
    adjust_display_post_noise,
)


# =============================================================================
# Physical augmentation parameters
# =============================================================================

params = {
    "elastic": {
        "alpha_ratio": (0.03, 0.07),
        "sigma_ratio": (0.03, 0.05),
    },
    "perspective": {
        "max_ratio": (0.01, 0.04),
    },
    "carbon": {
        "cover": (0.2, 0.5),
        "alpha": (0.2, 0.4),
        "sigma_mask": 30,
        "sigma_field": 30,
        "coarse_down": (8, 16),
        "focus_enable": True,
        "focus_center": ((0.4, 0.4), (0.6, 0.6)),
        "focus_radius_ratio": (0.2, 0.4),
        "focus_gain": (1.0, 2.0),
        "morph_kernel": 7,
        "morph_iter": 1,
    },
    "scan_jitter": {
        "pixel_size_A_orig": 0.09,
        "sigma_jitter_A": (0.05, 0.1),
        "line_freq_hz": (40.0, 70.0),
        "k": 1.0,
    },
    "poisson": {
        "beam_current_A": 3e-12,
        "dwell_time_s": 2e-6,
        "k": 1.0,
        "preserve_areal_dose": True,
    },
    "gaussian": {
        "sigma": (0.05, 0.15),
        "preserve_scale": True,
    },
    "display": {
        "gain_range": (0.8, 1.2),
        "bias_range": (-0.1, 0.1),
        "gamma_range": (0.8, 1.2),
    },
}


# =============================================================================
# Basic augmentation modules
# =============================================================================

class Compose:
    """Apply a sequence of image-mask transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return {"image": image, "mask": mask}


class Resize:
    """Resize image and mask to a fixed size."""

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image, mask):
        image = cv2.resize(
            image,
            (self.width, self.height),
            interpolation=cv2.INTER_LINEAR,
        )
        mask = cv2.resize(
            mask,
            (self.width, self.height),
            interpolation=cv2.INTER_NEAREST,
        )
        return image, mask


class CenterCrop:
    """Center-crop image and mask. Zero padding is applied if needed."""

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image, mask):
        h, w = image.shape[:2]

        if h < self.height or w < self.width:
            pad_h = max(0, self.height - h)
            pad_w = max(0, self.width - w)

            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left

            image = cv2.copyMakeBorder(
                image,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=0,
            )
            mask = cv2.copyMakeBorder(
                mask,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=0,
            )

            h, w = image.shape[:2]

        start_y = (h - self.height) // 2
        start_x = (w - self.width) // 2

        image = image[start_y:start_y + self.height, start_x:start_x + self.width]
        mask = mask[start_y:start_y + self.height, start_x:start_x + self.width]

        return image, mask


class RandomCrop:
    """Random-crop image and mask. The crop size can be fixed or callable."""

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image, mask):
        h, w = image.shape[:2]

        crop_h = self.height() if callable(self.height) else self.height
        crop_w = self.width() if callable(self.width) else self.width

        if h < crop_h or w < crop_w:
            pad_h = max(0, crop_h - h)
            pad_w = max(0, crop_w - w)

            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left

            image = cv2.copyMakeBorder(
                image,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=0,
            )
            mask = cv2.copyMakeBorder(
                mask,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=0,
            )

            h, w = image.shape[:2]

        max_y = h - crop_h
        max_x = w - crop_w

        start_y = np.random.randint(0, max(1, max_y + 1))
        start_x = np.random.randint(0, max(1, max_x + 1))

        image = image[start_y:start_y + crop_h, start_x:start_x + crop_w]
        mask = mask[start_y:start_y + crop_h, start_x:start_x + crop_w]

        return image, mask


class Rotate:
    """Randomly rotate image and mask by an angle within the specified range."""

    def __init__(self, limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5):
        self.limit = limit
        self.border_mode = border_mode
        self.p = p

    def __call__(self, image, mask):
        if np.random.random() > self.p:
            return image, mask

        angle = np.random.uniform(-self.limit, self.limit)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        image = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=self.border_mode,
            borderValue=0,
        )
        mask = cv2.warpAffine(
            mask,
            matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=self.border_mode,
            borderValue=0,
        )

        return image, mask


class RandomRotate90:
    """Randomly rotate image and mask by 90, 180, or 270 degrees."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if np.random.random() > self.p:
            return image, mask

        k = np.random.randint(1, 4)

        image = np.ascontiguousarray(np.rot90(image, k))
        mask = np.ascontiguousarray(np.rot90(mask, k))

        return image, mask


class HorizontalFlip:
    """Randomly flip image and mask horizontally."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if np.random.random() > self.p:
            return image, mask

        image = np.ascontiguousarray(np.fliplr(image))
        mask = np.ascontiguousarray(np.fliplr(mask))

        return image, mask


class VerticalFlip:
    """Randomly flip image and mask vertically."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if np.random.random() > self.p:
            return image, mask

        image = np.ascontiguousarray(np.flipud(image))
        mask = np.ascontiguousarray(np.flipud(mask))

        return image, mask


class Normalize:
    """Normalize image intensity."""

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        if self.mean is not None and self.std is not None:
            image = (image - self.mean) / self.std
        elif image.max() > 1.0:
            image = image / 255.0

        return image, mask


class ToTensorV2:
    """Convert NumPy image and mask arrays to PyTorch tensors."""

    def __call__(self, image, mask):
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        if image.ndim == 2:
            image = image[np.newaxis, :, :]
        elif image.ndim == 3:
            image = image.transpose(2, 0, 1)

        image = torch.from_numpy(image.copy()).float()
        mask = torch.from_numpy(mask.copy()).long()

        return image, mask


# =============================================================================
# Augmentation pipelines
# =============================================================================

def get_training_augmentation(dim):
    """Return the training augmentation pipeline."""

    def get_random_crop_size():
        return np.random.randint(450, 600)

    return Compose([
        Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        CenterCrop(height=750, width=750),
        RandomCrop(height=get_random_crop_size, width=get_random_crop_size),
        Resize(dim, dim),
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(),
        ToTensorV2(),
    ])


def get_validation_augmentation(dim):
    """Return the validation augmentation pipeline."""

    return Compose([
        CenterCrop(550, 550),
        Resize(dim, dim),
        ToTensorV2(),
    ])


# =============================================================================
# Dataset classes
# =============================================================================

class CustomDataset(Dataset):
    """
    Segmentation dataset for grayscale STEM images.

    Parameters
    ----------
    img_list : list[str]
        List of image paths.

    dim : int
        Output image size.

    data_type : str
        If 'train', physical augmentations are applied before geometric transforms.
        Otherwise, only validation transforms are applied.
    """

    def __init__(self, img_list, dim=256, data_type="train"):
        self.images = []
        self.masks = []
        self.dim = dim
        self.data_type = data_type
        self.rng = np.random.default_rng(None)

        print(f"Start filtering {data_type} samples...")

        for img_path in img_list:
            mask_path = os.path.splitext(img_path)[0] + ".png"

            if not os.path.exists(img_path):
                print(f"Skip: image not found: {img_path}")
                continue

            if not os.path.exists(mask_path):
                print(f"Skip: mask not found: {mask_path}")
                continue

            try:
                with Image.open(img_path) as img:
                    img_shape = (img.height, img.width)

                    if img.mode not in ["L", "RGB", "RGBA"]:
                        print(f"Skip: unsupported image mode {img.mode}: {img_path}")
                        continue

            except Exception as exc:
                print(f"Skip: failed to read image {img_path}, error: {exc}")
                continue

            try:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                if mask is None:
                    raise ValueError("empty mask")

                if mask.ndim != 2:
                    raise ValueError(f"invalid mask shape: {mask.shape}")

                mask_shape = mask.shape

            except Exception as exc:
                print(f"Skip: failed to read mask {mask_path}, error: {exc}")
                continue

            if img_shape != mask_shape:
                print(f"Skip: size mismatch image{img_shape} vs mask{mask_shape}: {img_path}")
                continue

            self.images.append(img_path)
            self.masks.append(mask_path)

        print(
            f"{data_type} samples filtered: "
            f"original={len(img_list)}, valid={len(self.images)}"
        )

        self.aug = (
            get_training_augmentation(dim)
            if data_type == "train"
            else get_validation_augmentation(dim)
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img_path = os.path.normpath(self.images[i])
        mask_path = os.path.normpath(self.masks[i])

        image = self._load_image(img_path)
        mask = self._load_mask(mask_path)

        if image.shape != mask.shape:
            raise ValueError(
                f"Sample {i} size mismatch: image{image.shape} vs mask{mask.shape}, "
                f"path={img_path}"
            )

        if self.data_type == "train":
            image, mask = self._apply_physical_augmentation(image, mask)

        image = image[..., None]
        augmented = self.aug(image=image, mask=mask)

        image = augmented["image"].float()
        mask = augmented["mask"].long()

        return image, mask

    @staticmethod
    def _load_image(img_path):
        """Load image as a normalized grayscale array."""
        try:
            with Image.open(img_path) as img:
                image = np.array(img.convert("L"), dtype=np.float32) / 255.0
                return np.clip(image, 0.0, 1.0)
        except Exception as exc:
            raise ValueError(f"Failed to read image {img_path}: {exc}")

    @staticmethod
    def _load_mask(mask_path):
        """Load mask as a uint8 grayscale array."""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise ValueError(f"Failed to read mask {mask_path}")

        return mask.astype(np.uint8)

    def _apply_physical_augmentation(self, image, mask):
        """Apply physics-inspired image augmentations and synchronized mask geometry."""

        image, mask = self._apply_elastic_deformation(image, mask)
        image, mask = self._apply_perspective_transform(image, mask)
        image = self._apply_carbon_background(image)
        image = self._apply_scan_noise(image)
        image = self._apply_poisson_noise(image)
        image = self._apply_gaussian_noise(image)
        image = self._apply_display_adjustment(image)

        return image, mask

    def _apply_elastic_deformation(self, image, mask):
        """Apply elastic deformation to both image and mask."""
        alpha_ratio = self.rng.uniform(*params["elastic"]["alpha_ratio"])
        sigma_ratio = self.rng.uniform(*params["elastic"]["sigma_ratio"])

        image, field = elastic_deform_cv(
            image,
            alpha_ratio=alpha_ratio,
            sigma_ratio=sigma_ratio,
            rng=self.rng,
            interpolation=cv2.INTER_LINEAR,
        )

        mask, _ = elastic_deform_cv(
            mask.astype(np.float32),
            alpha_ratio=alpha_ratio,
            sigma_ratio=sigma_ratio,
            rng=self.rng,
            interpolation=cv2.INTER_NEAREST,
            field=field,
        )

        return np.clip(image, 0.0, 1.0), mask.astype(np.uint8)

    def _apply_perspective_transform(self, image, mask):
        """Apply perspective warp to image and mask."""
        max_ratio = self.rng.uniform(*params["perspective"]["max_ratio"])

        image, matrix = perspective_warp_cv(
            image,
            max_ratio=max_ratio,
            rng=self.rng,
            interpolation=cv2.INTER_LINEAR,
        )

        mask = cv2.warpPerspective(
            mask,
            matrix,
            (mask.shape[1], mask.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return np.clip(image, 0.0, 1.0), mask.astype(np.uint8)

    def _apply_carbon_background(self, image):
        """Add non-uniform amorphous carbon background."""
        cfg = {
            "cover": self.rng.uniform(*params["carbon"]["cover"]),
            "alpha": self.rng.uniform(*params["carbon"]["alpha"]),
            "sigma_mask": params["carbon"]["sigma_mask"],
            "sigma_field": params["carbon"]["sigma_field"],
            "coarse_down": self.rng.integers(
                params["carbon"]["coarse_down"][0],
                params["carbon"]["coarse_down"][1] + 1,
            ),
            "focus_enable": params["carbon"]["focus_enable"],
            "focus_center": (
                self.rng.uniform(*params["carbon"]["focus_center"][0]),
                self.rng.uniform(*params["carbon"]["focus_center"][1]),
            ),
            "focus_radius_ratio": self.rng.uniform(*params["carbon"]["focus_radius_ratio"]),
            "focus_gain": self.rng.uniform(*params["carbon"]["focus_gain"]),
            "morph_kernel": params["carbon"]["morph_kernel"],
            "morph_iter": params["carbon"]["morph_iter"],
        }

        image = add_carbon_background(image, cfg=cfg, rng=self.rng)
        return np.clip(image, 0.0, 1.0)

    def _apply_scan_noise(self, image):
        """Add scan-line jitter noise."""
        image = add_scan_noise_from_original(
            image,
            pixel_size_A_orig=params["scan_jitter"]["pixel_size_A_orig"],
            sigma_jitter_A=self.rng.uniform(*params["scan_jitter"]["sigma_jitter_A"]),
            line_freq_hz=self.rng.uniform(*params["scan_jitter"]["line_freq_hz"]),
            k=params["scan_jitter"]["k"],
            rng=self.rng,
        )
        return np.clip(image, 0.0, 1.0)

    def _apply_poisson_noise(self, image):
        """Add Poisson shot noise."""
        image = add_poisson_noise_from_original(
            image,
            beam_current_A=params["poisson"]["beam_current_A"],
            dwell_time_s=params["poisson"]["dwell_time_s"],
            k=params["poisson"]["k"],
            preserve_areal_dose=params["poisson"]["preserve_areal_dose"],
            rng=self.rng,
        )
        return np.clip(image, 0.0, 1.0)

    def _apply_gaussian_noise(self, image):
        """Add Gaussian readout noise."""
        image = add_gaussian_noise(
            image,
            sigma=self.rng.uniform(*params["gaussian"]["sigma"]),
            preserve_scale=params["gaussian"]["preserve_scale"],
            rng=self.rng,
        )
        return np.clip(image, 0.0, 1.0)

    def _apply_display_adjustment(self, image):
        """Apply post-noise display adjustment."""
        image, _ = adjust_display_post_noise(
            image,
            gain=self.rng.uniform(*params["display"]["gain_range"]),
            bias=self.rng.uniform(*params["display"]["bias_range"]),
            gamma=self.rng.uniform(*params["display"]["gamma_range"]),
            rng=self.rng,
        )
        return np.clip(image, 0.0, 1.0)




# =============================================================================
# Optional augmentation preview
# =============================================================================

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
#     image_path = r"G:\Moire_Code\stem-learning-master\seg\seg\data\defect_labelled_600\train\sample_00000_aug_00_y0000_x0000.jpg"
#     mask_path = r"G:\Moire_Code\stem-learning-master\seg\seg\data\defect_labelled_600\train\sample_00000_aug_00_y0000_x0000.png"
#
#     if not os.path.exists(image_path):
#         raise FileNotFoundError(f"Image file not found: {image_path}")
#     if not os.path.exists(mask_path):
#         raise FileNotFoundError(f"Mask file not found: {mask_path}")
#
#     dataset = CustomDataset(
#         img_list=[image_path],
#         dim=256,
#         data_type="train",
#     )
#
#     with Image.open(image_path) as img:
#         image_orig = np.array(img.convert("L"))
#     mask_orig = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#
#     for idx in range(10):
#         image_tensor, mask_tensor = dataset[0]
#
#         image_np = image_tensor.squeeze().numpy()
#         mask_np = mask_tensor.numpy()
#
#         image_vis = np.clip(image_np * 255, 0, 255).astype(np.uint8)
#         mask_vis = np.clip(mask_np * 255, 0, 255).astype(np.uint8)
#
#         fig, axes = plt.subplots(2, 2, figsize=(10, 10))
#         fig.suptitle(f"Augmentation preview - Run {idx + 1}", fontsize=16)
#
#         axes[0, 0].imshow(image_orig, cmap="gray")
#         axes[0, 0].set_title("Original image")
#         axes[0, 0].axis("off")
#
#         axes[0, 1].imshow(mask_orig, cmap="gray")
#         axes[0, 1].set_title("Original mask")
#         axes[0, 1].axis("off")
#
#         axes[1, 0].imshow(image_vis, cmap="gray")
#         axes[1, 0].set_title("Augmented image")
#         axes[1, 0].axis("off")
#
#         axes[1, 1].imshow(mask_vis, cmap="gray")
#         axes[1, 1].set_title("Augmented mask")
#         axes[1, 1].axis("off")
#
#         plt.tight_layout()
#         plt.show()