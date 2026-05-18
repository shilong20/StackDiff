"""
Purpose: Provide an ImageFolder subclass with optional numeric filename sorting for deterministic StackDiff inference order. The module returns the standard torchvision samples without writing files.
Related files: src/core/guided_diffusion/diffusion.py and src/core/datasets/__init__.py.
CLI usage: This module is imported by the inference pipeline and is not intended to be executed directly.
"""

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision import transforms
import os

class SortedImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, sort=False):
        super(SortedImageFolder, self).__init__(root, transform, target_transform,
                                                loader, is_valid_file)
        if sort:

            self.imgs = sorted(self.imgs, key=lambda x: int(os.path.splitext(os.path.basename(x[0]))[0]))
            self.samples = self.imgs
        else:
            self.samples = self.imgs
