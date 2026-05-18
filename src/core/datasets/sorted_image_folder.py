# sorted_image_folder.py

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
            # 重写imgs和samples，使得它们按照文件名的数值顺序排序,注意，此时图片的命名必须是数字
            self.imgs = sorted(self.imgs, key=lambda x: int(os.path.splitext(os.path.basename(x[0]))[0]))
            self.samples = self.imgs
        else:
            self.samples = self.imgs
