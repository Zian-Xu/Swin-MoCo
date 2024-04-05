import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as f
from PIL import Image
from PIL import ImageFilter, ImageOps


class TwoCropsTransform:
    """Take two random crops of one image"""
    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""
    def __init__(self, sigma=None):
        if sigma is None:
            sigma = [.1, 2.]
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""
    def __call__(self, x):
        return ImageOps.solarize(x)


class MyDataset(Dataset):
    def __init__(self, image_path: str, output_size: int, crop_min: float):
        self.image_path = image_path
        self.image_list = os.listdir(image_path)
        self.output_size = output_size
        self.crop_min = crop_min
        self.transform = self.get_transform()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        image = np.load(os.path.join(self.image_path, self.image_list[item]))['image']
        image = np.expand_dims(image, -1).repeat(3, axis=-1)
        image = Image.fromarray(np.uint8(image * 255))
        image = self.transform(image)
        return image

    def get_transform(self):
        augmentation_1 = [
            transforms.RandomRotation(degrees=10, interpolation=f.InterpolationMode.BILINEAR),
            transforms.RandomResizedCrop(self.output_size, scale=(self.crop_min, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]
        augmentation_2 = [
            transforms.RandomRotation(degrees=10, interpolation=f.InterpolationMode.BILINEAR),
            transforms.RandomResizedCrop(self.output_size, scale=(self.crop_min, 1.)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
            transforms.RandomApply([Solarize()], p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]
        return TwoCropsTransform(
            transforms.Compose(augmentation_1),
            transforms.Compose(augmentation_2))


def collate_fn(batch):
    images_1, images_2 = tuple(zip(*batch))
    images_1 = torch.stack(images_1, dim=0)
    images_2 = torch.stack(images_2, dim=0)
    return images_1, images_2
