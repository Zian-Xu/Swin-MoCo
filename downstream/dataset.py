import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as f
from PIL import Image
from PIL import ImageFilter


class GaussianBlur(object):
    """
    Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709
    """

    def __init__(self, sigma=None):
        if sigma is None:
            sigma = [.1, 2.]
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class MyDataset(Dataset):
    def __init__(self, dataset_path: str, output_size: int):
        self.dataset_path = dataset_path
        self.data_list = os.listdir(dataset_path)
        output_size = (output_size,) * 2
        self.transform = transforms.Compose([
                transforms.Resize(output_size),
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
                transforms.RandomApply([GaussianBlur([0, 2])], p=1.0),
                transforms.ToTensor()])
        self.resize = transforms.Resize(
            output_size,
            interpolation=transforms.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        data = np.load(os.path.join(self.dataset_path, self.data_list[item]))
        image = data['image']
        label = data['label']

        image = np.expand_dims(image, -1).repeat(3, axis=-1)
        image = Image.fromarray(np.uint8(image * 255))
        label = Image.fromarray(np.uint8(label))

        image, label = self.flip(image=image, label=label, p=0.5)
        image, label = self.rotate(image=image, label=label, degree=5, crop=True)
        image = self.transform(image)
        label = self.resize(label)
        label = torch.tensor(np.uint8(label))
        return image, label

    @staticmethod
    def flip(image, label, p):
        if torch.rand(1) < p:
            image = f.hflip(image)
            label = f.hflip(label)
        return image, label

    @staticmethod
    def rotate(image, label, degree, crop):
        # 随机旋转角度
        degree = random.uniform(-degree, degree)
        # 旋转
        image = f.rotate(image, angle=degree, interpolation=f.InterpolationMode.BILINEAR)
        label = f.rotate(label, angle=degree, interpolation=f.InterpolationMode.NEAREST)
        # 是否裁剪掉黑边
        if crop:
            angle_crop = abs(degree) % 180  # 裁剪角度的等效周期是180°
            if degree > 90:
                angle_crop = 180 - angle_crop
            # 转化角度为弧度
            theta = angle_crop * np.pi / 180
            # 计算高宽比
            (w, h) = image.size
            hw_ratio = float(h) / float(w)
            # 计算裁剪边长系数的分子项
            tan_theta = np.tan(theta)
            numerator = np.cos(theta) + np.sin(theta) * np.tan(theta)
            # 计算分母中和高宽比相关的项
            r = hw_ratio if h > w else 1 / hw_ratio
            # 计算分母项
            denominator = r * tan_theta + 1
            # 最终的边长系数
            crop_ratio = numerator / denominator
            # 裁剪
            height = int(h * crop_ratio)
            width = int(w * crop_ratio)
            image = f.center_crop(image, output_size=[height, width])
            label = f.center_crop(label, output_size=[height, width])
        return image, label


def collate_fn(batch):
    images, labels = tuple(zip(*batch))
    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)
    return images, labels
