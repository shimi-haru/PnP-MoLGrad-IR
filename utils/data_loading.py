import torch
import torch.nn as nn
# import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
# from torchvision import transforms
import torchvision.transforms.functional as F
# import math


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1, mix_no_noise=False):
        self.std = std
        self.mean = mean
        self.mix_no_noise = mix_no_noise

    def __call__(self, tensor, seed=[]):
        if seed == []:
            if self.std == "random":
                std = torch.distributions.Uniform(0, 0.2).sample().item()
            elif isinstance(self.std, dict):
                if self.std["mode"] == "add_0":
                    std = (torch.rand(1) >= self.std["rate"]).int(
                    ).item()*self.std["std"]
            else:
                std = self.std
            noise_map = torch.randn(tensor.size()) * std + self.mean
            noise_img = tensor + noise_map
        else:
            if self.std == "random":
                std = torch.distributions.Uniform(
                    0, 0.2).sample().item()
            elif isinstance(self.std, dict):
                if self.std["mode"] == "add_0":
                    std = (torch.rand(1) >= self.std["rate"]).int(
                    ).item()*self.std["std"]
            else:
                std = self.std
            generator = torch.Generator().manual_seed(seed)
            noise_map = torch.randn(
                tensor.size(), generator=generator) * std + self.mean
            noise_img = tensor + noise_map

        if self.mix_no_noise:
            n = torch.rand(1).item()
            if n > 0.8:
                return tensor, std, noise_map
            else:
                return torch.clamp(noise_img, 0, 1), std, noise_map
                # return noise_img, std, noise_map
        else:
            return torch.clamp(noise_img, 0, 1), std, noise_map
            # return noise_img, std, noise_map

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class AddBlur(object):
    def __init__(self, blur_layer, device):
        self.blur_layer = blur_layer
        self.device = device

    def __call__(self, tensor):
        if self.blur_layer is None:
            return tensor
        else:
            self.blur_layer.eval()
            tensor = tensor.unsqueeze(0).to(self.device)
            blur_image = self.blur_layer(tensor)
            blur_image = blur_image.squeeze(0).cpu()
            return blur_image

    def __repr__(self):
        return self.__class__.__name__ + '(blur_kernel={0})'.format(self.blur_layer)


class data_loading(Dataset):

    def __init__(self, device, blur_layer, img_dirs, noise_lev, gray=True, crop_size=256, mix_no_noise=False):
        self.img_paths = []
        self.image_channels = 1 if gray else 3
        for img_dir in img_dirs:
            self.img_paths.extend(self._get_img_paths(img_dir))
        self.crop_size = crop_size
        image_size = (crop_size, crop_size)
        self.noise_lev = noise_lev

        if gray:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    size=image_size),
                # transforms.RandomResizedCrop(size=image_size),
                transforms.Grayscale(num_output_channels=1),
                # transforms.RandomRotation(degrees=30),
                transforms.ToTensor(),
                # transforms.Normalize(mean=0.45, std=0.226)
            ])
        else:
            self.transform = transforms.Compose([
                # transforms.RandomResizedCrop(size = image_size, scale = (0.8, 1.2)),
                # transforms.RandomRotation(degrees=30),
                transforms.RandomResizedCrop(
                    size=image_size, scale=(0.8, 1)),
                transforms.ToTensor(),
                # transforms.Normalize(mean=0.45, std=0.226)
            ])
        self.add_noise = transforms.Compose([
            AddGaussianNoise(0, self.noise_lev, mix_no_noise=mix_no_noise)
        ])
        self.blur_image = transforms.Compose([
            AddBlur(blur_layer, device)
        ])

    def __getitem__(self, index):
        path = self.img_paths[index]
        img = Image.open(path)
        true_img = self.transform(img)

        channels, height, width = true_img.shape

        if (channels, height, width) != (self.image_channels, self.crop_size, self.crop_size):
            return None  # または、空の辞書: {}

        blur_img = self.blur_image(true_img)
        noise_img, noise_lev, noise_map = self.add_noise(blur_img)
        return {"true": true_img, "noise": noise_img, "blur": blur_img, "noise_lev": noise_lev, "noise_map": noise_map}

    def _get_img_paths(self, img_dir):
        """指定したディレクトリ内の画像ファイルのパス一覧を取得する。
        """
        img_dir = Path(img_dir)
        exts = [".jpg", ".png", ".bmp"]
        img_paths = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts]

        return img_paths

    def __len__(self):
        return len(self.img_paths)


class CustomCrop:
    def __init__(self, top, left, height, width):
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, img):
        return F.crop(img, self.top, self.left, self.height, self.width)


class data_loading_test(Dataset):

    def __init__(self, device, blur_layer, img_dirs, noise_lev, gray=True, crop_size=256, seed=None, crop_point=None):
        self.img_paths = []
        self.image_channels = 1 if gray else 3
        for img_dir in img_dirs:
            self.img_paths.extend(self._get_img_paths(img_dir))

        image_size = (crop_size, crop_size)
        self.crop_size = crop_size
        if noise_lev == "random":
            self.noise_lev = 0.1
        else:
            self.noise_lev = noise_lev
        # print(len(self.img_paths))
        if seed == None:
            self.seed = torch.randint(
                0, int(len(self.img_paths)*100), [len(self.img_paths)])
        else:
            self.seed = seed*torch.ones(len(self.img_paths))

        if gray:
            self.transform = transforms.Compose([
                transforms.CenterCrop(size=image_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                # transforms.Normalize(mean=0.45, std=0.226)
            ])
        else:
            if crop_point is None:
                self.transform = transforms.Compose([
                    transforms.CenterCrop(size=image_size),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=0.45, std=0.226)
                ])
            else:
                self.transform = transforms.Compose([
                    CustomCrop(
                        top=crop_point[0], left=crop_point[1], height=crop_size, width=crop_size),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=0.45, std=0.226)
                ])
        self.add_noise = AddGaussianNoise(
            0, self.noise_lev)
        self.blur_image = transforms.Compose([
            AddBlur(blur_layer, device)
        ])

    def __getitem__(self, index):
        path = self.img_paths[index]
        img = Image.open(path)
        true_img = self.transform(img)
        channels, height, width = true_img.shape
        # サイズが [3, 128, 128] 以外の場合に棄却
        # print(f"Image shape: {true_img.shape}, expected: {(self.image_channels, self.crop_size, self.crop_size)}")
        if (channels, height, width) != (self.image_channels, self.crop_size, self.crop_size):
            # print(f"Size mismatch, returning None for index {index}, path {path}")
            return None  # または、空の辞書: {}
        blur_img = self.blur_image(true_img)
        noise_img, _, noise_map = self.add_noise(
            blur_img, int(self.seed[index].item()))
        return {"true": true_img, "noise": noise_img,  "blur": blur_img, "noise_map": noise_map, "name": path.stem}

    def _get_img_paths(self, img_dir):
        """指定したディレクトリ内の画像ファイルのパス一覧を取得する。
        """
        img_dir = Path(img_dir)
        exts = [".jpg", ".png", ".bmp"]
        img_paths = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts]

        return img_paths

    def __len__(self):
        return len(self.img_paths)

###
