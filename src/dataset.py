import os

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from PIL import Image
from skimage import color


class FreshDataset(Dataset):
    def __init__(self, dir, test=False):
        self.dir = dir
        self.gray_path = dir + 'gray/'
        self.color_path = dir + 'color/'
        self.test_path = dir + 'test/'
        self.test = test

        self.path = []
        if test:
            for _, _, files in os.walk(self.test_path):
                for file in files:
                    self.path.append(file)
        else:
            for root, dirs, _ in os.walk(self.gray_path):
                for dir in dirs:
                    for _, _, files in os.walk(self.gray_path+dir):
                        for file in files:
                            self.path.append(dir + '/' + file)

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        img_name = self.path[idx]
        if self.test:
            image = torchvision.io.read_image(self.test_path + img_name)
        else:
            image = torchvision.io.read_image(self.gray_path + img_name)
        image = image.unsqueeze(0)
        image = F.interpolate(image, (160, 160))
        image = image.squeeze(0)
        image = image.permute(1, 2, 0)
        image = image.repeat(1, 1, 3)
        image = image.permute(2, 0, 1)
        image = torch.tensor(color.rgb2lab(image.permute(1, 2, 0)/255))
        image = (image + torch.tensor([0, 128, 128])
                 ) / torch.tensor([100, 255, 255])
        image = image.permute(2, 0, 1)
        image = image[:1, :, :]  # grayscale channel as input
        if self.test:
            return image

        label = torchvision.io.read_image(self.color_path + img_name)
        label = label.unsqueeze(0)
        label = F.interpolate(label, (160, 160))
        label = label.squeeze(0)
        label = label.permute(1, 2, 0)
        label = label.permute(2, 0, 1)
        label = torch.tensor(color.rgb2lab(label.permute(1, 2, 0)/255))
        label = (label + torch.tensor([0, 128, 128])
                 ) / torch.tensor([100, 255, 255])
        label = label.permute(2, 0, 1)
        label = label[1:, :, :]  # a, b channels as label

        return image, label


class LandscapeDataset(Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.gray_path = dir + 'gray/'
        self.color_path = dir + 'color/'

    def __len__(self):
        return len(os.listdir(self.gray_path))

    def __getitem__(self, idx):
        img_name = str(idx) + '.jpg'
        image = torchvision.io.read_image(self.gray_path + img_name)
        image = image.unsqueeze(0)
        image = F.interpolate(image, (160, 160))
        image = image.squeeze(0)
        image = image.permute(1, 2, 0)
        image = image.repeat(1, 1, 3)
        image = image.permute(2, 0, 1)
        label = torchvision.io.read_image(self.color_path + img_name)
        label = label.unsqueeze(0)
        label = F.interpolate(label, (160, 160))
        label = label.squeeze(0)
        label = label.permute(1, 2, 0)
        label = label.permute(2, 0, 1)
        image = torch.tensor(color.rgb2lab(image.permute(1, 2, 0)/255))
        label = torch.tensor(color.rgb2lab(label.permute(1, 2, 0)/255))

        image = (image + torch.tensor([0, 128, 128])
                 ) / torch.tensor([100, 255, 255])
        label = (label + torch.tensor([0, 128, 128])
                 ) / torch.tensor([100, 255, 255])

        image = image.permute(2, 0, 1)
        label = label.permute(2, 0, 1)
        image = image[:1, :, :]  # grayscale channel as input
        label = label[1:, :, :]  # a, b channels as label

        return image, label
