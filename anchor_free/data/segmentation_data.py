import os
import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from data.abstract_data import AbstractDataModule
from consts import *


class SegmentationDataModule(AbstractDataModule):
    def __init__(self):
        super().__init__()
        self.data_dir = "../input/people-clothing-segmentation"

    @property
    def transform(self):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std),]
        )

    def get_dataset(self, phase):
        image_dataset = ClothesDataset(self.data_dir, phase, transform=self.transform,)
        return image_dataset


class ClothesDataset(Dataset):
    def __init__(self, data_dir, phase, transform):
        self.image_folder = f"{data_dir}/png_images/IMAGES"
        self.mask_folder = f"{data_dir}/png_masks/MASKS"
        if phase == "train":
            self.list_img = sorted(os.listdir(self.image_folder))[:800]
            self.list_mask = sorted(os.listdir(self.mask_folder))[:800]
        elif phase == "val":
            self.list_img = sorted(os.listdir(self.image_folder))[800:900]
            self.list_mask = sorted(os.listdir(self.mask_folder))[800:900]
        else:
            self.list_img = sorted(os.listdir(self.image_folder))[900:]
            self.list_mask = sorted(os.listdir(self.mask_folder))[900:]
        self.transform = transform

    def __len__(self):
        return len(self.list_img)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_folder, self.list_img[idx]))
        if len(np.shape(image)) != 3 or np.shape(image)[2] != 3:
            image = image.convert("RGB")
        image = image.resize(DATA_IMAGE_SIZE_SEGMENTATION, Image.LANCZOS)
        image = np.asarray(image)
        image = self.transform(image)

        mask = Image.open(os.path.join(self.mask_folder, self.list_mask[idx]))
        mask = mask.resize(DATA_IMAGE_SIZE_SEGMENTATION, Image.NEAREST)
        mask = torch.clamp(torch.from_numpy(np.array(mask)), 0, 1).long()

        return (image, mask)
