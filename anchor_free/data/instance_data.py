from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.datasets.cityscapes import Cityscapes

from data.abstract_data import AbstractDataModule
from data.augmentation import augmentations
from consts import *


class InstanceDataModule(AbstractDataModule):
    def __init__(self):
        super().__init__()
        self.data_dir = "datasets/cityscapes"

    def get_dataset(self, phase):
        image_dataset = CityscapesDataset(
            self.data_dir, phase, transform=self.transform
        )
        return image_dataset


class CityscapesDataset(Dataset):
    def __init__(self, data_dir, phase, transform):
        super().__init__()
        self.dataset = Cityscapes(
            data_dir, phase, mode="fine", target_type=["instance"]
        )
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        image, mask = self.dataset.__getitem__(index)

        image = image.resize(DATA_IMAGE_SIZE_INSTANCE, Image.LANCZOS)
        image = np.array(image, np.float32)

        mask = mask.resize(DATA_IMAGE_SIZE_INSTANCE, Image.NEAREST)
        mask = np.array(mask, np.int32)

        if self.phase == "train":
            result = augmentations(image=image, mask=mask)
            image = result["image"]
            mask = result["mask"]

        image = self.transform(image)

        return (image, mask)
