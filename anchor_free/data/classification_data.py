import os
from functools import partial
import numpy as np
from torchvision import datasets, transforms

from data.abstract_data import AbstractDataModule
from data.augmentation import augmentations


class ClassificationDataModule(AbstractDataModule):
    def __init__(self):
        super().__init__()
        self.data_dir = "../input/imagenette/imagenette"

    @staticmethod
    def augment(image, augmentation_pipeline):
        image = np.asarray(image)
        return augmentation_pipeline(image=image)["image"]

    @staticmethod
    def transforms(phase):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        if phase == "train":
            return transforms.Compose(
                [
                    partial(
                        ClassificationDataModule.augment,
                        augmentation_pipeline=augmentations,
                    ),
                    transforms.ToPILImage(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )

    def get_dataset(self, phase):
        image_dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, phase),
            ClassificationDataModule.transforms(phase),
        )
        return image_dataset
