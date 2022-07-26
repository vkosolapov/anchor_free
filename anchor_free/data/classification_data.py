import os
from functools import partial
import numpy as np
from torchvision import datasets, transforms

from data.abstract_data import AbstractDataModule
from data.augmentation import augmentations


class ClassificationDataModule(AbstractDataModule):
    def __init__(self):
        super().__init__()
        self.data_dir = "datasets/imagenette2"

    @staticmethod
    def augment(image, augmentation_pipeline):
        image = np.asarray(image)
        return augmentation_pipeline(image=image)["image"]

    def transform(self, phase):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        return transforms.Compose(
            (
                [
                    partial(self.augment, augmentation_pipeline=augmentations),
                    transforms.ToPILImage(),
                ]
                if phase == "train"
                else []
            ).extend(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        )

    def get_dataset(self, phase):
        image_dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, phase), self.transform(phase)
        )
        return image_dataset
