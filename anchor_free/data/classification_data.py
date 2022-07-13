import os
from torchvision import datasets, transforms

from data.abstract_data import AbstractDataModule


class ClassificationDataModule(AbstractDataModule):
    def __init__(self):
        super().__init__()
        self.data_dir = "datasets/imagenette2"

    @property
    def transform(self):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
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
            os.path.join(self.data_dir, phase), self.transform
        )
        return image_dataset
