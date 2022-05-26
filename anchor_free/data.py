import os
import torch
from torchvision import datasets, transforms

from pytorch_lightning import LightningDataModule


class Imagenette(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.data_dir = "/home/vladimir_kosolapov/image_classification/data/imagenette2"
        self.num_workers = 8
        self.batch_size = 64

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

    def get_dataloader(self, phase):
        image_dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, phase), self.transform
        )
        data_loader = torch.utils.data.DataLoader(
            image_dataset,
            batch_size=self.batch_size,
            shuffle=(phase == "train"),
            drop_last=(phase == "train"),
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )
        return data_loader

    def train_dataloader(self):
        return self.get_dataloader("train")

    def test_dataloader(self):
        return self.get_dataloader("val")
