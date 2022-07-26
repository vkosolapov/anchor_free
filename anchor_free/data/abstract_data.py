from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from pytorch_lightning import LightningDataModule

from consts import *


class AbstractDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.num_workers = DATA_NUM_WORKERS
        self.batch_size = DATA_BATCH_SIZE

    @property
    def transform(self):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

    def get_dataset(self, phase):
        return Dataset()

    def get_dataloader(self, phase):
        image_dataset = self.get_dataset(phase)
        data_loader = DataLoader(
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

    def val_dataloader(self):
        return self.get_dataloader("val")

    def test_dataloader(self):
        return self.get_dataloader("test")
