from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from pytorch_lightning import LightningDataModule


class AbstractDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.num_workers = 8
        self.batch_size = 64

    @property
    def transform(self):
        return transforms.Compose([])

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
