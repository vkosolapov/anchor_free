import torch
from torch.nn import functional as F

from pytorch_lightning import LightningModule


class AbstractModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.metrics = {
            "train": {},
            "val": {},
            "test": {},
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

    def step(self, batch, batch_idx, phase):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y.long())
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        self.step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self.step(batch, batch_idx, "test")
