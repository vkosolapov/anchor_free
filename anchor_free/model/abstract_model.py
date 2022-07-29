import torch
import torch.nn as nn
from torch.nn import functional as F

from pytorch_lightning import LightningModule

from model.norm import CBatchNorm2d


class AbstractModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.metrics = {
            "train": {},
            "val": {},
            "test": {},
        }

    def configure_model(self):
        prev_module = None
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                prev_module = module
            if isinstance(module, CBatchNorm2d):
                module.prev_module_weight = prev_module.weight

    def configure_optimizers(self):
        optimizer = torch.optim.Optimizer(self.model.parameters)
        scheduler = torch.optim.lr_scheduler._LRScheduler(optimizer)
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
