import torch
from torch.nn import functional as F

from timm.models.resnet import _create_resnet, Bottleneck
from torchmetrics import Accuracy, AUROC

from model.abstract_model import AbstractModel
from consts import *


class ClassificationModel(AbstractModel):
    def __init__(self):
        super().__init__()
        self.num_classes = 10
        model_args = dict(
            block=Bottleneck,
            layers=[2, 2, 2, 2],
            cardinality=32,
            base_width=4,
            block_args=dict(attn_layer="eca"),
            stem_width=32,
            stem_type="deep",
            avg_down=True,
            num_classes=self.num_classes,
        )
        self.model = _create_resnet("resnet50", False, **model_args)
        self.metrics = {
            "train": {
                "accuracy": Accuracy(num_classes=self.num_classes),
                "rocauc": AUROC(num_classes=self.num_classes, average="macro"),
            },
            "val": {
                "accuracy": Accuracy(num_classes=self.num_classes),
                "rocauc": AUROC(num_classes=self.num_classes, average="macro"),
            },
            "test": {
                "accuracy": Accuracy(num_classes=self.num_classes),
                "rocauc": AUROC(num_classes=self.num_classes, average="macro"),
            },
        }
        self.train_accuracy = self.metrics["train"]["accuracy"]
        self.train_rocauc = self.metrics["train"]["rocauc"]
        self.val_accuracy = self.metrics["val"]["accuracy"]
        self.val_rocauc = self.metrics["val"]["rocauc"]
        self.test_accuracy = self.metrics["test"]["accuracy"]
        self.test_rocauc = self.metrics["test"]["rocauc"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

    def step(self, batch, batch_idx, phase):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y.long(), label_smoothing=0.1)
        self.log(
            f"loss/{phase}",
            loss,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        for metric in self.metrics[phase]:
            self.metrics[phase][metric](logits.cpu(), y.cpu())
            self.log(
                f"{metric}/{phase}",
                self.metrics[phase][metric],
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
            )
        return loss
