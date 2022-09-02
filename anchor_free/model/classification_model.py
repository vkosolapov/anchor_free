import torch
from torch.nn import functional as F

from timm.models.resnet import _create_resnet, Bottleneck
from timm.models import create_model
from torchmetrics import Accuracy, AUROC

from model.abstract_model import AbstractModel
from model.norm import CBatchNorm2d
from model.loss import LabelSmoothingFocalLoss
from optim.ranger import Ranger
from optim.adan import Adan
from optim.cyclic_cosine import CyclicCosineLR
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
            norm_layer=CBatchNorm2d,
            act_layer=torch.nn.Mish,
            drop_block_rate=0.01,
            drop_path_rate=0.01,
            stem_width=32,
            stem_type="deep",
            avg_down=True,
            num_classes=self.num_classes,
        )
        self.model = _create_resnet("resnet18", False, **model_args)
        # self.model = create_model(
        #    "gmlp_ti16_224", pretrained=False, num_classes=self.num_classes,
        # )
        self.classification_loss = LabelSmoothingFocalLoss(
            self.num_classes, need_one_hot=True, gamma=2, alpha=0.25, smoothing=0.1
        )
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
        optimizer = Adan(self.parameters(), lr=0.01, weight_decay=0.0001)
        scheduler = CyclicCosineLR(
            optimizer,
            warmup_epochs=5,
            warmup_start_lr=0.005,
            warmup_linear=False,
            init_decay_epochs=5,
            min_decay_lr=0.001,
            restart_lr=0.01,
            restart_interval=10,
            # restart_interval_multiplier=1.2,
        )
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

    def step(self, batch, batch_idx, phase):
        x, y = batch
        logits = self.forward(x)
        loss = self.classification_loss(logits, y.long())
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
