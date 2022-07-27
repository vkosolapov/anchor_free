import torch

from timm.models import create_model
from timm.models.resnet import _create_resnet, Bottleneck
from torchmetrics import JaccardIndex

from model.abstract_model import AbstractModel
from model.backbone import TIMMBackbone
from model.unet import UNet
from optim.ranger import Ranger
from consts import *


class SegmentationModel(AbstractModel):
    def __init__(self):
        super().__init__()
        self.num_classes = 2  # 59
        backbone_args = dict(
            block=Bottleneck,
            layers=[2, 2, 2, 2],
            # cardinality=32,
            # base_width=4,
            # block_args=dict(attn_layer="eca"),
            stem_width=32,
            stem_type="deep",
            avg_down=True,
            num_classes=self.num_classes,
            features_only=True,
        )
        # self.backbone = _create_resnet("resnet50", False, **backbone_args)
        self.backbone = create_model(
            "resnet18",
            pretrained=True,
            features_only=True,
            num_classes=self.num_classes,
        )
        self.backbone = TIMMBackbone(self.backbone, multi_output=True)
        channels = self.backbone.get_output_channels()
        self.head = UNet(num_classes=self.num_classes, bilinear=True, channels=channels)
        self.metrics = {
            "train": {
                "jaccard": JaccardIndex(
                    num_classes=self.num_classes,
                    multilabel=False,
                    average="macro",
                    threshold=MODEL_CLASSIFICATION_THRESHOLD,
                )
            },
            "val": {
                "jaccard": JaccardIndex(
                    num_classes=self.num_classes,
                    multilabel=False,
                    average="macro",
                    threshold=MODEL_CLASSIFICATION_THRESHOLD,
                )
            },
            "test": {
                "jaccard": JaccardIndex(
                    num_classes=self.num_classes,
                    multilabel=False,
                    average="macro",
                    threshold=MODEL_CLASSIFICATION_THRESHOLD,
                )
            },
        }
        self.train_jaccard = self.metrics["train"]["jaccard"]
        self.val_jaccard = self.metrics["val"]["jaccard"]
        self.test_jaccard = self.metrics["test"]["jaccard"]

    def configure_optimizers(self):
        optimizer = Ranger(self.model.parameters(), lr=0.01, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        return [optimizer], [scheduler]

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

    def step(self, batch, batch_idx, phase):
        x, y = batch
        logits = self.forward(x)
        loss, separate_losses = self.head.loss(logits, y)
        self.log(
            f"loss/{phase}",
            loss,
            prog_bar=True,
            logger=True,
            on_step=False,
            on_epoch=True,
        )
        for sep_loss in separate_losses:
            self.log(
                f"{sep_loss}/{phase}",
                separate_losses[sep_loss],
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
            )

        for metric in self.metrics[phase]:
            self.metrics[phase][metric](torch.sigmoid(logits), y)
            self.log(
                f"{metric}/{phase}",
                self.metrics[phase][metric],
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
            )
        return loss
