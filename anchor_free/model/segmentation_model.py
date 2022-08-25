import torch

from timm.models.resnet import _create_resnet, Bottleneck
from timm.models import create_model
from torchmetrics import JaccardIndex

from model.abstract_model import AbstractModel
from model.backbone import TIMMBackbone
from model.unet import UNet
from model.norm import CBatchNorm2d
from model.pidnet import FullModel, PIDNet, OhemCrossEntropy, BoundaryLoss
from model.metric import BoundaryIoU
from optim.ranger import Ranger
from optim.cyclic_cosine import CyclicCosineLR
from consts import *


class SegmentationModel(AbstractModel):
    def __init__(self):
        super().__init__()
        self.num_classes = 2  # 59
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
            features_only=True,
            num_classes=self.num_classes,
        )
        self.backbone = _create_resnet("resnet18", False, **model_args)
        # self.backbone = create_model(
        #    "resnet18",
        #    pretrained=True,
        #    features_only=True,
        #    num_classes=self.num_classes,
        # )
        self.backbone = TIMMBackbone(self.backbone, multi_output=True)
        channels = self.backbone.get_output_channels()
        # self.head = UNet(
        #    num_classes=self.num_classes,
        #    bilinear=True,
        #    channels=channels,
        #    norm_layer=CBatchNorm2d,
        #    act_layer=torch.nn.Mish,
        #    drop_block_rate=0.01,
        #    drop_path_rate=0.01,
        # )
        model = PIDNet(
            m=2,
            num_classes=self.num_classes,
            planes=channels,
            ppm_planes=96,
            head_planes=128,
            augment=True,
            act_layer=torch.nn.Mish,
            norm_layer=CBatchNorm2d,
            drop_block_rate=0.01,
            drop_path_rate=0.01,
        )
        sem_criterion = OhemCrossEntropy(
            ignore_label=-1, thres=0.9, min_kept=100000, weight=None
        )
        bd_criterion = BoundaryLoss()
        self.head = FullModel(model, sem_criterion, bd_criterion)
        self.metrics = {
            "train": {
                "jaccard": JaccardIndex(
                    num_classes=self.num_classes,
                    multilabel=False,
                    average="macro",
                    threshold=MODEL_CLASSIFICATION_THRESHOLD,
                ),
                "boundary_iou": BoundaryIoU(
                    num_classes=self.num_classes,
                    threshold=MODEL_CLASSIFICATION_THRESHOLD,
                ),
            },
            "val": {
                "jaccard": JaccardIndex(
                    num_classes=self.num_classes,
                    multilabel=False,
                    average="macro",
                    threshold=MODEL_CLASSIFICATION_THRESHOLD,
                ),
                "boundary_iou": BoundaryIoU(
                    num_classes=self.num_classes,
                    threshold=MODEL_CLASSIFICATION_THRESHOLD,
                ),
            },
            "test": {
                "jaccard": JaccardIndex(
                    num_classes=self.num_classes,
                    multilabel=False,
                    average="macro",
                    threshold=MODEL_CLASSIFICATION_THRESHOLD,
                ),
                "boundary_iou": BoundaryIoU(
                    num_classes=self.num_classes,
                    threshold=MODEL_CLASSIFICATION_THRESHOLD,
                ),
            },
        }
        self.train_jaccard = self.metrics["train"]["jaccard"]
        self.train_boundary_iou = self.metrics["train"]["boundary_iou"]
        self.val_jaccard = self.metrics["val"]["jaccard"]
        self.val_boundary_iou = self.metrics["val"]["boundary_iou"]
        self.test_jaccard = self.metrics["test"]["jaccard"]
        self.test_boundary_iou = self.metrics["test"]["boundary_iou"]

    def configure_optimizers(self):
        optimizer = Ranger(self.parameters(), lr=0.01, weight_decay=0.0001)
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
        x = self.backbone(x)
        x = self.head(x)
        return x

    def step(self, batch, batch_idx, phase):
        x, y = batch
        logits = self.forward(x)
        predictions = self.head.postprocess_predictions(logits)
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
            self.metrics[phase][metric](torch.sigmoid(predictions), y)
            self.log(
                f"{metric}/{phase}",
                self.metrics[phase][metric],
                prog_bar=False,
                logger=True,
                on_step=False,
                on_epoch=True,
            )
        return loss
