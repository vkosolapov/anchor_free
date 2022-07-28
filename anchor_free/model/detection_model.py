import torch

from timm.models import create_model
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from model.abstract_model import AbstractModel
from model.backbone import TIMMBackbone
from model.centernet import CenterNet
from optim.ranger import Ranger
from optim.cyclic_cosine import CyclicCosineLR
from consts import *


class DetectionModel(AbstractModel):
    def __init__(self):
        super().__init__()
        self.num_classes = 6
        self.backbone = create_model(
            "resnet18",
            pretrained=True,
            features_only=False,
            num_classes=self.num_classes,
        )
        self.backbone = TIMMBackbone(self.backbone, multi_output=False)
        channels = self.backbone.get_output_channels()
        self.head = CenterNet(
            num_classes=self.num_classes,
            input_channels=channels,
            act_layer=torch.nn.Mish,
        )
        self.metrics = {
            "train": {},
            "val": {
                "map_05": MeanAveragePrecision(
                    box_format="xyxy",
                    iou_thresholds=[0.5],
                    rec_thresholds=[0.0],
                    max_detection_thresholds=[100],
                    class_metrics=False,
                ),
                "map_05_095": MeanAveragePrecision(
                    box_format="xyxy",
                    iou_thresholds=[(50.0 + th * 5.0) / 100.0 for th in range(10)],
                    rec_thresholds=[0.0],
                    max_detection_thresholds=[100],
                    class_metrics=False,
                ),
            },
            "test": {
                "map_05": MeanAveragePrecision(
                    box_format="xyxy",
                    iou_thresholds=[0.5],
                    rec_thresholds=[0.0],
                    max_detection_thresholds=[100],
                    class_metrics=False,
                ),
                "map_05_095": MeanAveragePrecision(
                    box_format="xyxy",
                    iou_thresholds=[(50.0 + th * 5.0) / 100.0 for th in range(10)],
                    rec_thresholds=[0.0],
                    max_detection_thresholds=[100],
                    class_metrics=False,
                ),
            },
        }
        self.val_map_05 = self.metrics["val"]["map_05"]
        self.val_map_05_095 = self.metrics["val"]["map_05_095"]
        self.test_map_05 = self.metrics["test"]["map_05"]
        self.test_map_05_095 = self.metrics["test"]["map_05_095"]

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
        x, y, labels_count = batch
        targets = self.head.preprocess_targets(y, labels_count)
        logits = self.forward(x)
        predictions = self.head.postprocess_predictions(logits)
        loss, separate_losses = self.head.loss(logits, targets)
        y = y.cpu()
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
        for i in range(len(predictions)):
            if not predictions[i] is None:
                preds = [
                    {
                        "boxes": torch.Tensor(predictions[i][:, :4]),
                        "scores": torch.Tensor(predictions[i][:, 4]),
                        "labels": torch.Tensor(predictions[i][:, 5]),
                    }
                ]
            else:
                preds = [
                    {
                        "boxes": torch.Tensor(),
                        "scores": torch.Tensor(),
                        "labels": torch.Tensor(),
                    }
                ]
            if not y[i] is None:
                labels = [
                    {
                        "boxes": torch.Tensor(y[i, : labels_count[i], :4]).view(
                            labels_count[i], 4
                        ),
                        "labels": torch.Tensor(y[i, : labels_count[i], 4]).view(
                            labels_count[i]
                        ),
                    }
                ]
            else:
                labels = [{"boxes": torch.Tensor(), "labels": torch.Tensor(),}]
            for metric in self.metrics[phase]:
                if isinstance(self.metrics[phase][metric], MeanAveragePrecision):
                    self.metrics[phase][metric](preds, labels)

        for metric in self.metrics[phase]:
            if isinstance(self.metrics[phase][metric], MeanAveragePrecision):
                self.log(
                    f"{metric}/{phase}",
                    self.metrics[phase][metric]["map"],
                    metric_attribute=f"{phase}_{metric}",
                    prog_bar=False,
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                )
        return loss
