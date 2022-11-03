import torch

from timm.models.resnet import _create_resnet, Bottleneck
from timm.models import create_model
from torchmetrics import Precision, Recall
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from model.abstract_model import AbstractModel
from model.backbone import TIMMBackbone
from model.fcos import FCOS
from model.centernet import CenterNet
from model.norm import CBatchNorm2d
from optim.ranger import Ranger
from optim.cyclic_cosine import CyclicCosineLR
from consts import *


class DetectionModel(AbstractModel):
    def __init__(self):
        super().__init__()
        self.num_classes = 6
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
        #    features_only=False,
        #    num_classes=self.num_classes,
        # )
        self.backbone = TIMMBackbone(self.backbone, multi_output=True)
        channels = self.backbone.get_output_channels()
        # self.head = CenterNet(
        #    num_classes=self.num_classes,
        #    input_channels=channels,
        #    norm_layer=CBatchNorm2d,
        #    act_layer=torch.nn.Mish,
        #    drop_block_rate=0.01,
        #    drop_path_rate=0.01,
        # )
        self.head = FCOS(channels, self.num_classes)
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
                "precision": Precision(
                    num_classes=self.num_classes,
                    threshold=MODEL_CLASSIFICATION_THRESHOLD,
                    average="macro",
                ),
                "recall": Recall(
                    num_classes=self.num_classes,
                    threshold=MODEL_CLASSIFICATION_THRESHOLD,
                    average="macro",
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
                "precision": Precision(
                    num_classes=self.num_classes,
                    threshold=MODEL_CLASSIFICATION_THRESHOLD,
                    average="macro",
                ),
                "recall": Recall(
                    num_classes=self.num_classes,
                    threshold=MODEL_CLASSIFICATION_THRESHOLD,
                    average="macro",
                ),
            },
        }
        self.val_map_05 = self.metrics["val"]["map_05"]
        self.val_map_05_095 = self.metrics["val"]["map_05_095"]
        self.val_precision = self.metrics["val"]["precision"]
        self.val_recall = self.metrics["val"]["recall"]
        self.test_map_05 = self.metrics["test"]["map_05"]
        self.test_map_05_095 = self.metrics["test"]["map_05_095"]
        self.test_precision = self.metrics["test"]["precision"]
        self.test_recall = self.metrics["test"]["recall"]

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
        logits = self.forward(x)
        predictions = self.head.postprocess_predictions(logits)
        # boxes_count = sum(
        #    [pred.shape[0] if not pred is None else 0 for pred in predictions]
        # )
        # print(boxes_count)
        # self.log(
        #    f"boxes/{phase}",
        #    boxes_count,
        #    prog_bar=False,
        #    logger=True,
        #    on_step=False,
        #    on_epoch=True,
        # )
        targets = self.head.preprocess_targets(y, labels_count)
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
        batch_preds = []
        batch_labels = []
        for i in range(len(predictions)):
            if not predictions[i] is None:
                preds = {
                    "boxes": torch.Tensor(predictions[i][:, :4]),
                    "scores": torch.Tensor(predictions[i][:, 4]),
                    "labels": torch.Tensor(predictions[i][:, 5]),
                }
            else:
                preds = {
                    "boxes": torch.Tensor(),
                    "scores": torch.Tensor(),
                    "labels": torch.Tensor(),
                }
            batch_preds.append(preds)
            if not y[i] is None:
                labels = {
                    "boxes": torch.Tensor(y[i, : labels_count[i], :4]).view(
                        labels_count[i], 4
                    ),
                    "labels": torch.Tensor(y[i, : labels_count[i], 4]).view(
                        labels_count[i]
                    ),
                }
            else:
                labels = {"boxes": torch.Tensor(), "labels": torch.Tensor()}
            batch_labels.append(labels)

        for metric in self.metrics[phase]:
            if isinstance(self.metrics[phase][metric], MeanAveragePrecision):
                self.metrics[phase][metric](batch_preds, batch_labels)
                self.log(
                    f"{metric}/{phase}",
                    self.metrics[phase][metric]["map"],
                    metric_attribute=f"{phase}_{metric}",
                    prog_bar=False,
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                )
            elif isinstance(self.metrics[phase][metric], Precision) or isinstance(
                self.metrics[phase][metric], Recall
            ):
                self.metrics[phase][metric](
                    torch.sigmoid(logits["cls"])
                    .permute(0, 2, 3, 1)
                    .reshape([-1, self.num_classes]),
                    targets["cls"]
                    .type(torch.int32)
                    .permute(0, 2, 3, 1)
                    .reshape([-1, self.num_classes]),
                )
                self.log(
                    f"{metric}/{phase}",
                    self.metrics[phase][metric],
                    metric_attribute=f"{phase}_{metric}",
                    prog_bar=False,
                    logger=True,
                    on_step=False,
                    on_epoch=True,
                )
        return loss
