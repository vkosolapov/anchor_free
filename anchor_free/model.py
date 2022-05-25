import torch
from torch.nn import functional as F

from timm.models.resnet import _create_resnet, Bottleneck
from torchmetrics import Accuracy

from pytorch_lightning import LightningModule


class Model(LightningModule):
    def __init__(self):
        super().__init__()
        model_args = dict(
            block=Bottleneck,
            layers=[2, 2, 2, 2],
            cardinality=32,
            base_width=4,
            block_args=dict(attn_layer="eca"),
            stem_width=32,
            stem_type="deep",
            avg_down=True,
            num_classes=10,
        )
        self.model = _create_resnet("resnet50", False, **model_args)
        self.test_acc = Accuracy()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y.long())
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y.long())
        self.test_acc(logits, y)
        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc)
