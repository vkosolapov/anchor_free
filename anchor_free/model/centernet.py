import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

from consts import *


class RegressionLossWithMask(nn.Module):
    def __init__(self, smooth=False):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target, mask):
        pred = pred.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
        expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)
        if self.smooth:
            loss_func = F.smooth_l1_loss
        else:
            loss_func = F.l1_loss
        loss = loss_func(pred * expand_mask, target * expand_mask, reduction="sum")
        loss = loss / (expand_mask.sum() + 1e-4)
        return loss


class CenterNet(nn.Module):
    def __init__(
        self, num_classes, input_channels, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d
    ):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = DATA_IMAGE_SIZE
        self.stride = 4
        self.output_size = int(self.image_size / self.stride)
        self.channels = input_channels
        self.act_layer = act_layer
        self.norm_layer = norm_layer

        self.decoder = self._make_decoder(
            num_layers=3,
            channels_list=[128, 64, 32],
            kernels_list=[4, 4, 4],
        )

        self.classification_head = self._make_head(
            input_channels=32, output_channels=num_classes
        )
        self.offset_head = self._make_head(input_channels=32, output_channels=2)
        self.size_head = self._make_head(input_channels=32, output_channels=2)

    def _make_decoder(self, num_layers, channels_list, kernels_list):
        layers = []
        for i in range(num_layers):
            channels = channels_list[i]
            kernel = kernels_list[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.channels,
                    out_channels=channels,
                    kernel_size=kernel,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False,
                )
            )
            layers.append(self.norm_layer(channels))
            layers.append(self.act_layer(inplace=True))
            self.channels = channels
        return nn.Sequential(*layers)

    def _make_head(self, input_channels, output_channels):
        return nn.Sequential(
            nn.Conv2d(
                input_channels, input_channels, kernel_size=3, padding=1, bias=False
            ),
            self.norm_layer(input_channels),
            self.act_layer(inplace=True),
            nn.Conv2d(
                input_channels, output_channels, kernel_size=1, stride=1, padding=0
            ),
        )

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, self.norm_layer):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.decoder(x)
        cls = self.classification_head(x)
        offset = self.offset_head(x)
        size = self.size_head(x)
        return {"cls": cls, "offset": offset, "size": size}

    def loss(self, logits, targets):
        loss_cls = sigmoid_focal_loss(
            logits["cls"], targets["cls"], alpha=0.25, gamma=2.0, reduction="mean"
        )
        regression_loss = RegressionLossWithMask(smooth=True)
        loss_offset = regression_loss(
            logits["offset"], targets["offset"], targets["mask"]
        )
        loss_size = regression_loss(logits["size"], targets["size"], targets["mask"])
        loss = loss_cls * 1.0 + loss_offset * 0.1 + loss_size * 0.1
        return loss, {
            "loss_cls": loss_cls,
            "loss_offset": loss_offset,
            "loss_size": loss_size,
        }

    def preprocess_targets(self, labels):
        return labels

    def postprocess_predictions(self, logits):
        return logits
