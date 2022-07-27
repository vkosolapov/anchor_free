import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.losses import DiceLoss

from model.loss import LabelSmoothingFocalLoss
from consts import *


class UNet(nn.Module):
    def __init__(self, num_classes, bilinear, channels):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.up1 = Up(channels[4], channels[3], bilinear)
        self.up2 = Up(channels[3], channels[2], bilinear)
        self.up3 = Up(channels[2], channels[1], bilinear)
        self.up4 = Up(channels[1], channels[0], bilinear)
        self.outc = OutConv(channels[0], num_classes)

        self.classification_loss = LabelSmoothingFocalLoss(
            num_classes, need_one_hot=True, gamma=2, alpha=0.25, smoothing=0.1
        )
        self.iou_loss = DiceLoss(
            mode="binary", log_loss=True, from_logits=True, smooth=0.1
        )

        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x1, x2, x3, x4, x5 = x
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def loss(self, logits, mask):
        loss_cls = self.classification_loss(logits, mask)
        loss_iou = self.iou_loss(logits, mask)
        loss = loss_cls * 1.0 + loss_iou * 0.1
        return (
            loss,
            {"loss_cls": loss_cls, "loss_iou": loss_iou},
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(
                in_channels + out_channels, out_channels, in_channels // 2
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels // 2 + out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):
        x = self.conv(x)
        return self.up(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
