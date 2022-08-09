from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers.drop import DropBlock2d, DropPath
from segmentation_models_pytorch.losses import DiceLoss

from model.loss import LabelSmoothingFocalLoss
from consts import *


class UNet(nn.Module):
    def __init__(
        self,
        num_classes,
        bilinear,
        channels,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        drop_block_rate=0.0,
        drop_path_rate=0.0,
    ):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.drop_block = (
            partial(
                DropBlock2d,
                drop_prob=drop_block_rate,
                block_size=3,
                gamma_scale=1.0,
                fast=True,
            )
            if drop_block_rate > 0.0
            else nn.Identity
        )
        self.drop_path = (
            partial(DropPath, drop_prob=drop_path_rate)
            if drop_path_rate > 0.0
            else nn.Identity
        )
        up_args = {
            "act_layer": act_layer,
            "norm_layer": norm_layer,
            "drop_block": self.drop_block,
            "drop_path": self.drop_path,
        }

        self.up1 = Up(channels[4], channels[3], bilinear, **up_args)
        self.up2 = Up(channels[3], channels[2], bilinear, **up_args)
        self.up3 = Up(channels[2], channels[1], bilinear, **up_args)
        self.up4 = Up(channels[1], channels[0], bilinear, **up_args)
        self.outc = OutConv(channels[0], num_classes)

        self.classification_loss = LabelSmoothingFocalLoss(
            num_classes, need_one_hot=True, gamma=2, alpha=0.25, smoothing=0.1
        )
        self.iou_loss = DiceLoss(
            mode="multiclass", log_loss=True, from_logits=True, smooth=0.1
        )

        self.initialize()

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
        loss = loss_cls * 1.0 + loss_iou * 1.0
        return (
            loss,
            {"loss_cls": loss_cls, "loss_iou": loss_iou},
        )

    def postprocess_predictions(self, logits):
        return logits


class Up(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bilinear=True,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        drop_block=nn.Identity,
        drop_path=nn.Identity,
    ):
        super().__init__()
        dc_args = {
            "act_layer": act_layer,
            "norm_layer": norm_layer,
            "drop_block": drop_block,
            "drop_path": drop_path,
        }
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(
                in_channels + out_channels, out_channels, in_channels // 2, **dc_args,
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(
                in_channels // 2 + out_channels, out_channels, **dc_args,
            )

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
    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        drop_block=nn.Identity,
        drop_path=nn.Identity,
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(mid_channels),
            drop_block(),
            act_layer(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            act_layer(inplace=True),
            drop_path(),
        )

    def forward(self, x):
        return self.double_conv(x)
