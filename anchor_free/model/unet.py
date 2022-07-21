import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.losses import FocalLoss


class UNet(nn.Module):
    def __init__(self, num_classes, bilinear, channels):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.loss_function = FocalLoss(mode="multiclass")

        self.up1 = Up(channels[4], channels[3], bilinear)
        self.up2 = Up(channels[3], channels[2], bilinear)
        self.up3 = Up(channels[2], channels[1], bilinear)
        self.up4 = Up(channels[1], channels[0], bilinear)
        self.outc = OutConv(channels[0], num_classes)

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
        return focal_loss_with_logits(logits, mask)


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
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.up(x)
        diffY = 825 - x.size()[2]
        diffX = 550 - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        # if not mid_channels:
        mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


def focal_loss_with_logits(
    output,
    target,
    gamma=2.0,
    alpha=0.25,
    reduction="mean",
    normalized=False,
    reduced_threshold=None,
    eps=1e-6,
):
    target = target.float()
    logpt = F.binary_cross_entropy_with_logits(output, target, reduction="none")
    pt = torch.exp(-logpt)
    if reduced_threshold is None:
        focal_term = (1.0 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term[pt < reduced_threshold] = 1
    loss = focal_term * logpt
    if alpha is not None:
        loss *= alpha * target + (1 - alpha) * (1 - target)
    if normalized:
        norm_factor = focal_term.sum().clamp_min(eps)
        loss /= norm_factor
    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)
    return loss
