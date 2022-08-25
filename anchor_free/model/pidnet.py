from functools import partial

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers.drop import DropBlock2d, DropPath

from consts import *

bn_mom = 0.1
algc = False
balance_weights = [0.5, 0.5]
sb_weights = 0.5
y_k_size = 6
x_k_size = 6


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        no_act=False,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        drop_block=nn.Identity,
        drop_path=nn.Identity,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = norm_layer(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_layer(planes, momentum=bn_mom)
        self.act = act_layer(inplace=True)
        self.drop_block = drop_block()
        self.drop_path = drop_path()
        self.stride = stride
        self.downsample = downsample
        self.no_act = no_act

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.drop_block(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        if not self.no_act:
            out = self.act(out)
        out = self.drop_path(out)
        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        no_act=True,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        drop_block=nn.Identity,
        drop_path=nn.Identity,
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = norm_layer(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = norm_layer(planes * self.expansion, momentum=bn_mom)
        self.act = act_layer(inplace=True)
        self.drop_block = drop_block()
        self.drop_path = drop_path()
        self.stride = stride
        self.downsample = downsample
        self.no_act = no_act

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop_block(out)
        out = self.act(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        if not self.no_act:
            out = self.act(out)
        out = self.drop_path(out)
        return out


class PagFM(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        after_act=False,
        with_channel=False,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
    ):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_act = after_act
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            norm_layer(mid_channels),
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            norm_layer(mid_channels),
        )
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
                norm_layer(in_channels),
            )
        if after_act:
            self.act = act_layer(inplace=True)

    def forward(self, x, y):
        input_size = x.size()
        if self.after_act:
            y = self.act(y)
            x = self.act(x)
        y_q = self.f_y(y)
        y_q = F.interpolate(
            y_q,
            size=[input_size[2], input_size[3]],
            mode="bilinear",
            align_corners=algc,
        )
        x_k = self.f_x(x)
        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))
        y = F.interpolate(
            y, size=[input_size[2], input_size[3]], mode="bilinear", align_corners=algc
        )
        x = (1 - sim_map) * x + sim_map * y
        return x


class PAPPM(nn.Module):
    def __init__(
        self,
        inplanes,
        branch_planes,
        outplanes,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
    ):
        super(PAPPM, self).__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            norm_layer(inplanes, momentum=bn_mom),
            act_layer(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            norm_layer(inplanes, momentum=bn_mom),
            act_layer(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            norm_layer(inplanes, momentum=bn_mom),
            act_layer(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            norm_layer(inplanes, momentum=bn_mom),
            act_layer(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale0 = nn.Sequential(
            norm_layer(inplanes, momentum=bn_mom),
            act_layer(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale_process = nn.Sequential(
            norm_layer(branch_planes * 4, momentum=bn_mom),
            act_layer(inplace=True),
            nn.Conv2d(
                branch_planes * 4,
                branch_planes * 4,
                kernel_size=3,
                padding=1,
                groups=4,
                bias=False,
            ),
        )
        self.compression = nn.Sequential(
            norm_layer(branch_planes * 5, momentum=bn_mom),
            act_layer(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            norm_layer(inplanes, momentum=bn_mom),
            act_layer(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        scale_list = []
        x_ = self.scale0(x)
        scale_list.append(
            F.interpolate(
                self.scale1(x),
                size=[height, width],
                mode="bilinear",
                align_corners=algc,
            )
            + x_
        )
        scale_list.append(
            F.interpolate(
                self.scale2(x),
                size=[height, width],
                mode="bilinear",
                align_corners=algc,
            )
            + x_
        )
        scale_list.append(
            F.interpolate(
                self.scale3(x),
                size=[height, width],
                mode="bilinear",
                align_corners=algc,
            )
            + x_
        )
        scale_list.append(
            F.interpolate(
                self.scale4(x),
                size=[height, width],
                mode="bilinear",
                align_corners=algc,
            )
            + x_
        )
        scale_out = self.scale_process(torch.cat(scale_list, 1))
        out = self.compression(torch.cat([x_, scale_out], 1)) + self.shortcut(x)
        return out


class DAPPM(nn.Module):
    def __init__(
        self,
        inplanes,
        branch_planes,
        outplanes,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
    ):
        super(DAPPM, self).__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            norm_layer(inplanes, momentum=bn_mom),
            act_layer(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            norm_layer(inplanes, momentum=bn_mom),
            act_layer(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            norm_layer(inplanes, momentum=bn_mom),
            act_layer(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            norm_layer(inplanes, momentum=bn_mom),
            act_layer(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale0 = nn.Sequential(
            norm_layer(inplanes, momentum=bn_mom),
            act_layer(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.process1 = nn.Sequential(
            norm_layer(branch_planes, momentum=bn_mom),
            act_layer(inplace=True),
            nn.Conv2d(
                branch_planes, branch_planes, kernel_size=3, padding=1, bias=False
            ),
        )
        self.process2 = nn.Sequential(
            norm_layer(branch_planes, momentum=bn_mom),
            act_layer(inplace=True),
            nn.Conv2d(
                branch_planes, branch_planes, kernel_size=3, padding=1, bias=False
            ),
        )
        self.process3 = nn.Sequential(
            norm_layer(branch_planes, momentum=bn_mom),
            act_layer(inplace=True),
            nn.Conv2d(
                branch_planes, branch_planes, kernel_size=3, padding=1, bias=False
            ),
        )
        self.process4 = nn.Sequential(
            norm_layer(branch_planes, momentum=bn_mom),
            act_layer(inplace=True),
            nn.Conv2d(
                branch_planes, branch_planes, kernel_size=3, padding=1, bias=False
            ),
        )
        self.compression = nn.Sequential(
            norm_layer(branch_planes * 5, momentum=bn_mom),
            act_layer(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            norm_layer(inplanes, momentum=bn_mom),
            act_layer(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        x_list = []
        x_list.append(self.scale0(x))
        x_list.append(
            self.process1(
                (
                    F.interpolate(
                        self.scale1(x),
                        size=[height, width],
                        mode="bilinear",
                        align_corners=algc,
                    )
                    + x_list[0]
                )
            )
        )
        x_list.append(
            (
                self.process2(
                    (
                        F.interpolate(
                            self.scale2(x),
                            size=[height, width],
                            mode="bilinear",
                            align_corners=algc,
                        )
                        + x_list[1]
                    )
                )
            )
        )
        x_list.append(
            self.process3(
                (
                    F.interpolate(
                        self.scale3(x),
                        size=[height, width],
                        mode="bilinear",
                        align_corners=algc,
                    )
                    + x_list[2]
                )
            )
        )
        x_list.append(
            self.process4(
                (
                    F.interpolate(
                        self.scale4(x),
                        size=[height, width],
                        mode="bilinear",
                        align_corners=algc,
                    )
                    + x_list[3]
                )
            )
        )
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out


class Light_Bag(nn.Module):
    def __init__(
        self, in_channels, out_channels, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d
    ):
        super(Light_Bag, self).__init__()
        self.conv_p = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
        )

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        p_add = self.conv_p((1 - edge_att) * i + p)
        i_add = self.conv_i(i + edge_att * p)
        return p_add + i_add


class Bag(nn.Module):
    def __init__(
        self, in_channels, out_channels, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d
    ):
        super(Bag, self).__init__()
        self.conv = nn.Sequential(
            norm_layer(in_channels),
            act_layer(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return self.conv(edge_att * p + (1 - edge_att) * i)


class SegmentHead(nn.Module):
    def __init__(
        self,
        inplanes,
        interplanes,
        outplanes,
        scale_factor=None,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
    ):
        super(SegmentHead, self).__init__()
        self.bn1 = norm_layer(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(
            inplanes, interplanes, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = norm_layer(interplanes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(
            interplanes, outplanes, kernel_size=1, padding=0, bias=True
        )
        self.act = act_layer(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv1(self.act(self.bn1(x)))
        out = self.conv2(self.act(self.bn2(x)))
        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(
                out, size=[height, width], mode="bilinear", align_corners=algc
            )
        return out


class PIDNet(nn.Module):
    def __init__(
        self,
        m=2,
        num_classes=2,
        planes=64,
        ppm_planes=96,
        head_planes=128,
        augment=True,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        drop_block_rate=0.0,
        drop_path_rate=0.0,
    ):
        super(PIDNet, self).__init__()
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
        self.params = {"act_layer": act_layer, "norm_layer": norm_layer}
        self.block_params = {
            "act_layer": act_layer,
            "norm_layer": norm_layer,
            "drop_block": self.drop_block,
            "drop_path": self.drop_path,
        }
        self.augment = augment
        self.act = act_layer(inplace=True)
        self.compression3 = nn.Sequential(
            nn.Conv2d(planes[2], planes[1], kernel_size=1, bias=False),
            norm_layer(planes[1], momentum=bn_mom),
        )
        self.compression4 = nn.Sequential(
            nn.Conv2d(planes[3], planes[1], kernel_size=1, bias=False),
            norm_layer(planes[1], momentum=bn_mom),
        )
        self.pag3 = PagFM(planes[1], planes[1], **self.params)
        self.pag4 = PagFM(planes[1], planes[1], **self.params)
        self.layer3_ = self._make_layer(BasicBlock, planes[1], planes[1], m)
        self.layer4_ = self._make_layer(BasicBlock, planes[1], planes[1], m)
        self.layer5_ = self._make_single_layer(
            Bottleneck, planes[1], int(planes[1] / 2)
        )
        if m == 2:
            self.layer3_d = self._make_single_layer(BasicBlock, planes[1], planes[0])
            self.layer4_d = self._make_single_layer(
                Bottleneck, planes[0], int(planes[1] / 2)
            )
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes[2], planes[0], kernel_size=3, padding=1, bias=False),
                norm_layer(planes[0], momentum=bn_mom),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes[3], planes[1], kernel_size=3, padding=1, bias=False),
                norm_layer(planes[1], momentum=bn_mom),
            )
            self.spp = PAPPM(planes[4], ppm_planes, planes[1], **self.params)
            self.dfm = Light_Bag(planes[1], planes[1], **self.params)
        else:
            self.layer3_d = self._make_single_layer(BasicBlock, planes[1], planes[1])
            self.layer4_d = self._make_single_layer(BasicBlock, planes[1], planes[1])
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes[2], planes[1], kernel_size=3, padding=1, bias=False),
                norm_layer(planes[1], momentum=bn_mom),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes[3], planes[1], kernel_size=3, padding=1, bias=False),
                norm_layer(planes[1], momentum=bn_mom),
            )
            self.spp = DAPPM(planes[4], ppm_planes, planes[1], **self.params)
            self.dfm = Bag(planes[1], planes[1], **self.params)
        self.layer5_d = self._make_single_layer(
            Bottleneck, planes[1], int(planes[1] / 2)
        )
        if self.augment:
            self.seghead_p = SegmentHead(
                planes[1], head_planes, num_classes, **self.params
            )
            self.seghead_d = SegmentHead(planes[1], planes[0], 1, **self.params)
        self.final_layer = SegmentHead(
            planes[1], head_planes, num_classes, **self.params
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, norm_layer):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                self.norm_layer(planes * block.expansion, momentum=bn_mom),
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample, **self.block_params))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(
                    block(inplanes, planes, stride=1, no_act=True, **self.block_params)
                )
            else:
                layers.append(
                    block(inplanes, planes, stride=1, no_act=False, **self.block_params)
                )
            inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                self.norm_layer(planes * block.expansion, momentum=bn_mom),
            )
        layer = block(
            inplanes, planes, stride, downsample, no_act=True, **self.block_params
        )
        return layer

    def forward(self, x):
        x1, x2, x3, x4, x5 = x
        width_output = DATA_IMAGE_SIZE_SEGMENTATION[0] // 4
        height_output = DATA_IMAGE_SIZE_SEGMENTATION[1] // 4
        x_ = self.layer3_(x2)
        x_d = self.layer3_d(x2)
        x_ = self.pag3(x_, self.compression3(x3))
        x_d = x_d + F.interpolate(
            self.diff3(x3),
            size=[height_output, width_output],
            mode="bilinear",
            align_corners=algc,
        )
        if self.augment:
            temp_p = x_
        x_ = self.layer4_(self.act(x_))
        x_d = self.layer4_d(self.act(x_d))
        x_ = self.pag4(x_, self.compression4(x4))
        x_d = x_d + F.interpolate(
            self.diff4(x4),
            size=[height_output, width_output],
            mode="bilinear",
            align_corners=algc,
        )
        if self.augment:
            temp_d = x_d
        x_ = self.layer5_(self.act(x_))
        x_d = self.layer5_d(self.act(x_d))
        x = F.interpolate(
            self.spp(x5),
            size=[height_output, width_output],
            mode="bilinear",
            align_corners=algc,
        )
        x_ = self.final_layer(self.dfm(x_, x, x_d))
        if self.augment:
            x_extra_p = self.seghead_p(temp_p)
            x_extra_d = self.seghead_d(temp_d)
            return [x_extra_p, x_, x_extra_d]
        else:
            return x_


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

    def _forward(self, score, target):
        loss = self.criterion(score, target)
        return loss

    def forward(self, score, target):
        if len(balance_weights) == len(score):
            return sum(
                [w * self._forward(x, target) for (w, x) in zip(balance_weights, score)]
            )
        elif len(score) == 1:
            return sb_weights * self._forward(score[0], target)
        else:
            raise ValueError("lengths of prediction and target are not identical!")


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7, min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_label, reduction="none"
        )

    def _ce_forward(self, score, target):
        loss = self.criterion(score, target)
        return loss

    def _ohem_forward(self, score, target, **kwargs):
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label
        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)
        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):
        if not (isinstance(score, list) or isinstance(score, tuple)):
            score = [score]
        if len(balance_weights) == len(score):
            functions = [self._ce_forward] * (len(balance_weights) - 1) + [
                self._ohem_forward
            ]
            return sum(
                [
                    w * func(x, target)
                    for (w, x, func) in zip(balance_weights, score, functions)
                ]
            )
        elif len(score) == 1:
            return sb_weights * self._ohem_forward(score[0], target)
        else:
            raise ValueError("lengths of prediction and target are not identical!")


class BoundaryLoss(nn.Module):
    def __init__(self, coeff_bce=20.0):
        super(BoundaryLoss, self).__init__()
        self.coeff_bce = coeff_bce

    def forward(self, bd_pre, bd_gt):
        bce_loss = self.coeff_bce * BoundaryLoss.weighted_bce(bd_pre, bd_gt)
        loss = bce_loss
        return loss

    @staticmethod
    def weighted_bce(bd_pre, target):
        log_p = bd_pre.permute(0, 2, 3, 1).contiguous().view(1, -1)
        target_t = target.view(1, -1)
        pos_index = target_t == 1
        neg_index = target_t == 0
        weight = torch.zeros_like(log_p).type(target.type())
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num
        loss = F.binary_cross_entropy_with_logits(
            log_p, target_t, weight, reduction="mean"
        )
        return loss


class FullModel(nn.Module):
    def __init__(self, model, sem_loss, bd_loss):
        super(FullModel, self).__init__()
        self.model = model
        self.sem_loss = sem_loss
        self.bd_loss = bd_loss

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc

    def forward(self, inputs, *args, **kwargs):
        outputs = self.model(inputs, *args, **kwargs)
        return outputs

    @staticmethod
    def get_edge(
        label, edge_pad=True, edge_size=4,
    ):
        label = np.uint8(label)
        edge = cv2.Canny(label, 0.1, 0.2)
        kernel = np.ones((edge_size, edge_size), np.uint8)
        if edge_pad:
            edge = edge[y_k_size:-y_k_size, x_k_size:-x_k_size]
            edge = np.pad(
                edge, ((y_k_size, y_k_size), (x_k_size, x_k_size)), mode="constant"
            )
        edge = (cv2.dilate(edge, kernel, iterations=1) > 50) * 1.0
        return edge

    def loss(self, outputs, labels):
        edges = []
        for label in labels:
            edge = FullModel.get_edge(label.cpu().detach().numpy())
            edges.append(edge)
        edges = np.stack(edges)
        edges = torch.Tensor(edges).to(labels.device)
        h, w = labels.size(1), labels.size(2)
        ph, pw = outputs[0].size(2), outputs[0].size(3)
        if ph != h or pw != w:
            for i in range(len(outputs)):
                outputs[i] = F.interpolate(
                    outputs[i], size=(h, w), mode="bilinear", align_corners=algc,
                )
        acc = self.pixel_acc(outputs[-2], labels)
        loss_s = self.sem_loss(outputs[:-1], labels)
        loss_b = self.bd_loss(outputs[-1], edges)
        filler = torch.ones_like(labels) * -1
        bd_label = torch.where(F.sigmoid(outputs[-1][:, 0, :, :]) > 0.8, labels, filler)
        loss_sb = self.sem_loss(outputs[-2], bd_label)
        loss = loss_s + loss_b + loss_sb
        return (
            torch.unsqueeze(loss, 0).mean(),
            {"acc": acc.mean(), "loss_s": loss_s.mean(), "loss_b": loss_b.mean()},
        )

    def postprocess_predictions(self, logits):
        pred = F.interpolate(
            input=logits[1], scale_factor=4, mode="bilinear", align_corners=algc
        )
        return pred
