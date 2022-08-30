from functools import partial
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

from timm.models.layers.drop import DropBlock2d, DropPath

from model.loss import LabelSmoothingFocalLoss, RegressionLossWithMask, IoULossWithMask
from consts import *


class CenterNet(nn.Module):
    def __init__(
        self,
        num_classes,
        input_channels,
        act_layer=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        drop_block_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.classification_threshold = MODEL_CLASSIFICATION_THRESHOLD
        self.image_size = DATA_IMAGE_SIZE_DETECTION
        self.stride = 4
        self.output_size = int(self.image_size / self.stride)
        self.channels = input_channels
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

        self.decoder = self._make_decoder(
            num_layers=3, channels_list=[128, 64, 32], kernels_list=[4, 4, 4],
        )

        self.classification_head = self._make_head(
            input_channels=32, output_channels=num_classes
        )
        self.offset_head = self._make_head(input_channels=32, output_channels=2)
        self.size_head = self._make_head(input_channels=32, output_channels=2)

        self.classification_loss = LabelSmoothingFocalLoss(
            num_classes, need_one_hot=False, gamma=2, alpha=0.9, smoothing=0.0
        )
        self.regression_loss = RegressionLossWithMask(smooth=True)
        self.bbox_loss = IoULossWithMask(CIoU=True)

        self.initialize()

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
            layers.append(self.drop_block())
            layers.append(self.act_layer(inplace=True))
            layers.append(self.drop_path())
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
        if logits["cls"].is_cuda:
            device = logits["cls"].get_device()
            targets["cls"] = targets["cls"].to(device)
            targets["offset"] = targets["offset"].to(device)
            targets["size"] = targets["size"].to(device)
            targets["mask"] = targets["mask"].to(device)
        loss_cls = self.classification_loss(logits["cls"], targets["cls"])
        loss_offset = self.regression_loss(
            logits["offset"], targets["offset"], targets["mask"]
        )
        loss_size = self.regression_loss(
            logits["size"], targets["size"], targets["mask"]
        )
        pred_coord = self.offset_to_coord(logits["offset"])
        labels_coord = self.offset_to_coord(targets["offset"])
        loss_bbox = self.bbox_loss(
            torch.cat([pred_coord, logits["size"]], dim=1),
            torch.cat([labels_coord, targets["size"]], dim=1),
            targets["mask"],
        )
        loss = loss_cls * 100.0 + loss_offset * 0.1 + loss_size * 0.01 + loss_bbox * 0.1
        return (
            loss,
            {
                "loss_cls": loss_cls,
                "loss_offset": loss_offset,
                "loss_size": loss_size,
                "loss_bbox": loss_bbox,
            },
        )

    def offset_to_coord(self, pred):
        b, c, output_w, output_h = pred.shape
        xv, yv = torch.meshgrid(torch.arange(0, output_w), torch.arange(0, output_h))
        coords = torch.stack([xv, yv]).repeat(b, 1, 1, 1)
        if pred.is_cuda:
            device = pred.get_device()
            coords = coords.to(device)
        else:
            coords = coords.cpu()
        return pred + coords

    def gaussian_radius(self, det_size, min_overlap=0.7):
        width, height = det_size
        a1 = 1
        b1 = width + height
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2
        a2 = 4
        b2 = 2 * (width + height)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2
        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (width + height)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
        return min(r1, r2, r3)

    def gaussian2D(self, shape, sigma=1):
        m, n = [(ss - 1.0) / 2.0 for ss in shape]
        x, y = np.ogrid[-m : m + 1, -n : n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def draw_gaussian(self, heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = self.gaussian2D((diameter, diameter), sigma=diameter / 6)
        x, y = int(center[0]), int(center[1])
        width, height = heatmap.shape[0:2]
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)
        masked_heatmap = heatmap[x - left : x + right, y - top : y + bottom]
        masked_gaussian = gaussian[
            radius - left : radius + right, radius - top : radius + bottom
        ]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            masked_heatmap = np.maximum(
                masked_heatmap, np.expand_dims(masked_gaussian * k, axis=2)
            )
            heatmap[x - left : x + right, y - top : y + bottom] = masked_heatmap
        return heatmap

    def preprocess_targets(self, labels, labels_count):
        batch_size = labels.size()[0]
        target_cls = np.zeros(
            (batch_size, self.output_size, self.output_size, self.num_classes),
            dtype=np.float32,
        )
        target_offset = np.zeros(
            (batch_size, self.output_size, self.output_size, 2), dtype=np.float32
        )
        target_size = np.zeros(
            (batch_size, self.output_size, self.output_size, 2), dtype=np.float32
        )
        target_regression_mask = np.zeros(
            (batch_size, self.output_size, self.output_size), dtype=np.float32
        )

        boxes = np.array(labels[:, :, :4].cpu(), dtype=np.float32)
        boxes = np.clip(
            boxes / self.image_size * self.output_size, 0, self.output_size - 1,
        )

        for i in range(batch_size):
            for j in range(labels_count[i]):
                box = boxes[i, j].copy()
                cls_id = int(labels[i, j, -1])
                w, h = box[2] - box[0], box[3] - box[1]
                if w > 0 and h > 0:
                    radius = self.gaussian_radius((math.ceil(w), math.ceil(h)))
                    radius = max(0, int(radius))
                    center = np.array(
                        [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32
                    )
                    center_int = center.astype(np.int32)
                    target_cls[:, :, :, cls_id] = self.draw_gaussian(
                        target_cls[:, :, :, cls_id], center_int, radius
                    )
                    target_offset[i, center_int[0], center_int[1]] = center - center_int
                    target_size[i, center_int[0], center_int[1]] = 1.0 * w, 1.0 * h
                    target_regression_mask[i, center_int[0], center_int[1]] = 1
        target_cls = torch.tensor(np.transpose(target_cls, (0, 3, 1, 2)))
        target_offset = torch.tensor(np.transpose(target_offset, (0, 3, 1, 2)))
        target_size = torch.tensor(np.transpose(target_size, (0, 3, 1, 2)))
        target_regression_mask = torch.tensor(target_regression_mask)

        return {
            "cls": target_cls,
            "offset": target_offset,
            "size": target_size,
            "mask": target_regression_mask,
        }

    def postprocess_predictions(self, logits):
        pred_cls = logits["cls"]
        pred_offset = logits["offset"]
        pred_size = logits["size"]

        kernel = 3
        pad = (kernel - 1) // 2
        hmax = F.max_pool2d(pred_cls, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == pred_cls).float()
        pred_cls = pred_cls * keep

        b, c, output_w, output_h = pred_cls.shape
        detects = []
        for batch in range(b):
            # heat_map = F.softmax(pred_cls[batch], dim=0)
            heat_map = torch.sigmoid(pred_cls[batch])
            heat_map = heat_map.permute(1, 2, 0).view([-1, c])
            pred_off = pred_offset[batch].permute(1, 2, 0).view([-1, 2])
            pred_wh = pred_size[batch].permute(1, 2, 0).view([-1, 2])

            xv, yv = torch.meshgrid(
                torch.arange(0, output_w), torch.arange(0, output_h)
            )
            xv, yv = xv.flatten().float(), yv.flatten().float()
            if pred_cls.is_cuda:
                device = pred_cls.get_device()
                xv = xv.to(device)
                yv = yv.to(device)

            class_conf, class_pred = torch.max(heat_map, dim=-1)
            mask = class_conf > self.classification_threshold

            pred_offset_mask = pred_off[mask]
            pred_wh_mask = pred_wh[mask]
            if len(pred_wh_mask) == 0:
                detects.append([])
                continue

            xv_mask = torch.unsqueeze(xv[mask] + pred_offset_mask[..., 0], -1)
            yv_mask = torch.unsqueeze(yv[mask] + pred_offset_mask[..., 1], -1)
            half_w, half_h = pred_wh_mask[..., 0:1] / 2, pred_wh_mask[..., 1:2] / 2
            bboxes = torch.cat(
                [
                    xv_mask - half_w,
                    yv_mask - half_h,
                    xv_mask + half_w,
                    yv_mask + half_h,
                ],
                dim=1,
            )
            bboxes[:, [0, 2]] /= output_w
            bboxes[:, [1, 3]] /= output_h
            detect = torch.cat(
                [
                    bboxes,
                    torch.unsqueeze(class_conf[mask], -1),
                    torch.unsqueeze(class_pred[mask], -1).float(),
                ],
                dim=-1,
            )
            detects.append(detect)

        output = [None for _ in range(len(detects))]

        for i, detections in enumerate(detects):
            if len(detections) == 0:
                continue
            unique_labels = detections[:, -1].unique()

            for c in unique_labels:
                detections_class = detections[detections[:, -1] == c]
                if MODEL_NMS:
                    keep = nms(
                        detections_class[:, :4],
                        detections_class[:, 4],
                        MODEL_NMS_THRESHOLD,
                    )
                    max_detections = detections_class[keep]
                else:
                    max_detections = detections_class
                output[i] = (
                    max_detections
                    if output[i] is None
                    else torch.cat((output[i], max_detections))
                )

            if output[i] is not None:
                output[i] = output[i].cpu().detach().numpy()
                box_xy, box_wh = (
                    (output[i][:, 0:2] + output[i][:, 2:4]) / 2,
                    output[i][:, 2:4] - output[i][:, 0:2],
                )

                image_shape = np.array(
                    [self.image_size, self.image_size, self.image_size, self.image_size]
                )

                box_mins = box_xy - (box_wh / 2.0)
                box_maxes = box_xy + (box_wh / 2.0)
                boxes = np.concatenate(
                    [
                        box_mins[..., 0:1],
                        box_mins[..., 1:2],
                        box_maxes[..., 0:1],
                        box_maxes[..., 1:2],
                    ],
                    axis=-1,
                )
                boxes *= image_shape
                output[i][:, :4] = boxes

        return output
