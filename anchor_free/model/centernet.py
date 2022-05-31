import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss, nms

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
        self.classification_threshold = MODEL_CLASSIFICATION_THRESHOLD
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

        boxes = np.array(labels[:, :, :4], dtype=np.float32)
        boxes = np.clip(
            boxes / self.image_size * self.output_size,
            0,
            self.output_size - 1,
        )

        for i in range(batch_size):
            for j in range(labels_count[i]):
                box = boxes[i, j].copy()
                cls_id = int(labels[i, j, -1])
                w, h = box[2] - box[0], box[3] - box[1]
                if w > 0 and h > 0:
                    # radius = gaussian_radius((math.ceil(w), math.ceil(h)))
                    # radius = max(0, int(radius))
                    center = np.array(
                        [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32
                    )
                    center_int = center.astype(np.int32)
                    # target_cls[:, :, :, cls_id] = draw_gaussian(
                    #    target_cls[:, :, :, cls_id], center_int, radius
                    # )
                    target_cls[i, center_int[0], center_int[1], cls_id] = 1.0
                    target_offset[i, center_int[0], center_int[1]] = center - center_int
                    target_size[i, center_int[0], center_int[1]] = 1.0 * w, 1.0 * h
                    target_regression_mask[i, center_int[0], center_int[1]] = 1
        target_cls = np.transpose(target_cls, (0, 3, 1, 2))
        target_offset = np.transpose(target_offset, (0, 3, 1, 2))
        target_size = np.transpose(target_size, (0, 3, 1, 2))

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

        # kernel = 3
        # pad = (kernel - 1) // 2
        # hmax = F.max_pool2d(pred_cls, (kernel, kernel), stride=1, padding=pad)
        # keep = (hmax == pred_cls).float()
        # pred_cls = pred_cls * keep

        b, c, output_w, output_h = pred_cls.shape
        detects = []
        for batch in range(b):
            heat_map = torch.sigmoid(pred_cls[batch])
            heat_map = heat_map.permute(1, 2, 0).view([-1, c])
            pred_off = pred_offset[batch].permute(1, 2, 0).view([-1, 2])
            pred_wh = pred_size[batch].permute(1, 2, 0).view([-1, 2])

            xv, yv = torch.meshgrid(
                torch.arange(0, output_w), torch.arange(0, output_h)
            )
            xv, yv = xv.flatten().float(), yv.flatten().float()

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
                # if need_nms:
                #    keep = nms(
                #        detections_class[:, :4], detections_class[:, 4], nms_thres
                #    )
                #    max_detections = detections_class[keep]
                # else:
                #    max_detections = detections_class
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

                input_shape = np.array([self.output_size, self.output_size])
                image_shape = np.array(self.image_size, self.image_size)

                # if letterbox_image:
                #    new_shape = np.round(
                #        image_shape * np.min(input_shape / image_shape)
                #    )
                #    offset = (input_shape - new_shape) / 2.0 / input_shape
                #    scale = input_shape / new_shape
                #    box_xy = (box_xy - offset) * scale
                #    box_wh *= scale

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
