import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import ops

from model.fpn import FPN, FPNTopP6P7
from consts import *


INF = 100000000
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


def init_conv_std(module, std=0.01):
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class Scale(nn.Module):
    def __init__(self, init=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([init], dtype=torch.float32))

    def forward(self, input):
        return input * self.scale


class IOULoss(nn.Module):
    def __init__(self, loc_loss_type):
        super().__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, out, target, weight=None):
        pred_left, pred_top, pred_right, pred_bottom = out.unbind(1)
        target_left, target_top, target_right, target_bottom = target.unbind(1)

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(
            pred_right, target_right
        )
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(
            pred_top, target_top
        )

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        ious = (area_intersect + 1) / (area_union + 1)

        if self.loc_loss_type == "iou":
            loss = -torch.log(ious)
        elif self.loc_loss_type == "giou":
            g_w_intersect = torch.max(pred_left, target_left) + torch.max(
                pred_right, target_right
            )
            g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(
                pred_top, target_top
            )
            g_intersect = g_w_intersect * g_h_intersect + 1e-7
            gious = ious - (g_intersect - area_union) / g_intersect
            loss = 1 - gious

        if weight is not None and weight.sum() > 0:
            return (loss * weight).sum() / weight.sum()
        else:
            return loss.mean()


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, out, target):
        n_class = out.shape[1]
        class_ids = torch.arange(
            1, n_class + 1, dtype=target.dtype, device=target.device
        ).unsqueeze(0)

        t = target.unsqueeze(1)
        p = torch.sigmoid(out)

        gamma = self.gamma
        alpha = self.alpha

        term1 = (1 - p) ** gamma * torch.log(p)
        term2 = p**gamma * torch.log(1 - p)

        loss = (
            -(t == class_ids).float() * alpha * term1
            - ((t != class_ids) * (t >= 0)).float() * (1 - alpha) * term2
        )

        return loss.sum()


class FCOSLoss(nn.Module):
    def __init__(
        self, sizes, gamma, alpha, iou_loss_type, center_sample, fpn_strides, pos_radius
    ):
        super().__init__()
        self.sizes = sizes
        self.cls_loss = SigmoidFocalLoss(gamma, alpha)
        self.box_loss = IOULoss(iou_loss_type)
        self.center_loss = nn.BCEWithLogitsLoss()
        self.center_sample = center_sample
        self.strides = fpn_strides
        self.radius = pos_radius

    def prepare_target(self, points, targets):
        ex_size_of_interest = []

        for i, point_per_level in enumerate(points):
            size_of_interest_per_level = point_per_level.new_tensor(self.sizes[i])
            ex_size_of_interest.append(
                size_of_interest_per_level[None].expand(len(point_per_level), -1)
            )

        ex_size_of_interest = torch.cat(ex_size_of_interest, 0)
        n_point_per_level = [len(point_per_level) for point_per_level in points]
        point_all = torch.cat(points, dim=0)
        label, box_target = self.compute_target_for_location(
            point_all, targets, ex_size_of_interest, n_point_per_level
        )

        for i in range(len(label)):
            label[i] = torch.split(label[i], n_point_per_level, 0)
            box_target[i] = torch.split(box_target[i], n_point_per_level, 0)

        label_level_first = []
        box_target_level_first = []

        for level in range(len(points)):
            label_level_first.append(
                torch.cat([label_per_img[level] for label_per_img in label], 0)
            )
            box_target_level_first.append(
                torch.cat(
                    [box_target_per_img[level] for box_target_per_img in box_target], 0
                )
            )

        return label_level_first, box_target_level_first

    def get_sample_region(self, gt, strides, n_point_per_level, xs, ys, radius=1):
        n_gt = gt.shape[0]
        n_loc = len(xs)
        gt = gt[None].expand(n_loc, n_gt, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2

        if center_x[..., 0].sum() == 0:
            return xs.new_zeros(xs.shape, dtype=torch.uint8)

        begin = 0
        center_gt = gt.new_zeros(gt.shape)

        for level, n_p in enumerate(n_point_per_level):
            end = begin + n_p
            stride = strides[level] * radius
            x_min = center_x[begin:end] - stride
            y_min = center_y[begin:end] - stride
            x_max = center_x[begin:end] + stride
            y_max = center_y[begin:end] + stride

            center_gt[begin:end, :, 0] = torch.where(
                x_min > gt[begin:end, :, 0], x_min, gt[begin:end, :, 0]
            )
            center_gt[begin:end, :, 1] = torch.where(
                y_min > gt[begin:end, :, 1], y_min, gt[begin:end, :, 1]
            )
            center_gt[begin:end, :, 2] = torch.where(
                x_max > gt[begin:end, :, 2], gt[begin:end, :, 2], x_max
            )
            center_gt[begin:end, :, 3] = torch.where(
                y_max > gt[begin:end, :, 3], gt[begin:end, :, 3], y_max
            )

            begin = end

        left = xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - xs[:, None]
        top = ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        is_in_boxes = center_bbox.min(-1)[0] > 0

        return is_in_boxes

    def compute_target_for_location(
        self, locations, targets, sizes_of_interest, n_point_per_level
    ):
        labels = []
        box_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        for i in range(len(targets)):
            targets_per_img = targets[i]
            assert targets_per_img["mode"] == "xyxy"
            bboxes = targets_per_img["box"]
            labels_per_img = targets_per_img["labels"]
            area = targets_per_img["area"]
            print(locations.size(), bboxes.size())

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]

            box_targets_per_img = torch.stack([l, t, r, b], 2)

            if self.center_sample:
                is_in_boxes = self.get_sample_region(
                    bboxes, self.strides, n_point_per_level, xs, ys, radius=self.radius
                )
            else:
                is_in_boxes = box_targets_per_img.min(2)[0] > 0

            max_box_targets_per_img = box_targets_per_img.max(2)[0]

            is_cared_in_level = (
                max_box_targets_per_img >= sizes_of_interest[:, [0]]
            ) & (max_box_targets_per_img <= sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_level == 0] = INF

            locations_to_min_area, locations_to_gt_id = locations_to_gt_area.min(1)

            box_targets_per_img = box_targets_per_img[
                range(len(locations)), locations_to_gt_id
            ]
            labels_per_img = labels_per_img[locations_to_gt_id]
            labels_per_img[locations_to_min_area == INF] = 0

            labels.append(labels_per_img)
            box_targets.append(box_targets_per_img)

        return labels, box_targets

    def compute_centerness_targets(self, box_targets):
        left_right = box_targets[:, [0, 2]]
        top_bottom = box_targets[:, [1, 3]]
        centerness = (left_right.min(-1)[0] / left_right.max(-1)[0]) * (
            top_bottom.min(-1)[0] / top_bottom.max(-1)[0]
        )
        return torch.sqrt(centerness)

    def forward(self, locations, cls_pred, box_pred, center_pred, targets):
        batch = cls_pred[0].shape[0]
        n_class = cls_pred[0].shape[1]

        labels, box_targets = self.prepare_target(locations, targets)

        cls_flat = []
        box_flat = []
        center_flat = []
        labels_flat = []
        box_targets_flat = []

        for i in range(len(labels)):
            cls_flat.append(cls_pred[i].permute(0, 2, 3, 1).reshape(-1, n_class))
            box_flat.append(box_pred[i].permute(0, 2, 3, 1).reshape(-1, 4))
            center_flat.append(center_pred[i].permute(0, 2, 3, 1).reshape(-1))
            labels_flat.append(labels[i].reshape(-1))
            box_targets_flat.append(box_targets[i].reshape(-1, 4))

        cls_flat = torch.cat(cls_flat, 0)
        box_flat = torch.cat(box_flat, 0)
        center_flat = torch.cat(center_flat, 0)

        labels_flat = torch.cat(labels_flat, 0)
        box_targets_flat = torch.cat(box_targets_flat, 0)

        pos_id = torch.nonzero(labels_flat > 0).squeeze(1)

        cls_loss = self.cls_loss(cls_flat, labels_flat.int()) / (pos_id.numel() + batch)

        box_flat = box_flat[pos_id]
        center_flat = center_flat[pos_id]

        box_targets_flat = box_targets_flat[pos_id]

        if pos_id.numel() > 0:
            center_targets = self.compute_centerness_targets(box_targets_flat)
            box_loss = self.box_loss(box_flat, box_targets_flat, center_targets)
            center_loss = self.center_loss(center_flat, center_targets)
        else:
            box_loss = box_flat.sum()
            center_loss = center_flat.sum()

        return cls_loss, box_loss, center_loss


class FCOSPostprocessor(nn.Module):
    def __init__(self, threshold, top_n, nms_threshold, post_top_n, min_size, n_class):
        super().__init__()

        self.threshold = threshold
        self.top_n = top_n
        self.nms_threshold = nms_threshold
        self.post_top_n = post_top_n
        self.min_size = min_size
        self.n_class = n_class

    def forward_single_feature_map(
        self, location, cls_pred, box_pred, center_pred, image_sizes
    ):
        batch, channel, height, width = cls_pred.shape

        cls_pred = cls_pred.view(batch, channel, height, width).permute(0, 2, 3, 1)
        cls_pred = cls_pred.reshape(batch, -1, channel).sigmoid()

        box_pred = box_pred.view(batch, 4, height, width).permute(0, 2, 3, 1)
        box_pred = box_pred.reshape(batch, -1, 4)

        center_pred = center_pred.view(batch, 1, height, width).permute(0, 2, 3, 1)
        center_pred = center_pred.reshape(batch, -1).sigmoid()

        candid_ids = cls_pred > self.threshold

        top_ns = candid_ids.reshape(batch, -1).sum(1)
        top_ns = top_ns.clamp(max=self.top_n)

        cls_pred = cls_pred * center_pred[:, :, None]

        results = []

        for i in range(batch):
            cls_p = cls_pred[i]
            candid_id = candid_ids[i]
            cls_p = cls_p[candid_id]
            candid_nonzero = candid_id.nonzero()
            box_loc = candid_nonzero[:, 0]
            class_id = candid_nonzero[:, 1] + 1

            box_p = box_pred[i]
            box_p = box_p[box_loc]
            loc = location[box_loc]

            top_n = top_ns[i]

            if candid_id.sum().item() > top_n.item():
                cls_p, top_k_id = cls_p.topk(top_n, sorted=False)
                class_id = class_id[top_k_id]
                box_p = box_p[top_k_id]
                loc = loc[top_k_id]

            detections = torch.stack(
                [
                    loc[:, 0] - box_p[:, 0],
                    loc[:, 1] - box_p[:, 1],
                    loc[:, 0] + box_p[:, 2],
                    loc[:, 1] + box_p[:, 3],
                ],
                1,
            )

            height, width = image_sizes  # [i]

            boxlist = BoxList(detections, (int(width), int(height)), mode="xyxy")
            boxlist.fields["labels"] = class_id
            boxlist.fields["scores"] = torch.sqrt(cls_p)
            boxlist = boxlist.clip(remove_empty=False)
            boxlist = remove_small_box(boxlist, self.min_size)

            results.append(boxlist)

        return results

    def forward(self, location, cls_pred, box_pred, center_pred, image_sizes):
        boxes = []

        for loc, cls_p, box_p, center_p in zip(
            location, cls_pred, box_pred, center_pred
        ):
            boxes.append(
                self.forward_single_feature_map(
                    loc, cls_p, box_p, center_p, image_sizes
                )
            )

        boxlists = list(zip(*boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_scales(boxlists)

        return boxlists

    def select_over_scales(self, boxlists):
        results = []

        for boxlist in boxlists:
            scores = boxlist.fields["scores"]
            labels = boxlist.fields["labels"]
            box = boxlist.box

            result = []

            for j in range(1, self.n_class):
                id = (labels == j).nonzero().view(-1)
                score_j = scores[id]
                box_j = box[id, :].view(-1, 4)
                box_by_class = BoxList(box_j, boxlist.size, mode="xyxy")
                box_by_class.fields["scores"] = score_j
                box_by_class = boxlist_nms(box_by_class, score_j, self.nms_threshold)
                n_label = len(box_by_class)
                box_by_class.fields["labels"] = torch.full(
                    (n_label,), j, dtype=torch.int64, device=scores.device
                )
                result.append(box_by_class)

            result = cat_boxlist(result)
            n_detection = len(result)

            if n_detection > self.post_top_n > 0:
                scores = result.fields["scores"]
                img_threshold, _ = torch.kthvalue(
                    scores.cpu(), n_detection - self.post_top_n + 1
                )
                keep = scores >= img_threshold.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]

            results.append(result)

        return results


class BoxList:
    def __init__(self, box, image_size, mode="xyxy"):
        device = box.device if hasattr(box, "device") else "cpu"
        box = torch.as_tensor(box, dtype=torch.float32, device=device)

        self.box = box
        self.size = image_size
        self.mode = mode

        self.fields = {}

    def convert(self, mode):
        if mode == self.mode:
            return self

        x_min, y_min, x_max, y_max = self.split_to_xyxy()

        if mode == "xyxy":
            box = torch.cat([x_min, y_min, x_max, y_max], -1)
            box = BoxList(box, self.size, mode=mode)
        elif mode == "xywh":
            remove = 1
            box = torch.cat(
                [x_min, y_min, x_max - x_min + remove, y_max - y_min + remove], -1
            )
            box = BoxList(box, self.size, mode=mode)

        box.copy_field(self)

        return box

    def copy_field(self, box):
        for k, v in box.fields.items():
            self.fields[k] = v

    def area(self):
        box = self.box

        if self.mode == "xyxy":
            remove = 1
            area = (box[:, 2] - box[:, 0] + remove) * (box[:, 3] - box[:, 1] + remove)
        elif self.mode == "xywh":
            area = box[:, 2] * box[:, 3]

        return area

    def split_to_xyxy(self):
        if self.mode == "xyxy":
            x_min, y_min, x_max, y_max = self.box.split(1, dim=-1)
            return x_min, y_min, x_max, y_max
        elif self.mode == "xywh":
            remove = 1
            x_min, y_min, w, h = self.box.split(1, dim=-1)
            return (
                x_min,
                y_min,
                x_min + (w - remove).clamp(min=0),
                y_min + (h - remove).clamp(min=0),
            )

    def __len__(self):
        return self.box.shape[0]

    def __getitem__(self, index):
        box = BoxList(self.box[index], self.size, self.mode)
        for k, v in self.fields.items():
            box.fields[k] = v[index]
        return box

    def resize(self, size, *args, **kwargs):
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))

        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled = self.box * ratio
            box = BoxList(scaled, size, mode=self.mode)

            for k, v in self.fields.items():
                if not isinstance(v, torch.Tensor):
                    v = v.resize(size, *args, **kwargs)
                box.fields[k] = v

            return box

        ratio_w, ratio_h = ratios
        x_min, y_min, x_max, y_max = self.split_to_xyxy()
        scaled_x_min = x_min * ratio_w
        scaled_x_max = x_max * ratio_w
        scaled_y_min = y_min * ratio_h
        scaled_y_max = y_max * ratio_h
        scaled = torch.cat([scaled_x_min, scaled_y_min, scaled_x_max, scaled_y_max], -1)
        box = BoxList(scaled, size, mode="xyxy")

        for k, v in self.fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)
            box.fields[k] = v

        return box.convert(self.mode)

    def transpose(self, method):
        width, height = self.size
        x_min, y_min, x_max, y_max = self.split_to_xyxy()

        if method == FLIP_LEFT_RIGHT:
            remove = 1
            transpose_x_min = width - x_max - remove
            transpose_x_max = width - x_min - remove
            transpose_y_min = y_min
            transpose_y_max = y_max
        elif method == FLIP_TOP_BOTTOM:
            transpose_x_min = x_min
            transpose_x_max = x_max
            transpose_y_min = height - y_max
            transpose_y_max = height - y_min

        transpose_box = torch.cat(
            [transpose_x_min, transpose_y_min, transpose_x_max, transpose_y_max], -1
        )
        box = BoxList(transpose_box, self.size, mode="xyxy")

        for k, v in self.fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)
            box.fields[k] = v

        return box.convert(self.mode)

    def clip(self, remove_empty=True):
        remove = 1

        max_width = self.size[0] - remove
        max_height = self.size[1] - remove

        self.box[:, 0].clamp_(min=0, max=max_width)
        self.box[:, 1].clamp_(min=0, max=max_height)
        self.box[:, 2].clamp_(min=0, max=max_width)
        self.box[:, 3].clamp_(min=0, max=max_height)

        if remove_empty:
            box = self.box
            keep = (box[:, 3] > box[:, 1]) & (box[:, 2] > box[:, 0])
            return self[keep]
        else:
            return self

    def to(self, device):
        box = BoxList(self.box.to(device), self.size, self.mode)

        for k, v in self.fields.items():
            if hasattr(v, "to"):
                v = v.to(device)
            box.fields[k] = v

        return box


def remove_small_box(boxlist, min_size):
    box = boxlist.convert("xywh").box
    _, _, w, h = box.unbind(dim=1)
    keep = (w >= min_size) & (h >= min_size)
    keep = keep.nonzero().squeeze(1)

    return boxlist[keep]


def cat_boxlist(boxlists):
    size = boxlists[0].size
    mode = boxlists[0].mode
    field_keys = boxlists[0].fields.keys()

    box_cat = torch.cat([boxlist.box for boxlist in boxlists], 0)
    new_boxlist = BoxList(box_cat, size, mode)

    for field in field_keys:
        data = torch.cat([boxlist.fields[field] for boxlist in boxlists], 0)
        new_boxlist.fields[field] = data

    return new_boxlist


def boxlist_nms(boxlist, scores, threshold, max_proposal=-1):
    if threshold <= 0:
        return boxlist

    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    box = boxlist.box
    keep = ops.nms(box, scores, threshold)

    if max_proposal > 0:
        keep = keep[:max_proposal]

    boxlist = boxlist[keep]

    return boxlist.convert(mode)


class FCOS(nn.Module):
    def __init__(self, channels, n_class):
        super().__init__()
        in_channel = 256
        self.n_class = n_class
        cls_tower = []
        bbox_tower = []
        prior = 0.01

        for i in range(4):
            cls_tower.append(
                nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False)
            )
            cls_tower.append(nn.GroupNorm(32, in_channel))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(in_channel, in_channel, 3, padding=1, bias=False)
            )
            bbox_tower.append(nn.GroupNorm(32, in_channel))
            bbox_tower.append(nn.ReLU())

        self.cls_tower = nn.Sequential(*cls_tower)
        self.bbox_tower = nn.Sequential(*bbox_tower)
        self.cls_pred = nn.Conv2d(in_channel, n_class, 3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channel, 4, 3, padding=1)
        self.center_pred = nn.Conv2d(in_channel, 1, 3, padding=1)
        self.apply(init_conv_std)
        prior_bias = -math.log((1 - prior) / prior)
        nn.init.constant_(self.cls_pred.bias, prior_bias)
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(5)])

        self.fpn_strides = [8, 16, 32, 64, 128, 256]
        fpn_top = FPNTopP6P7(2048, 256, use_p5=True)
        self.fpn = FPN(channels, 256, fpn_top)
        self.fcos_loss = FCOSLoss(
            sizes=[
                [-1, 64],
                [64, 128],
                [128, 256],
                [256, 512],
                [512, 1024],
                [1024, 2048],
                [2048, 100000000],
            ],
            gamma=2.0,
            alpha=0.25,
            iou_loss_type="giou",
            center_sample=True,
            fpn_strides=self.fpn_strides,
            pos_radius=1.5,
        )
        self.postprocessor = FCOSPostprocessor(
            threshold=MODEL_CLASSIFICATION_THRESHOLD,
            top_n=1000,
            nms_threshold=MODEL_NMS_THRESHOLD,
            post_top_n=100,
            min_size=0,
            n_class=self.n_class,
        )

    def forward(self, input):
        logits = []
        bboxes = []
        centers = []

        input = self.fpn(input)
        self.location = self.compute_location(input)

        for feat, scale in zip(input, self.scales):
            cls_out = self.cls_tower(feat)
            logits.append(self.cls_pred(cls_out))
            centers.append(self.center_pred(cls_out))
            bbox_out = self.bbox_tower(feat)
            bbox_out = torch.exp(scale(self.bbox_pred(bbox_out)))
            bboxes.append(bbox_out)

        return {"cls": logits, "bboxes": bboxes, "centers": centers}

    def loss(self, logits, targets):
        loss_cls, loss_box, loss_center = self.fcos_loss(
            self.location, logits["cls"], logits["bboxes"], logits["centers"], targets
        )
        losses = {
            "loss_cls": loss_cls,
            "loss_box": loss_box,
            "loss_center": loss_center,
        }
        return loss_cls * 1.0 + loss_center * 1.0 + loss_box * 0.01, losses

    def preprocess_targets(self, targets, labels_count):
        result = []
        index = 0
        for i in range(targets.size()[0]):
            result.append(
                {
                    "mode": "xyxy",
                    "box": targets[:4, index : index + labels_count[i]],
                    "labels": targets[4, index : index + labels_count[i]],
                    "area": (
                        (
                            targets[2, index : index + labels_count[i]]
                            - targets[0, index : index + labels_count[i]]
                        )
                        * (
                            targets[3, index : index + labels_count[i]]
                            - targets[1, index : index + labels_count[i]]
                        )
                    ),
                }
            )
            index += labels_count[i]
        return result

    def postprocess_predictions(self, logits):
        boxes = self.postprocessor(
            self.location,
            logits["cls"],
            logits["bboxes"],
            logits["centers"],
            (DATA_IMAGE_SIZE_DETECTION, DATA_IMAGE_SIZE_DETECTION),
        )
        return boxes

    def compute_location(self, features):
        locations = []
        for i, feat in enumerate(features):
            _, _, height, width = feat.shape
            location_per_level = self.compute_location_per_level(
                height, width, self.fpn_strides[i - 1], feat.device
            )
            locations.append(location_per_level)
        return locations

    def compute_location_per_level(self, height, width, stride, device):
        shift_x = torch.arange(
            0, width * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y = torch.arange(
            0, height * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        location = torch.stack((shift_x, shift_y), 1) + stride // 2
        return location
