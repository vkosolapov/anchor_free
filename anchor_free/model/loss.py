import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingFocalLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        need_one_hot=True,
        gamma=0,
        alpha=None,
        smoothing=0.0,
        size_average=True,
        ignore_index=None,
    ):
        super().__init__()
        self._num_classes = num_classes
        self.need_one_hot = need_one_hot
        self._gamma = gamma
        self._alpha = alpha
        self._smoothing = smoothing
        self._size_average = size_average
        self._ignore_index = ignore_index

        if self._num_classes <= 1:
            raise ValueError("The number of classes must be 2 or higher")
        if self._gamma < 0:
            raise ValueError("Gamma must be 0 or higher")
        if self._alpha is not None:
            if self._alpha <= 0 or self._alpha >= 1:
                raise ValueError("Alpha must be 0 <= alpha <= 1")

    def forward(self, logits, label):
        logits = logits.float()
        label = label.long()
        difficulty_level = self._estimate_difficulty_level(logits, label)

        with torch.no_grad():
            label = label.clone().detach()
            if self.need_one_hot:
                if self._ignore_index is not None:
                    ignore = label.eq(self._ignore_index)
                    label[ignore] = 0
                lb_pos, lb_neg = (
                    1.0 - self._smoothing,
                    self._smoothing / (self._num_classes - 1),
                )
                lb_one_hot = (
                    torch.empty_like(logits)
                    .fill_(lb_neg)
                    .scatter_(1, label.unsqueeze(1), lb_pos)
                    .detach()
                )
            else:
                lb_one_hot = label
        logs = F.log_softmax(logits, dim=1)
        loss = -torch.sum(difficulty_level * logs * lb_one_hot, dim=1)
        if self._ignore_index is not None:
            loss[ignore] = 0
        return loss.mean()

    def _estimate_difficulty_level(self, logits, label):
        if self.need_one_hot:
            one_hot_key = torch.nn.functional.one_hot(
                label, num_classes=self._num_classes
            )
            if len(one_hot_key.size()) == 4:
                one_hot_key = one_hot_key.permute((0, 3, 1, 2))
        else:
            one_hot_key = label
        if one_hot_key.device != logits.device:
            one_hot_key = one_hot_key.to(logits.device)
        pt = one_hot_key * F.softmax(logits, dim=1)
        difficulty_level = torch.pow(1 - pt, self._gamma)
        return difficulty_level


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


class IoULossWithMask(nn.Module):
    def __init__(self, GIoU=False, DIoU=False, CIoU=False):
        super().__init__()
        self.GIoU = GIoU
        self.DIoU = DIoU
        self.CIoU = CIoU

    def forward(self, pred, target, mask):
        pred = pred.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
        expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 4)
        loss = bbox_iou(pred * expand_mask, target * expand_mask, CIoU=True)
        loss = loss.sum() / (expand_mask.sum() + 1e-4)
        return 1 - loss


def bbox_iou(box1, box2, x1y1x2y2=False, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = (
            box1[..., 0],
            box1[..., 1],
            box1[..., 2],
            box1[..., 3],
        )
        b2_x1, b2_y1, b2_x2, b2_y2 = (
            box2[..., 0],
            box2[..., 1],
            box2[..., 2],
            box2[..., 3],
        )
    else:
        b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
        b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
        b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4
            if CIoU:
                v = (4 / math.pi ** 2) * torch.pow(
                    torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
                )
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            return iou - rho2 / c2
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
    return iou
