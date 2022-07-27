import torch
import torch.nn as nn
import torch.nn.functional as F


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

