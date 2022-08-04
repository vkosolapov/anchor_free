import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchmetrics


def mask_to_boundary(mask, boundary_size=3, dilation_ratio=0.02):
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((boundary_size, boundary_size), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    return mask - mask_erode


class BoundaryIoU(torchmetrics.Metric):
    def __init__(
        self,
        num_classes,
        threshold,
        boundary_size=3,
        compute_on_step=True,
        dist_sync_on_step=False,
        process_group=None,
        dist_sync_fn=None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.num_classes = num_classes
        self.threshold = threshold
        self.boundary_size = boundary_size
        self.iou = torchmetrics.JaccardIndex(num_classes=2)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        target = (
            F.one_hot(target, num_classes=self.num_classes)
            .permute(0, 3, 1, 2)
            .cpu()
            .detach()
            .numpy()
            .astype("uint8")
        )
        b, c, h, w = target.shape
        preds = (preds > self.threshold).cpu().detach().numpy().astype("uint8")
        for i in range(b):
            for j in range(c):
                boundary_target = mask_to_boundary(
                    target[i][j], boundary_size=self.boundary_size,
                )
                boundary_target = (
                    torch.Tensor(boundary_target).int().to(self.iou.device)
                )
                boundary_preds = mask_to_boundary(
                    preds[i][j], boundary_size=self.boundary_size,
                )
                boundary_preds = torch.Tensor(boundary_preds).int().to(self.iou.device)
                self.iou(boundary_preds, boundary_target)

    def compute(self):
        res = self.iou.compute()
        self.iou.reset()
        return res

    def reset(self):
        self.iou.reset()


if __name__ == "__main__":
    from PIL import Image

    file_path = "datasets/clothing_segmentation/png_masks/MASKS/seg_0001.png"
    mask = Image.open(file_path)
    mask = mask.resize((288, 416), Image.NEAREST)
    mask = np.array(mask, np.int32)
    mask = torch.clamp(torch.from_numpy(np.array(mask)), 0, 1).long()
    mask = torch.unsqueeze(mask, dim=0)
    pred = F.one_hot(mask, num_classes=2).permute(0, 3, 1, 2)
    iou = BoundaryIoU(num_classes=2, threshold=0.5)
    print(iou(pred, mask))

