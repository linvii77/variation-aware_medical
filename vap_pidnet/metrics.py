"""Segmentation metrics."""

from __future__ import annotations

import torch


class MeanIoU:
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = torch.zeros(
            num_classes, num_classes, dtype=torch.int64
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds = preds.detach().cpu().long()
        targets = targets.detach().cpu().long()
        valid = (
            (targets != self.ignore_index)
            & (targets >= 0)
            & (targets < self.num_classes)
        )
        if not valid.any():
            return
        indices = self.num_classes * targets[valid] + preds[valid]
        hist = torch.bincount(
            indices, minlength=self.num_classes * self.num_classes
        )
        self.confusion_matrix += hist.view(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, torch.Tensor]:
        hist = self.confusion_matrix.float()
        true_positive = torch.diag(hist)
        union = hist.sum(dim=1) + hist.sum(dim=0) - true_positive
        valid = union > 0
        iou = torch.zeros(self.num_classes, dtype=torch.float32)
        iou[valid] = true_positive[valid] / union[valid]
        miou = iou[valid].mean() if valid.any() else torch.tensor(0.0)
        pixel_acc = true_positive.sum() / hist.sum().clamp_min(1.0)
        return {
            "miou": miou,
            "iou": iou,
            "pixel_acc": pixel_acc,
        }
