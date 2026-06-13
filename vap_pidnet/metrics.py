"""Segmentation metrics."""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage


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


class DiceHD95:
    """Per-class Dice and HD95 for 3D medical segmentation.

    Background is skipped by default. HD95 is computed from binary mask surface
    distances with voxel spacing assumed to be isotropic.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        include_background: bool = False,
    ) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.include_background = include_background
        self.class_start = 0 if include_background else 1
        self.dice_values: list[list[float]] = [
            [] for _ in range(self.class_start, num_classes)
        ]
        self.hd95_values: list[list[float]] = [
            [] for _ in range(self.class_start, num_classes)
        ]

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds_np = preds.detach().cpu().numpy().astype(np.int64)
        targets_np = targets.detach().cpu().numpy().astype(np.int64)
        if preds_np.ndim == 3:
            preds_np = preds_np[None]
            targets_np = targets_np[None]

        for pred, target in zip(preds_np, targets_np):
            valid = target != self.ignore_index
            for offset, class_index in enumerate(range(self.class_start, self.num_classes)):
                pred_mask = (pred == class_index) & valid
                target_mask = (target == class_index) & valid
                if not pred_mask.any() and not target_mask.any():
                    continue
                self.dice_values[offset].append(_dice(pred_mask, target_mask))
                hd95 = _hd95(pred_mask, target_mask)
                if np.isfinite(hd95):
                    self.hd95_values[offset].append(hd95)

    def compute(self) -> dict[str, torch.Tensor]:
        dice = torch.tensor(
            [_mean_or_zero(values) for values in self.dice_values],
            dtype=torch.float32,
        )
        hd95 = torch.tensor(
            [_mean_or_zero(values) for values in self.hd95_values],
            dtype=torch.float32,
        )
        return {
            "dice": dice,
            "mean_dice": dice.mean() if dice.numel() else torch.tensor(0.0),
            "hd95": hd95,
            "mean_hd95": hd95.mean() if hd95.numel() else torch.tensor(0.0),
        }


def _dice(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    intersection = np.logical_and(pred_mask, target_mask).sum(dtype=np.float64)
    denominator = pred_mask.sum(dtype=np.float64) + target_mask.sum(dtype=np.float64)
    if denominator == 0:
        return 1.0
    return float(2.0 * intersection / denominator)


def _hd95(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    if not pred_mask.any() or not target_mask.any():
        return float("inf")
    pred_surface = _surface(pred_mask)
    target_surface = _surface(target_mask)
    if not pred_surface.any() or not target_surface.any():
        return float("inf")

    target_distance = ndimage.distance_transform_edt(~target_surface)
    pred_distance = ndimage.distance_transform_edt(~pred_surface)
    distances = np.concatenate(
        [target_distance[pred_surface], pred_distance[target_surface]]
    )
    if distances.size == 0:
        return float("inf")
    return float(np.percentile(distances, 95))


def _surface(mask: np.ndarray) -> np.ndarray:
    eroded = ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3)), border_value=0)
    return mask & ~eroded


def _mean_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def keep_largest_connected_component(
    preds: np.ndarray,
    num_classes: int,
    include_background: bool = False,
) -> np.ndarray:
    """Zero out all but the largest connected component per foreground class.

    ``preds`` is an integer array of class indices with shape ``[D, H, W]``.
    """
    class_start = 0 if include_background else 1
    out = preds.copy()
    for class_index in range(class_start, num_classes):
        mask = preds == class_index
        if not mask.any():
            continue
        labeled, num_features = ndimage.label(mask)
        if num_features <= 1:
            continue
        sizes = ndimage.sum(mask, labeled, index=range(1, num_features + 1))
        largest_label = int(np.argmax(sizes)) + 1
        out[mask & (labeled != largest_label)] = 0
    return out
