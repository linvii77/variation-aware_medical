"""Auxiliary segmentation loss terms shared across model variants."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
    include_background: bool = False,
    eps: float = 1.0e-5,
) -> torch.Tensor:
    """Soft Dice loss averaged over classes and batch.

    ``logits`` has shape ``[B, C, ...]`` and ``targets`` has shape ``[B, ...]``
    with integer class indices (possibly containing ``ignore_index``).

    With epsilon smoothing, a class with no ground-truth voxels in a sample
    but some predicted voxels gets ``dice ~= 0`` (loss ~= 1), directly
    penalizing hallucinated false positives. A class absent from both
    prediction and target gets ``dice == 1`` (loss == 0).
    """

    probs = torch.softmax(logits, dim=1)
    valid = (targets != ignore_index) & (targets >= 0) & (targets < num_classes)
    targets_clamped = targets.clamp(min=0, max=num_classes - 1)

    one_hot = F.one_hot(targets_clamped, num_classes=num_classes)
    one_hot = one_hot.movedim(-1, 1).float()

    valid = valid.unsqueeze(1).float()
    probs = probs * valid
    one_hot = one_hot * valid

    dims = tuple(range(2, probs.ndim))
    intersection = (probs * one_hot).sum(dim=dims)
    union = probs.sum(dim=dims) + one_hot.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (union + eps)

    class_start = 0 if include_background else 1
    dice = dice[:, class_start:]
    return (1.0 - dice).mean()
