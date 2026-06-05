"""Inference utilities for 3D medical segmentation."""

from __future__ import annotations

import torch
from torch.nn import functional as F


@torch.no_grad()
def sliding_window_logits_3d(
    model: torch.nn.Module,
    image: torch.Tensor,
    num_classes: int,
    patch_size: tuple[int, int, int],
    stride: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """Run sliding-window inference for one 3D image.

    Args:
        model: Segmentation model returning ``{"logits": tensor}``.
        image: Tensor shaped ``[C, D, H, W]``.
        num_classes: Number of output classes.
        patch_size: Sliding window size.
        stride: Sliding window stride.
        device: Model device.

    Returns:
        Full-volume logits shaped ``[num_classes, D, H, W]``.
    """

    if image.ndim != 4:
        raise ValueError("image must have shape [C, D, H, W].")

    original_size = tuple(int(v) for v in image.shape[-3:])
    patch_size = tuple(int(v) for v in patch_size)
    stride = tuple(int(v) for v in stride)
    padded, crop_slices = _pad_image_to_patch(image, patch_size)
    padded_size = tuple(int(v) for v in padded.shape[-3:])

    score = torch.zeros(
        (num_classes, *padded_size),
        device=device,
        dtype=torch.float32,
    )
    count = torch.zeros((1, *padded_size), device=device, dtype=torch.float32)

    model_device_image = padded.to(device=device, dtype=torch.float32)
    for start_d in _window_starts(padded_size[0], patch_size[0], stride[0]):
        for start_h in _window_starts(padded_size[1], patch_size[1], stride[1]):
            for start_w in _window_starts(padded_size[2], patch_size[2], stride[2]):
                patch = model_device_image[
                    :,
                    start_d : start_d + patch_size[0],
                    start_h : start_h + patch_size[1],
                    start_w : start_w + patch_size[2],
                ].unsqueeze(0)
                logits = model(patch)["logits"]
                if logits.shape[-3:] != patch_size:
                    logits = F.interpolate(
                        logits,
                        size=patch_size,
                        mode="trilinear",
                        align_corners=False,
                    )
                score[
                    :,
                    start_d : start_d + patch_size[0],
                    start_h : start_h + patch_size[1],
                    start_w : start_w + patch_size[2],
                ] += logits[0].float()
                count[
                    :,
                    start_d : start_d + patch_size[0],
                    start_h : start_h + patch_size[1],
                    start_w : start_w + patch_size[2],
                ] += 1.0

    score = score / count.clamp_min(1.0)
    return score[(slice(None), *crop_slices)].reshape(num_classes, *original_size)


def _pad_image_to_patch(
    image: torch.Tensor, patch_size: tuple[int, int, int]
) -> tuple[torch.Tensor, tuple[slice, slice, slice]]:
    spatial = tuple(int(v) for v in image.shape[-3:])
    pads = []
    crop_slices = []
    for dim, size in reversed(list(zip(spatial, patch_size))):
        missing = max(size - dim, 0)
        before = missing // 2
        after = missing - before
        pads.extend([before, after])
        crop_slices.append(slice(before, before + dim))
    padded = F.pad(image, pads)
    return padded, tuple(reversed(crop_slices))  # type: ignore[return-value]


def _window_starts(length: int, window: int, stride: int) -> list[int]:
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive.")
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return starts
