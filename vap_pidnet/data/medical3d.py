"""3D Synapse and AMOS volume datasets for medical VAPL experiments."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


MedicalDatasetName = Literal["synapse", "amos"]

SYNAPSE_NUM_CLASSES_DHC = 14
AMOS_NUM_CLASSES = 16
MEDICAL_IGNORE_INDEX = 255


def read_split_list(path: str | Path) -> list[str]:
    split_path = Path(path)
    if not split_path.exists():
        raise FileNotFoundError(f"Split list not found: {split_path}")
    entries = []
    for raw_line in split_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    if not entries:
        raise ValueError(f"Split list is empty: {split_path}")
    return entries


def case_id_from_entry(entry: str) -> str:
    """Normalize DHC/TransUNet-style list entries to volume IDs."""

    item = entry.strip()
    if item.startswith("case"):
        item = item[4:]
    if "_slice" in item:
        item = item.split("_slice", maxsplit=1)[0]
    return item


class MedicalVolumeDataset(Dataset):
    """Load 3D Synapse ``.h5`` or AMOS ``.npy`` volumes.

    The returned image tensor has shape ``[1, D, H, W]`` and the target has
    shape ``[D, H, W]``. The axis order follows the preprocessed arrays used by
    SCDL/DHC loaders.
    """

    def __init__(
        self,
        root: str | Path,
        dataset: MedicalDatasetName,
        split_file: str | Path,
        patch_size: tuple[int, int, int] = (96, 96, 96),
        train: bool = True,
        random_flip: bool = True,
        random_rotate: bool = True,
        foreground_prob: float = 0.75,
        foreground_margin: int = 4,
        full_volume: bool = False,
    ) -> None:
        self.root = Path(root)
        self.dataset = dataset
        self.entries = read_split_list(split_file)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.train = train
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.full_volume = full_volume
        if not 0.0 <= foreground_prob <= 1.0:
            raise ValueError("foreground_prob must be in [0, 1].")
        self.foreground_prob = foreground_prob
        self.foreground_margin = foreground_margin

        if dataset not in {"synapse", "amos"}:
            raise ValueError("dataset must be 'synapse' or 'amos'.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        entry = self.entries[index]
        image, target, case_id = self._load_case(entry)

        if self.train:
            image, target = random_crop_3d(
                image,
                target,
                self.patch_size,
                foreground_prob=self.foreground_prob,
                foreground_margin=self.foreground_margin,
            )
            if self.random_rotate:
                image, target = random_rot90_flip_3d(image, target)
            elif self.random_flip:
                image, target = random_flip_3d(image, target)
        elif not self.full_volume:
            image, target = center_crop_3d(image, target, self.patch_size)

        image = np.ascontiguousarray(image[None].astype(np.float32))
        target = np.ascontiguousarray(target.astype(np.int64))
        return {
            "image": torch.from_numpy(image),
            "target": torch.from_numpy(target),
            "case_id": case_id,
        }

    def _load_case(self, entry: str) -> tuple[np.ndarray, np.ndarray, str]:
        if self.dataset == "synapse":
            case_id = case_id_from_entry(entry)
            path = self.root / f"{case_id}.h5"
            if not path.exists():
                raise FileNotFoundError(f"Missing Synapse case: {path}")
            with h5py.File(path, "r") as h5f:
                image = h5f["image"][:].astype(np.float32)
                target = h5f["label"][:].astype(np.int64)
            return image, target, case_id

        case_id = entry.strip()
        image_path = self.root / f"{case_id}_image.npy"
        target_path = self.root / f"{case_id}_label.npy"
        if not image_path.exists() or not target_path.exists():
            raise FileNotFoundError(f"Missing AMOS case: {image_path} / {target_path}")
        image = np.load(image_path).astype(np.float32)
        target = np.load(target_path).astype(np.int64)
        image = np.clip(image, -125.0, 275.0)
        image = (image + 125.0) / 400.0
        return image, target, case_id


def pad_to_patch(
    image: np.ndarray,
    target: np.ndarray,
    patch_size: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    pads = []
    for dim, size in zip(image.shape, patch_size):
        missing = max(size - dim, 0)
        before = missing // 2
        after = missing - before
        pads.append((before, after))
    if any(before or after for before, after in pads):
        image = np.pad(image, pads, mode="constant", constant_values=0)
        target = np.pad(target, pads, mode="constant", constant_values=0)
    return image, target


def random_crop_3d(
    image: np.ndarray,
    target: np.ndarray,
    patch_size: tuple[int, int, int],
    foreground_prob: float = 0.0,
    foreground_margin: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    image, target = pad_to_patch(image, target, patch_size)
    starts = None
    if foreground_prob > 0.0 and random.random() < foreground_prob:
        starts = foreground_crop_starts(target, patch_size, foreground_margin)
    if starts is None:
        starts = [
            random.randint(0, dim - size) if dim > size else 0
            for dim, size in zip(image.shape, patch_size)
        ]
    slices = tuple(slice(start, start + size) for start, size in zip(starts, patch_size))
    return image[slices], target[slices]


def foreground_crop_starts(
    target: np.ndarray,
    patch_size: tuple[int, int, int],
    margin: int = 4,
) -> list[int] | None:
    classes = np.unique(target)
    classes = classes[classes > 0]
    if classes.size == 0:
        return None

    # Pick a foreground class uniformly first, then a voxel of that class.
    # This gives rare small-volume classes (e.g. esophagus, adrenal glands)
    # the same chance of being the patch center as large organs (e.g.
    # liver), instead of weighting by raw voxel count.
    chosen_class = classes[random.randrange(len(classes))]
    foreground = np.argwhere(target == chosen_class)
    center = foreground[random.randrange(len(foreground))]
    starts = []
    for axis, (coord, dim, size) in enumerate(zip(center, target.shape, patch_size)):
        max_start = max(dim - size, 0)
        if max_start == 0:
            starts.append(0)
            continue

        low = max(int(coord) - size + 1 + margin, 0)
        high = min(int(coord) - margin, max_start)
        if low > high:
            start = min(max(int(coord) - size // 2, 0), max_start)
        else:
            start = random.randint(low, high)
        starts.append(start)
    return starts


def center_crop_3d(
    image: np.ndarray,
    target: np.ndarray,
    patch_size: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    image, target = pad_to_patch(image, target, patch_size)
    starts = [(dim - size) // 2 for dim, size in zip(image.shape, patch_size)]
    slices = tuple(slice(start, start + size) for start, size in zip(starts, patch_size))
    return image[slices], target[slices]


def random_rot90_flip_3d(
    image: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    axes = random.choice(((0, 1), (0, 2), (1, 2)))
    k = random.randint(0, 3)
    image = np.rot90(image, k=k, axes=axes)
    target = np.rot90(target, k=k, axes=axes)
    return random_flip_3d(image, target)


def random_flip_3d(
    image: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    for axis in range(3):
        if random.random() < 0.5:
            image = np.flip(image, axis=axis)
            target = np.flip(target, axis=axis)
    return image, target
