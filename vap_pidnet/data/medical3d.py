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
    ) -> None:
        self.root = Path(root)
        self.dataset = dataset
        self.entries = read_split_list(split_file)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.train = train
        self.random_flip = random_flip
        self.random_rotate = random_rotate

        if dataset not in {"synapse", "amos"}:
            raise ValueError("dataset must be 'synapse' or 'amos'.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        entry = self.entries[index]
        image, target, case_id = self._load_case(entry)

        if self.train:
            image, target = random_crop_3d(image, target, self.patch_size)
            if self.random_rotate:
                image, target = random_rot90_flip_3d(image, target)
            elif self.random_flip:
                image, target = random_flip_3d(image, target)
        else:
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
) -> tuple[np.ndarray, np.ndarray]:
    image, target = pad_to_patch(image, target, patch_size)
    starts = [
        random.randint(0, dim - size) if dim > size else 0
        for dim, size in zip(image.shape, patch_size)
    ]
    slices = tuple(slice(start, start + size) for start, size in zip(starts, patch_size))
    return image[slices], target[slices]


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
