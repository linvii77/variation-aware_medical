"""Cityscapes dataset utilities for PIDNet/VAPL experiments."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CITYSCAPES_NUM_CLASSES = 19
CITYSCAPES_IGNORE_INDEX = 255
CITYSCAPES_MEAN = (0.485, 0.456, 0.406)
CITYSCAPES_STD = (0.229, 0.224, 0.225)

ID_TO_TRAIN_ID = np.full(256, CITYSCAPES_IGNORE_INDEX, dtype=np.uint8)
for label_id, train_id in {
    7: 0,
    8: 1,
    11: 2,
    12: 3,
    13: 4,
    17: 5,
    19: 6,
    20: 7,
    21: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    31: 16,
    32: 17,
    33: 18,
}.items():
    ID_TO_TRAIN_ID[label_id] = train_id


class Cityscapes(Dataset):
    """Cityscapes fine annotations.

    Expected root:

    ```text
    root/
      leftImg8bit/train/<city>/*_leftImg8bit.png
      leftImg8bit/val/<city>/*_leftImg8bit.png
      gtFine/train/<city>/*_gtFine_labelIds.png
      gtFine/val/<city>/*_gtFine_labelIds.png
    ```

    If ``*_gtFine_labelTrainIds.png`` exists, it is used directly.
    Otherwise, ``labelIds`` are mapped to the standard 19 train IDs.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Callable[[Image.Image, Image.Image], tuple[torch.Tensor, torch.Tensor]]
        | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.images = sorted(
            (self.root / "leftImg8bit" / split).glob("*/*_leftImg8bit.png")
        )
        if not self.images:
            raise FileNotFoundError(
                f"No Cityscapes images found under "
                f"{self.root / 'leftImg8bit' / split}"
            )
        self.targets = [self._target_path(path) for path in self.images]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path = self.images[index]
        target_path = self.targets[index]
        image = Image.open(image_path).convert("RGB")
        target = Image.open(target_path)

        if target_path.name.endswith("_labelIds.png"):
            target_np = np.asarray(target, dtype=np.uint8)
            target = Image.fromarray(ID_TO_TRAIN_ID[target_np], mode="L")

        if self.transform is not None:
            image_tensor, target_tensor = self.transform(image, target)
        else:
            image_tensor, target_tensor = CityscapesTransform(train=False)(
                image, target
            )

        return {
            "image": image_tensor,
            "target": target_tensor,
            "image_path": str(image_path),
            "target_path": str(target_path),
        }

    def _target_path(self, image_path: Path) -> Path:
        city = image_path.parent.name
        stem = image_path.name.removesuffix("_leftImg8bit.png")
        target_dir = self.root / "gtFine" / self.split / city
        train_id_path = target_dir / f"{stem}_gtFine_labelTrainIds.png"
        label_id_path = target_dir / f"{stem}_gtFine_labelIds.png"
        if train_id_path.exists():
            return train_id_path
        if label_id_path.exists():
            return label_id_path
        raise FileNotFoundError(
            f"Missing Cityscapes target for {image_path}. Expected "
            f"{train_id_path} or {label_id_path}."
        )


class CityscapesTransform:
    def __init__(
        self,
        train: bool,
        crop_size: tuple[int, int] = (1024, 1024),
        scale_range: tuple[float, float] = (0.5, 2.0),
        mean: tuple[float, float, float] = CITYSCAPES_MEAN,
        std: tuple[float, float, float] = CITYSCAPES_STD,
        ignore_index: int = CITYSCAPES_IGNORE_INDEX,
    ) -> None:
        self.train = train
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.ignore_index = ignore_index

    def __call__(
        self, image: Image.Image, target: Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.train:
            image, target = self._random_scale(image, target)
            image, target = self._random_crop(image, target)
            image, target = self._random_flip(image, target)

        image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))
        image_tensor = image_tensor.permute(2, 0, 1).contiguous() / 255.0
        image_tensor = (image_tensor - self.mean) / self.std

        target_tensor = torch.from_numpy(np.asarray(target, dtype=np.int64))
        return image_tensor, target_tensor

    def _random_scale(
        self, image: Image.Image, target: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        scale = random.uniform(*self.scale_range)
        width, height = image.size
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        image = image.resize(size, Image.BILINEAR)
        target = target.resize(size, Image.NEAREST)
        return image, target

    def _random_crop(
        self, image: Image.Image, target: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        crop_h, crop_w = self.crop_size
        width, height = image.size
        pad_w = max(crop_w - width, 0)
        pad_h = max(crop_h - height, 0)
        if pad_w > 0 or pad_h > 0:
            image = self._pad_image(image, pad_w, pad_h, fill=(0, 0, 0))
            target = self._pad_image(target, pad_w, pad_h, fill=self.ignore_index)
            width, height = image.size

        left = random.randint(0, width - crop_w)
        top = random.randint(0, height - crop_h)
        box = (left, top, left + crop_w, top + crop_h)
        return image.crop(box), target.crop(box)

    @staticmethod
    def _random_flip(
        image: Image.Image, target: Image.Image
    ) -> tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            target = target.transpose(Image.FLIP_LEFT_RIGHT)
        return image, target

    @staticmethod
    def _pad_image(
        image: Image.Image,
        pad_w: int,
        pad_h: int,
        fill: int | tuple[int, int, int],
    ) -> Image.Image:
        width, height = image.size
        mode = image.mode
        padded = Image.new(mode, (width + pad_w, height + pad_h), color=fill)
        padded.paste(image, (0, 0))
        return padded
