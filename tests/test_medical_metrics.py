from __future__ import annotations

import torch

from vap_pidnet.metrics import DiceHD95


def test_dice_hd95_perfect_prediction() -> None:
    metric = DiceHD95(num_classes=3)
    targets = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    targets[:, 2:6, 2:6, 2:6] = 1

    metric.update(targets, targets)
    result = metric.compute()

    assert result["mean_dice"].item() == 0.5
    assert result["dice"][0].item() == 1.0
    assert result["dice"][1].item() == 0.0


def test_dice_hd95_empty_foreground_prediction() -> None:
    metric = DiceHD95(num_classes=2)
    preds = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    targets = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    targets[:, 2:6, 2:6, 2:6] = 1

    metric.update(preds, targets)
    result = metric.compute()

    assert result["mean_dice"].item() == 0.0
