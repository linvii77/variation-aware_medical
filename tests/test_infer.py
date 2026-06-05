from __future__ import annotations

import torch

from vap_pidnet.infer import sliding_window_logits_3d


class TinySegModel(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv3d(1, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"logits": self.conv(x)}


def test_sliding_window_logits_3d_shape_cpu() -> None:
    model = TinySegModel(num_classes=3)
    image = torch.randn(1, 10, 12, 14)

    logits = sliding_window_logits_3d(
        model=model,
        image=image,
        num_classes=3,
        patch_size=(8, 8, 8),
        stride=(4, 4, 4),
        device=torch.device("cpu"),
    )

    assert logits.shape == (3, 10, 12, 14)
