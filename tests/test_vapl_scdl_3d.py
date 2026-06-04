from __future__ import annotations

import torch

from vap_pidnet import build_vapl_scdl_3d
from vap_pidnet.models import scdl_vnet_3d


def test_scdl_vnet_3d_backbone_forward_cpu() -> None:
    model = scdl_vnet_3d(num_classes=3, base_channels=4)
    model.eval()

    images = torch.randn(1, 1, 32, 32, 32)

    with torch.no_grad():
        outputs = model(images)

    assert outputs["logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["features"].shape == (1, 32, 4, 4, 4)
    assert outputs["aux_logits_p"] is None
    assert outputs["aux_logits_d"] is None


def test_vapl_scdl_3d_forward_backward_cpu() -> None:
    model = build_vapl_scdl_3d(
        num_classes=3,
        base_channels=4,
        embedding_dim=32,
    )
    model.train()

    images = torch.randn(1, 1, 32, 32, 32)
    targets = torch.randint(0, 3, (1, 32, 32, 32))

    outputs = model(images, targets)
    losses = outputs["losses"]
    assert losses is not None
    assert outputs["logits"].shape == (1, 3, 32, 32, 32)
    assert outputs["embeddings"].shape == (1, 32, 4, 4, 4)

    losses["loss_total"].backward()
    assert model.cs_loss.variation_vectors.grad is not None
