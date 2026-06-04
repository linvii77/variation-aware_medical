from __future__ import annotations

import torch

from vap_pidnet import build_vapl_pidnet_m


def test_vapl_pidnet_forward_backward_cpu() -> None:
    model = build_vapl_pidnet_m(num_classes=3, embedding_dim=64)
    model.train()

    images = torch.randn(2, 3, 128, 128)
    targets = torch.randint(0, 3, (2, 128, 128))

    outputs = model(images, targets)
    losses = outputs["losses"]
    assert losses is not None
    assert outputs["logits"].shape == (2, 3, 128, 128)
    assert outputs["embeddings"].shape[-2:] == (16, 16)

    losses["loss_total"].backward()
    assert model.cs_loss.variation_vectors.grad is not None
