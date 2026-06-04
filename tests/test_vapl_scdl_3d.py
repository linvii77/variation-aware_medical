from __future__ import annotations

import torch

from vap_pidnet import build_vapl_scdl_3d
from vap_pidnet.models import scdl_vnet_3d
from vap_pidnet.scdl import SemanticClassDistributionLoss


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


def test_semantic_class_distribution_loss_3d_cpu() -> None:
    loss_fn = SemanticClassDistributionLoss(
        in_channels=8,
        num_classes=3,
        embedding_dim=16,
        proxy_samples=4,
    )
    features = torch.randn(1, 8, 4, 4, 4)
    targets = torch.randint(0, 3, (1, 32, 32, 32))

    loss, prior, stats = loss_fn(features, targets)

    assert loss.ndim == 0
    assert prior.shape == (1, 16, 4, 4, 4)
    assert stats.scdl_valid_tokens.item() > 0

    loss.backward()
    assert loss_fn.proxies.grad is not None


def test_vapl_scdl_3d_scdl_only_forward_backward_cpu() -> None:
    model = build_vapl_scdl_3d(
        num_classes=3,
        base_channels=4,
        embedding_dim=32,
        lambda_cs=0.0,
        lambda_scdl=1.0,
    )
    model.train()

    images = torch.randn(1, 1, 32, 32, 32)
    targets = torch.randint(0, 3, (1, 32, 32, 32))

    outputs = model(images, targets)
    losses = outputs["losses"]
    assert losses is not None
    assert outputs["scdl_prior"].shape == (1, 32, 4, 4, 4)
    assert losses["loss_cs"].item() == 0.0
    assert losses["loss_scdl"].item() > 0.0

    losses["loss_total"].backward()
    assert model.scdl_loss.proxies.grad is not None
