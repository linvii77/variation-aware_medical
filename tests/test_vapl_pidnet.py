from __future__ import annotations

import torch

from vap_pidnet import build_vapl_pidnet_m
from vap_pidnet.vapl import CompositionalSimilarityLoss


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
    assert model.cs_loss.proxy_dist.grad is not None


def test_compositional_similarity_loss_proxy_distribution_cpu() -> None:
    torch.manual_seed(0)
    loss_fn = CompositionalSimilarityLoss(num_classes=4, embedding_dim=8, num_variations=3)

    embeddings = torch.randn(1, 8, 4, 5)
    targets = torch.randint(0, 4, (1, 4, 5))

    loss, stats = loss_fn(embeddings, targets)
    loss.backward()

    # The new (mu, sigma) proxy distribution must receive gradient: it now
    # participates in a cross-class softmax instead of being cancelled out
    # by the per-variation softmax.
    assert loss_fn.proxy_dist.grad is not None
    assert loss_fn.proxy_dist.grad.norm().item() > 0
    assert 0.0 <= stats.proxy_assignment_accuracy.item() <= 1.0
    assert stats.proxy_sigma_mean.item() >= loss_fn.proxy_sigma_min

    # combined[n, c, k] = q_c(x_n) * p_sub(x_n, v_{c,k} | c) must be a valid
    # joint distribution over (class, variation) for each pixel.
    resized_targets = loss_fn._resize_targets(targets, embeddings.shape[2:])
    flat_embeddings, _ = loss_fn._flatten_valid(embeddings, resized_targets)
    x = torch.nn.functional.normalize(flat_embeddings, p=2, dim=1)
    q, _ = loss_fn._proxy_assignment_probabilities(x)
    p_sub = loss_fn._variation_subdistribution(x)
    combined = q.unsqueeze(-1) * p_sub
    assert torch.allclose(
        combined.sum(dim=(1, 2)), torch.ones(combined.shape[0]), atol=1e-5
    )
