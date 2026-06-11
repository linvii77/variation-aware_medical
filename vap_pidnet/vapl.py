"""Variation-Aware Proxy Learning modules.

The default loss follows the Methodology section of
"Variation-aware proxy learning for semantic segmentation":

* one representative proxy per class
* K variation vectors per class, default K=5
* factorized similarity score
* negative-only focal modulation
* attraction + repulsion compositional similarity loss
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


SoftmaxScope = Literal["per_class", "global"]


class ProjectionHead(nn.Module):
    """Small projection head used only during training."""

    def __init__(
        self,
        in_channels: int,
        embedding_dim: int = 256,
        hidden_channels: int | None = None,
        spatial_dims: int = 2,
    ) -> None:
        super().__init__()
        if spatial_dims not in {2, 3}:
            raise ValueError("spatial_dims must be 2 or 3.")
        hidden_channels = hidden_channels or embedding_dim
        conv = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        norm = nn.BatchNorm3d if spatial_dims == 3 else nn.BatchNorm2d
        self.proj = nn.Sequential(
            conv(in_channels, hidden_channels, kernel_size=1, bias=False),
            norm(hidden_channels),
            nn.ReLU(inplace=True),
            conv(hidden_channels, embedding_dim, kernel_size=1, bias=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(features), p=2, dim=1)


@dataclass(frozen=True)
class VAPLStats:
    loss_cs: torch.Tensor
    loss_attraction: torch.Tensor
    loss_repulsion: torch.Tensor
    positive_probability: torch.Tensor
    negative_probability: torch.Tensor
    hard_fraction: torch.Tensor
    valid_pixels: torch.Tensor
    proxy_assignment_accuracy: torch.Tensor
    proxy_sigma_mean: torch.Tensor


class CompositionalSimilarityLoss(nn.Module):
    """Compositional Similarity Loss from the paper methodology.

    ``softmax_scope="per_class"`` is the literal formula in the PDF:
    p_sub is normalized across variation vectors inside each class.
    ``softmax_scope="global"`` is kept only for diagnostics and is not
    used by the default reproduction path.
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 256,
        num_variations: int = 5,
        lambda_var: float = 1.0,
        tau: float = 10.0,
        gamma: float = 2.0,
        tau_r: float = 0.8,
        lambda_r: float = 1.0,
        ignore_index: int = 255,
        softmax_scope: SoftmaxScope = "per_class",
        proxy_sigma_min: float = 0.05,
        eps: float = 1.0e-7,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive.")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive.")
        if num_variations < 1:
            raise ValueError("num_variations must be positive.")
        if softmax_scope not in {"per_class", "global"}:
            raise ValueError("softmax_scope must be 'per_class' or 'global'.")

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.num_variations = num_variations
        self.lambda_var = lambda_var
        self.tau = tau
        self.gamma = gamma
        self.tau_r = tau_r
        self.lambda_r = lambda_r
        self.ignore_index = ignore_index
        self.softmax_scope = softmax_scope
        self.proxy_sigma_min = proxy_sigma_min
        self.eps = eps

        # Replaces the single-point representative proxy with an SCDL-style
        # per-class Gaussian (mu, sigma), stored as a single [C, 2*D] tensor.
        self.proxy_dist = nn.Parameter(
            torch.empty(num_classes, embedding_dim * 2)
        )
        self.variation_vectors = nn.Parameter(
            torch.empty(num_classes, num_variations, embedding_dim)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proxy_dist)
        nn.init.xavier_uniform_(self.variation_vectors)

    def forward(
        self, embeddings: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, VAPLStats]:
        if embeddings.ndim not in {4, 5}:
            raise ValueError("embeddings must have shape [B, D, H, W] or [B, D, Z, H, W].")
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"expected embedding dim {self.embedding_dim}, "
                f"got {embeddings.shape[1]}"
            )

        targets = self._resize_targets(targets, embeddings.shape[2:])
        flat_embeddings, flat_targets = self._flatten_valid(embeddings, targets)

        if flat_embeddings.numel() == 0:
            zero = embeddings.sum() * 0.0
            stats = VAPLStats(
                loss_cs=zero,
                loss_attraction=zero,
                loss_repulsion=zero,
                positive_probability=zero,
                negative_probability=zero,
                hard_fraction=zero,
                valid_pixels=torch.zeros((), device=embeddings.device),
                proxy_assignment_accuracy=zero,
                proxy_sigma_mean=zero,
            )
            return zero, stats

        x = F.normalize(flat_embeddings, p=2, dim=1)
        arange = torch.arange(flat_targets.numel(), device=flat_targets.device)

        # SCDL-style distribution proxy: q[:, c] is the probability that a
        # token belongs to class c, derived from the per-class Gaussian
        # (mu_c, sigma_c). This replaces the single-point representative
        # proxy from the original factorized similarity score.
        q, sigma_c = self._proxy_assignment_probabilities(x)
        # Per-class softmax over variation vectors (unchanged from the
        # original formulation, minus the additive proxy_sim term that was
        # cancelled by this same softmax).
        p_sub = self._variation_subdistribution(x)
        # Joint distribution over (class, variation): combined[n, c, k] =
        # q_c(x_n) * p_sub(x_n, v_{c,k} | c). Sums to 1 over (c, k).
        combined = q.unsqueeze(-1) * p_sub

        p_pos = combined[arange, flat_targets].amax(dim=1).clamp_min(self.eps)
        loss_attraction = -torch.log(p_pos).mean()

        p_neg = self._negative_probability(combined, flat_targets)
        p_neg_for_log = p_neg.clamp(min=0.0, max=1.0 - self.eps)
        ratio = p_neg_for_log / p_pos.clamp_min(self.eps)
        hard_mask = ratio > self.tau_r

        if hard_mask.any():
            focal_weight = p_neg_for_log[hard_mask].pow(self.gamma)
            loss_repulsion = -(
                focal_weight * torch.log1p(-p_neg_for_log[hard_mask])
            ).mean()
        else:
            loss_repulsion = loss_attraction.new_zeros(())

        loss_cs = loss_attraction + self.lambda_r * loss_repulsion
        proxy_assignment_accuracy = (q.argmax(dim=1) == flat_targets).float().mean()
        stats = VAPLStats(
            loss_cs=loss_cs,
            loss_attraction=loss_attraction,
            loss_repulsion=loss_repulsion,
            positive_probability=p_pos.detach().mean(),
            negative_probability=p_neg_for_log.detach().mean(),
            hard_fraction=hard_mask.float().detach().mean(),
            valid_pixels=torch.as_tensor(
                flat_targets.numel(), device=embeddings.device, dtype=torch.float32
            ),
            proxy_assignment_accuracy=proxy_assignment_accuracy.detach(),
            proxy_sigma_mean=sigma_c.detach().mean(),
        )
        return loss_cs, stats

    def _resize_targets(
        self, targets: torch.Tensor, size: tuple[int, ...]
    ) -> torch.Tensor:
        if targets.ndim in {4, 5} and targets.shape[1] == 1:
            targets = targets[:, 0]
        expected_ndim = len(size) + 1
        if targets.ndim != expected_ndim:
            raise ValueError(
                "targets must have shape [B, ...] or [B, 1, ...] matching embeddings."
            )
        if targets.shape[1:] == size:
            return targets.long()
        resized = F.interpolate(
            targets.unsqueeze(1).float(), size=size, mode="nearest"
        )
        return resized[:, 0].long()

    def _flatten_valid(
        self, embeddings: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_embeddings = embeddings.movedim(1, -1).reshape(-1, embeddings.shape[1])
        flat_targets = targets.reshape(-1)
        valid = (
            (flat_targets != self.ignore_index)
            & (flat_targets >= 0)
            & (flat_targets < self.num_classes)
        )
        return flat_embeddings[valid], flat_targets[valid]

    def _proxy_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.proxy_dist[:, : self.embedding_dim]
        sigma = F.softplus(self.proxy_dist[:, self.embedding_dim :])
        sigma = sigma.clamp_min(self.proxy_sigma_min)
        return mu, sigma

    def _proxy_assignment_probabilities(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """SCDL-style class assignment probability q_c(x) = softmax_c(sim(x, mu_c) / sigma_c)."""
        mu, sigma = self._proxy_params()
        mu_norm = F.normalize(mu, p=2, dim=1)
        sigma_c = sigma.mean(dim=1)
        proxy_logits = torch.matmul(x, mu_norm.t()) / sigma_c.unsqueeze(0)
        q = torch.softmax(proxy_logits, dim=1)
        return q, sigma_c

    def _variation_subdistribution(self, x: torch.Tensor) -> torch.Tensor:
        variations = F.normalize(self.variation_vectors, p=2, dim=-1)
        variation_sim = torch.einsum("nd,ckd->nck", x, variations)
        scores = self.lambda_var * variation_sim

        if self.softmax_scope == "per_class":
            return torch.softmax(self.tau * scores, dim=2)

        flat_scores = scores.flatten(1)
        flat_prob = torch.softmax(self.tau * flat_scores, dim=1)
        return flat_prob.view(-1, self.num_classes, self.num_variations)

    def _negative_probability(
        self, joint_prob: torch.Tensor, flat_targets: torch.Tensor
    ) -> torch.Tensor:
        if self.num_classes == 1:
            return torch.zeros_like(joint_prob[:, 0, 0])

        neg = joint_prob.clone()
        arange = torch.arange(flat_targets.numel(), device=flat_targets.device)
        neg[arange, flat_targets, :] = -torch.inf
        return neg.flatten(1).amax(dim=1)
