"""SCDL-style semantic class distribution learning modules."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SCDLStats:
    loss_scdl: torch.Tensor
    loss_scdl_align: torch.Tensor
    loss_scdl_proxy: torch.Tensor
    loss_scdl_anchor: torch.Tensor
    scdl_true_probability: torch.Tensor
    scdl_valid_tokens: torch.Tensor


class SemanticClassDistributionLoss(nn.Module):
    """Proxy-distribution loss adapted from SCDL for 3D feature maps.

    Each semantic class owns a learnable Gaussian proxy distribution in the
    embedding space. Feature tokens are aligned to class proxies and class
    centers from annotated regions act as semantic anchors.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        embedding_dim: int = 256,
        proxy_samples: int = 8,
        tau_e2p: float = 0.2,
        tau_p2e: float = 0.2,
        proxy_loss_weight: float = 1.0,
        anchor_loss_weight: float = 0.1,
        ignore_index: int = 255,
        eps: float = 1.0e-7,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be positive.")
        if num_classes < 1:
            raise ValueError("num_classes must be positive.")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive.")
        if proxy_samples < 1:
            raise ValueError("proxy_samples must be positive.")

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.proxy_samples = proxy_samples
        self.tau_e2p = tau_e2p
        self.tau_p2e = tau_p2e
        self.proxy_loss_weight = proxy_loss_weight
        self.anchor_loss_weight = anchor_loss_weight
        self.ignore_index = ignore_index
        self.eps = eps

        self.projector = nn.Sequential(
            nn.Conv3d(in_channels, embedding_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=_groups_for(embedding_dim), num_channels=embedding_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(embedding_dim, embedding_dim, kernel_size=1, bias=True),
        )
        self.proxies = nn.Parameter(torch.empty(num_classes, embedding_dim * 2))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proxies)

    def forward(
        self, features: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, SCDLStats]:
        if features.ndim != 5:
            raise ValueError("features must have shape [B, C, D, H, W].")

        embeddings = F.normalize(self.projector(features), p=2, dim=1)
        targets = self._resize_targets(targets, features.shape[2:])
        flat_embeddings, flat_targets = self._flatten_valid(embeddings, targets)

        if flat_embeddings.numel() == 0:
            zero = features.sum() * 0.0
            stats = SCDLStats(
                loss_scdl=zero,
                loss_scdl_align=zero,
                loss_scdl_proxy=zero,
                loss_scdl_anchor=zero,
                scdl_true_probability=zero.detach(),
                scdl_valid_tokens=torch.zeros((), device=features.device),
            )
            return zero, embeddings.detach() * 0.0, stats

        mu, sigma = self._proxy_params()
        mu_norm = F.normalize(mu, p=2, dim=1)
        proxy_samples = self._sample_proxies(mu, sigma)

        logits_e2p = torch.matmul(flat_embeddings, mu_norm.t()) / self.tau_e2p
        prob_e2p = torch.softmax(logits_e2p, dim=1)
        logits_p2e = torch.matmul(flat_embeddings, mu_norm.t()) / self.tau_p2e
        prob_p2e = torch.softmax(logits_p2e, dim=1)

        arange = torch.arange(flat_targets.numel(), device=flat_targets.device)
        true_prob = prob_e2p[arange, flat_targets].clamp_min(self.eps)
        loss_align = -torch.log(true_prob).mean()
        loss_align = loss_align + F.kl_div(
            prob_e2p.clamp_min(self.eps).log(),
            prob_p2e.detach().clamp_min(self.eps),
            reduction="batchmean",
        )

        similarity = torch.einsum("nd,csd->ncs", flat_embeddings, proxy_samples)
        pos = similarity[arange, flat_targets].mean(dim=1)
        neg = self._negative_similarity(similarity, flat_targets)
        loss_proxy = torch.exp(-(pos - neg)).mean()

        loss_anchor = self._anchor_loss(flat_embeddings, flat_targets, mu_norm)
        loss_scdl = loss_align
        loss_scdl = loss_scdl + self.proxy_loss_weight * loss_proxy
        loss_scdl = loss_scdl + self.anchor_loss_weight * loss_anchor

        prior_map = self._prior_map(embeddings, mu_norm)
        stats = SCDLStats(
            loss_scdl=loss_scdl,
            loss_scdl_align=loss_align,
            loss_scdl_proxy=loss_proxy,
            loss_scdl_anchor=loss_anchor,
            scdl_true_probability=true_prob.detach().mean(),
            scdl_valid_tokens=torch.as_tensor(
                flat_targets.numel(), device=features.device, dtype=torch.float32
            ),
        )
        return loss_scdl, prior_map, stats

    def _proxy_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.proxies[:, : self.embedding_dim]
        sigma = F.softplus(self.proxies[:, self.embedding_dim :]).clamp_min(self.eps)
        return mu, sigma

    def _sample_proxies(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(
            self.num_classes,
            self.proxy_samples,
            self.embedding_dim,
            device=mu.device,
            dtype=mu.dtype,
        )
        samples = mu.unsqueeze(1) + sigma.unsqueeze(1) * noise
        return F.normalize(samples, p=2, dim=-1)

    def _resize_targets(self, targets: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
        if targets.ndim == 5 and targets.shape[1] == 1:
            targets = targets[:, 0]
        if targets.ndim != 4:
            raise ValueError("targets must have shape [B, D, H, W] or [B, 1, D, H, W].")
        if targets.shape[1:] == size:
            return targets.long()
        resized = F.interpolate(targets.unsqueeze(1).float(), size=size, mode="nearest")
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

    def _negative_similarity(
        self, similarity: torch.Tensor, flat_targets: torch.Tensor
    ) -> torch.Tensor:
        if self.num_classes == 1:
            return torch.zeros_like(similarity[:, 0, 0])
        neg = similarity.clone()
        arange = torch.arange(flat_targets.numel(), device=flat_targets.device)
        neg[arange, flat_targets, :] = -torch.inf
        return neg.amax(dim=(1, 2))

    def _anchor_loss(
        self,
        flat_embeddings: torch.Tensor,
        flat_targets: torch.Tensor,
        mu_norm: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        for class_index in torch.unique(flat_targets):
            class_mask = flat_targets == class_index
            if not class_mask.any():
                continue
            center = flat_embeddings[class_mask].mean(dim=0)
            center = F.normalize(center, p=2, dim=0)
            proxy = mu_norm[class_index]
            losses.append(1.0 - torch.sum(center * proxy))
        if not losses:
            return flat_embeddings.sum() * 0.0
        return torch.stack(losses).mean()

    def _prior_map(self, embeddings: torch.Tensor, mu_norm: torch.Tensor) -> torch.Tensor:
        flat_embeddings = embeddings.movedim(1, -1).reshape(-1, embeddings.shape[1])
        probs = torch.softmax(torch.matmul(flat_embeddings, mu_norm.t()) / self.tau_p2e, dim=1)
        prior_tokens = torch.matmul(probs, mu_norm)
        prior = prior_tokens.view(*embeddings.shape[0:1], *embeddings.shape[2:], self.embedding_dim)
        return F.normalize(prior.movedim(-1, 1).contiguous(), p=2, dim=1)


def _groups_for(channels: int, max_groups: int = 8) -> int:
    groups = min(max_groups, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups
