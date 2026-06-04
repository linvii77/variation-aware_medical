"""End-to-end VAPL models using segmentation backbones."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .models import PIDNet, SCDLVNet3D, pidnet_m, scdl_vnet_3d
from .scdl import SemanticClassDistributionLoss
from .vapl import CompositionalSimilarityLoss, ProjectionHead, SoftmaxScope


class VAPLPIDNetM(nn.Module):
    """PIDNet-M with Variation-Aware Proxy Learning for training.

    Inference with ``targets=None`` only runs the PIDNet-M segmentation path.
    Projection/proxy modules are evaluated only when targets are supplied or
    ``return_embeddings=True`` is explicitly requested.
    """

    def __init__(
        self,
        num_classes: int = 19,
        embedding_dim: int = 256,
        lambda_cs: float = 1.0,
        aux_loss_weight: float = 0.4,
        ignore_index: int = 255,
        augment: bool = True,
        num_variations: int = 5,
        lambda_var: float = 1.0,
        tau: float = 10.0,
        gamma: float = 2.0,
        tau_r: float = 0.8,
        lambda_r: float = 1.0,
        softmax_scope: SoftmaxScope = "per_class",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.lambda_cs = lambda_cs
        self.aux_loss_weight = aux_loss_weight
        self.ignore_index = ignore_index

        self.backbone: PIDNet = pidnet_m(num_classes=num_classes, augment=augment)
        self.projection_head = ProjectionHead(
            in_channels=self.backbone.feature_channels,
            embedding_dim=embedding_dim,
        )
        self.cs_loss = CompositionalSimilarityLoss(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            num_variations=num_variations,
            lambda_var=lambda_var,
            tau=tau,
            gamma=gamma,
            tau_r=tau_r,
            lambda_r=lambda_r,
            ignore_index=ignore_index,
            softmax_scope=softmax_scope,
        )

    def forward(
        self,
        images: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_embeddings: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor] | None]:
        pid_out = self.backbone(images)

        logits_lowres = pid_out["logits"]
        logits = self._upsample_logits(
            logits_lowres,
            targets.shape[-2:] if targets is not None else images.shape[-2:],
        )

        outputs: dict[str, torch.Tensor | dict[str, torch.Tensor] | None] = {
            "logits": logits,
            "logits_lowres": logits_lowres,
            "features": pid_out["features"],
            "aux_logits_p": pid_out["aux_logits_p"],
            "aux_logits_d": pid_out["aux_logits_d"],
            "embeddings": None,
            "losses": None,
        }

        if targets is None and not return_embeddings:
            return outputs

        embeddings = None
        if self.lambda_cs > 0.0 or return_embeddings:
            embeddings = self.projection_head(pid_out["features"])
            outputs["embeddings"] = embeddings

        if targets is None:
            return outputs

        targets_3d = self._targets_3d(targets)
        loss_seg = F.cross_entropy(
            logits,
            targets_3d.long(),
            ignore_index=self.ignore_index,
        )

        loss_aux = logits.new_zeros(())
        aux_logits_p = pid_out["aux_logits_p"]
        if self.aux_loss_weight > 0.0 and aux_logits_p is not None:
            aux_up = self._upsample_logits(aux_logits_p, targets_3d.shape[-2:])
            loss_aux = F.cross_entropy(
                aux_up,
                targets_3d.long(),
                ignore_index=self.ignore_index,
            )

        if self.lambda_cs > 0.0:
            if embeddings is None:
                raise RuntimeError("VAPL embeddings were not computed.")
            loss_cs, stats = self.cs_loss(embeddings, targets_3d)
            stats_dict = vars(stats)
        else:
            loss_cs = logits.new_zeros(())
            stats_dict = {
                "loss_cs": loss_cs,
                "loss_attraction": loss_cs,
                "loss_repulsion": loss_cs,
                "positive_probability": loss_cs.detach(),
                "negative_probability": loss_cs.detach(),
                "hard_fraction": loss_cs.detach(),
                "valid_pixels": loss_cs.detach(),
            }
        total_loss = loss_seg + self.aux_loss_weight * loss_aux
        total_loss = total_loss + self.lambda_cs * loss_cs

        losses = {
            "loss_total": total_loss,
            "loss_seg": loss_seg,
            "loss_aux_p": loss_aux,
            "loss_cs": loss_cs,
        }
        losses.update(stats_dict)
        outputs["losses"] = losses
        return outputs

    @staticmethod
    def _upsample_logits(
        logits: torch.Tensor | None, size: tuple[int, int]
    ) -> torch.Tensor | None:
        if logits is None:
            return None
        if logits.shape[-2:] == size:
            return logits
        return F.interpolate(
            logits,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    @staticmethod
    def _targets_3d(targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 4 and targets.shape[1] == 1:
            return targets[:, 0]
        if targets.ndim != 3:
            raise ValueError("targets must have shape [B, H, W] or [B, 1, H, W].")
        return targets


def build_vapl_pidnet_m(
    num_classes: int = 19,
    embedding_dim: int = 256,
    lambda_cs: float = 1.0,
    ignore_index: int = 255,
) -> VAPLPIDNetM:
    """Convenience builder with the paper's default VAPL hyperparameters."""

    return VAPLPIDNetM(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        lambda_cs=lambda_cs,
        ignore_index=ignore_index,
    )


class VAPLSCDL3D(nn.Module):
    """3D SCDL-style medical backbone with VAPL training loss.

    This wrapper is the 3D counterpart of ``VAPLPIDNetM``. It keeps the same
    output dictionary contract but expects volumes shaped ``[B, C, D, H, W]``
    and targets shaped ``[B, D, H, W]`` or ``[B, 1, D, H, W]``.
    """

    def __init__(
        self,
        num_classes: int = 14,
        in_channels: int = 1,
        base_channels: int = 16,
        embedding_dim: int = 256,
        lambda_cs: float = 1.0,
        lambda_scdl: float = 0.0,
        ignore_index: int = 255,
        num_variations: int = 5,
        lambda_var: float = 1.0,
        tau: float = 10.0,
        gamma: float = 2.0,
        tau_r: float = 0.8,
        lambda_r: float = 1.0,
        softmax_scope: SoftmaxScope = "per_class",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.lambda_cs = lambda_cs
        self.lambda_scdl = lambda_scdl
        self.ignore_index = ignore_index

        self.backbone: SCDLVNet3D = scdl_vnet_3d(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
        )
        self.projection_head = ProjectionHead(
            in_channels=self.backbone.feature_channels,
            embedding_dim=embedding_dim,
            spatial_dims=3,
        )
        self.cs_loss = CompositionalSimilarityLoss(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            num_variations=num_variations,
            lambda_var=lambda_var,
            tau=tau,
            gamma=gamma,
            tau_r=tau_r,
            lambda_r=lambda_r,
            ignore_index=ignore_index,
            softmax_scope=softmax_scope,
        )
        self.scdl_loss = SemanticClassDistributionLoss(
            in_channels=self.backbone.feature_channels,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            ignore_index=ignore_index,
        )

    def forward(
        self,
        images: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_embeddings: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor] | None]:
        scdl_out = self.backbone(images)

        logits_raw = scdl_out["logits"]
        logits = self._upsample_logits_3d(
            logits_raw,
            targets.shape[-3:] if targets is not None else images.shape[-3:],
        )

        outputs: dict[str, torch.Tensor | dict[str, torch.Tensor] | None] = {
            "logits": logits,
            "logits_lowres": logits_raw,
            "features": scdl_out["features"],
            "aux_logits_p": None,
            "aux_logits_d": None,
            "embeddings": None,
            "scdl_prior": None,
            "losses": None,
        }

        if targets is None and not return_embeddings:
            return outputs

        embeddings = None
        if self.lambda_cs > 0.0 or return_embeddings:
            embeddings = self.projection_head(scdl_out["features"])
            outputs["embeddings"] = embeddings

        if targets is None:
            return outputs

        targets_4d = self._targets_4d(targets)
        loss_seg = F.cross_entropy(
            logits,
            targets_4d.long(),
            ignore_index=self.ignore_index,
        )

        if self.lambda_cs > 0.0:
            if embeddings is None:
                raise RuntimeError("VAPL embeddings were not computed.")
            loss_cs, stats = self.cs_loss(embeddings, targets_4d)
            stats_dict = vars(stats)
        else:
            loss_cs = logits.new_zeros(())
            stats_dict = {
                "loss_cs": loss_cs,
                "loss_attraction": loss_cs,
                "loss_repulsion": loss_cs,
                "positive_probability": loss_cs.detach(),
                "negative_probability": loss_cs.detach(),
                "hard_fraction": loss_cs.detach(),
                "valid_pixels": loss_cs.detach(),
            }

        if self.lambda_scdl > 0.0:
            loss_scdl, scdl_prior, scdl_stats = self.scdl_loss(
                scdl_out["features"], targets_4d
            )
            outputs["scdl_prior"] = scdl_prior
            scdl_stats_dict = vars(scdl_stats)
        else:
            loss_scdl = logits.new_zeros(())
            scdl_stats_dict = {
                "loss_scdl": loss_scdl,
                "loss_scdl_align": loss_scdl,
                "loss_scdl_proxy": loss_scdl,
                "loss_scdl_anchor": loss_scdl,
                "scdl_true_probability": loss_scdl.detach(),
                "scdl_valid_tokens": loss_scdl.detach(),
            }

        total_loss = loss_seg + self.lambda_cs * loss_cs
        total_loss = total_loss + self.lambda_scdl * loss_scdl
        losses = {
            "loss_total": total_loss,
            "loss_seg": loss_seg,
            "loss_aux_p": logits.new_zeros(()),
            "loss_cs": loss_cs,
            "loss_scdl": loss_scdl,
        }
        losses.update(stats_dict)
        losses.update(scdl_stats_dict)
        outputs["losses"] = losses
        return outputs

    @staticmethod
    def _upsample_logits_3d(
        logits: torch.Tensor | None, size: tuple[int, int, int]
    ) -> torch.Tensor | None:
        if logits is None:
            return None
        if logits.shape[-3:] == size:
            return logits
        return F.interpolate(
            logits,
            size=size,
            mode="trilinear",
            align_corners=False,
        )

    @staticmethod
    def _targets_4d(targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim == 5 and targets.shape[1] == 1:
            return targets[:, 0]
        if targets.ndim != 4:
            raise ValueError("targets must have shape [B, D, H, W] or [B, 1, D, H, W].")
        return targets


def build_vapl_scdl_3d(
    num_classes: int = 14,
    in_channels: int = 1,
    base_channels: int = 16,
    embedding_dim: int = 256,
    lambda_cs: float = 1.0,
    lambda_scdl: float = 0.0,
    ignore_index: int = 255,
) -> VAPLSCDL3D:
    """Build the 3D SCDL-style medical backbone with VAPL loss."""

    return VAPLSCDL3D(
        num_classes=num_classes,
        in_channels=in_channels,
        base_channels=base_channels,
        embedding_dim=embedding_dim,
        lambda_cs=lambda_cs,
        lambda_scdl=lambda_scdl,
        ignore_index=ignore_index,
    )
