"""3D medical segmentation backbone for SCDL-style experiments.

The network is a compact V-Net/MagicNet-style encoder-decoder that exposes
intermediate decoder features for VAPL. It is intentionally kept independent
from PIDNet so the original 2D Cityscapes path remains unchanged.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


Normalization = Literal["none", "batchnorm", "instancenorm", "groupnorm"]
FeatureStage = Literal["decoder_12", "decoder_24", "decoder_48", "decoder_full"]


def _normalization(channels: int, normalization: Normalization) -> nn.Module:
    if normalization == "batchnorm":
        return nn.BatchNorm3d(channels)
    if normalization == "instancenorm":
        return nn.InstanceNorm3d(channels)
    if normalization == "groupnorm":
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if normalization == "none":
        return nn.Identity()
    raise ValueError(f"unsupported normalization: {normalization}")


class ConvBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        normalization: Normalization,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for index in range(num_layers):
            current_in = in_channels if index == 0 else out_channels
            layers.extend(
                [
                    nn.Conv3d(current_in, out_channels, kernel_size=3, padding=1),
                    _normalization(out_channels, normalization),
                    nn.ReLU(inplace=True),
                ]
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownsampleBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: Normalization,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2),
            _normalization(out_channels, normalization),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpsampleBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalization: Normalization,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            _normalization(out_channels, normalization),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SCDLVNet3D(nn.Module):
    """V-Net/MagicNet-style 3D segmentation backbone.

    ``features`` are exposed from a configurable decoder stage for proxy losses.
    The default ``decoder_12`` stage is stride-8 for a 96^3 patch, matching the
    token grid used by SCDL-style distribution modules.
    """

    _stage_channels = {
        "decoder_12": 8,
        "decoder_24": 4,
        "decoder_48": 2,
        "decoder_full": 1,
    }

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 14,
        base_channels: int = 16,
        normalization: Normalization = "instancenorm",
        feature_stage: FeatureStage = "decoder_12",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if feature_stage not in self._stage_channels:
            raise ValueError(f"unsupported feature_stage: {feature_stage}")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.feature_stage = feature_stage
        self.feature_channels = base_channels * self._stage_channels[feature_stage]

        c = base_channels
        self.enc1 = ConvBlock3d(in_channels, c, 1, normalization)
        self.down1 = DownsampleBlock3d(c, c * 2, normalization)
        self.enc2 = ConvBlock3d(c * 2, c * 2, 2, normalization)
        self.down2 = DownsampleBlock3d(c * 2, c * 4, normalization)
        self.enc3 = ConvBlock3d(c * 4, c * 4, 3, normalization)
        self.down3 = DownsampleBlock3d(c * 4, c * 8, normalization)
        self.enc4 = ConvBlock3d(c * 8, c * 8, 3, normalization)
        self.down4 = DownsampleBlock3d(c * 8, c * 16, normalization)
        self.bottleneck = ConvBlock3d(c * 16, c * 16, 3, normalization)

        self.up4 = UpsampleBlock3d(c * 16, c * 8, normalization)
        self.dec4 = ConvBlock3d(c * 8, c * 8, 3, normalization)
        self.up3 = UpsampleBlock3d(c * 8, c * 4, normalization)
        self.dec3 = ConvBlock3d(c * 4, c * 4, 3, normalization)
        self.up2 = UpsampleBlock3d(c * 4, c * 2, normalization)
        self.dec2 = ConvBlock3d(c * 2, c * 2, 2, normalization)
        self.up1 = UpsampleBlock3d(c * 2, c, normalization)
        self.dec1 = ConvBlock3d(c, c, 1, normalization)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0.0 else nn.Identity()
        self.head = nn.Conv3d(c, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm3d, nn.InstanceNorm3d, nn.GroupNorm)):
                if getattr(module, "weight", None) is not None:
                    nn.init.ones_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        x1 = self.enc1(x)
        x2 = self.enc2(self.down1(x1))
        x3 = self.enc3(self.down2(x2))
        x4 = self.enc4(self.down3(x3))
        x5 = self.bottleneck(self.down4(x4))

        d4 = self.dec4(self.up4(x5) + x4)
        d3 = self.dec3(self.up3(d4) + x3)
        d2 = self.dec2(self.up2(d3) + x2)
        d1 = self.dec1(self.up1(d2) + x1)
        d1 = self.dropout(d1)
        logits = self.head(d1)

        features_by_stage = {
            "decoder_12": d4,
            "decoder_24": d3,
            "decoder_48": d2,
            "decoder_full": d1,
        }
        return {
            "logits": logits,
            "features": features_by_stage[self.feature_stage],
            "aux_logits_p": None,
            "aux_logits_d": None,
        }


def scdl_vnet_3d(
    in_channels: int = 1,
    num_classes: int = 14,
    base_channels: int = 16,
    normalization: Normalization = "instancenorm",
    feature_stage: FeatureStage = "decoder_12",
    dropout: float = 0.0,
) -> SCDLVNet3D:
    return SCDLVNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        normalization=normalization,
        feature_stage=feature_stage,
        dropout=dropout,
    )
