"""PIDNet backbone with feature output for VAPL training.

This file follows the official PIDNet structure for PIDNet-M:
``m=2, n=3, planes=64, ppm_planes=96, head_planes=128``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .pidnet_blocks import (
    ALIGN_CORNERS,
    BN_MOMENTUM,
    Bag,
    BasicBlock,
    Bottleneck,
    DAPPM,
    LightBag,
    PAPPM,
    PagFM,
    SegmentHead,
)


class PIDNet(nn.Module):
    """Proportional-Integral-Derivative Network for semantic segmentation."""

    def __init__(
        self,
        m: int = 2,
        n: int = 3,
        num_classes: int = 19,
        planes: int = 64,
        ppm_planes: int = 96,
        head_planes: int = 128,
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.augment = augment
        self.num_classes = num_classes
        self.feature_channels = planes * 4

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)

        # Integral branch.
        self.layer1 = self._make_layer(BasicBlock, planes, planes, m)
        self.layer2 = self._make_layer(BasicBlock, planes, planes * 2, m, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes * 2, planes * 4, n, stride=2)
        self.layer4 = self._make_layer(BasicBlock, planes * 4, planes * 8, n, stride=2)
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 2, stride=2)

        # Proportional branch.
        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, planes * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 2, momentum=BN_MOMENTUM),
        )
        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, planes * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 2, momentum=BN_MOMENTUM),
        )
        self.pag3 = PagFM(planes * 2, planes)
        self.pag4 = PagFM(planes * 2, planes)
        self.layer3_p = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer4_p = self._make_layer(BasicBlock, planes * 2, planes * 2, m)
        self.layer5_p = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        # Derivative branch. PIDNet-M shares the lightweight configuration with S.
        if m == 2:
            self.layer3_d = self._make_single_layer(BasicBlock, planes * 2, planes)
            self.layer4_d = self._make_layer(Bottleneck, planes, planes, 1)
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4, planes, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(planes * 2, momentum=BN_MOMENTUM),
            )
            self.spp = PAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = LightBag(planes * 4, planes * 4)
        else:
            self.layer3_d = self._make_single_layer(
                BasicBlock, planes * 2, planes * 2
            )
            self.layer4_d = self._make_single_layer(
                BasicBlock, planes * 2, planes * 2
            )
            self.diff3 = nn.Sequential(
                nn.Conv2d(planes * 4, planes * 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(planes * 2, momentum=BN_MOMENTUM),
            )
            self.diff4 = nn.Sequential(
                nn.Conv2d(planes * 8, planes * 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(planes * 2, momentum=BN_MOMENTUM),
            )
            self.spp = DAPPM(planes * 16, ppm_planes, planes * 4)
            self.dfm = Bag(planes * 4, planes * 4)

        self.layer5_d = self._make_layer(Bottleneck, planes * 2, planes * 2, 1)

        if self.augment:
            self.seghead_p = SegmentHead(planes * 2, head_planes, num_classes)
            self.seghead_d = SegmentHead(planes * 2, planes, 1)
        self.final_layer = SegmentHead(planes * 4, head_planes, num_classes)

        self._init_weights()

    def _make_layer(
        self,
        block: type[BasicBlock] | type[Bottleneck],
        inplanes: int,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers: list[nn.Module] = [
            block(inplanes, planes, stride=stride, downsample=downsample)
        ]
        inplanes = planes * block.expansion
        for index in range(1, blocks):
            layers.append(block(inplanes, planes, no_relu=index == blocks - 1))
        return nn.Sequential(*layers)

    def _make_single_layer(
        self,
        block: type[BasicBlock] | type[Bottleneck],
        inplanes: int,
        planes: int,
        stride: int = 1,
    ) -> nn.Module:
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        return block(inplanes, planes, stride=stride, downsample=downsample, no_relu=True)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        output_size = (x.shape[-2] // 8, x.shape[-1] // 8)

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))

        p = self.layer3_p(x)
        d = self.layer3_d(x)

        x = self.relu(self.layer3(x))
        p = self.pag3(p, self.compression3(x))
        d = d + F.interpolate(
            self.diff3(x),
            size=output_size,
            mode="bilinear",
            align_corners=ALIGN_CORNERS,
        )
        aux_p_feature = p

        x = self.relu(self.layer4(x))
        p = self.layer4_p(self.relu(p))
        d = self.layer4_d(self.relu(d))

        p = self.pag4(p, self.compression4(x))
        d = d + F.interpolate(
            self.diff4(x),
            size=output_size,
            mode="bilinear",
            align_corners=ALIGN_CORNERS,
        )
        aux_d_feature = d

        p = self.layer5_p(self.relu(p))
        d = self.layer5_d(self.relu(d))
        i = F.interpolate(
            self.spp(self.layer5(x)),
            size=output_size,
            mode="bilinear",
            align_corners=ALIGN_CORNERS,
        )

        fused = self.dfm(p, i, d)
        logits = self.final_layer(fused)

        out: dict[str, torch.Tensor | None] = {
            "logits": logits,
            "features": fused,
            "aux_logits_p": None,
            "aux_logits_d": None,
        }
        if self.augment:
            out["aux_logits_p"] = self.seghead_p(aux_p_feature)
            out["aux_logits_d"] = self.seghead_d(aux_d_feature)
        return out


def pidnet_m(num_classes: int = 19, augment: bool = True) -> PIDNet:
    """Build PIDNet-M with the configuration used by the VAPL paper."""

    return PIDNet(
        m=2,
        n=3,
        num_classes=num_classes,
        planes=64,
        ppm_planes=96,
        head_planes=128,
        augment=augment,
    )
