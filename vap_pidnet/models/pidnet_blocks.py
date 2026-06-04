"""PIDNet building blocks.

The PIDNet backbone follows the official MIT-licensed implementation:
https://github.com/XuJiacong/PIDNet
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.1
ALIGN_CORNERS = False


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        no_relu: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        if self.no_relu:
            return out
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        no_relu: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        if self.no_relu:
            return out
        return self.relu(out)


class SegmentHead(nn.Module):
    def __init__(
        self,
        inplanes: int,
        interplanes: int,
        outplanes: int,
        scale_factor: int | None = None,
    ) -> None:
        super().__init__()
        self.bn1 = BatchNorm2d(inplanes, momentum=BN_MOMENTUM)
        self.conv1 = nn.Conv2d(
            inplanes, interplanes, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = BatchNorm2d(interplanes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(
                out,
                size=(height, width),
                mode="bilinear",
                align_corners=ALIGN_CORNERS,
            )
        return out


class DAPPM(nn.Module):
    def __init__(
        self,
        inplanes: int,
        branch_planes: int,
        outplanes: int,
        batch_norm: type[nn.Module] = BatchNorm2d,
    ) -> None:
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            batch_norm(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            batch_norm(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            batch_norm(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            batch_norm(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale0 = nn.Sequential(
            batch_norm(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )

        self.process1 = self._process(branch_planes, batch_norm)
        self.process2 = self._process(branch_planes, batch_norm)
        self.process3 = self._process(branch_planes, batch_norm)
        self.process4 = self._process(branch_planes, batch_norm)
        self.compression = nn.Sequential(
            batch_norm(branch_planes * 5, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            batch_norm(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    @staticmethod
    def _process(
        branch_planes: int, batch_norm: type[nn.Module]
    ) -> nn.Sequential:
        return nn.Sequential(
            batch_norm(branch_planes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]

        x0 = self.scale0(x)
        x1 = self.process1(
            F.interpolate(
                self.scale1(x), size=(height, width), mode="bilinear",
                align_corners=ALIGN_CORNERS
            )
            + x0
        )
        x2 = self.process2(
            F.interpolate(
                self.scale2(x), size=(height, width), mode="bilinear",
                align_corners=ALIGN_CORNERS
            )
            + x1
        )
        x3 = self.process3(
            F.interpolate(
                self.scale3(x), size=(height, width), mode="bilinear",
                align_corners=ALIGN_CORNERS
            )
            + x2
        )
        x4 = self.process4(
            F.interpolate(
                self.scale4(x), size=(height, width), mode="bilinear",
                align_corners=ALIGN_CORNERS
            )
            + x3
        )

        out = self.compression(torch.cat([x0, x1, x2, x3, x4], dim=1))
        return out + self.shortcut(x)


class PAPPM(nn.Module):
    def __init__(
        self,
        inplanes: int,
        branch_planes: int,
        outplanes: int,
        batch_norm: type[nn.Module] = BatchNorm2d,
    ) -> None:
        super().__init__()
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            batch_norm(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            batch_norm(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            batch_norm(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            batch_norm(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale0 = nn.Sequential(
            batch_norm(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        self.scale_process = nn.Sequential(
            batch_norm(branch_planes * 4, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                branch_planes * 4,
                branch_planes * 4,
                kernel_size=3,
                padding=1,
                groups=4,
                bias=False,
            ),
        )
        self.compression = nn.Sequential(
            batch_norm(branch_planes * 5, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            batch_norm(inplanes, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        x0 = self.scale0(x)
        scale_list = [
            F.interpolate(
                self.scale1(x), size=(height, width), mode="bilinear",
                align_corners=ALIGN_CORNERS
            )
            + x0,
            F.interpolate(
                self.scale2(x), size=(height, width), mode="bilinear",
                align_corners=ALIGN_CORNERS
            )
            + x0,
            F.interpolate(
                self.scale3(x), size=(height, width), mode="bilinear",
                align_corners=ALIGN_CORNERS
            )
            + x0,
            F.interpolate(
                self.scale4(x), size=(height, width), mode="bilinear",
                align_corners=ALIGN_CORNERS
            )
            + x0,
        ]
        scale_out = self.scale_process(torch.cat(scale_list, dim=1))
        out = self.compression(torch.cat([x0, scale_out], dim=1))
        return out + self.shortcut(x)


class PagFM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        after_relu: bool = False,
        with_channel: bool = False,
        batch_norm: type[nn.Module] = BatchNorm2d,
    ) -> None:
        super().__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            batch_norm(mid_channels),
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            batch_norm(mid_channels),
        )
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
                batch_norm(in_channels),
            )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        input_size = x.size()
        if self.after_relu:
            x = self.relu(x)
            y = self.relu(y)

        y_q = self.f_y(y)
        y_q = F.interpolate(
            y_q,
            size=(input_size[2], input_size[3]),
            mode="bilinear",
            align_corners=ALIGN_CORNERS,
        )
        x_k = self.f_x(x)
        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1, keepdim=True))

        y = F.interpolate(
            y,
            size=(input_size[2], input_size[3]),
            mode="bilinear",
            align_corners=ALIGN_CORNERS,
        )
        return (1 - sim_map) * x + sim_map * y


class LightBag(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        batch_norm: type[nn.Module] = BatchNorm2d,
    ) -> None:
        super().__init__()
        self.conv_p = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            batch_norm(out_channels),
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            batch_norm(out_channels),
        )

    def forward(
        self, p: torch.Tensor, i: torch.Tensor, d: torch.Tensor
    ) -> torch.Tensor:
        edge_att = torch.sigmoid(d)
        p_add = self.conv_p((1 - edge_att) * i + p)
        i_add = self.conv_i(i + edge_att * p)
        return p_add + i_add


class Bag(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        batch_norm: type[nn.Module] = BatchNorm2d,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            batch_norm(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(
        self, p: torch.Tensor, i: torch.Tensor, d: torch.Tensor
    ) -> torch.Tensor:
        edge_att = torch.sigmoid(d)
        return self.conv(edge_att * p + (1 - edge_att) * i)
