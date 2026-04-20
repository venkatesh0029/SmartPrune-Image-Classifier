"""CIFAR-10 classifiers with prunable linear and/or convolutional layers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import PrunableConv2d, PrunableLinear


class SelfPruningMLP(nn.Module):
    """Flatten -> PrunableLinear -> ReLU x2 -> PrunableLinear -> logits."""

    def __init__(self, num_classes: int = 10, hidden1: int = 512, hidden2: int = 256) -> None:
        super().__init__()
        flat = 32 * 32 * 3
        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(flat, hidden1)
        self.fc2 = PrunableLinear(hidden1, hidden2)
        self.fc3 = PrunableLinear(hidden2, num_classes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)


class _PrunableBasicBlock(nn.Module):
    """Pre-activation style residual block with prunable convolutions."""

    def __init__(self, in_planes: int, planes: int, stride: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = PrunableConv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = PrunableConv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                PrunableConv2d(in_planes, planes, 1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x), inplace=True)
        out = self.conv1(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        return out


class SelfPruningResNet(nn.Module):
    """
    Wide-ResNet-style network with prunable convolutions and a prunable classifier.
    Designed for CIFAR-10 (32x32); uses strong capacity (width) for high accuracy.
    """

    def __init__(
        self,
        num_classes: int = 10,
        width: int = 10,
        blocks_per_stage: tuple[int, int, int] = (4, 4, 4),
    ) -> None:
        super().__init__()
        w = width
        c1 = 16 * w
        c2 = 32 * w
        c3 = 64 * w
        self.conv1 = PrunableConv2d(3, c1, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.layer1 = self._make_stage(c1, c1, blocks_per_stage[0], stride=1)
        self.layer2 = self._make_stage(c1, c2, blocks_per_stage[1], stride=2)
        self.layer3 = self._make_stage(c2, c3, blocks_per_stage[2], stride=2)
        self.bn_out = nn.BatchNorm2d(c3)
        self.fc = PrunableLinear(c3, num_classes)
        self._reset_params()

    def _reset_params(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_stage(
        self,
        in_planes: int,
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        layers.append(_PrunableBasicBlock(in_planes, planes, stride))
        for _ in range(1, num_blocks):
            layers.append(_PrunableBasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(self.bn_out(x), inplace=True)
        x = x.mean(dim=(2, 3))
        return self.fc(x)
