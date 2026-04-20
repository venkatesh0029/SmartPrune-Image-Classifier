"""Learnable gated layers: y uses (W ⊙ σ(G)) instead of W for linear and conv."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """
    Each weight is multiplied by a sigmoid gate so the network can drive
    unimportant connections toward zero during training.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / fan_in**0.5 if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.uniform_(self.gate_scores, -0.1, 0.1)

    def effective_weight(self) -> torch.Tensor:
        return self.weight * torch.sigmoid(self.gate_scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)

    def gate_values(self) -> torch.Tensor:
        """σ(G), same shape as weight; detached for metrics/plots."""
        return torch.sigmoid(self.gate_scores.detach())


class PrunableConv2d(nn.Module):
    """2D convolution with per-weight gates: effective kernel = W ⊙ σ(G)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kh, kw)
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kh, kw))
        self.gate_scores = nn.Parameter(torch.empty(out_channels, in_channels, kh, kw))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / fan_in**0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        nn.init.uniform_(self.gate_scores, -0.1, 0.1)

    def effective_weight(self) -> torch.Tensor:
        return self.weight * torch.sigmoid(self.gate_scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.effective_weight(),
            self.bias,
            self.stride,
            self.padding,
        )

    def gate_values(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores.detach())
