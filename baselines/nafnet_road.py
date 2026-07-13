from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Compact NAFNet-style block for a locally trained road baseline."""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, hidden * 2, 1)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2)
        self.gate = SimpleGate()
        self.channel_attn = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden, hidden, 1))
        self.conv2 = nn.Conv2d(hidden, channels, 1)
        self.norm2 = LayerNorm2d(channels)
        self.ffn1 = nn.Conv2d(channels, hidden * 2, 1)
        self.ffn2 = nn.Conv2d(hidden, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.gate(y)
        y = y * self.channel_attn(y)
        y = self.conv2(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.ffn1(y)
        y = self.gate(y)
        y = self.ffn2(y)
        return x + y * self.gamma


class NAFNetRoad(nn.Module):
    """Small NAFNet-style encoder-decoder trained on the same road pairs."""

    def __init__(self, width: int = 32, blocks_per_stage: int = 2) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        self.enc1 = nn.Sequential(*[NAFBlock(width) for _ in range(blocks_per_stage)])
        self.down1 = nn.Conv2d(width, width * 2, 2, stride=2)
        self.enc2 = nn.Sequential(*[NAFBlock(width * 2) for _ in range(blocks_per_stage)])
        self.down2 = nn.Conv2d(width * 2, width * 4, 2, stride=2)
        self.mid = nn.Sequential(*[NAFBlock(width * 4) for _ in range(blocks_per_stage + 1)])
        self.up2 = nn.Conv2d(width * 4, width * 2, 1)
        self.dec2 = nn.Sequential(*[NAFBlock(width * 2) for _ in range(blocks_per_stage)])
        self.up1 = nn.Conv2d(width * 2, width, 1)
        self.dec1 = nn.Sequential(*[NAFBlock(width) for _ in range(blocks_per_stage)])
        self.head = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(self.stem(image))
        x2 = self.enc2(self.down1(x1))
        x3 = self.mid(self.down2(x2))
        y = F.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec2(self.up2(y) + x2)
        y = F.interpolate(y, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec1(self.up1(y) + x1)
        return torch.clamp(image + self.head(y), 0.0, 1.0)
