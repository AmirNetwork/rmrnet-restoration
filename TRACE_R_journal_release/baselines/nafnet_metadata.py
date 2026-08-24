# TRACE-R release integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
from __future__ import annotations

"""Metadata-conditioned NAFNet control with a declared conditioning budget."""

import torch
import torch.nn.functional as F
from torch import nn

from .nafnet_road import NAFBlock


class CodeFiLM(nn.Module):
    """Apply FiLM(x, z) = (1 + gamma(z)) * x + beta(z)."""

    def __init__(self, code_dim: int, channels: int) -> None:
        super().__init__()
        self.affine = nn.Linear(code_dim, 2 * channels)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, features: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.affine(code).chunk(2, dim=1)
        return (1.0 + gamma[:, :, None, None]) * features + beta[:, :, None, None]


class MetadataNAFNetRoad(nn.Module):
    """NAFNet-road with stage-wise FiLM from the same metadata packet as RMR-Net.

    This control has no image-code estimator, reliability gate, task-evidence
    branch, or detail skip. The matched trainer can still apply the same
    external frozen-detector losses used for every restorer. It isolates
    whether stage-wise FiLM from the same continuous sensor packet is enough to
    explain a gain. ``code_dim`` is stored in every checkpoint so inference
    reconstructs the exact declared input budget.
    """

    def __init__(self, width: int = 32, code_dim: int = 8, blocks_per_stage: int = 2) -> None:
        super().__init__()
        self.width = int(width)
        self.code_dim = int(code_dim)
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
        self.film1 = CodeFiLM(code_dim, width)
        self.film2 = CodeFiLM(code_dim, width * 2)
        self.film3 = CodeFiLM(code_dim, width * 4)
        self.film4 = CodeFiLM(code_dim, width * 2)
        self.film5 = CodeFiLM(code_dim, width)

    def forward(self, image: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        if code.dim() == 1:
            code = code.unsqueeze(0)
        if code.shape[0] == 1 and image.shape[0] > 1:
            code = code.expand(image.shape[0], -1)
        x1 = self.film1(self.enc1(self.stem(image)), code)
        x2 = self.film2(self.enc2(self.down1(x1)), code)
        x3 = self.film3(self.mid(self.down2(x2)), code)
        y = F.interpolate(x3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.film4(self.dec2(self.up2(y) + x2), code)
        y = F.interpolate(y, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.film5(self.dec1(self.up1(y) + x1), code)
        return torch.clamp(image + self.head(y), 0.0, 1.0)
