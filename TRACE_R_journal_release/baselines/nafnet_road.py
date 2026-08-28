"""Faithful NAFNet baseline and legacy compact compatibility model.

``NAFNetRoad`` follows the official ECCV 2022 NAFNet implementation and the
released GoPro width-32 configuration exactly:

    width=32, enc_blk_nums=(1, 1, 1, 28), middle_blk_num=1,
    dec_blk_nums=(1, 1, 1, 1).

The implementation is dependency-free but preserves the official parameter
names and tensor shapes, so the authors' released ``params`` state dictionary
loads with ``strict=True``. See https://github.com/megvii-research/NAFNet
(MIT license). ``CompactNAFNetRoad`` is retained only to read historical local
checkpoints; it must not be reported as the paper's NAFNet baseline.
"""

from __future__ import annotations

# TRACE-R integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>.
# NAFNet architecture: Liangyu Chen, Xiaojie Chu, Xiangyu Zhang, and Jian Sun.

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


OFFICIAL_GOPRO_WIDTH32 = {
    "variant": "official_eccv2022_gopro_width32",
    "width": 32,
    "enc_blk_nums": [1, 1, 1, 28],
    "middle_blk_num": 1,
    "dec_blk_nums": [1, 1, 1, 1],
}


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm matching the official NAFNet parameter layout."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = float(eps)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        mean = image.mean(dim=1, keepdim=True)
        variance = (image - mean).pow(2).mean(dim=1, keepdim=True)
        normalized = (image - mean) * torch.rsqrt(variance + self.eps)
        return normalized * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        first, second = features.chunk(2, dim=1)
        return first * second


class NAFBlock(nn.Module):
    """Official nonlinear-activation-free block from Chen et al. (ECCV 2022)."""

    def __init__(
        self,
        channels: int,
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
    ) -> None:
        super().__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand
        self.conv1 = nn.Conv2d(channels, dw_channels, 1)
        self.conv2 = nn.Conv2d(
            dw_channels, dw_channels, 3, padding=1, groups=dw_channels
        )
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, 1)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channels // 2, dw_channels // 2, 1),
        )
        self.sg = SimpleGate()
        self.conv4 = nn.Conv2d(channels, ffn_channels, 1)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, 1)
        self.norm1 = LayerNorm2d(channels)
        self.norm2 = LayerNorm2d(channels)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.conv1(self.norm1(inputs))
        features = self.conv2(features)
        features = self.sg(features)
        features = features * self.sca(features)
        features = self.dropout1(self.conv3(features))
        residual = inputs + features * self.beta
        features = self.conv4(self.norm2(residual))
        features = self.sg(features)
        features = self.dropout2(self.conv5(features))
        return residual + features * self.gamma


class NAFNetRoad(nn.Module):
    """Official NAFNet initialized from the released GoPro deblurring model."""

    def __init__(
        self,
        width: int = 32,
        enc_blk_nums: Sequence[int] = (1, 1, 1, 28),
        middle_blk_num: int = 1,
        dec_blk_nums: Sequence[int] = (1, 1, 1, 1),
    ) -> None:
        super().__init__()
        if len(enc_blk_nums) != len(dec_blk_nums):
            raise ValueError("NAFNet encoder and decoder depths must have equal length")
        self.width = int(width)
        self.enc_blk_nums = tuple(int(value) for value in enc_blk_nums)
        self.middle_blk_num = int(middle_blk_num)
        self.dec_blk_nums = tuple(int(value) for value in dec_blk_nums)
        self.intro = nn.Conv2d(3, self.width, 3, padding=1)
        self.ending = nn.Conv2d(self.width, 3, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        channels = self.width
        for block_count in self.enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(channels) for _ in range(block_count)])
            )
            self.downs.append(nn.Conv2d(channels, 2 * channels, 2, stride=2))
            channels *= 2
        self.middle_blks = nn.Sequential(
            *[NAFBlock(channels) for _ in range(self.middle_blk_num)]
        )
        for block_count in self.dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(channels, 2 * channels, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            channels //= 2
            self.decoders.append(
                nn.Sequential(*[NAFBlock(channels) for _ in range(block_count)])
            )
        self.padder_size = 2 ** len(self.encoders)

    @property
    def architecture(self) -> dict[str, object]:
        return {
            "variant": "official_eccv2022_gopro_width32",
            "width": self.width,
            "enc_blk_nums": list(self.enc_blk_nums),
            "middle_blk_num": self.middle_blk_num,
            "dec_blk_nums": list(self.dec_blk_nums),
        }

    def check_image_size(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        pad_h = (self.padder_size - height % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - width % self.padder_size) % self.padder_size
        return F.pad(image, (0, pad_w, 0, pad_h))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        padded = self.check_image_size(image)
        features = self.intro(padded)
        skips: list[torch.Tensor] = []
        for encoder, downsample in zip(self.encoders, self.downs):
            features = encoder(features)
            skips.append(features)
            features = downsample(features)
        features = self.middle_blks(features)
        for decoder, upsample, skip in zip(self.decoders, self.ups, reversed(skips)):
            features = decoder(upsample(features) + skip)
        restored = self.ending(features) + padded
        return restored[:, :, :height, :width]


class _CompactLayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = float(eps)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        mean = features.mean(dim=1, keepdim=True)
        variance = (features - mean).pow(2).mean(dim=1, keepdim=True)
        return (features - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class CompactNAFBlock(nn.Module):
    """Historical local block retained only for checkpoint compatibility."""

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        self.norm1 = _CompactLayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, hidden * 2, 1)
        self.dwconv = nn.Conv2d(
            hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2
        )
        self.gate = SimpleGate()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden, hidden, 1)
        )
        self.conv2 = nn.Conv2d(hidden, channels, 1)
        self.norm2 = _CompactLayerNorm2d(channels)
        self.ffn1 = nn.Conv2d(channels, hidden * 2, 1)
        self.ffn2 = nn.Conv2d(hidden, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.gate(self.dwconv(self.conv1(self.norm1(inputs))))
        features = self.conv2(features * self.channel_attn(features))
        residual = inputs + features * self.beta
        features = self.ffn2(self.gate(self.ffn1(self.norm2(residual))))
        return residual + features * self.gamma


class CompactNAFNetRoad(nn.Module):
    """Legacy 1.21M-parameter NAF-style model; not the NAFNet benchmark."""

    def __init__(self, width: int = 32, blocks_per_stage: int = 2) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, width, 3, padding=1)
        self.enc1 = nn.Sequential(*[CompactNAFBlock(width) for _ in range(blocks_per_stage)])
        self.down1 = nn.Conv2d(width, width * 2, 2, stride=2)
        self.enc2 = nn.Sequential(*[CompactNAFBlock(width * 2) for _ in range(blocks_per_stage)])
        self.down2 = nn.Conv2d(width * 2, width * 4, 2, stride=2)
        self.mid = nn.Sequential(*[CompactNAFBlock(width * 4) for _ in range(blocks_per_stage + 1)])
        self.up2 = nn.Conv2d(width * 4, width * 2, 1)
        self.dec2 = nn.Sequential(*[CompactNAFBlock(width * 2) for _ in range(blocks_per_stage)])
        self.up1 = nn.Conv2d(width * 2, width, 1)
        self.dec1 = nn.Sequential(*[CompactNAFBlock(width) for _ in range(blocks_per_stage)])
        self.head = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        first = self.enc1(self.stem(image))
        second = self.enc2(self.down1(first))
        features = self.mid(self.down2(second))
        features = F.interpolate(features, size=second.shape[-2:], mode="bilinear", align_corners=False)
        features = self.dec2(self.up2(features) + second)
        features = F.interpolate(features, size=first.shape[-2:], mode="bilinear", align_corners=False)
        features = self.dec1(self.up1(features) + first)
        return torch.clamp(image + self.head(features), 0.0, 1.0)


def build_nafnet_from_payload(
    checkpoint: dict[str, object],
) -> tuple[nn.Module, dict[str, torch.Tensor], bool]:
    """Instantiate the checkpoint-declared NAF architecture without guessing."""

    arch_value = checkpoint.get("arch", {})
    arch = dict(arch_value) if isinstance(arch_value, dict) else {}
    state: object = checkpoint.get("model")
    if not isinstance(state, dict):
        state = checkpoint.get("params_ema")
    if not isinstance(state, dict):
        state = checkpoint.get("params")
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise TypeError("Checkpoint does not contain a valid NAFNet state dictionary")
    official = (
        str(arch.get("variant", "")).startswith("official_eccv2022")
        or "intro.weight" in state
    )
    if official:
        model: nn.Module = NAFNetRoad(
            width=int(arch.get("width", 32)),
            enc_blk_nums=arch.get("enc_blk_nums", (1, 1, 1, 28)),
            middle_blk_num=int(arch.get("middle_blk_num", 1)),
            dec_blk_nums=arch.get("dec_blk_nums", (1, 1, 1, 1)),
        )
    else:
        model = CompactNAFNetRoad(width=int(arch.get("width", 32)))
    typed_state = dict(state)
    model.load_state_dict(typed_state, strict=True)
    return model, typed_state, official


__all__ = [
    "build_nafnet_from_payload",
    "CompactNAFBlock",
    "CompactNAFNetRoad",
    "LayerNorm2d",
    "NAFBlock",
    "NAFNetRoad",
    "OFFICIAL_GOPRO_WIDTH32",
]
