"""Sparse wavelet refinement for telemetry-aware road-image restoration.

The module adapts the residual-domain insight of ISTA-Net to restoration: a
strong matched restorer supplies the base image, while a small unrolled module
learns a sparse correction in an invertible Haar-wavelet domain.  The public
camera/IMU/vehicle packet is fused with image evidence by the parent TRACE-R
controller; hidden renderer parameters and benchmark scenario names are never
inference inputs.

For stage ``t`` the implemented update is

    c^(t+1) = c^t + eta_t g_t S_lambda_t(A_t(c^t, c_b, c_d, z)),

where ``c_b`` and ``c_d`` are the wavelet coefficients of the base restoration
and degraded image, ``z`` is the joint image--telemetry corruption state,
``S`` is a smooth sparsity-inducing shrinkage operator, and ``g_t`` is a
spatial compatibility gate.  The low-frequency update is separately bounded;
the three detail bands receive most of the correction capacity.
"""

from __future__ import annotations

# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from models.rmrp_metadata_demoe import RMRPMetadataDeMoE


def haar_dwt(image: torch.Tensor) -> torch.Tensor:
    """Return channel-major ``LL, LH, HL, HH`` Haar coefficients.

    The normalization makes :func:`haar_iwt` an exact inverse up to floating
    point round-off.  Inputs must have even spatial dimensions.
    """

    if image.ndim != 4:
        raise ValueError(f"Expected BCHW image, received {tuple(image.shape)}")
    if image.shape[-2] % 2 or image.shape[-1] % 2:
        raise ValueError("Haar DWT requires even height and width")
    x00 = image[..., 0::2, 0::2]
    x01 = image[..., 0::2, 1::2]
    x10 = image[..., 1::2, 0::2]
    x11 = image[..., 1::2, 1::2]
    ll = 0.5 * (x00 + x01 + x10 + x11)
    lh = 0.5 * (-x00 - x01 + x10 + x11)
    hl = 0.5 * (-x00 + x01 - x10 + x11)
    hh = 0.5 * (x00 - x01 - x10 + x11)
    return torch.cat((ll, lh, hl, hh), dim=1)


def haar_iwt(coefficients: torch.Tensor) -> torch.Tensor:
    """Invert channel-major ``LL, LH, HL, HH`` Haar coefficients."""

    if coefficients.ndim != 4 or coefficients.shape[1] % 4:
        raise ValueError(
            "Expected BCHW coefficients with a channel count divisible by four"
        )
    ll, lh, hl, hh = coefficients.chunk(4, dim=1)
    x00 = 0.5 * (ll - lh - hl + hh)
    x01 = 0.5 * (ll - lh + hl - hh)
    x10 = 0.5 * (ll + lh - hl - hh)
    x11 = 0.5 * (ll + lh + hl + hh)
    batch, channels, height, width = ll.shape
    image = coefficients.new_empty(batch, channels, 2 * height, 2 * width)
    image[..., 0::2, 0::2] = x00
    image[..., 0::2, 1::2] = x01
    image[..., 1::2, 0::2] = x10
    image[..., 1::2, 1::2] = x11
    return image


def smooth_shrink(
    value: torch.Tensor,
    threshold: torch.Tensor,
    temperature: float = 0.02,
) -> torch.Tensor:
    """Differentiable approximation to soft thresholding.

    Unlike a hard support mask, the sigmoid transition retains useful
    gradients near zero during identity-initialized training.
    """

    if temperature <= 0.0:
        raise ValueError("shrinkage temperature must be positive")
    support = torch.sigmoid((value.abs() - threshold) / float(temperature))
    return value * support


class SparseWaveletUpdate(nn.Module):
    """One metadata-conditioned proximal residual update."""

    def __init__(self, channels: int, state_dim: int) -> None:
        super().__init__()
        if channels % 4:
            raise ValueError("hidden channel count must be divisible by four")
        self.input = nn.Conv2d(48, channels, 3, padding=1)
        self.body = nn.Sequential(
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.affine = nn.Linear(state_dim, 2 * channels)
        self.threshold = nn.Linear(state_dim, 9)
        self.step = nn.Linear(state_dim, 4)
        self.output = nn.Conv2d(channels, 13, 3, padding=1)

        # Exact base-restorer initialization. The state pathway may learn from
        # its supervised corruption code before changing any image pixel.
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)
        nn.init.zeros_(self.threshold.weight)
        nn.init.constant_(self.threshold.bias, -4.0)
        nn.init.zeros_(self.step.weight)
        nn.init.zeros_(self.step.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        current: torch.Tensor,
        base: torch.Tensor,
        degraded: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        evidence = torch.cat((current, base, degraded, degraded - current), dim=1)
        features = self.input(evidence)
        scale, shift = self.affine(state).chunk(2, dim=1)
        features = features * (1.0 + 0.20 * torch.tanh(scale)[:, :, None, None])
        features = features + 0.20 * torch.tanh(shift)[:, :, None, None]
        features = features + self.body(features)
        raw = self.output(features)

        gate = torch.sigmoid(raw[:, 12:13])
        low = 0.25 * torch.tanh(raw[:, :3])
        detail = raw[:, 3:12]
        threshold = 0.06 * torch.sigmoid(self.threshold(state))[:, :, None, None]
        detail = smooth_shrink(detail, threshold)
        proposal = torch.cat((low, detail), dim=1)

        step = (0.25 + 0.75 * torch.sigmoid(self.step(state)))[:, :, None, None]
        step = step.repeat_interleave(3, dim=1)
        update = gate * step * proposal
        return current + update, gate, threshold


class TelemetrySparseWaveletRefiner(nn.Module):
    """Apply a bounded sequence of sparse wavelet residual updates."""

    def __init__(
        self,
        state_dim: int = 8,
        hidden_channels: int = 48,
        stages: int = 3,
        max_residual: float = 0.16,
    ) -> None:
        super().__init__()
        if stages < 1:
            raise ValueError("at least one sparse refinement stage is required")
        if not 0.0 < max_residual <= 0.5:
            raise ValueError("max_residual must lie in (0, 0.5]")
        self.state_dim = int(state_dim)
        self.hidden_channels = int(hidden_channels)
        self.stages_count = int(stages)
        self.max_residual = float(max_residual)
        self.stages = nn.ModuleList(
            SparseWaveletUpdate(hidden_channels, state_dim) for _ in range(stages)
        )
        # The residual itself is zero-initialized, so a neutral gain preserves
        # exact identity while giving the first task-driven updates useful
        # gradient scale.  A strongly negative gain made the pilot correction
        # quantize to only one or two 8-bit levels and could not change
        # full-frame detector evidence meaningfully.
        self.output_gain = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _pad_even(image: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        height, width = image.shape[-2:]
        pad_h, pad_w = height % 2, width % 2
        if pad_h or pad_w:
            mode = "reflect" if height > 1 and width > 1 else "replicate"
            image = F.pad(image, (0, pad_w, 0, pad_h), mode=mode)
        return image, (height, width)

    def forward(
        self,
        degraded: torch.Tensor,
        base_restored: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if degraded.shape != base_restored.shape:
            raise ValueError("degraded and base-restored tensors must have the same shape")
        if state.ndim != 2 or state.shape != (degraded.shape[0], self.state_dim):
            raise ValueError(
                f"Expected state shape ({degraded.shape[0]}, {self.state_dim}), "
                f"received {tuple(state.shape)}"
            )
        degraded_even, original_size = self._pad_even(degraded)
        base_even, _ = self._pad_even(base_restored)
        degraded_coefficients = haar_dwt(degraded_even)
        base_coefficients = haar_dwt(base_even)
        coefficients = base_coefficients
        gates, thresholds = [], []
        for stage in self.stages:
            coefficients, gate, threshold = stage(
                coefficients,
                base_coefficients,
                degraded_coefficients,
                state,
            )
            gates.append(gate)
            thresholds.append(threshold)

        # Reconstruct only the learned coefficient difference. This makes the
        # zero-initialized path bit-exactly equal to ``base_restored`` instead
        # of introducing round-off from a needless base DWT/IWT cycle.
        candidate_residual = haar_iwt(coefficients - base_coefficients)
        height, width = original_size
        residual = candidate_residual[..., :height, :width]
        residual = self.max_residual * torch.tanh(residual / self.max_residual)
        gain = torch.sigmoid(self.output_gain)
        restored = torch.clamp(base_restored + gain * residual, 0.0, 1.0)
        return restored, {
            "wavelet_residual": gain * residual,
            "wavelet_gate": torch.stack(gates, dim=1),
            "wavelet_threshold": torch.stack(thresholds, dim=1),
            "wavelet_output_gain": gain.expand(degraded.shape[0]),
        }


class TRACERSparseDeMoE(RMRPMetadataDeMoE):
    """TRACE-R with a matched DeMoE backbone and sparse wavelet adapter."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        wavelet_hidden_channels: int = 48,
        wavelet_stages: int = 3,
        wavelet_max_residual: float = 0.16,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("top_k", 1)
        kwargs.setdefault("use_refiner", False)
        kwargs.setdefault("backbone_route_mode", "sensor_task")
        kwargs.setdefault("use_cause_refiners", False)
        kwargs.setdefault("use_semantic_adapters", False)
        super().__init__(backbone, **kwargs)
        self.wavelet_refiner = TelemetrySparseWaveletRefiner(
            state_dim=8,
            hidden_channels=wavelet_hidden_channels,
            stages=wavelet_stages,
            max_residual=wavelet_max_residual,
        )
        self.wavelet_hidden_channels = int(wavelet_hidden_channels)
        self.wavelet_stages = int(wavelet_stages)
        self.wavelet_max_residual = float(wavelet_max_residual)

    def forward(
        self,
        image: torch.Tensor,
        metadata: torch.Tensor | None = None,
        *,
        return_dict: bool = False,
        return_aux: bool = False,
        prompt_teacher_weights: torch.Tensor | None = None,
        prompt_teacher_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | dict[str, torch.Tensor | None]:
        result = super().forward(
            image,
            metadata,
            return_dict=True,
            return_aux=return_aux,
            prompt_teacher_weights=prompt_teacher_weights,
            prompt_teacher_mask=prompt_teacher_mask,
            **kwargs,
        )
        assert isinstance(result, dict)
        base = result["restored"]
        state = result["posterior_degradation_code"]
        if not isinstance(base, torch.Tensor) or not isinstance(state, torch.Tensor):
            raise RuntimeError("TRACE-R base model did not expose restoration/state tensors")
        restored, wavelet = self.wavelet_refiner(image, base, state)
        if not return_dict:
            return restored
        result["base_restored"] = base
        result["restored"] = restored
        result.update(wavelet)
        return result


__all__ = [
    "haar_dwt",
    "haar_iwt",
    "smooth_shrink",
    "SparseWaveletUpdate",
    "TelemetrySparseWaveletRefiner",
    "TRACERSparseDeMoE",
]
