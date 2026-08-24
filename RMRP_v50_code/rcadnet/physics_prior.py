from __future__ import annotations

"""Metadata-derived physical priors used by RMR-Net.

The motion branch consumes the corrected ``axial_v2`` code:

    z_0 = .5 (1 + cos 2 theta) m,
    z_1 = .5 (1 + sin 2 theta) m,
    z_2 = m = min(L / 25, 1).

The point-spread function is fixed by measured blur length ``L`` and axial
angle ``theta``.  Its Wiener inversion is differentiable with respect to the
input image.  Kernel construction is deliberately non-differentiable with
respect to telemetry because sensor metadata is an observed condition, not a
quantity optimized by image loss.
"""

import math

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _motion_psf(length_px: float, angle_deg: float) -> np.ndarray:
    """Match the controlled-benchmark OpenCV line-spread generator."""

    size = max(3, int(round(length_px)))
    if size % 2 == 0:
        size += 1
    kernel = np.zeros((size, size), dtype=np.float32)
    center = (size - 1) / 2.0
    radius = (size - 1) / 2.0
    angle = np.deg2rad(float(angle_deg))
    dx, dy = np.cos(angle) * radius, np.sin(angle) * radius
    p1 = (int(round(center - dx)), int(round(center - dy)))
    p2 = (int(round(center + dx)), int(round(center + dy)))
    cv2.line(kernel, p1, p2, color=1.0, thickness=1, lineType=cv2.LINE_AA)
    kernel /= max(float(kernel.sum()), 1e-8)
    return kernel


class MetadataMotionWienerPrior(nn.Module):
    """Cause-gated exact-kernel motion prior.

    The validation-selected defaults (K=0.0075, eta=0.90) are recorded in the
    PCM validation audit.  ``nuisance_decay`` prevents the motion inverse from
    being applied to compound low-light/noise inputs, where ringing was found
    to damage detector evidence.
    """

    def __init__(
        self,
        *,
        regularization: float = 0.0075,
        blend: float = 0.90,
        nuisance_decay: float = 30.0,
        compound_gate_floor: float = 0.0,
        adaptive_regularization_gain: float = 0.0,
        max_length_px: int = 25,
        solver: str = "wiener",
        richardson_lucy_iterations: int = 4,
    ) -> None:
        super().__init__()
        if regularization <= 0:
            raise ValueError("Wiener regularization must be positive")
        if not 0.0 <= blend <= 1.0:
            raise ValueError("Motion-prior blend must be in [0, 1]")
        self.regularization = float(regularization)
        self.blend = float(blend)
        self.nuisance_decay = float(nuisance_decay)
        self.compound_gate_floor = float(compound_gate_floor)
        self.adaptive_regularization_gain = float(adaptive_regularization_gain)
        self.max_length_px = int(max_length_px)
        self.solver = str(solver)
        self.richardson_lucy_iterations = int(richardson_lucy_iterations)
        if not 0.0 <= self.compound_gate_floor <= 1.0:
            raise ValueError("compound_gate_floor must be in [0, 1]")
        if self.adaptive_regularization_gain < 0.0:
            raise ValueError("adaptive_regularization_gain must be non-negative")
        if self.solver not in {"wiener", "richardson_lucy"}:
            raise ValueError("motion-prior solver must be wiener or richardson_lucy")
        if self.richardson_lucy_iterations < 1:
            raise ValueError("Richardson-Lucy iterations must be positive")

    @staticmethod
    def _decode_axial(code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        magnitude = code[:, 2].clamp(0.0, 1.0)
        denominator = magnitude.clamp_min(1e-6)
        cos_2theta = (2.0 * code[:, 0] / denominator - 1.0).clamp(-1.0, 1.0)
        sin_2theta = (2.0 * code[:, 1] / denominator - 1.0).clamp(-1.0, 1.0)
        angle = 0.5 * torch.atan2(sin_2theta, cos_2theta)
        angle = torch.where(magnitude > 1e-6, angle, torch.zeros_like(angle))
        return magnitude, angle

    @staticmethod
    def _transfer_function(
        psf: np.ndarray,
        height: int,
        width: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        kernel = np.zeros((height, width), dtype=np.float32)
        kh, kw = psf.shape
        kernel[:kh, :kw] = psf
        kernel = np.roll(kernel, -(kh // 2), axis=0)
        kernel = np.roll(kernel, -(kw // 2), axis=1)
        return torch.fft.fft2(torch.from_numpy(kernel).to(device=device))

    def cause_gate(self, code: torch.Tensor) -> torch.Tensor:
        """Use exact causes to suppress unsafe compound-degradation inversion."""

        motion = code[:, 2].clamp(0.0, 1.0)
        nuisance = code[:, 3:7].clamp(0.0, 1.0).sum(dim=1)
        present = (motion > (1.0 / self.max_length_px)).to(code.dtype)
        attenuated = torch.exp(-self.nuisance_decay * nuisance)
        compound_safe = self.compound_gate_floor + (1.0 - self.compound_gate_floor) * attenuated
        return present * compound_safe

    @staticmethod
    def _richardson_lucy(
        source: torch.Tensor,
        psf: np.ndarray,
        iterations: int,
    ) -> torch.Tensor:
        """Run a bounded differentiable Richardson-Lucy image update.

        The measured per-image PSF remains fixed. Ratio clipping and image
        clamping keep the short unrolled solver stable on low-light pavement
        patches, where unconstrained Richardson-Lucy updates amplify noise.
        """

        channels = source.shape[1]
        kernel = torch.from_numpy(psf).to(
            device=source.device,
            dtype=source.dtype,
        )
        kernel = kernel[None, None].expand(channels, 1, -1, -1).contiguous()
        flipped = torch.flip(kernel, dims=(-2, -1))
        radius_y = psf.shape[0] // 2
        radius_x = psf.shape[1] // 2

        def blur(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            padded = F.pad(
                value,
                (radius_x, radius_x, radius_y, radius_y),
                mode="reflect",
            )
            return F.conv2d(padded, weight, groups=channels)

        estimate = source.clamp(1.0e-4, 1.0)
        for _ in range(iterations):
            ratio = source / blur(estimate, kernel).clamp_min(1.0e-4)
            ratio = ratio.clamp(0.0, 4.0)
            correction = blur(ratio, flipped).clamp(0.25, 4.0)
            estimate = (estimate * correction).clamp(0.0, 1.0)
        return estimate

    def forward(self, image: torch.Tensor, metadata_code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if metadata_code.ndim == 1:
            metadata_code = metadata_code[None, :].expand(image.shape[0], -1)
        metadata_code = metadata_code.to(device=image.device, dtype=torch.float32)
        source_dtype = image.dtype
        source = image.float()
        magnitude, angle = self._decode_axial(metadata_code)
        deconvolved_items = []
        # Samples are processed independently because their PSF radii differ.
        # This exactly matches the standalone validation operator's reflection
        # boundary while retaining gradients from each output to its input.
        nuisance = metadata_code[:, 3:7].clamp(0.0, 1.0).sum(dim=1)
        for index, (motion, theta) in enumerate(
            zip(magnitude.detach().cpu().tolist(), angle.detach().cpu().tolist())
        ):
            length = max(1.0, min(float(motion) * self.max_length_px, float(self.max_length_px)))
            psf = _motion_psf(length, math.degrees(float(theta)))
            radius = psf.shape[0] // 2
            sample = source[index : index + 1]
            if self.solver == "richardson_lucy":
                item = self._richardson_lucy(
                    sample,
                    psf,
                    self.richardson_lucy_iterations,
                )
            else:
                padded = F.pad(sample, (radius, radius, radius, radius), mode="reflect")
                transfer = self._transfer_function(
                    psf,
                    padded.shape[-2],
                    padded.shape[-1],
                    device=image.device,
                )
                adaptive_k = self.regularization * (
                    1.0
                    + self.adaptive_regularization_gain
                    * float(nuisance[index].detach().cpu())
                )
                inverse = transfer.conj() / (transfer.abs().square() + adaptive_k)
                spectrum = torch.fft.fft2(padded)
                item = torch.fft.ifft2(spectrum * inverse[None, None]).real
                if radius:
                    item = item[:, :, radius:-radius, radius:-radius]
            deconvolved_items.append(item)
        deconvolved = torch.cat(deconvolved_items, dim=0)
        prior = source + self.blend * (deconvolved - source)
        prior = prior.clamp(0.0, 1.0).to(source_dtype)
        return prior, self.cause_gate(metadata_code).to(source_dtype)
