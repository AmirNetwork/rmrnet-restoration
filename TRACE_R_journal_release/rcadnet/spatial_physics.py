# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
from __future__ import annotations

"""Differentiable exposure-motion physics for PI-RMR-Net.

This module implements the rotation-observable subset of the image-formation
model proposed for PI-RMR-Net:

    I_d(x) = (1 / M) sum_j I_r(W(x; R_j, K)),

where ``R_j`` is obtained by integrating exposure-synchronised angular rate
on SO(3), and ``K`` is a declared camera calibration. Translation is not used
unless depth/road-plane calibration is supplied by a future packet version.
Keeping that boundary explicit is preferable to silently inventing depth.

The 82-value practical packet stores normalized measurements. Exposure is
decoded to seconds before integration; angular rate is restored to rad/s by
the fixed sensor full-scale stored with the checkpoint. The resulting field
therefore has a stable pixel interpretation across datasets.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from .practical_metadata import (
    CONTEXT_START,
    GYRO_END,
    PRACTICAL_SENSOR_DIM,
    TRAJECTORY_STEPS,
    clamp_sensor_packet,
)


def decode_log_exposure_ms(
    normalized: torch.Tensor,
    *,
    minimum_ms: float = 0.05,
    maximum_ms: float = 40.0,
) -> torch.Tensor:
    """Invert the public log exposure encoding in physical milliseconds."""

    if minimum_ms <= 0.0 or maximum_ms <= minimum_ms:
        raise ValueError("Exposure bounds must satisfy 0 < minimum < maximum")
    value = normalized.clamp(0.0, 1.0)
    log_min = math.log(float(minimum_ms))
    log_span = math.log(float(maximum_ms)) - log_min
    return torch.exp(value * log_span + log_min)


def encode_log_exposure_ms(
    exposure_ms: float,
    *,
    minimum_ms: float = 0.05,
    maximum_ms: float = 40.0,
) -> float:
    """Encode a physical exposure using the inverse of the function above."""

    if minimum_ms <= 0.0 or maximum_ms <= minimum_ms:
        raise ValueError("Exposure bounds must satisfy 0 < minimum < maximum")
    clipped = min(max(float(exposure_ms), minimum_ms), maximum_ms)
    return (math.log(clipped) - math.log(minimum_ms)) / (
        math.log(maximum_ms) - math.log(minimum_ms)
    )


@dataclass
class ExposurePhysicsState:
    """Auditable outputs of the deterministic exposure operator."""

    flow: torch.Tensor
    summary: torch.Tensor
    reliability: torch.Tensor
    motion_px: torch.Tensor
    exposure_ms: torch.Tensor
    samples_per_exposure: torch.Tensor
    constant_rate_reliability: torch.Tensor
    trajectory_reliability: torch.Tensor
    temporal_reliability: torch.Tensor
    constant_rate_error_px: torch.Tensor
    valid: torch.Tensor


def _skew(vector: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(vector[..., 0])
    x, y, z = vector.unbind(dim=-1)
    return torch.stack(
        (
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ),
        dim=-1,
    ).reshape(*vector.shape[:-1], 3, 3)


def _so3_exp(rotation_vector: torch.Tensor) -> torch.Tensor:
    """Stable Rodrigues exponential for batched small camera rotations."""

    angle = rotation_vector.norm(dim=-1, keepdim=True)
    axis = rotation_vector / angle.clamp_min(1.0e-8)
    k = _skew(axis)
    eye = torch.eye(3, device=rotation_vector.device, dtype=rotation_vector.dtype)
    eye = eye.expand(*rotation_vector.shape[:-1], 3, 3)
    sin = torch.sin(angle)[..., None]
    cos = torch.cos(angle)[..., None]
    rodrigues = eye + sin * k + (1.0 - cos) * (k @ k)
    # The exact zero-angle limit is identity. Selecting it explicitly avoids
    # numerical axis noise in clean/short-exposure safety cases.
    return torch.where((angle < 1.0e-8)[..., None], eye, rodrigues)


class RotationExposurePhysics(nn.Module):
    """Build a spatial camera-rotation field and differentiable reblur.

    The camera axes in the practical packet are already calibrated by the
    dataset builder. ``focal_ratio`` is ``f_x / image_width`` for square pixels;
    it is a checkpoint-level calibration constant, not a per-image target.
    ``calibration_reliability`` must be reduced when this value is estimated
    rather than measured.
    """

    def __init__(
        self,
        *,
        samples: int = 5,
        gyro_full_scale: float = 4.0,
        exposure_min_ms: float = 0.05,
        exposure_max_ms: float = 40.0,
        context_window_ms: float = 50.0,
        focal_ratio: float = 0.75,
        calibration_reliability: float = 0.50,
        maximum_motion_px: float = 32.0,
        activation_motion_px: float = 0.10,
        imu_frequency_hz: float = 200.0,
        constant_rate_error_tolerance_px: float = 0.25,
    ) -> None:
        super().__init__()
        if samples < 3 or samples % 2 == 0:
            raise ValueError("Exposure samples must be an odd integer >= 3")
        if gyro_full_scale <= 0.0:
            raise ValueError("gyro_full_scale must be positive")
        if context_window_ms <= 0.0 or focal_ratio <= 0.0:
            raise ValueError("Context window and focal ratio must be positive")
        if not 0.0 <= calibration_reliability <= 1.0:
            raise ValueError("calibration_reliability must be in [0, 1]")
        self.samples = int(samples)
        self.gyro_full_scale = float(gyro_full_scale)
        self.exposure_min_ms = float(exposure_min_ms)
        self.exposure_max_ms = float(exposure_max_ms)
        self.context_window_ms = float(context_window_ms)
        self.focal_ratio = float(focal_ratio)
        self.calibration_reliability = float(calibration_reliability)
        self.maximum_motion_px = float(maximum_motion_px)
        self.activation_motion_px = float(activation_motion_px)
        if imu_frequency_hz <= 0.0 or constant_rate_error_tolerance_px <= 0.0:
            raise ValueError("IMU frequency and constant-rate tolerance must be positive")
        self.imu_frequency_hz = float(imu_frequency_hz)
        self.constant_rate_error_tolerance_px = float(
            constant_rate_error_tolerance_px
        )

    def _sample_gyro(
        self,
        gyro: torch.Tensor,
        exposure_seconds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = gyro.shape[0]
        unit_times = torch.linspace(
            -0.5,
            0.5,
            self.samples,
            device=gyro.device,
            dtype=gyro.dtype,
        )
        times = exposure_seconds[:, None] * unit_times[None]
        window_seconds = self.context_window_ms / 1000.0
        indices = (times / window_seconds + 0.5) * float(TRAJECTORY_STEPS - 1)
        indices = indices.clamp(0.0, float(TRAJECTORY_STEPS - 1))
        left = indices.floor().long()
        right = (left + 1).clamp(max=TRAJECTORY_STEPS - 1)
        alpha = (indices - left.to(indices.dtype))[..., None]
        gather_index_left = left[..., None].expand(batch, self.samples, 3)
        gather_index_right = right[..., None].expand(batch, self.samples, 3)
        sampled = torch.gather(gyro, 1, gather_index_left) * (1.0 - alpha)
        sampled = sampled + torch.gather(gyro, 1, gather_index_right) * alpha
        return sampled, times

    @staticmethod
    def _integrate_about_midpoint(
        angular_rate: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        dt = times[:, 1:] - times[:, :-1]
        increments = 0.5 * (angular_rate[:, 1:] + angular_rate[:, :-1])
        increments = increments * dt[..., None]
        cumulative = torch.cat(
            (torch.zeros_like(angular_rate[:, :1]), increments.cumsum(dim=1)),
            dim=1,
        )
        midpoint = cumulative[:, cumulative.shape[1] // 2 : cumulative.shape[1] // 2 + 1]
        return cumulative - midpoint

    def build(
        self,
        packet: torch.Tensor,
        height: int,
        width: int,
    ) -> ExposurePhysicsState:
        if packet.ndim == 1:
            packet = packet.unsqueeze(0)
        if packet.shape[-1] != PRACTICAL_SENSOR_DIM:
            raise ValueError(
                f"Expected {PRACTICAL_SENSOR_DIM} sensor values, got {packet.shape[-1]}"
            )
        p = clamp_sensor_packet(packet).float()
        batch = p.shape[0]
        gyro = p[:, :GYRO_END].reshape(batch, TRAJECTORY_STEPS, 3)
        gyro = gyro * self.gyro_full_scale
        context = p[:, CONTEXT_START:]
        exposure_ms = decode_log_exposure_ms(
            context[:, 0],
            minimum_ms=self.exposure_min_ms,
            maximum_ms=self.exposure_max_ms,
        )
        sampled_rate, times = self._sample_gyro(gyro, exposure_ms / 1000.0)
        rotation_vectors = self._integrate_about_midpoint(sampled_rate, times)
        rotations = _so3_exp(rotation_vectors)

        yy, xx = torch.meshgrid(
            torch.arange(height, device=p.device, dtype=torch.float32),
            torch.arange(width, device=p.device, dtype=torch.float32),
            indexing="ij",
        )
        focal_px = self.focal_ratio * float(width)
        cx = 0.5 * float(width - 1)
        cy = 0.5 * float(height - 1)
        rays = torch.stack(
            ((xx - cx) / focal_px, (yy - cy) / focal_px, torch.ones_like(xx)),
            dim=-1,
        )
        warped = torch.einsum("bmij,hwj->bmihw", rotations, rays)
        z = warped[:, :, 2].clamp_min(1.0e-6)
        warped_x = focal_px * warped[:, :, 0] / z + cx
        warped_y = focal_px * warped[:, :, 1] / z + cy
        flow_x = warped_x - xx
        flow_y = warped_y - yy
        flow = torch.stack((flow_x, flow_y), dim=-1)
        valid = (
            (warped_x >= 0.0)
            & (warped_x <= float(width - 1))
            & (warped_y >= 0.0)
            & (warped_y <= float(height - 1))
        )

        endpoint = flow[:, -1] - flow[:, 0]
        motion_px = endpoint.square().sum(dim=-1).sqrt().amax(dim=(1, 2))
        path_step = flow[:, 1:] - flow[:, :-1]
        path_length = path_step.square().sum(dim=-1).sqrt().sum(dim=1)
        direction = torch.atan2(endpoint[..., 1], endpoint[..., 0])
        summary = torch.stack(
            (
                endpoint[..., 0] / self.maximum_motion_px,
                endpoint[..., 1] / self.maximum_motion_px,
                path_length / self.maximum_motion_px,
                torch.sin(direction),
                torch.cos(direction),
                (
                    exposure_ms[:, None, None]
                    .expand(batch, height, width)
                    / self.exposure_max_ms
                ),
            ),
            dim=1,
        ).clamp(-1.0, 1.0)

        camera_rel = context[:, 12].clamp(0.0, 1.0)
        imu_rel = context[:, 13].clamp(0.0, 1.0)
        available = context[:, 15].clamp(0.0, 1.0)
        sync_rel = (1.0 - context[:, 10].abs()).clamp(0.0, 1.0)
        imu_noise_rel = (1.0 - context[:, 11].abs()).clamp(0.0, 1.0)
        # Paper Eq. (observability): q_phys = q_d q_temporal q_s q_k q_a.
        # q_k includes declared camera/IMU calibration and packet noise support.
        support = camera_rel * imu_rel * imu_noise_rel * available * sync_rel
        support = support * self.calibration_reliability

        # Temporal observability has two distinct routes. A short exposure can
        # still be approximated by omega(tc) * Te when local angular
        # acceleration is small, even though the IMU does not resolve the
        # intra-exposure trajectory. Conversely, a longer exposure containing
        # multiple IMU intervals supports direct trajectory reconstruction.
        imu_dt = 1.0 / self.imu_frequency_hz
        angular_acceleration = (gyro[:, 1:] - gyro[:, :-1]) / imu_dt
        angular_acceleration = angular_acceleration.norm(dim=-1).amax(dim=1)
        constant_rate_error_px = (
            0.5
            * focal_px
            * angular_acceleration
            * (exposure_ms / 1000.0).square()
        )
        constant_rate_reliability = torch.exp(
            -(
                constant_rate_error_px
                / self.constant_rate_error_tolerance_px
            ).square()
        )
        samples_per_exposure = exposure_ms * self.imu_frequency_hz / 1000.0
        trajectory_reliability = torch.sigmoid(
            (samples_per_exposure - 1.5) / 0.35
        )
        temporal_reliability = 1.0 - (
            (1.0 - constant_rate_reliability)
            * (1.0 - trajectory_reliability)
        )
        # Below the declared pixel support, the INS cannot justify an image
        # correction. A smooth gate preserves differentiability near the limit.
        motion_support = torch.sigmoid(
            (motion_px - self.activation_motion_px)
            / max(0.25 * self.activation_motion_px, 1.0e-3)
        )
        reliability = support * motion_support * temporal_reliability
        reliability_map = reliability[:, None, None, None].expand(batch, 1, height, width)
        summary = torch.cat((summary, reliability_map), dim=1)
        return ExposurePhysicsState(
            flow=flow,
            summary=summary,
            reliability=reliability_map,
            motion_px=motion_px,
            exposure_ms=exposure_ms,
            samples_per_exposure=samples_per_exposure,
            constant_rate_reliability=constant_rate_reliability,
            trajectory_reliability=trajectory_reliability,
            temporal_reliability=temporal_reliability,
            constant_rate_error_px=constant_rate_error_px,
            valid=valid,
        )

    @staticmethod
    def _warp_average(
        image: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a supplied spatial exposure flow by warp-and-average."""

        batch, _, height, width = image.shape
        if flow.shape[:4] != (batch, flow.shape[1], height, width):
            raise ValueError("Physics flow and image dimensions do not match")
        yy, xx = torch.meshgrid(
            torch.arange(height, device=image.device, dtype=image.dtype),
            torch.arange(width, device=image.device, dtype=image.dtype),
            indexing="ij",
        )
        sample_x = xx[None, None] + flow[..., 0].to(image.dtype)
        sample_y = yy[None, None] + flow[..., 1].to(image.dtype)
        grid_x = 2.0 * sample_x / max(width - 1, 1) - 1.0
        grid_y = 2.0 * sample_y / max(height - 1, 1) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1)
        samples = flow.shape[1]
        expanded = image[:, None].expand(batch, samples, -1, -1, -1)
        warped = F.grid_sample(
            expanded.reshape(batch * samples, image.shape[1], height, width),
            grid.reshape(batch * samples, height, width, 2),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).reshape(batch, samples, image.shape[1], height, width)
        return warped.mean(dim=1)

    @staticmethod
    def reblur(image: torch.Tensor, state: ExposurePhysicsState) -> torch.Tensor:
        """Apply ``H_phys`` by differentiable warp-and-average."""

        return RotationExposurePhysics._warp_average(image, state.flow)

    @staticmethod
    def inverse_candidate(
        observed: torch.Tensor,
        state: ExposurePhysicsState,
        *,
        iterations: int = 3,
        step_size: float = 0.65,
        maximum_residual: float = 0.20,
    ) -> torch.Tensor:
        """Regularized unrolled Landweber candidate from the measured operator.

        x_{k+1} = x_k + alpha H^T (y - H x_k).

        The reverse-flow average is a stable approximation to H^T for the
        exposure-centred rotation operator. Each update is bounded and scaled
        by physical reliability, so unobservable or mismatched packets cannot
        force a large correction. The neural network refines this proposal; it
        does not receive a hidden kernel.
        """

        if iterations < 1 or not 0.0 < step_size <= 1.0:
            raise ValueError("iterations must be positive and step_size in (0, 1]")
        estimate = observed
        reliability = state.reliability.to(observed.dtype)
        for _ in range(iterations):
            residual = observed - RotationExposurePhysics.reblur(estimate, state)
            adjoint = RotationExposurePhysics._warp_average(
                residual,
                -state.flow,
            )
            update = (step_size * reliability * adjoint).clamp(
                -maximum_residual,
                maximum_residual,
            )
            estimate = (estimate + update).clamp(0.0, 1.0)
        return estimate


class PhysicsFeatureEncoder(nn.Module):
    """Zero-initialized spatial conditioning from the physical field."""

    def __init__(self, output_channels: int, input_channels: int = 7) -> None:
        super().__init__()
        hidden = max(output_channels // 2, 16)
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, output_channels, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        return self.net(summary)


def physics_reblur_loss(
    reblurred: torch.Tensor,
    observed: torch.Tensor,
    reliability: torch.Tensor,
    *,
    epsilon: float = 1.0e-3,
) -> torch.Tensor:
    """Confidence-weighted robust image-formation residual ``L_phys``."""

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if reliability.ndim == 1:
        reliability = reliability[:, None, None, None]
    # Saturated observations are not invertible under the omitted camera
    # response. Their contribution is smoothly reduced rather than asserted.
    luminance = observed.mean(dim=1, keepdim=True)
    camera_valid = ((luminance > 0.01) & (luminance < 0.99)).to(observed.dtype)
    weight = reliability.to(observed.dtype) * camera_valid
    residual = torch.sqrt((reblurred - observed).square() + epsilon**2)
    denominator = weight.sum() * observed.shape[1]
    if bool((denominator <= 1.0e-8).detach().cpu()):
        return residual.new_zeros(())
    return (residual * weight).sum() / denominator
