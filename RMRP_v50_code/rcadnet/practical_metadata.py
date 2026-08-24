from __future__ import annotations

"""Practical camera/IMU/vehicle metadata for RMR-P.

Paired synthetic data retain exact rendering parameters for cause supervision,
but a deployed camera does not observe a hidden blur kernel. This module defines
the observable exposure-synchronized sensor packet used by RMR-P:

* eleven gyroscope and accelerometer samples in a capture-centred window;
* camera exposure, ISO/gain, lens state, and rolling-shutter readout;
* vehicle speed/yaw context and synchronization uncertainty;
* explicit camera/IMU/vehicle reliability and availability.

The short window covers the exposure when the sensor rate permits it and is a
50 ms capture-centred context window for the 200 Hz field SBG unit. All values
are normalized by the dataset builder. Signed inertial quantities
remain in [-1, 1]; camera settings and reliability values are in [0, 1].
The encoder converts this packet into the same eight causal coordinates used
by the restoration backbone. Exact generator parameters may supervise this
conversion during training, but they are never an inference input.
"""

from collections.abc import Mapping, Sequence
import math
from typing import Any

import torch
from torch import nn


TRAJECTORY_STEPS = 11
GYRO_PACKET_NAMES = tuple(
    f"gyro_t{step:02d}_{axis}"
    for step in range(TRAJECTORY_STEPS)
    for axis in ("x", "y", "z")
)
ACCEL_PACKET_NAMES = tuple(
    f"accel_t{step:02d}_{axis}"
    for step in range(TRAJECTORY_STEPS)
    for axis in ("x", "y", "z")
)
CONTEXT_PACKET_NAMES = (
    "log_exposure",
    "log_iso",
    "analog_gain",
    "focal_length",
    "aperture",
    "focus_error_proxy",
    "autofocus_confidence",
    "rolling_readout",
    "vehicle_speed",
    "vehicle_yaw_rate",
    "timestamp_offset",
    "imu_noise",
    "camera_reliability",
    "imu_reliability",
    "vehicle_reliability",
    "metadata_available",
)
SENSOR_PACKET_NAMES = GYRO_PACKET_NAMES + ACCEL_PACKET_NAMES + CONTEXT_PACKET_NAMES
PRACTICAL_SENSOR_DIM = len(SENSOR_PACKET_NAMES)
GYRO_END = TRAJECTORY_STEPS * 3
ACCEL_END = GYRO_END + TRAJECTORY_STEPS * 3
CONTEXT_START = ACCEL_END
AVAILABILITY_PATTERNS = (
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (0, 1, 1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
)


def clamp_sensor_packet(packet: torch.Tensor) -> torch.Tensor:
    """Clamp each sensor group to its declared normalized range."""

    if packet.shape[-1] != PRACTICAL_SENSOR_DIM:
        raise ValueError(
            f"Expected practical sensor dimension {PRACTICAL_SENSOR_DIM}, "
            f"got {packet.shape[-1]}"
        )
    clamped = packet.clone()
    clamped[..., :ACCEL_END] = clamped[..., :ACCEL_END].clamp(-1.0, 1.0)
    clamped[..., CONTEXT_START : CONTEXT_START + 9] = clamped[
        ..., CONTEXT_START : CONTEXT_START + 9
    ].clamp(0.0, 1.0)
    clamped[..., CONTEXT_START + 9 : CONTEXT_START + 11] = clamped[
        ..., CONTEXT_START + 9 : CONTEXT_START + 11
    ].clamp(-1.0, 1.0)
    clamped[..., CONTEXT_START + 11 :] = clamped[
        ..., CONTEXT_START + 11 :
    ].clamp(0.0, 1.0)
    return clamped


def perturb_sensor_packet(packet: torch.Tensor, std: float) -> torch.Tensor:
    """Add measurement noise without repopulating unavailable modalities."""

    if std <= 0:
        return clamp_sensor_packet(packet)
    original = clamp_sensor_packet(packet)
    noisy = original + torch.randn_like(original) * float(std)
    # camera/IMU/vehicle reliability and global availability describe the
    # packet, so measurement noise must not silently rewrite those flags.
    noisy[..., CONTEXT_START + 12 : CONTEXT_START + 16] = original[
        ..., CONTEXT_START + 12 : CONTEXT_START + 16
    ]
    noisy = clamp_sensor_packet(noisy)
    # Dropout is applied before noise in the trainer. Reapply its field mask so
    # an unavailable IMU/camera/vehicle cannot reappear as random measurements
    # merely because Gaussian noise was added to an all-zero channel.
    masked = apply_sensor_modality_mask(
        noisy,
        camera_available=original[..., CONTEXT_START + 12] > 0,
        imu_available=original[..., CONTEXT_START + 13] > 0,
        vehicle_available=original[..., CONTEXT_START + 14] > 0,
    )
    masked[..., CONTEXT_START + 12 : CONTEXT_START + 16] = original[
        ..., CONTEXT_START + 12 : CONTEXT_START + 16
    ]
    return clamp_sensor_packet(masked)


def apply_sensor_modality_mask(
    packet: torch.Tensor,
    *,
    camera_available: torch.Tensor | float | bool,
    imu_available: torch.Tensor | float | bool,
    vehicle_available: torch.Tensor | float | bool,
) -> torch.Tensor:
    """Apply field-consistent partial-metadata availability masks.

    Missing metadata is not a binary operating mode. Camera, IMU, and vehicle
    fields can fail independently. This function zeroes each unavailable
    measurement group and its matching reliability declaration:

    * camera: exposure/lens/readout fields and camera reliability;
    * IMU: gyro/accelerometer trajectories, timing/noise, IMU reliability;
    * vehicle: speed/yaw and vehicle reliability.

    ``metadata_available`` is one when at least one modality remains. The
    downstream cause-wise reliability gate then falls back to image evidence
    only for unsupported degradation coordinates.
    """

    squeeze = packet.ndim == 1
    if squeeze:
        packet = packet.unsqueeze(0)
    masked = clamp_sensor_packet(packet)
    batch = masked.shape[0]

    def availability_tensor(value: torch.Tensor | float | bool) -> torch.Tensor:
        tensor = torch.as_tensor(
            value,
            device=masked.device,
            dtype=masked.dtype,
        )
        if tensor.ndim == 0:
            tensor = tensor.expand(batch)
        tensor = tensor.reshape(batch, 1).clamp(0.0, 1.0)
        return tensor

    camera = availability_tensor(camera_available)
    imu = availability_tensor(imu_available)
    vehicle = availability_tensor(vehicle_available)

    masked[:, :GYRO_END] = masked[:, :GYRO_END] * imu
    masked[:, GYRO_END:ACCEL_END] = (
        masked[:, GYRO_END:ACCEL_END] * imu
    )
    masked[:, CONTEXT_START : CONTEXT_START + 8] = (
        masked[:, CONTEXT_START : CONTEXT_START + 8] * camera
    )
    masked[:, CONTEXT_START + 8 : CONTEXT_START + 10] = (
        masked[:, CONTEXT_START + 8 : CONTEXT_START + 10] * vehicle
    )
    masked[:, CONTEXT_START + 10 : CONTEXT_START + 12] = (
        masked[:, CONTEXT_START + 10 : CONTEXT_START + 12] * imu
    )
    masked[:, CONTEXT_START + 12] = (
        masked[:, CONTEXT_START + 12] * camera[:, 0]
    )
    masked[:, CONTEXT_START + 13] = (
        masked[:, CONTEXT_START + 13] * imu[:, 0]
    )
    masked[:, CONTEXT_START + 14] = (
        masked[:, CONTEXT_START + 14] * vehicle[:, 0]
    )
    masked[:, CONTEXT_START + 15] = torch.maximum(
        camera,
        torch.maximum(imu, vehicle),
    )[:, 0]
    masked = clamp_sensor_packet(masked)
    return masked[0] if squeeze else masked


def structured_sensor_dropout(
    packet: torch.Tensor,
    probability: float,
) -> torch.Tensor:
    """Randomly drop complete sensor modalities during training."""

    if probability <= 0:
        return clamp_sensor_packet(packet)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Structured sensor dropout probability must be in [0, 1]")
    squeeze = packet.ndim == 1
    batch = 1 if squeeze else packet.shape[0]
    device = packet.device
    camera = (torch.rand(batch, device=device) >= probability).to(packet.dtype)
    imu = (torch.rand(batch, device=device) >= probability).to(packet.dtype)
    vehicle = (torch.rand(batch, device=device) >= probability).to(packet.dtype)
    return apply_sensor_modality_mask(
        packet,
        camera_available=camera,
        imu_available=imu,
        vehicle_available=vehicle,
    )


def balanced_sensor_dropout(
    packet: torch.Tensor,
    nonfull_probability: float,
) -> torch.Tensor:
    """Sample full, partial, and unavailable metadata packets explicitly.

    The packet stays complete with probability ``1 - nonfull_probability``.
    Otherwise one of the seven remaining camera/IMU/vehicle availability
    patterns is sampled uniformly. This gives every partial-metadata state
    direct training support while retaining continuous reliability values
    inside each available modality.
    """

    if not 0.0 <= nonfull_probability <= 1.0:
        raise ValueError("Balanced sensor dropout probability must be in [0, 1]")
    if nonfull_probability <= 0:
        return clamp_sensor_packet(packet)

    squeeze = packet.ndim == 1
    values = packet.unsqueeze(0) if squeeze else packet
    batch = values.shape[0]
    device = values.device
    dtype = values.dtype

    # Index seven is the complete 111 packet. The other indices cover:
    # 000, 100, 010, 001, 110, 101, and 011.
    use_nonfull = torch.rand(batch, device=device) < nonfull_probability
    pattern_index = torch.randint(0, 7, (batch,), device=device)
    pattern_index = torch.where(
        use_nonfull,
        pattern_index,
        torch.full_like(pattern_index, 7),
    )
    patterns = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        device=device,
        dtype=dtype,
    )
    availability = patterns[pattern_index]
    masked = apply_sensor_modality_mask(
        values,
        camera_available=availability[:, 0],
        imu_available=availability[:, 1],
        vehicle_available=availability[:, 2],
    )
    return masked[0] if squeeze else masked


def sensor_availability_index(packet: torch.Tensor) -> torch.Tensor:
    """Return the camera/IMU/vehicle availability pattern in ``[0, 7]``.

    The index is the binary code ``4*C + 2*I + V``. Reliability values remain
    continuous inputs, while this discrete index only selects a small
    calibration residual for the observed modality combination. It therefore
    distinguishes, for example, camera+IMU from camera+vehicle packets without
    pretending that metadata are globally on or off.
    """

    squeeze = packet.ndim == 1
    values = packet.unsqueeze(0) if squeeze else packet
    if values.shape[-1] != PRACTICAL_SENSOR_DIM:
        raise ValueError(
            f"Expected practical sensor dimension {PRACTICAL_SENSOR_DIM}, "
            f"got {values.shape[-1]}"
        )
    camera = (values[:, CONTEXT_START + 12] > 0).long()
    imu = (values[:, CONTEXT_START + 13] > 0).long()
    vehicle = (values[:, CONTEXT_START + 14] > 0).long()
    index = 4 * camera + 2 * imu + vehicle
    return index[0] if squeeze else index


def counterfactual_sensor_packet(packet: torch.Tensor) -> torch.Tensor:
    """Construct a cause-changing but range-valid sensor hard negative.

    Randomly pairing two telemetry records can leave the degradation cause
    almost unchanged when a mini-batch contains the same scenario. This
    deterministic intervention instead rotates the exposure-time gyro
    trajectory by 90 degrees, attenuates and reverses accelerometer evidence,
    and contradicts observable camera settings. Timestamp, noise,
    availability, and reliability declarations are preserved.

    The packet is used only for reliability-gate supervision during training.
    It is never an inference input.
    """

    squeeze = packet.ndim == 1
    if squeeze:
        packet = packet.unsqueeze(0)
    original = clamp_sensor_packet(packet)
    wrong = original.clone()

    gyro = original[:, :GYRO_END].reshape(-1, TRAJECTORY_STEPS, 3)
    gyro = torch.flip(gyro, dims=(1,))
    rotated_gyro = torch.empty_like(gyro)
    rotated_gyro[:, :, 0] = -gyro[:, :, 1]
    rotated_gyro[:, :, 1] = gyro[:, :, 0]
    rotated_gyro[:, :, 2] = -gyro[:, :, 2]
    wrong[:, :GYRO_END] = rotated_gyro.reshape(-1, GYRO_END)

    accel = original[:, GYRO_END:ACCEL_END].reshape(
        -1, TRAJECTORY_STEPS, 3
    )
    counter_accel = -0.25 * torch.flip(accel, dims=(1,))
    wrong[:, GYRO_END:ACCEL_END] = counter_accel.reshape(
        -1, ACCEL_END - GYRO_END
    )

    # These indices are observable camera settings rather than hidden
    # renderer parameters.
    for context_index in (0, 1, 2, 4, 5, 7):
        index = CONTEXT_START + context_index
        wrong[:, index] = 1.0 - original[:, index]
    wrong[:, CONTEXT_START + 9] = -original[:, CONTEXT_START + 9]

    wrong = clamp_sensor_packet(wrong)
    return wrong[0] if squeeze else wrong


def sensor_packet_from_mapping(
    metadata: Mapping[str, Any],
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Read a normalized practical packet from one public metadata sidecar."""

    packet = metadata.get("practical_sensor_packet")
    if packet is None:
        packet = metadata.get("sensor_packet")
    if isinstance(packet, Mapping):
        values = [float(packet.get(name, 0.0)) for name in SENSOR_PACKET_NAMES]
    elif isinstance(packet, Sequence) and not isinstance(packet, (str, bytes)):
        values = [float(value) for value in packet]
    else:
        raise KeyError(
            "Metadata sidecar has no practical_sensor_packet. "
            "Use the practical-metadata builder or an explicit renderer-metadata path."
        )
    if len(values) != PRACTICAL_SENSOR_DIM:
        raise ValueError(
            f"Expected {PRACTICAL_SENSOR_DIM} practical sensor values, got {len(values)}"
        )
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    return clamp_sensor_packet(tensor)


def observable_code_from_packet(
    packet: torch.Tensor,
    *,
    gyro_full_scale: float = 1.0,
) -> torch.Tensor:
    """Convert observable sensor measurements to a physical eight-value code.

    This deterministic path is deliberately based on the measured packet:

        Delta theta = T_exp * trapz(omega),
        m = ||Delta theta_xy||,
        z_x = .5 (1 + cos 2 theta) m,
        z_y = .5 (1 + sin 2 theta) m.

    It never reads hidden blur length, angle, noise sigma, or defocus sigma.
    The learned encoder predicts a bounded residual around this physical code.
    """

    squeeze = packet.ndim == 1
    if squeeze:
        packet = packet.unsqueeze(0)
    if packet.shape[-1] != PRACTICAL_SENSOR_DIM:
        raise ValueError(
            f"Expected practical sensor dimension {PRACTICAL_SENSOR_DIM}, "
            f"got {packet.shape[-1]}"
        )

    if gyro_full_scale <= 0.0:
        raise ValueError("gyro_full_scale must be greater than zero")
    p = clamp_sensor_packet(packet)
    gyro = p[:, :GYRO_END].reshape(-1, TRAJECTORY_STEPS, 3)
    accel = p[:, GYRO_END:ACCEL_END].reshape(-1, TRAJECTORY_STEPS, 3)
    context = p[:, CONTEXT_START:]
    exposure = context[:, 0].clamp(0.0, 1.0)
    exposure_scale = 0.20 + 0.80 * exposure
    integrated = torch.trapezoid(gyro, dim=1) / float(TRAJECTORY_STEPS - 1)
    # The packet stores gyro readings normalized by the calibrated sensor
    # full-scale. Equation Delta theta = T_exp * integral(omega dt) therefore
    # restores physical units before exposure integration.
    integrated = integrated * float(gyro_full_scale)
    integrated = integrated * exposure_scale[:, None]
    rotation_x = integrated[:, 0].clamp(-1.0, 1.0)
    rotation_y = integrated[:, 1].clamp(-1.0, 1.0)
    motion = torch.sqrt(rotation_x.square() + rotation_y.square() + 1e-8).clamp(0.0, 1.0)
    theta = torch.atan2(rotation_y, rotation_x)
    motion_x = 0.5 * (1.0 + torch.cos(2.0 * theta)) * motion
    motion_y = 0.5 * (1.0 + torch.sin(2.0 * theta)) * motion
    accel_variation = accel.std(dim=1, unbiased=False).square().mean(dim=1).sqrt()
    gyro_variation = gyro.std(dim=1, unbiased=False).square().mean(dim=1).sqrt()
    vibration = (0.65 * accel_variation + 0.35 * gyro_variation).clamp(0.0, 1.0)

    # Measurement value and measurement reliability are separate variables:
    #
    #   z_m = h(m),  alpha = q(m, I),  z = z_I + alpha (z_m - z_I).
    #
    # Reliability therefore must not shrink h(m) before the fusion layer uses
    # q again. Missing modalities are already zeroed by
    # apply_sensor_modality_mask().
    focus_error = context[:, 5].clamp(0.0, 1.0)
    iso = context[:, 1].clamp(0.0, 1.0)
    gain = context[:, 2].clamp(0.0, 1.0)
    noise = (0.65 * iso + 0.35 * gain).clamp(0.0, 1.0)
    low_light = (0.55 * exposure + 0.45 * iso).clamp(0.0, 1.0)

    code = torch.zeros(packet.shape[0], 8, device=packet.device, dtype=packet.dtype)
    code[:, 0] = motion_x
    code[:, 1] = motion_y
    code[:, 2] = torch.maximum(motion, 0.35 * vibration)
    code[:, 3] = focus_error
    code[:, 4] = noise
    code[:, 5] = low_light
    code[:, 6] = 0.0
    code[:, 7] = code[:, :7].amax(dim=1)
    return code[0] if squeeze else code


def conditioning_code_from_packet(
    packet: torch.Tensor,
    *,
    gyro_full_scale: float = 1.0,
) -> torch.Tensor:
    """Return checkpoint-compatible conditioning coordinates from sensors.

    Earlier checkpoint-compatible conditioning used

        z_0 = |cos(theta)| m,  z_1 = |sin(theta)| m,

    while the physical Wiener branch requires the axial representation
    ``cos(2 theta), sin(2 theta)``. The public sensor trajectory supports both.
    This function converts the physical code to the historical conditioning
    coordinates without exposing a hidden blur kernel.
    """

    squeeze = packet.ndim == 1
    if squeeze:
        packet = packet.unsqueeze(0)
    p = clamp_sensor_packet(packet)
    axial = observable_code_from_packet(
        p,
        gyro_full_scale=gyro_full_scale,
    )
    magnitude = axial[:, 2].clamp(0.0, 1.0)
    denominator = magnitude.clamp_min(1e-6)
    cos_2theta = (2.0 * axial[:, 0] / denominator - 1.0).clamp(-1.0, 1.0)
    sin_2theta = (2.0 * axial[:, 1] / denominator - 1.0).clamp(-1.0, 1.0)
    theta = 0.5 * torch.atan2(sin_2theta, cos_2theta)

    gyro = p[:, :GYRO_END].reshape(-1, TRAJECTORY_STEPS, 3)
    accel = p[:, GYRO_END:ACCEL_END].reshape(-1, TRAJECTORY_STEPS, 3)
    accel_variation = accel.std(dim=1, unbiased=False).square().mean(dim=1).sqrt()
    gyro_variation = gyro.std(dim=1, unbiased=False).square().mean(dim=1).sqrt()
    vibration = (0.65 * accel_variation + 0.35 * gyro_variation).clamp(0.0, 1.0)

    code = axial.clone()
    code[:, 0] = theta.cos().abs() * magnitude
    code[:, 1] = theta.sin().abs() * magnitude
    code[:, 2] = 0.25 * vibration
    code[:, 7] = torch.maximum(magnitude, code[:, 3:7].amax(dim=1))
    return code[0] if squeeze else code


class PracticalSensorEncoder(nn.Module):
    """Encode raw practical metadata while preserving a physical initialization.

    Two temporal branches process the exposure-synchronized gyro and
    accelerometer trajectories. Compact context branches process camera,
    vehicle/synchronization, and reliability fields. Their fused residual is
    bounded around ``observable_code_from_packet``. This keeps initialization
    interpretable and lets training calibrate sensor bias without replacing
    physics with an unconstrained metadata MLP.
    """

    def __init__(
        self,
        sensor_dim: int = PRACTICAL_SENSOR_DIM,
        code_dim: int = 8,
        gyro_full_scale: float = 1.0,
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if sensor_dim != PRACTICAL_SENSOR_DIM:
            raise ValueError(
                f"The released practical packet has {PRACTICAL_SENSOR_DIM} values; "
                f"received sensor_dim={sensor_dim}"
            )
        if code_dim != 8:
            raise ValueError("PracticalSensorEncoder currently targets the eight RMR causes")
        self.sensor_dim = int(sensor_dim)
        self.code_dim = int(code_dim)
        if gyro_full_scale <= 0.0:
            raise ValueError("gyro_full_scale must be greater than zero")
        self.gyro_full_scale = float(gyro_full_scale)
        if not 0.0 <= residual_scale <= 0.25:
            raise ValueError("residual_scale must be in [0, 0.25]")
        self.residual_scale = float(residual_scale)
        self.gyro_temporal = nn.Sequential(
            nn.Conv1d(3, 24, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(24, 32, 3, padding=1),
            nn.GELU(),
        )
        self.accel_temporal = nn.Sequential(
            nn.Conv1d(3, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(16, 24, 3, padding=1),
            nn.GELU(),
        )
        self.camera = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 12))
        self.vehicle = nn.Sequential(nn.Linear(4, 12), nn.GELU(), nn.Linear(12, 8))
        self.reliability = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 8))
        self.fuse = nn.Sequential(
            nn.Linear(140, 64),
            nn.GELU(),
            nn.Linear(64, code_dim),
        )
        self.physical_fuse = nn.Sequential(
            nn.Linear(140, 64),
            nn.GELU(),
            nn.Linear(64, code_dim),
        )
        # Availability-conditioned residual calibration:
        #
        #   r(m, a) = r_shared(m) + r_a(m),
        #   a = 4 * 1_camera + 2 * 1_IMU + 1_vehicle.
        #
        # The eight experts are deliberately small and zero-initialized.
        # Consequently an older checkpoint has identical predictions when
        # transferred, while subsequent training can correct the systematic
        # bias of each full/partial/missing packet pattern independently.
        self.availability_conditioning_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(140, 24),
                    nn.GELU(),
                    nn.Linear(24, code_dim),
                )
                for _ in AVAILABILITY_PATTERNS
            ]
        )
        self.availability_physical_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(140, 24),
                    nn.GELU(),
                    nn.Linear(24, code_dim),
                )
                for _ in AVAILABILITY_PATTERNS
            ]
        )
        self._init_residual()
        self.last_direct_code: torch.Tensor | None = None
        self.last_conditioning_direct_code: torch.Tensor | None = None
        self.last_calibrated_physical_code: torch.Tensor | None = None
        self.last_reliability: torch.Tensor | None = None
        self.last_cause_reliability: torch.Tensor | None = None

    def _init_residual(self) -> None:
        for branch in (self.fuse, self.physical_fuse):
            final = branch[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
        for expert in (
            *self.availability_conditioning_experts,
            *self.availability_physical_experts,
        ):
            final = expert[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

    @staticmethod
    def _route_availability_experts(
        features: torch.Tensor,
        pattern_index: torch.Tensor,
        experts: nn.ModuleList,
    ) -> torch.Tensor:
        """Evaluate only the expert associated with each packet pattern."""

        routed = features.new_zeros(features.shape[0], 8)
        for index, expert in enumerate(experts):
            selected = pattern_index == index
            if selected.any():
                routed[selected] = expert(features[selected])
        return routed

    def forward(self, packet: torch.Tensor) -> torch.Tensor:
        if packet.ndim == 1:
            packet = packet.unsqueeze(0)
        if packet.shape[-1] != self.sensor_dim:
            raise ValueError(
                f"Expected sensor packet dimension {self.sensor_dim}, got {packet.shape[-1]}"
            )
        packet = clamp_sensor_packet(packet.to(dtype=next(self.parameters()).dtype))
        physical_direct = observable_code_from_packet(
            packet,
            gyro_full_scale=self.gyro_full_scale,
        )
        conditioning_direct = conditioning_code_from_packet(
            packet,
            gyro_full_scale=self.gyro_full_scale,
        )
        gyro = packet[:, :GYRO_END].reshape(-1, TRAJECTORY_STEPS, 3).transpose(1, 2)
        accel = packet[:, GYRO_END:ACCEL_END].reshape(-1, TRAJECTORY_STEPS, 3).transpose(1, 2)
        context = packet[:, CONTEXT_START:]
        gyro_features = self.gyro_temporal(gyro)
        accel_features = self.accel_temporal(accel)
        gyro_summary = torch.cat(
            [gyro_features.mean(dim=2), gyro_features.amax(dim=2)],
            dim=1,
        )
        accel_summary = torch.cat(
            [accel_features.mean(dim=2), accel_features.amax(dim=2)],
            dim=1,
        )
        features = torch.cat(
            [
                gyro_summary,
                accel_summary,
                self.camera(context[:, :8]),
                self.vehicle(context[:, 8:12]),
                self.reliability(context[:, 12:16]),
            ],
            dim=1,
        )
        pattern_index = sensor_availability_index(packet)
        conditioning_expert = self._route_availability_experts(
            features,
            pattern_index,
            self.availability_conditioning_experts,
        )
        physical_expert = self._route_availability_experts(
            features,
            pattern_index,
            self.availability_physical_experts,
        )
        # Paper: z_sensor = z_physics + r * Delta z.  residual_scale bounds
        # the learned calibration Delta z; zero is the direct-physics control.
        residual = self.residual_scale * torch.tanh(
            self.fuse(features) + conditioning_expert
        )
        physical_residual = self.residual_scale * torch.tanh(
            self.physical_fuse(features) + physical_expert
        )
        reliability = context[:, 15:16].clamp(0.0, 1.0)
        output = (conditioning_direct + reliability * residual).clamp(0.0, 1.0)
        calibrated_physical = (
            physical_direct + reliability * physical_residual
        ).clamp(0.0, 1.0)
        timestamp_quality = 1.0 - context[:, 10:11].abs().clamp(0.0, 1.0)
        imu_noise_quality = 1.0 - context[:, 11:12].clamp(0.0, 1.0)
        imu_quality = (
            context[:, 13:14].clamp(0.0, 1.0)
            * timestamp_quality
            * (0.5 + 0.5 * imu_noise_quality)
            * reliability
        )
        camera_quality = context[:, 12:13].clamp(0.0, 1.0) * reliability
        vehicle_quality = (
            context[:, 14:15].clamp(0.0, 1.0) * reliability
        )
        # The exposure-synchronized gyro directly observes camera rotation.
        # Vehicle speed/yaw provide useful context to the joint image posterior
        # but cannot alone determine pixel blur without depth and camera
        # extrinsics, so they are not labelled as direct PSF support.
        motion_quality = imu_quality
        autofocus_quality = (
            0.25 + 0.75 * context[:, 6:7].clamp(0.0, 1.0)
        ) * camera_quality
        cause_reliability = torch.zeros_like(output)
        cause_reliability[:, :3] = motion_quality
        cause_reliability[:, 3:4] = autofocus_quality
        cause_reliability[:, 4:6] = camera_quality
        # Compression has no released sensor field in this protocol. It must
        # therefore be inferred from the image rather than borrowed from IMU.
        cause_reliability[:, 6:7] = 0.0
        cause_reliability[:, 7:8] = cause_reliability[:, :7].amax(dim=1, keepdim=True)
        self.last_direct_code = physical_direct
        self.last_conditioning_direct_code = conditioning_direct
        self.last_calibrated_physical_code = calibrated_physical
        self.last_reliability = reliability
        self.last_cause_reliability = cause_reliability
        return output


def packet_as_named_dict(packet: Sequence[float]) -> dict[str, float]:
    """Serialize a packet with stable, reviewer-readable field names."""

    if len(packet) != PRACTICAL_SENSOR_DIM:
        raise ValueError(f"Expected {PRACTICAL_SENSOR_DIM} values, got {len(packet)}")
    return {name: float(value) for name, value in zip(SENSOR_PACKET_NAMES, packet)}


def axial_code_from_length_angle(length_px: float, angle_deg: float) -> list[float]:
    """Training-target helper used only by the controlled dataset builder."""

    magnitude = max(0.0, min(float(length_px) / 25.0, 1.0))
    angle = math.radians(float(angle_deg))
    return [
        0.5 * (1.0 + math.cos(2.0 * angle)) * magnitude,
        0.5 * (1.0 + math.sin(2.0 * angle)) * magnitude,
        magnitude,
    ]
