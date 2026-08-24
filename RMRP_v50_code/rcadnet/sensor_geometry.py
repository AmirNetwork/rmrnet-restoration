from __future__ import annotations

"""Geometry-aware supervision for practical motion metadata.

RMR-Net stores axial motion in the first three degradation coordinates:

    z_0 = 0.5 m (1 + cos 2 theta),
    z_1 = 0.5 m (1 + sin 2 theta),
    z_2 = m = min(L / L_max, 1).

Coordinate-wise regression is poorly matched to the Wiener operator: a small
error in ``z_0`` or ``z_1`` can produce a large angular error after decoding.
The loss below therefore supervises blur length and the axial direction vector
directly. Direction is evaluated modulo 180 degrees, as required for a line
spread function.
"""

import math

import torch
import torch.nn.functional as F


def axial_vector(code: torch.Tensor) -> torch.Tensor:
    """Return ``m [cos(2 theta), sin(2 theta)]`` without unstable division."""

    magnitude = code[:, 2:3].clamp(0.0, 1.0)
    return torch.cat(
        [
            2.0 * code[:, 0:1] - magnitude,
            2.0 * code[:, 1:2] - magnitude,
        ],
        dim=1,
    )


def axial_angle(code: torch.Tensor) -> torch.Tensor:
    """Decode the axial angle in radians over ``[-pi/2, pi/2]``."""

    vector = axial_vector(code)
    magnitude = code[:, 2].clamp(0.0, 1.0)
    angle = 0.5 * torch.atan2(vector[:, 1], vector[:, 0])
    return torch.where(magnitude > 1e-6, angle, torch.zeros_like(angle))


def wrapped_axial_angle_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Absolute axial direction error in radians, wrapped modulo pi."""

    difference = axial_angle(prediction) - axial_angle(target)
    difference = torch.remainder(
        difference + math.pi / 2.0,
        math.pi,
    ) - math.pi / 2.0
    return difference.abs()


def practical_psf_geometry_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
    *,
    sample_available: torch.Tensor | None = None,
    length_weight: float = 4.0,
    vector_weight: float = 2.0,
    direction_weight: float = 0.5,
    max_length_px: float = 25.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Supervise motion-kernel geometry under partial metadata support.

    The optimization objective is

        L_PSF = w_L rho(m, m*) + w_v rho(v, v*)
                + w_theta 1[m* > 1/L_max] (1 - cos(2(theta-theta*))).

    ``support`` is cause-wise camera/IMU/vehicle reliability. Its first three
    coordinates are averaged into a detached motion-observability weight. An
    unavailable motion packet therefore contributes no sensor-only gradient;
    the deployed fusion path falls back to image evidence for those samples.
    """

    if prediction.shape != target.shape:
        raise ValueError(
            f"PSF geometry shapes differ: {prediction.shape} vs {target.shape}"
        )
    if prediction.ndim != 2 or prediction.shape[1] < 3:
        raise ValueError("PSF geometry expects [batch, >=3] degradation codes")
    if support.shape != prediction.shape:
        raise ValueError(
            f"PSF support shape {support.shape} does not match {prediction.shape}"
        )

    motion_support = support[:, :3].detach().to(
        device=prediction.device,
        dtype=prediction.dtype,
    ).mean(dim=1).clamp(0.0, 1.0)
    if sample_available is not None:
        available = sample_available.to(
            device=prediction.device,
            dtype=prediction.dtype,
        ).reshape(prediction.shape[0])
        motion_support = motion_support * available

    predicted_magnitude = prediction[:, 2].clamp(0.0, 1.0)
    target_magnitude = target[:, 2].clamp(0.0, 1.0)
    length_pointwise = F.smooth_l1_loss(
        predicted_magnitude,
        target_magnitude,
        reduction="none",
    )
    vector_pointwise = F.smooth_l1_loss(
        axial_vector(prediction),
        axial_vector(target),
        reduction="none",
    ).mean(dim=1)

    predicted_unit = F.normalize(axial_vector(prediction), dim=1, eps=1e-6)
    target_unit = F.normalize(axial_vector(target), dim=1, eps=1e-6)
    direction_pointwise = (
        1.0 - (predicted_unit * target_unit).sum(dim=1).clamp(-1.0, 1.0)
    )
    motion_present = (
        target_magnitude > (1.0 / float(max_length_px))
    ).to(prediction.dtype)

    denominator = motion_support.sum().clamp_min(1.0)
    direction_denominator = (
        motion_support * motion_present
    ).sum().clamp_min(1.0)
    length_loss = (motion_support * length_pointwise).sum() / denominator
    vector_loss = (motion_support * vector_pointwise).sum() / denominator
    direction_loss = (
        motion_support * motion_present * direction_pointwise
    ).sum() / direction_denominator
    total = (
        float(length_weight) * length_loss
        + float(vector_weight) * vector_loss
        + float(direction_weight) * direction_loss
    )

    angle_error = wrapped_axial_angle_error(prediction, target)
    angle_error_deg = angle_error * (180.0 / math.pi)
    metrics = {
        "length": length_loss,
        "vector": vector_loss,
        "direction": direction_loss,
        "length_error_px": (
            motion_support
            * (predicted_magnitude - target_magnitude).abs()
            * float(max_length_px)
        ).sum()
        / denominator,
        "angle_error_deg": (
            motion_support * motion_present * angle_error_deg
        ).sum()
        / direction_denominator,
        "motion_support": motion_support.mean(),
    }
    return total, metrics
