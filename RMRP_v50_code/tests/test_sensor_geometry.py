from __future__ import annotations

import math
import unittest

import torch

from rcadnet.sensor_geometry import (
    practical_psf_geometry_loss,
    wrapped_axial_angle_error,
)


def axial_code(magnitude: float, angle_deg: float) -> torch.Tensor:
    theta = math.radians(angle_deg)
    return torch.tensor(
        [
            0.5 * magnitude * (1.0 + math.cos(2.0 * theta)),
            0.5 * magnitude * (1.0 + math.sin(2.0 * theta)),
            magnitude,
            0.0,
            0.0,
            0.0,
            0.0,
            magnitude,
        ],
        dtype=torch.float32,
    )


class SensorGeometryTests(unittest.TestCase):
    def test_axial_angle_wraps_at_180_degrees(self) -> None:
        first = axial_code(0.5, 89.0).unsqueeze(0)
        second = axial_code(0.5, -89.0).unsqueeze(0)
        error_deg = wrapped_axial_angle_error(first, second) * (180.0 / math.pi)
        self.assertTrue(
            torch.allclose(error_deg, torch.tensor([2.0]), atol=1e-4)
        )

    def test_geometry_loss_is_zero_for_exact_kernel(self) -> None:
        target = axial_code(0.52, -23.0).unsqueeze(0)
        support = torch.ones_like(target)
        loss, metrics = practical_psf_geometry_loss(target, target, support)
        self.assertLess(float(loss), 1e-7)
        self.assertLess(float(metrics["length_error_px"]), 1e-7)
        self.assertLess(float(metrics["angle_error_deg"]), 1e-5)

    def test_missing_motion_metadata_masks_sensor_only_gradient(self) -> None:
        prediction = axial_code(0.80, 35.0).unsqueeze(0).requires_grad_(True)
        target = axial_code(0.52, -23.0).unsqueeze(0)
        support = torch.ones_like(target)
        support[:, :3] = 0.0
        loss, _ = practical_psf_geometry_loss(prediction, target, support)
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(int(torch.count_nonzero(prediction.grad)), 0)


if __name__ == "__main__":
    unittest.main()
