# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
from __future__ import annotations

import json
import unittest

import numpy as np
import torch

from models.rmrnet import RMRNet
from rcadnet.model import SensorImagePSFRefiner
from rcadnet.practical_metadata import (
    ACCEL_END,
    CONTEXT_START,
    GYRO_END,
    PRACTICAL_SENSOR_DIM,
    TRAJECTORY_STEPS,
    PracticalSensorEncoder,
    apply_sensor_modality_mask,
    balanced_sensor_dropout,
    clamp_sensor_packet,
    conditioning_code_from_packet,
    counterfactual_sensor_packet,
    observable_code_from_packet,
    perturb_sensor_packet,
    sensor_availability_index,
    structured_sensor_dropout,
)
from tools.build_practical_metadata_benchmark import (
    practical_sidecar,
    private_calibration_target,
)


def one_hidden_sample() -> dict:
    """Return a deterministic private renderer record for an isolated test."""

    return {
        "scenario_family": "mixed",
        "gyro_x": 0.20,
        "gyro_y": 0.10,
        "accel_norm": 0.35,
        "speed_mps": 12.0,
        "exposure_ms": 18.0,
        "defocus_score": 0.15,
        "noise_score": 0.25,
        "blur_angle_deg": 22.0,
        "blur_length_px": 11.0,
        "low_light_score": 0.70,
        "jpeg_quality": 70.0,
    }


def one_oxts_sample() -> dict:
    """Return sensor-observable vehicle context without requiring KITTI data."""

    return {
        "drive": "unit_test_drive",
        "frame": "0000000000",
        "vf": 12.0,
        "wl": 0.01,
        "wf": -0.02,
        "wu": 0.015,
        "af": 0.20,
        "al": -0.10,
        "au": 0.02,
        "velacc": 0.10,
    }


class PracticalMetadataTest(unittest.TestCase):
    def test_gyro_full_scale_restores_physical_motion_magnitude(self) -> None:
        packet = torch.zeros(PRACTICAL_SENSOR_DIM)
        packet[:GYRO_END].reshape(TRAJECTORY_STEPS, 3)[:, 0] = 0.20
        packet[CONTEXT_START] = 1.0
        packet[CONTEXT_START + 12 : CONTEXT_START + 16] = 1.0
        normalized = observable_code_from_packet(packet, gyro_full_scale=1.0)
        calibrated = observable_code_from_packet(packet, gyro_full_scale=4.0)
        self.assertAlmostEqual(float(normalized[2]), 0.20, places=5)
        self.assertAlmostEqual(float(calibrated[2]), 0.80, places=5)

    def test_sensor_ranges_preserve_signed_trajectory(self) -> None:
        packet = torch.full((2, PRACTICAL_SENSOR_DIM), 2.0)
        packet[:, :ACCEL_END:2] = -2.0
        packet[:, CONTEXT_START + 9 : CONTEXT_START + 11] = -2.0
        clamped = clamp_sensor_packet(packet)
        self.assertGreaterEqual(float(clamped[:, :ACCEL_END].min()), -1.0)
        self.assertLessEqual(float(clamped[:, :ACCEL_END].max()), 1.0)
        self.assertGreaterEqual(
            float(clamped[:, CONTEXT_START : CONTEXT_START + 9].min()),
            0.0,
        )
        self.assertLessEqual(
            float(clamped[:, CONTEXT_START : CONTEXT_START + 9].max()),
            1.0,
        )
        self.assertEqual(GYRO_END, 33)

    def test_counterfactual_packet_changes_cause_but_preserves_reliability(self) -> None:
        packet = torch.zeros(PRACTICAL_SENSOR_DIM)
        gyro = packet[:GYRO_END].reshape(11, 3)
        gyro[:, 0] = torch.linspace(0.1, 0.8, 11)
        gyro[:, 1] = torch.linspace(-0.2, 0.3, 11)
        accel = packet[GYRO_END:ACCEL_END].reshape(11, 3)
        accel[:, 2] = torch.linspace(-0.6, 0.6, 11)
        packet[CONTEXT_START + 0] = 0.7
        packet[CONTEXT_START + 1] = 0.6
        packet[CONTEXT_START + 2] = 0.5
        packet[CONTEXT_START + 5] = 0.2
        packet[CONTEXT_START + 12 : CONTEXT_START + 16] = torch.tensor(
            [0.9, 0.8, 0.7, 1.0]
        )

        counterfactual = counterfactual_sensor_packet(packet)
        original_code = observable_code_from_packet(packet)
        counterfactual_code = observable_code_from_packet(counterfactual)

        self.assertTrue(
            torch.equal(
                packet[CONTEXT_START + 12 : CONTEXT_START + 16],
                counterfactual[CONTEXT_START + 12 : CONTEXT_START + 16],
            )
        )
        self.assertGreater(
            float((original_code - counterfactual_code).abs().mean()),
            0.05,
        )

    def test_temporal_encoder_receives_gradients(self) -> None:
        encoder = PracticalSensorEncoder()
        packet = torch.zeros(2, PRACTICAL_SENSOR_DIM)
        packet[:, :GYRO_END] = torch.randn(2, GYRO_END) * 0.1
        packet[:, CONTEXT_START + 12 :] = 1.0
        prediction = encoder(packet)
        prediction.square().mean().backward()
        gradients = [
            parameter.grad
            for parameter in encoder.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(any(grad is not None and torch.isfinite(grad).all() for grad in gradients))

    def test_partial_modality_mask_updates_matching_reliability(self) -> None:
        packet = torch.ones(2, PRACTICAL_SENSOR_DIM)
        masked = apply_sensor_modality_mask(
            packet,
            camera_available=torch.tensor([1.0, 0.0]),
            imu_available=torch.tensor([0.0, 1.0]),
            vehicle_available=torch.tensor([1.0, 0.0]),
        )
        self.assertEqual(float(masked[0, :ACCEL_END].abs().sum()), 0.0)
        self.assertEqual(
            float(masked[1, CONTEXT_START : CONTEXT_START + 8].abs().sum()),
            0.0,
        )
        self.assertEqual(float(masked[0, CONTEXT_START + 13]), 0.0)
        self.assertEqual(float(masked[1, CONTEXT_START + 12]), 0.0)
        self.assertEqual(float(masked[1, CONTEXT_START + 14]), 0.0)
        self.assertEqual(float(masked[:, CONTEXT_START + 15].min()), 1.0)

    def test_noise_preserves_reliability_and_structured_dropout_is_valid(self) -> None:
        torch.manual_seed(4)
        packet = torch.rand(8, PRACTICAL_SENSOR_DIM)
        packet[:, :ACCEL_END] = 2.0 * packet[:, :ACCEL_END] - 1.0
        packet = clamp_sensor_packet(packet)
        noisy = perturb_sensor_packet(packet, 0.1)
        self.assertTrue(
            torch.equal(
                noisy[:, CONTEXT_START + 12 : CONTEXT_START + 16],
                packet[:, CONTEXT_START + 12 : CONTEXT_START + 16],
            )
        )
        dropped = structured_sensor_dropout(packet, 0.5)
        self.assertEqual(tuple(dropped.shape), tuple(packet.shape))
        self.assertTrue(torch.isfinite(dropped).all())

        camera_only = apply_sensor_modality_mask(
            packet,
            camera_available=True,
            imu_available=False,
            vehicle_available=False,
        )
        noisy_camera_only = perturb_sensor_packet(camera_only, 0.1)
        self.assertEqual(
            float(noisy_camera_only[:, :ACCEL_END].abs().sum()),
            0.0,
        )
        self.assertEqual(
            float(
                noisy_camera_only[
                    :,
                    CONTEXT_START + 8 : CONTEXT_START + 12,
                ].abs().sum()
            ),
            0.0,
        )

    def test_balanced_dropout_covers_partial_and_missing_packets(self) -> None:
        torch.manual_seed(9)
        packet = torch.ones(4096, PRACTICAL_SENSOR_DIM)
        dropped = balanced_sensor_dropout(packet, 1.0)
        camera = dropped[:, CONTEXT_START + 12] > 0
        imu = dropped[:, CONTEXT_START + 13] > 0
        vehicle = dropped[:, CONTEXT_START + 14] > 0
        patterns = {
            (bool(c), bool(i), bool(v))
            for c, i, v in zip(camera, imu, vehicle)
        }
        self.assertEqual(
            patterns,
            {
                (False, False, False),
                (True, False, False),
                (False, True, False),
                (False, False, True),
                (True, True, False),
                (True, False, True),
                (False, True, True),
            },
        )

    def test_availability_experts_route_all_eight_patterns(self) -> None:
        packet = torch.ones(8, PRACTICAL_SENSOR_DIM)
        patterns = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=torch.float32,
        )
        packet = apply_sensor_modality_mask(
            packet,
            camera_available=patterns[:, 0],
            imu_available=patterns[:, 1],
            vehicle_available=patterns[:, 2],
        )
        self.assertTrue(
            torch.equal(
                sensor_availability_index(packet),
                torch.arange(8),
            )
        )

        encoder = PracticalSensorEncoder()
        output = encoder(packet)
        output.sum().backward()
        for index in range(1, 8):
            gradient = (
                encoder.availability_conditioning_experts[index][-1]
                .weight.grad
            )
            self.assertIsNotNone(gradient)
            self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_joint_physical_refiner_uses_coordinatewise_fallback(self) -> None:
        torch.manual_seed(12)
        refiner = SensorImagePSFRefiner(image_feature_dim=16, code_dim=8)
        image_features = torch.randn(2, 16)
        image_code = torch.rand(2, 8)
        # Matching image and metadata evidence keeps nominal reliability.
        sensor_cause = image_code.clone()
        sensor_physical = torch.rand(2, 8)
        reliability = torch.stack(
            [
                torch.ones(8),
                torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ]
        )
        posterior, image_physical, sensor_weight, joint_cause = refiner(
            image_features,
            image_code,
            sensor_cause,
            sensor_physical,
            reliability,
        )
        # The residual head starts at zero. Fully supported coordinates must
        # therefore reproduce the calibrated sensor state exactly, while
        # missing coordinates use the image-derived physical fallback.
        self.assertTrue(
            torch.allclose(posterior[0], sensor_physical[0], atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(posterior[1, :3], sensor_physical[1, :3], atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(posterior[1, 3:], image_physical[1, 3:], atol=1e-6)
        )
        self.assertTrue(torch.equal(sensor_weight, reliability))
        self.assertTrue(
            torch.allclose(joint_cause[0], sensor_cause[0], atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(joint_cause[1, :3], sensor_cause[1, :3], atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(joint_cause[1, 3:], image_code[1, 3:], atol=1e-6)
        )
        (posterior.mean() + joint_cause.mean()).backward()
        gradients = [
            parameter.grad
            for parameter in refiner.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(
            any(
                gradient is not None and torch.isfinite(gradient).all()
                for gradient in gradients
            )
        )

    def test_joint_physical_refiner_suppresses_contradicted_sensor_risk(self) -> None:
        refiner = SensorImagePSFRefiner(
            image_feature_dim=16,
            code_dim=8,
            compatibility_floor=0.10,
            compatibility_temperature=4.0,
        )
        image_features = torch.zeros(1, 16)
        image_code = torch.zeros(1, 8)
        sensor_cause = torch.zeros(1, 8)
        sensor_cause[:, 4] = 1.0  # High ISO implies noise risk.
        sensor_physical = sensor_cause.clone()
        reliability = torch.ones(1, 8)

        posterior, image_physical, sensor_weight, joint_cause = refiner(
            image_features,
            image_code,
            sensor_cause,
            sensor_physical,
            reliability,
        )

        expected_weight = 0.10 + 0.90 * torch.exp(torch.tensor(-4.0))
        self.assertAlmostEqual(
            float(sensor_weight[0, 4]),
            float(expected_weight),
            places=6,
        )
        self.assertLess(float(joint_cause[0, 4].detach()), 0.12)
        self.assertLess(float(posterior[0, 4].detach()), 0.14)
        self.assertGreater(float(image_physical[0, 4].detach()), 0.0)

    def test_joint_sensor_state_is_the_deployed_conditioning_code(self) -> None:
        model = RMRNet(
            width=8,
            blocks_per_stage=1,
            use_estimated_code=True,
            code_fusion="sensor_fused",
            conditioning="residual_basis",
            use_practical_sensor_encoder=True,
            use_sensor_image_psf_refiner=True,
            enable_aux_contour=False,
        )
        image = torch.rand(1, 3, 32, 32)
        packet = torch.zeros(1, PRACTICAL_SENSOR_DIM)
        result = model.physical_state_forward(image, packet)
        self.assertTrue(
            torch.allclose(
                result["degradation_state"],
                result["joint_degradation_code"],
                atol=1e-7,
            )
        )

    def test_observable_state_is_not_attenuated_by_reliability(self) -> None:
        packet = torch.zeros(2, PRACTICAL_SENSOR_DIM)
        packet[:, :GYRO_END] = 0.4
        packet[:, CONTEXT_START + 15] = 1.0
        packet[0, CONTEXT_START + 13] = 1.0
        packet[1, CONTEXT_START + 13] = 0.5
        code = observable_code_from_packet(packet)
        self.assertTrue(torch.allclose(code[0, :3], code[1, :3], atol=1e-6))

    def test_sensor_keeps_separate_conditioning_and_physical_codes(self) -> None:
        packet = torch.zeros(2, PRACTICAL_SENSOR_DIM)
        packet[:, :GYRO_END] = 0.4
        packet[:, CONTEXT_START + 13] = 1.0
        packet[:, CONTEXT_START + 15] = 1.0
        encoder = PracticalSensorEncoder()
        output = encoder(packet)
        expected = conditioning_code_from_packet(packet)
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))
        self.assertIsNotNone(encoder.last_direct_code)
        self.assertIsNotNone(encoder.last_calibrated_physical_code)
        self.assertFalse(torch.allclose(output[:, :3], encoder.last_direct_code[:, :3]))

    def test_rmrnet_sensor_forward_and_old_checkpoint_transfer(self) -> None:
        old = RMRNet(
            width=8,
            blocks_per_stage=1,
            use_estimated_code=True,
            code_fusion="metadata_fused",
            enable_aux_contour=False,
        )
        model = RMRNet(
            width=8,
            blocks_per_stage=1,
            use_estimated_code=True,
            code_fusion="sensor_fused",
            use_practical_sensor_encoder=True,
            use_sensor_image_psf_refiner=True,
            enable_aux_contour=False,
        )
        audit = model.load_pretrained({"model": old.state_dict()}, strict=False)
        self.assertTrue(any(name.startswith("sensor_encoder.") for name in audit["missing_keys"]))

        image = torch.rand(1, 3, 32, 32)
        packet = torch.zeros(1, PRACTICAL_SENSOR_DIM)
        packet[:, CONTEXT_START + 12 :] = 1.0
        result = model(image, packet, return_dict=True, return_aux=False)
        self.assertEqual(tuple(result["restored"].shape), tuple(image.shape))
        self.assertEqual(tuple(result["sensor_code"].shape), (1, 8))
        self.assertEqual(
            tuple(result["sensor_calibrated_physical_code"].shape),
            (1, 8),
        )
        self.assertEqual(tuple(result["image_physical_code"].shape), (1, 8))
        self.assertEqual(
            tuple(result["physical_sensor_weight"].shape),
            (1, 8),
        )
        self.assertTrue(torch.isfinite(result["restored"]).all())

    def test_sensor_prior_fusion_is_bounded_and_trainable(self) -> None:
        model = RMRNet(
            width=8,
            blocks_per_stage=1,
            use_estimated_code=True,
            code_fusion="sensor_fused",
            use_motion_prior=True,
            use_practical_sensor_encoder=True,
            use_sensor_prior_fusion=True,
            use_sensor_image_psf_refiner=True,
            enable_aux_contour=False,
        )
        image = torch.rand(1, 3, 32, 32)
        packet = torch.zeros(1, PRACTICAL_SENSOR_DIM)
        packet[:, :GYRO_END] = 0.25
        packet[:, CONTEXT_START + 12 :] = 1.0
        result = model(image, packet, return_dict=True)
        alpha = result["sensor_prior_alpha"]
        self.assertIsNotNone(alpha)
        self.assertIsNotNone(result["sensor_only_physical_code"])
        self.assertIsNotNone(result["sensor_calibrated_physical_code"])
        self.assertGreaterEqual(float(alpha.detach().min()), 0.0)
        self.assertLessEqual(float(alpha.detach().max()), 1.0)
        result["restored"].mean().backward()
        gradients = [
            parameter.grad
            for parameter in model.sensor_prior_fusion.parameters()
        ]
        self.assertTrue(
            any(
                gradient is not None and torch.isfinite(gradient).all()
                for gradient in gradients
            )
        )

    def test_test_sidecar_excludes_training_target(self) -> None:
        hidden = one_hidden_sample()
        oxts = one_oxts_sample()
        sidecar = practical_sidecar(hidden, oxts, np.random.default_rng(5))
        self.assertEqual(len(sidecar["practical_sensor_packet"]), PRACTICAL_SENSOR_DIM)
        self.assertNotIn("training_cause_target_code", sidecar)
        self.assertNotIn("training_physical_target_code", sidecar)
        serialized = json.dumps(sidecar)
        for forbidden in ("blur_length_px", "blur_angle_deg", "defocus_sigma", "noise_sigma"):
            self.assertNotIn(forbidden, serialized)

        private_target = private_calibration_target(hidden)
        self.assertEqual(len(private_target["training_cause_target_code"]), 8)
        self.assertEqual(len(private_target["training_physical_target_code"]), 8)
        self.assertIs(private_target["training_target_is_model_input"], False)


if __name__ == "__main__":
    unittest.main()
