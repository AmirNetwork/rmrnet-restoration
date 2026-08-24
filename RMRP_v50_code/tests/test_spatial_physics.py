from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from models.rmrnet import RMRNet
from rcadnet.practical_metadata import (
    CONTEXT_START,
    GYRO_END,
    PRACTICAL_SENSOR_DIM,
    TRAJECTORY_STEPS,
)
from rcadnet.spatial_physics import (
    RotationExposurePhysics,
    encode_log_exposure_ms,
    physics_reblur_loss,
)
from tools.restore_native_yolo_split import load_rcadnet


def packet(exposure_ms: float, gyro_x: float = 0.0) -> torch.Tensor:
    value = torch.zeros(1, PRACTICAL_SENSOR_DIM)
    value[:, :GYRO_END].reshape(1, TRAJECTORY_STEPS, 3)[:, :, 0] = gyro_x
    value[:, CONTEXT_START] = encode_log_exposure_ms(exposure_ms)
    value[:, CONTEXT_START + 12 : CONTEXT_START + 16] = 1.0
    return value


class SpatialPhysicsTest(unittest.TestCase):
    def test_native_crid_exposure_is_preserved(self) -> None:
        operator = RotationExposurePhysics(gyro_full_scale=4.0)
        state = operator.build(packet(0.25, gyro_x=0.01), 32, 48)
        self.assertAlmostEqual(float(state.exposure_ms[0]), 0.25, places=5)
        self.assertLess(float(state.motion_px[0]), 0.1)
        self.assertLess(float(state.reliability[0, 0, 0, 0]), 0.02)
        self.assertAlmostEqual(float(state.samples_per_exposure[0]), 0.05, places=5)

    def test_temporal_observability_distinguishes_constant_and_trajectory(self) -> None:
        operator = RotationExposurePhysics(
            gyro_full_scale=4.0,
            calibration_reliability=1.0,
        )
        short = operator.build(packet(0.25, gyro_x=0.2), 32, 48)
        long = operator.build(packet(20.0, gyro_x=0.2), 32, 48)
        self.assertLess(float(short.trajectory_reliability[0]), 0.05)
        self.assertGreater(float(short.constant_rate_reliability[0]), 0.99)
        self.assertGreater(float(long.trajectory_reliability[0]), 0.99)

    def test_zero_motion_reblur_is_identity(self) -> None:
        operator = RotationExposurePhysics(gyro_full_scale=4.0)
        state = operator.build(packet(10.0), 32, 48)
        image = torch.rand(1, 3, 32, 48)
        reblurred = operator.reblur(image, state)
        self.assertTrue(torch.allclose(reblurred, image, atol=2.0e-6))

    def test_reblur_loss_has_finite_image_gradient(self) -> None:
        operator = RotationExposurePhysics(
            gyro_full_scale=4.0,
            activation_motion_px=0.01,
            calibration_reliability=1.0,
        )
        state = operator.build(packet(20.0, gyro_x=0.3), 32, 48)
        restored = torch.rand(1, 3, 32, 48, requires_grad=True)
        observed = torch.rand_like(restored)
        loss = physics_reblur_loss(
            operator.reblur(restored, state),
            observed,
            state.reliability,
        )
        loss.backward()
        self.assertIsNotNone(restored.grad)
        self.assertTrue(torch.isfinite(restored.grad).all())

    def test_inverse_candidate_is_identity_without_motion(self) -> None:
        operator = RotationExposurePhysics(gyro_full_scale=4.0)
        state = operator.build(packet(10.0), 32, 48)
        image = torch.rand(1, 3, 32, 48)
        candidate = operator.inverse_candidate(image, state)
        self.assertTrue(torch.allclose(candidate, image, atol=2.0e-6))

    def test_rmrnet_physics_outputs_and_checkpoint_compatibility(self) -> None:
        old = RMRNet(
            width=8,
            blocks_per_stage=1,
            use_estimated_code=True,
            code_fusion="sensor_fused",
            use_practical_sensor_encoder=True,
            use_sensor_image_psf_refiner=True,
            enable_aux_contour=False,
        )
        model = RMRNet(
            width=8,
            blocks_per_stage=1,
            use_estimated_code=True,
            code_fusion="sensor_fused",
            use_practical_sensor_encoder=True,
            use_sensor_image_psf_refiner=True,
            use_spatial_physics=True,
            enable_aux_contour=False,
        )
        report = model.load_pretrained({"model": old.state_dict()}, strict=False)
        self.assertTrue(
            any(name.startswith("physics_feature_encoder.") for name in report["missing_keys"])
        )
        image = torch.rand(1, 3, 32, 48)
        result = model(image, packet(8.0, 0.2), return_dict=True)
        self.assertEqual(tuple(result["restored"].shape), tuple(image.shape))
        self.assertEqual(tuple(result["physics_reblurred"].shape), tuple(image.shape))
        self.assertEqual(result["physics_flow"].shape[-1], 2)
        self.assertIsNotNone(result["physics_temporal_reliability"])

        reversed_packet = packet(8.0, -0.2)
        isolated = model(
            image,
            packet(8.0, 0.2),
            physics_code=reversed_packet,
            return_dict=True,
        )
        self.assertLess(
            float((result["physics_flow"] + isolated["physics_flow"]).abs().mean()),
            1.0e-4,
        )

    def test_inference_loader_recovers_first_run_physics_args(self) -> None:
        """The first physics run saved its flags in args, before arch was fixed."""

        model = RMRNet(
            width=8,
            use_estimated_code=True,
            code_fusion="sensor_fused",
            use_practical_sensor_encoder=True,
            use_sensor_image_psf_refiner=True,
            use_spatial_physics=True,
            enable_aux_contour=False,
        )
        checkpoint = {
            "model": model.state_dict(),
            "arch": {
                "width": 8,
                "use_estimated_code": True,
                "code_fusion": "sensor_fused",
                "use_practical_sensor_encoder": True,
                "use_sensor_image_psf_refiner": True,
            },
            "args": {
                "spatial_physics": True,
                "physics_samples": 5,
                "physics_exposure_min_ms": 0.05,
                "physics_exposure_max_ms": 40.0,
                "physics_focal_ratio": 0.75,
                "physics_calibration_reliability": 0.5,
                "physics_activation_motion_px": 0.1,
            },
        }
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pth"
            torch.save(checkpoint, path)
            loaded = load_rcadnet(path, torch.device("cpu"))
        self.assertTrue(loaded.use_spatial_physics)
        self.assertIsNotNone(loaded.physics_feature_encoder)

    def test_exclusive_trajectory_keeps_gyro_out_of_learned_encoder(self) -> None:
        model = RMRNet(
            width=8,
            use_estimated_code=True,
            code_fusion="sensor_fused",
            use_practical_sensor_encoder=True,
            use_spatial_physics=True,
            physics_exclusive_trajectory=True,
            enable_aux_contour=False,
        ).eval()
        image = torch.rand(1, 3, 32, 48)
        positive = packet(8.0, 0.2)
        negative = packet(8.0, -0.2)
        with torch.no_grad():
            out_positive = model(image, positive, return_dict=True)
            out_negative = model(image, negative, return_dict=True)
        self.assertTrue(
            torch.allclose(
                out_positive["sensor_code"], out_negative["sensor_code"], atol=1e-7
            )
        )
        self.assertFalse(
            torch.allclose(
                out_positive["physics_flow"], out_negative["physics_flow"]
            )
        )

    def test_image_estimated_model_accepts_physics_packet_separately(self) -> None:
        model = RMRNet(
            width=8,
            use_estimated_code=True,
            code_fusion="estimated",
            use_practical_sensor_encoder=False,
            use_spatial_physics=True,
            enable_aux_contour=False,
        ).eval()
        image = torch.rand(1, 3, 32, 48)
        with torch.no_grad():
            result = model(
                image,
                None,
                physics_code=packet(10.0, 0.2),
                return_dict=True,
            )
        self.assertIsNotNone(result["physics_flow"])
        self.assertIsNone(result["sensor_code"])

    def test_inverse_candidate_is_bounded_restoration_input(self) -> None:
        model = RMRNet(
            width=8,
            use_estimated_code=True,
            code_fusion="estimated",
            use_spatial_physics=True,
            use_physics_inverse_candidate=True,
            physics_inverse_blend=1.0,
            enable_aux_contour=False,
        ).eval()
        image = torch.rand(1, 3, 32, 48)
        with torch.no_grad():
            result = model(
                image,
                None,
                physics_code=packet(20.0, 0.2),
                return_dict=True,
            )
        self.assertIsNotNone(result["physics_inverse_candidate"])
        self.assertGreaterEqual(float(result["physics_restoration_input"].min()), 0.0)
        self.assertLessEqual(float(result["physics_restoration_input"].max()), 1.0)

    def test_decoder_gate_suppresses_subpixel_neural_correction(self) -> None:
        model = RMRNet(
            width=8,
            use_estimated_code=True,
            code_fusion="estimated",
            use_spatial_physics=True,
            use_physics_inverse_candidate=True,
            physics_decoder_motion_threshold_px=1.5,
            physics_decoder_motion_transition_px=0.35,
            enable_aux_contour=False,
        ).eval()
        # Batch size two guards the scalar-motion/spatial-reliability broadcast.
        image = torch.rand(2, 3, 32, 48)
        with torch.no_grad():
            short = model(
                image,
                None,
                physics_code=packet(0.25, 0.2).repeat(2, 1),
                return_dict=True,
            )
            long = model(
                image,
                None,
                physics_code=packet(20.0, 0.2).repeat(2, 1),
                return_dict=True,
            )
        self.assertIsNotNone(short["physics_decoder_gate"])
        self.assertLess(
            float(short["physics_decoder_gate"].mean()),
            float(long["physics_decoder_gate"].mean()),
        )


if __name__ == "__main__":
    unittest.main()
