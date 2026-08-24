from __future__ import annotations

import unittest

import torch

from train_rcadnet import masked_sensor_state_loss
from train_matched_restorer import (
    SensorRecordReservoir,
    cause_targets_to_prompt_weights,
    causewise_available_state_loss,
    metadata_robustness_scale,
    mismatch_sensor_records,
    route_teacher_probability,
)


class MaskedSensorStateLossTests(unittest.TestCase):
    def test_unsupported_coordinates_have_zero_gradient(self) -> None:
        prediction = torch.tensor(
            [[0.8, 0.7, 0.6, 0.5]],
            requires_grad=True,
        )
        target = torch.zeros_like(prediction)
        support = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        loss, active_fraction = masked_sensor_state_loss(
            prediction,
            target,
            support,
        )
        loss.backward()

        self.assertAlmostEqual(float(active_fraction), 0.5)
        self.assertGreater(float(prediction.grad[0, 0]), 0.0)
        self.assertEqual(float(prediction.grad[0, 1]), 0.0)
        self.assertGreater(float(prediction.grad[0, 2]), 0.0)
        self.assertEqual(float(prediction.grad[0, 3]), 0.0)

    def test_unavailable_calibration_sample_is_excluded(self) -> None:
        prediction = torch.tensor(
            [[0.5, 0.5], [0.5, 0.5]],
            requires_grad=True,
        )
        target = torch.zeros_like(prediction)
        support = torch.ones_like(prediction)
        available = torch.tensor([1.0, 0.0])
        loss, _ = masked_sensor_state_loss(
            prediction,
            target,
            support,
            sample_available=available,
        )
        loss.backward()

        self.assertTrue(torch.all(prediction.grad[0] > 0))
        self.assertTrue(torch.all(prediction.grad[1] == 0))

    def test_matched_trainer_detaches_learned_support(self) -> None:
        prediction = torch.tensor([[0.8, 0.7]], requires_grad=True)
        support = torch.tensor([[0.9, 0.0]], requires_grad=True)
        loss = causewise_available_state_loss(
            prediction,
            torch.zeros_like(prediction),
            support,
        )
        loss.backward()

        self.assertGreater(float(prediction.grad[0, 0]), 0.0)
        self.assertEqual(float(prediction.grad[0, 1]), 0.0)
        self.assertIsNone(support.grad)

    def test_mismatched_record_keeps_donor_sensor_target(self) -> None:
        metadata = torch.arange(164, dtype=torch.float32).reshape(2, 82)
        cause = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        physical = cause.clone()
        available = torch.ones(2)
        result = mismatch_sensor_records(
            metadata,
            cause,
            physical,
            available,
            probability=1.0,
        )
        wrong_metadata, sensor_cause, sensor_physical, _, mismatch = result

        self.assertTrue(torch.equal(wrong_metadata[0], metadata[1]))
        self.assertTrue(torch.equal(sensor_cause[0], cause[1]))
        self.assertTrue(torch.equal(sensor_physical[0], physical[1]))
        self.assertTrue(bool(mismatch.all()))

    def test_reservoir_enables_mismatch_for_micro_batch_one(self) -> None:
        reservoir = SensorRecordReservoir(capacity=4)
        donor_metadata = torch.full((1, 82), 0.75)
        donor_cause = torch.tensor([[1.0, 0.0]])
        donor_physical = torch.tensor([[0.8, 0.2]])
        donor_available = torch.ones(1)
        reservoir.update(
            donor_metadata,
            donor_cause,
            donor_physical,
            donor_available,
        )

        current_metadata = torch.full((1, 82), 0.10)
        current_cause = torch.tensor([[0.0, 1.0]])
        current_physical = torch.tensor([[0.1, 0.9]])
        result = mismatch_sensor_records(
            current_metadata,
            current_cause,
            current_physical,
            torch.ones(1),
            probability=1.0,
            reservoir=reservoir,
        )
        wrong_metadata, sensor_cause, sensor_physical, _, mismatch = result

        self.assertTrue(torch.equal(wrong_metadata, donor_metadata))
        self.assertTrue(torch.equal(sensor_cause, donor_cause))
        self.assertTrue(torch.equal(sensor_physical, donor_physical))
        self.assertTrue(bool(mismatch.item()))

    def test_metadata_curriculum_has_aligned_then_ramped_phases(self) -> None:
        self.assertEqual(metadata_robustness_scale(1, 8, 5), 0.0)
        self.assertEqual(metadata_robustness_scale(8, 8, 5), 0.0)
        self.assertAlmostEqual(metadata_robustness_scale(9, 8, 5), 0.2)
        self.assertAlmostEqual(metadata_robustness_scale(11, 8, 5), 0.6)
        self.assertEqual(metadata_robustness_scale(13, 8, 5), 1.0)
        self.assertEqual(metadata_robustness_scale(20, 8, 5), 1.0)

    def test_zero_length_curriculum_enables_robustness_immediately(self) -> None:
        self.assertEqual(metadata_robustness_scale(1, 0, 0), 1.0)

    def test_route_teacher_schedule_is_fully_withdrawn(self) -> None:
        self.assertEqual(route_teacher_probability(5, 5, 5), 1.0)
        self.assertAlmostEqual(route_teacher_probability(6, 5, 5), 0.8)
        self.assertEqual(route_teacher_probability(10, 5, 5), 0.0)

    def test_cause_targets_map_to_clean_and_compound_prompts(self) -> None:
        causes = torch.zeros(2, 8)
        causes[1, 0] = 0.5
        causes[1, 5] = 0.6
        weights = cause_targets_to_prompt_weights(causes, 6)
        assert weights is not None
        self.assertEqual(weights.argmax(dim=1).tolist(), [0, 5])


if __name__ == "__main__":
    unittest.main()
