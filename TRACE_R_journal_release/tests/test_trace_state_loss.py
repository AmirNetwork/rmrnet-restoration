from __future__ import annotations

import torch

from train_matched_restorer import (
    SensorRecordReservoir,
    causewise_available_state_loss,
    metadata_robustness_scale,
    mismatch_sensor_records,
)


def test_unsupported_state_coordinates_have_zero_gradient() -> None:
    prediction = torch.tensor([[0.8, 0.7, 0.6, 0.5]], requires_grad=True)
    support = torch.tensor([[1.0, 0.0, 1.0, 0.0]], requires_grad=True)
    loss = causewise_available_state_loss(
        prediction,
        torch.zeros_like(prediction),
        support,
    )
    loss.backward()

    assert float(prediction.grad[0, 0]) > 0.0
    assert float(prediction.grad[0, 1]) == 0.0
    assert float(prediction.grad[0, 2]) > 0.0
    assert float(prediction.grad[0, 3]) == 0.0
    assert support.grad is None


def test_packet_mismatch_preserves_donor_targets() -> None:
    reservoir = SensorRecordReservoir(capacity=4)
    donor_metadata = torch.full((1, 82), 0.75)
    donor_cause = torch.tensor([[1.0, 0.0]])
    donor_physical = torch.tensor([[0.8, 0.2]])
    donor_available = torch.ones(1)
    reservoir.update(donor_metadata, donor_cause, donor_physical, donor_available)

    result = mismatch_sensor_records(
        torch.full((1, 82), 0.10),
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[0.1, 0.9]]),
        torch.ones(1),
        probability=1.0,
        reservoir=reservoir,
    )
    wrong_metadata, sensor_cause, sensor_physical, _, mismatch = result
    assert torch.equal(wrong_metadata, donor_metadata)
    assert torch.equal(sensor_cause, donor_cause)
    assert torch.equal(sensor_physical, donor_physical)
    assert bool(mismatch.item())


def test_metadata_curriculum_has_aligned_and_ramp_phases() -> None:
    assert metadata_robustness_scale(1, 4, 4) == 0.0
    assert metadata_robustness_scale(4, 4, 4) == 0.0
    assert metadata_robustness_scale(5, 4, 4) == 0.25
    assert metadata_robustness_scale(8, 4, 4) == 1.0
