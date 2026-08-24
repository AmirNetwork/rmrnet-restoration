from __future__ import annotations

import types
import unittest

import torch
from torch import nn

from rcadnet.model import PostPriorEvidenceRefiner
from rcadnet.task_losses import FrozenDetectorSupervisedLoss


class _ToyDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)
        self.args = types.SimpleNamespace()

    def loss(self, batch: dict[str, torch.Tensor]):
        prediction = self.conv(batch["img"])
        box = prediction.square().mean()
        cls = (prediction - 0.25).square().mean()
        dfl = prediction.abs().mean()
        parts = torch.stack([box, cls, dfl])
        return parts, parts.detach()


class MatchedDetectorObjectiveTests(unittest.TestCase):
    def test_frozen_detector_still_provides_image_gradient(self) -> None:
        detector = _ToyDetector()
        objective = FrozenDetectorSupervisedLoss(
            detector,
            input_size=(16, 16),
            cqmix_prob=0.0,
        )
        restored = torch.rand(1, 3, 16, 16, requires_grad=True)
        clean = torch.rand(1, 3, 16, 16)
        classes = torch.tensor([[0]])
        boxes = torch.tensor([[[0.5, 0.5, 0.25, 0.25]]])
        valid = torch.tensor([[True]])

        loss = objective(
            restored,
            clean,
            classes=classes,
            bboxes=boxes,
            valid=valid,
        )
        loss.backward()

        self.assertGreater(float(restored.grad.abs().sum()), 0.0)
        self.assertTrue(all(not p.requires_grad for p in detector.parameters()))
        self.assertTrue(all(p.grad is None for p in detector.parameters()))

    def test_post_prior_refiner_starts_as_bounded_identity(self) -> None:
        refiner = PostPriorEvidenceRefiner(code_dim=8, max_gain=0.2)
        candidate = torch.rand(2, 3, 24, 24)
        restored, _, correction = refiner(
            torch.rand_like(candidate),
            torch.rand_like(candidate),
            candidate,
            torch.rand(2, 8),
            torch.ones(2),
        )

        self.assertTrue(torch.equal(restored, candidate))
        self.assertEqual(float(correction.detach().abs().max()), 0.0)
        dilations = [block[0].dilation for block in refiner.blocks]
        self.assertEqual(dilations, [(1, 1), (2, 2), (4, 4)])


if __name__ == "__main__":
    unittest.main()
