from __future__ import annotations

import types
import unittest

import torch
from torch import nn

from rcadnet.model import PostPriorEvidenceRefiner
from rcadnet.task_losses import (
    DetectorEvidenceDistillationLoss,
    FrozenDetectorSupervisedLoss,
)


class _ToyDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)
        self.args = types.SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    def loss(self, batch: dict[str, torch.Tensor]):
        prediction = self.conv(batch["img"])
        box = prediction.square().mean()
        cls = (prediction - 0.25).square().mean()
        dfl = prediction.abs().mean()
        parts = torch.stack([box, cls, dfl])
        return parts, parts.detach()


class _ToyDenseDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 1)
        self.score = nn.Conv2d(8, 2, 1)
        self.box = nn.Conv2d(8, 16, 1)

    def forward(self, image: torch.Tensor):
        feature = torch.nn.functional.adaptive_avg_pool2d(
            torch.relu(self.stem(image)),
            (2, 2),
        )
        scores = self.score(feature).flatten(2)
        boxes = self.box(feature).flatten(2)
        prediction = torch.cat((boxes[:, :4], scores), dim=1)
        return prediction, {"scores": scores, "boxes": boxes, "feats": [feature]}


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

    def test_dense_evidence_distillation_is_training_only_and_differentiable(self) -> None:
        detector = _ToyDenseDetector()
        objective = DetectorEvidenceDistillationLoss(
            detector,
            input_size=(16, 16),
            foreground_topk=2,
            background_topk=1,
        )
        restored = torch.rand(2, 3, 16, 16, requires_grad=True)
        clean = torch.rand(2, 3, 16, 16)

        loss = objective(restored, clean)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(restored.grad.abs().sum()), 0.0)
        self.assertTrue(all(not p.requires_grad for p in detector.parameters()))
        self.assertTrue(all(p.grad is None for p in detector.parameters()))

    def test_component_multipliers_are_applied_to_detector_gains(self) -> None:
        detector = _ToyDetector()
        FrozenDetectorSupervisedLoss(
            detector,
            box_multiplier=0.5,
            class_multiplier=3.0,
            dfl_multiplier=0.25,
        )
        self.assertAlmostEqual(detector.args.box, 3.75)
        self.assertAlmostEqual(detector.args.cls, 1.5)
        self.assertAlmostEqual(detector.args.dfl, 0.375)

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

    def test_full_frame_letterbox_updates_normalized_boxes(self) -> None:
        objective = FrozenDetectorSupervisedLoss(
            _ToyDetector(),
            input_size=(640, 640),
            letterbox=True,
            cqmix_prob=0.0,
        )
        image = torch.rand(1, 3, 360, 640)
        targets = {
            "cls": torch.tensor([0.0]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.5]]),
            "batch_idx": torch.tensor([0.0]),
        }
        prepared, adjusted = objective._prepare_image_and_targets(image, targets)

        self.assertEqual(tuple(prepared.shape), (1, 3, 640, 640))
        torch.testing.assert_close(
            adjusted["bboxes"],
            torch.tensor([[0.5, 0.5, 0.25, 0.28125]]),
        )


if __name__ == "__main__":
    unittest.main()
