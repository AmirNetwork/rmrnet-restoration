from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from tools.evaluate_crid46_crossfit_suite import (
    filter_confidence,
    select_confidence,
    standard_metrics,
)
from tools import run_crid46_sequence_disjoint_comparison as legacy_eval


def item(image: str, label: str, confidence: float, box: tuple[float, float, float, float]) -> dict:
    return {
        "image": image,
        "label": label,
        "primary": "pothole" if label == "Potholes" else "crack",
        "conf": confidence,
        "box": box,
    }


def test_standard_metrics_perfect_four_class_predictions() -> None:
    labels = [
        "Longitudinal Crack",
        "Transverse Crack",
        "Alligator Crack",
        "Potholes",
    ]
    gt = {
        f"image_{index}.jpg": [item(f"image_{index}.jpg", label, 1.0, (0, 0, 10, 10))]
        for index, label in enumerate(labels)
    }
    predictions = {
        image: [dict(target[0], conf=0.9)] for image, target in gt.items()
    }
    row, per_class = standard_metrics("perfect", gt, predictions, predictions)
    assert row["map50"] == pytest.approx(1.0)
    assert row["map50_95"] == pytest.approx(1.0)
    assert row["precision"] == pytest.approx(1.0)
    assert row["recall"] == pytest.approx(1.0)
    assert row["f1"] == pytest.approx(1.0)
    assert len(per_class) == 4
    assert all(result["ap50"] == pytest.approx(1.0) for result in per_class)


def test_validation_threshold_rejects_low_confidence_false_positive() -> None:
    image = "image.jpg"
    gt = {image: [item(image, "Potholes", 1.0, (0, 0, 10, 10))]}
    predictions = {
        image: [
            item(image, "Potholes", 0.9, (0, 0, 10, 10)),
            item(image, "Potholes", 0.04, (30, 30, 40, 40)),
        ]
    }
    threshold, f1 = select_confidence(gt, predictions)
    assert threshold >= 0.05
    assert f1 == pytest.approx(1.0)
    assert len(filter_confidence(predictions, threshold)[image]) == 1


def test_subset_gt_supports_standard_yolo_test_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_dir = tmp_path / "images" / "test"
    label_dir = tmp_path / "labels" / "test"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    name = "frame.jpg"
    Image.new("RGB", (100, 80), "gray").save(image_dir / name)
    (label_dir / "frame.txt").write_text("3 0.5 0.5 0.2 0.25\n", encoding="utf-8")
    monkeypatch.setattr(legacy_eval, "SPLIT_LABEL_ROOT", tmp_path)

    result = legacy_eval.subset_gt([name])

    assert result[name][0]["label"] == "Potholes"
    assert result[name][0]["box"] == pytest.approx((40.0, 30.0, 60.0, 50.0))
