"""Unit tests for TRACE-R's label-free detector-evidence guard."""

# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>

from models.tracer_detection_policy import (
    DetectorEvidencePolicy,
    apply_detector_evidence_policy,
)


def prediction(conf: float, box=(0.0, 0.0, 10.0, 10.0), primary="crack"):
    return {"conf": conf, "box": box, "primary": primary}


def test_native_is_kept_without_sufficient_evidence_gain():
    policy = DetectorEvidencePolicy(gate_margin=0.10)
    native = [prediction(0.50)]
    restored = [prediction(0.54)]
    output, decision = apply_detector_evidence_policy(native, restored, policy)
    assert decision["selected_view"] == "native"
    assert output == native


def test_stronger_restored_evidence_is_fused_without_duplicates():
    policy = DetectorEvidencePolicy(gate_margin=0.10, fusion_iou=0.35)
    native = [prediction(0.50)]
    restored = [prediction(0.80), prediction(0.60, box=(20.0, 20.0, 30.0, 30.0))]
    output, decision = apply_detector_evidence_policy(native, restored, policy)
    assert decision["selected_view"] == "native_plus_restored"
    assert [item["conf"] for item in output] == [0.80, 0.60]
