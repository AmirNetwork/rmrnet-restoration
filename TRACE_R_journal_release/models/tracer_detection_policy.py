"""Label-free detector-evidence guard used by TRACE-R in field deployment.

The policy compares detector evidence from the native and restored views.  It
keeps the native view unless restoration produces a sufficient evidence gain;
accepted restored detections are fused with native detections by confidence-
ordered, primary-family NMS.  Constants are selected on validation data only.
"""

from __future__ import annotations

# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>

from dataclasses import dataclass
from typing import Any


Prediction = dict[str, Any]


@dataclass(frozen=True)
class DetectorEvidencePolicy:
    """Validation-frozen TRACE-R field policy.

    Manuscript equations:

        a = 1[q(P_r) > (1 + tau) q(P_d)]
        P_o = P_d                         if a = 0
              NMS(P_d union P_r; kappa)   if a = 1
    """

    score_kind: str = "top10"
    confidence_floor: float = 0.05
    gate_margin: float = 0.10
    restored_confidence_floor: float = 0.075
    fusion_iou: float = 0.35

    def __post_init__(self) -> None:
        if self.score_kind not in {"top3", "top10", "sumsq"}:
            raise ValueError(f"Unknown evidence statistic: {self.score_kind}")
        for name in (
            "confidence_floor",
            "gate_margin",
            "restored_confidence_floor",
            "fusion_iou",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], received {value}")


def evidence_score(
    predictions: list[Prediction],
    kind: str,
    confidence_floor: float,
) -> float:
    """Return the compact confidence statistic q(P) used by the gate."""

    values = sorted(
        (
            float(item["conf"])
            for item in predictions
            if float(item["conf"]) >= confidence_floor
        ),
        reverse=True,
    )
    if kind == "top3":
        return sum(values[:3])
    if kind == "top10":
        return sum(values[:10])
    if kind == "sumsq":
        return sum(value * value for value in values)
    raise ValueError(f"Unknown evidence statistic: {kind}")


def box_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def fuse_predictions(
    native: list[Prediction],
    restored: list[Prediction],
    policy: DetectorEvidencePolicy,
) -> list[Prediction]:
    """Fuse detector views with confidence-ordered, primary-family NMS."""

    candidates = [dict(item) for item in native]
    candidates.extend(
        dict(item)
        for item in restored
        if float(item["conf"]) >= policy.restored_confidence_floor
    )
    candidates.sort(key=lambda item: float(item["conf"]), reverse=True)
    kept: list[Prediction] = []
    for candidate in candidates:
        duplicate = any(
            candidate["primary"] == accepted["primary"]
            and box_iou(candidate["box"], accepted["box"]) >= policy.fusion_iou
            for accepted in kept
        )
        if not duplicate:
            kept.append(candidate)
    return kept


def apply_detector_evidence_policy(
    native: list[Prediction],
    restored: list[Prediction],
    policy: DetectorEvidencePolicy | None = None,
) -> tuple[list[Prediction], dict[str, float | str]]:
    """Apply one TRACE-R native/restored field decision without annotations."""

    policy = policy or DetectorEvidencePolicy()
    native_score = evidence_score(
        native, policy.score_kind, policy.confidence_floor
    )
    restored_score = evidence_score(
        restored, policy.score_kind, policy.confidence_floor
    )
    use_restored = restored_score > (1.0 + policy.gate_margin) * native_score
    output = (
        fuse_predictions(native, restored, policy)
        if use_restored
        else [dict(item) for item in native]
    )
    decision: dict[str, float | str] = {
        "selected_view": "native_plus_restored" if use_restored else "native",
        "native_evidence": native_score,
        "restored_evidence": restored_score,
    }
    return output, decision


__all__ = [
    "DetectorEvidencePolicy",
    "apply_detector_evidence_policy",
    "box_iou",
    "evidence_score",
    "fuse_predictions",
]
