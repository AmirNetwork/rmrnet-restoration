# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from .model import RCADNet
from .scenario_codes import code_from_scenario, code_from_metadata
from .task_losses import (
    ActiveContourGeometryLoss,
    CompositeTaskLoss,
    DetectorInputAnchorLoss,
    FrozenDetectorFeatureExtractor,
    TaskLossWeights,
    TaskDrivenPerceptualLoss,
    cross_quality_patch_mix,
    hutchinson_jacobian_penalty,
    road_evidence_nonregression_loss,
    road_evidence_vector,
)

__all__ = [
    "RCADNet",
    "code_from_scenario",
    "code_from_metadata",
    "ActiveContourGeometryLoss",
    "CompositeTaskLoss",
    "DetectorInputAnchorLoss",
    "FrozenDetectorFeatureExtractor",
    "TaskLossWeights",
    "TaskDrivenPerceptualLoss",
    "cross_quality_patch_mix",
    "hutchinson_jacobian_penalty",
    "road_evidence_nonregression_loss",
    "road_evidence_vector",
]
