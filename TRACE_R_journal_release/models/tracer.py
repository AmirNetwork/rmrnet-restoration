"""Public TRACE-R model API.

TRACE-R stands for Telemetry-Routed Adaptive Corruption-Expert Restoration.
The implementation is defined in :mod:`models.rmrp_expert_fusion` because the
frozen v50 experiments predate the journal-method rename. Keeping one
implementation prevents numerical drift between the archived evidence and the
journal release.
"""

from __future__ import annotations

# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>

from .rmrp_expert_fusion import TRACERExpertFusion, TRACERPolicy

__all__ = ["TRACERExpertFusion", "TRACERPolicy"]
