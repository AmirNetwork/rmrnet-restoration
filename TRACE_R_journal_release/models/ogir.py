# TRACE-R release integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
"""Paper-facing entry point for Observability-Gated Inertial Restoration.

OGIR is the inertial-physics configuration of the existing restoration core.
The subclass keeps state-dict keys unchanged, so audited checkpoints remain
loadable while the new paper can use a distinct, scientifically descriptive
method name.

The implemented forward model is

    I_b(p) = (1 / M) * sum_j I_s(W(p; R_j, K)),

where R_j is obtained by integrating exposure-synchronised angular rate on
SO(3).  See ``rcadnet/spatial_physics.py`` for the operator and observability
terms, and ``rcadnet/model.py`` for bounded physics-to-neural fusion.
"""

from __future__ import annotations

from .rmrnet import RMRNet


class OGIR(RMRNet):
    """Observability-Gated Inertial Restoration.

    This is a naming and configuration boundary, not a duplicate network.
    Callers should enable ``use_spatial_physics`` and pass the synchronized
    practical sensor packet through the inherited ``forward`` method.
    """


__all__ = ["OGIR"]
