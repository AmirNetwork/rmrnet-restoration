from __future__ import annotations

from pathlib import Path

import pytest

from baselines.dfpir_adapter import DFPIRAdapter


def test_reported_dfpir_requires_official_clip_prompt() -> None:
    with pytest.raises(ValueError, match="CLIP degradation prompt"):
        DFPIRAdapter(
            Path("official-checkpoint-placeholder.pth"),
            device="cpu",
            smoke=False,
            use_clip=False,
        )

