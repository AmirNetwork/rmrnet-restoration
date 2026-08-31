#!/usr/bin/env python3
"""Export the frozen CRID-320 TRACE-R policy identity without model tensors.

Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path(
    r"E:\TRACE_R_experiments\crid320_trace_field_policy_v8_20260831"
    r"\trace_crid_field_policy_frozen.pth"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "crid320_field_policy_identity_20260831.json",
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    output = {
        "status": "FROZEN_CRID320_TRACE_R_FIELD_POLICY_IDENTITY",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "epoch": payload.get("epoch"),
        "arch": payload.get("arch"),
        "field_policy_freeze": payload.get("field_policy_freeze"),
        "model_tensors_redistributed": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
