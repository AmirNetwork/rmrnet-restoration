#!/usr/bin/env python3
"""Freeze the validation-selected CRID native policy before test access.

Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    selection_path = args.selection.resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("test_images_or_labels_read") is not False:
        raise RuntimeError("Selection report does not certify an unopened test split")
    selected = selection.get("selected", {})
    spec = selected.get("spec", {})
    if spec.get("kind") != "unsharp" or float(spec.get("alpha", 1.0)) >= 0.0:
        raise RuntimeError("Expected a validation-selected low-pass detail residual")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {
        key: value
        for key, value in payload["model"].items()
        if not key.startswith("refiner.")
    }
    arch = dict(payload["arch"])
    arch.update(
        {
            "use_refiner": False,
            "refiner_mode": "spatial",
            "native_output_filter": {
                "kind": "gaussian_detail",
                "sigma_native_px": float(spec["sigma"]),
                "detail_gain": float(spec["alpha"]),
                "metadata_threshold": 0.5,
                "camera_reliability_threshold": 0.5,
                "imu_reliability_threshold": 0.5,
                "automatic": True,
            },
            "native_output_filter_selection_sha256": sha256(selection_path),
        }
    )
    frozen = {
        "model": state,
        "arch": arch,
        "epoch": payload.get("epoch", 0),
        "field_policy_freeze": {
            "status": "FROZEN_BEFORE_SEALED_TEST",
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": sha256(checkpoint),
            "validation_selection": str(selection_path),
            "validation_selection_sha256": sha256(selection_path),
            "selected_validation_map50": float(selected["map50"]),
            "selected_validation_map50_95": float(selected["map50_95"]),
            "test_images_or_labels_read": False,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(frozen, args.out)
    print(
        json.dumps(
            {"checkpoint": str(args.out.resolve()), "sha256": sha256(args.out.resolve())},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
