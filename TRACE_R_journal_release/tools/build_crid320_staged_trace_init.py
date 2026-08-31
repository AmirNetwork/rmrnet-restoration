#!/usr/bin/env python3
# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
"""Compose the CRID staged TRACE-R initialization from a matched DFPIR stage.

The DFPIR backbone state is copied exactly. TRACE-R's sensor path and FiLM
layers retain their identity initialization, while the native gate starts at
0.5, reproducing the validation-selected DFPIR residual strength before any
sensor-only updates are made.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-template", type=Path, required=True)
    parser.add_argument("--dfpir-stage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--enable-refiner",
        action="store_true",
        help="Retain the zero-output post-prior refiner from the TRACE template.",
    )
    parser.add_argument("--refiner-support-floor", type=float, default=0.0)
    parser.add_argument(
        "--refiner-gain",
        type=float,
        default=0.12,
        help="Maximum absolute post-prior correction before the native gate.",
    )
    parser.add_argument(
        "--refiner-mode",
        choices=("spatial", "detail"),
        default="spatial",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    template_path = args.trace_template.resolve()
    dfpir_path = args.dfpir_stage.resolve()
    for path in (template_path, dfpir_path):
        if not path.exists():
            raise FileNotFoundError(path)

    template = torch.load(template_path, map_location="cpu", weights_only=False)
    dfpir = torch.load(dfpir_path, map_location="cpu", weights_only=False)
    if template.get("arch", {}).get("backbone") != "dfpir_sensor_prompt":
        raise RuntimeError("TRACE template is not a prompted-DFPIR checkpoint")
    if dfpir.get("arch", {}).get("model") != "dfpir":
        raise RuntimeError("Stage checkpoint is not the matched DFPIR baseline")

    trace_state = dict(template["model"])
    copied = []
    skipped = []
    for key, value in dfpir["model"].items():
        target = f"backbone.{key}"
        if target not in trace_state or trace_state[target].shape != value.shape:
            skipped.append(key)
            continue
        trace_state[target] = value.detach().clone()
        copied.append(key)
    if skipped:
        raise RuntimeError(f"DFPIR state did not map exactly into TRACE-R: {skipped[:8]}")
    if len(copied) != len(dfpir["model"]):
        raise RuntimeError("Not every DFPIR tensor was copied")

    if not 0.0 <= args.refiner_support_floor <= 1.0:
        raise ValueError("--refiner-support-floor must be in [0, 1]")
    if not 0.0 < args.refiner_gain <= 0.5:
        raise ValueError("--refiner-gain must be in (0, 0.5]")
    # The refiner output projection is zero initialized, so retaining it still
    # reproduces the matched DFPIR output exactly before field adaptation.
    if not args.enable_refiner or args.refiner_mode == "detail":
        trace_state = {
            key: value
            for key, value in trace_state.items()
            if not key.startswith("refiner.")
        }
    arch = dict(template["arch"])
    arch.update(
        {
            "use_refiner": bool(args.enable_refiner),
            "refiner_support_floor": float(args.refiner_support_floor),
            "refiner_gain": float(args.refiner_gain),
            "refiner_mode": str(args.refiner_mode),
            "use_native_gate": True,
            "native_gate_init": 0.50,
            "prompt_router": "fixed_deblur",
            "field_initialization": (
                "matched CRID DFPIR stage-1 backbone plus identity sensor adapters"
            ),
            "staged_training": True,
            "stage1_epochs": int(dfpir.get("epoch", 8)),
            "stage1_checkpoint_sha256": sha256(dfpir_path),
            "stage1_selected_eta": 0.50,
        }
    )
    payload = {
        "model": trace_state,
        "arch": arch,
        "epoch": int(dfpir.get("epoch", 8)),
        "stage_provenance": {
            "trace_template": str(template_path),
            "trace_template_sha256": sha256(template_path),
            "dfpir_stage": str(dfpir_path),
            "dfpir_stage_sha256": sha256(dfpir_path),
            "copied_deterministic_backbone_tensors": len(copied),
            "test_images_or_labels_read": False,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(
        {
            "out": str(args.out.resolve()),
            "copied_tensors": len(copied),
            "sha256": sha256(args.out.resolve()),
        }
    )


if __name__ == "__main__":
    main()
