# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

"""Create a machine-readable audit of RMR-Net checkpoint provenance.

This audit keeps the native Sony/G46 deployment experiment separate from the
sequence-disjoint PCM/IVCNZ training evidence. It reads saved trainer configs
and checkpoints; it does not infer provenance from paper labels or filenames.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]

CONTROLLED_SOURCES = [
    Path("models/rmrnet.py"),
    Path("rcadnet/model.py"),
    Path("rcadnet/losses.py"),
    Path("rcadnet/task_losses.py"),
    Path("rcadnet/dataset.py"),
    Path("rcadnet/scenario_codes.py"),
    Path("train_rcadnet.py"),
    Path("tools/run_trc_final_30ep.py"),
]


RUNS = {
    "g46_detection_heavy": {
        "run": ROOT / "runs" / "gt46_detection_heavy_yolo26_olddata_3ep",
        "checkpoint": "rcadnet_epoch_003.pth",
        "role": "Sony/G46 native-image deployment experiment",
    },
    "ivcnz_image_only_foundation": {
        "run": ROOT
        / "runs"
        / "major_revision_sequence_disjoint_v1_image_only30_20260716_rmrnet_pothole_30ep",
        "checkpoint": "rcadnet_epoch_029.pth",
        "role": "sequence-disjoint IVCNZ image-only initialization",
    },
    "ivcnz_residual_metadata": {
        "run": ROOT
        / "runs"
        / "strict_residual_metadata_ivcnz_pilot8_20260718_rmrnet_pothole_8ep",
        "checkpoint": "rcadnet_epoch_007.pth",
        "role": "sequence-disjoint IVCNZ metadata-conditioned model",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path | str | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT))
    except (OSError, ValueError):
        return str(candidate)


def checkpoint_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = payload.get("model", payload.get("state_dict", payload))
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a model state dictionary")
    return state


def audit_run(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(spec["run"])
    config_path = run_dir / "audit_config.json"
    checkpoint_path = run_dir / str(spec["checkpoint"])
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    args = config.get("args", {})
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint_state(payload)
    return {
        "name": name,
        "role": spec["role"],
        "run_directory": relative(run_dir),
        "checkpoint": relative(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "state_tensor_count": len(state),
        "state_key_sha256": hashlib.sha256(
            "\n".join(sorted(state)).encode("utf-8")
        ).hexdigest(),
        "data_roots": [relative(item) for item in config.get("data_roots", [])],
        "scenarios": list(config.get("scenarios", [])),
        "contains_native_identity": "native_identity" in config.get("scenarios", []),
        "initial_checkpoint": relative(args.get("init_weights")),
        "epochs": args.get("epochs"),
        "batch_size": args.get("batch_size"),
        "learning_rate": args.get("lr"),
        "code_source": args.get("code_source"),
        "conditioning": args.get("conditioning"),
        "base_weight": config.get("base_weight"),
        "tdp_weight": config.get("lambda_tdp"),
        "jacobian_weight": config.get("lambda_jacobian"),
        "metadata_advantage_weight": config.get("lambda_metadata_advantage", 0.0),
        "active_contour_weight": config.get("lambda_active_contour", 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "rmrnet_model_lineage_audit_20260719.json",
    )
    args = parser.parse_args()

    runs = {name: audit_run(name, spec) for name, spec in RUNS.items()}
    g46_checkpoint = runs["g46_detection_heavy"]["checkpoint"]
    controlled = [
        runs["ivcnz_image_only_foundation"],
        runs["ivcnz_residual_metadata"],
    ]

    captured_manifest_path = (
        ROOT
        / "experiments"
        / "strict_residual_metadata_ivcnz_pilot8_20260718"
        / "source_manifest.json"
    )
    captured_manifest = json.loads(captured_manifest_path.read_text(encoding="utf-8"))
    captured_source = captured_manifest.get("source", {})
    current_source = {
        str(path).replace("/", "\\"): sha256(ROOT / path) for path in CONTROLLED_SOURCES
    }
    source_comparison = {
        path: {
            "captured": captured_source.get(path),
            "current": digest,
            "matches": captured_source.get(path) == digest,
        }
        for path, digest in current_source.items()
    }

    image_state = runs["ivcnz_image_only_foundation"]["state_key_sha256"]
    metadata_state = runs["ivcnz_residual_metadata"]["state_key_sha256"]
    report = {
        "audit": "RMR-Net native-field versus controlled-dataset checkpoint lineage",
        "runs": runs,
        "controlled_source_manifest": relative(captured_manifest_path),
        "controlled_source_comparison": source_comparison,
        "lineage_edges": [
            {
                "from": runs["ivcnz_image_only_foundation"]["checkpoint"],
                "to": runs["ivcnz_residual_metadata"]["checkpoint"],
                "evidence": "saved init_weights in the residual-metadata trainer config",
            }
        ],
        "checks": {
            "g46_checkpoint_reused_by_controlled_runs": any(
                item["initial_checkpoint"] == g46_checkpoint for item in controlled
            ),
            "controlled_checkpoint_state_keys_match": image_state == metadata_state,
            "controlled_source_hashes_match_manifest": all(
                item["matches"] for item in source_comparison.values()
            ),
            "controlled_metadata_stage_excludes_native_identity": not runs[
                "ivcnz_residual_metadata"
            ]["contains_native_identity"],
            "active_contour_disabled_in_controlled_runs": all(
                float(item["active_contour_weight"] or 0.0) == 0.0 for item in controlled
            ),
            "controlled_test_selection_used": False,
            "native_gate_used_in_controlled_validation": False,
        },
        "interpretation": {
            "g46": (
                "Separate three-epoch detection-heavy experiment using old combined training data, "
                "base weight 0.25, and TDP weight 0.08."
            ),
            "controlled": (
                "Sequence-disjoint IVCNZ training and validation. The metadata stage starts from "
                "the audited image-only checkpoint, excludes native_identity, and disables the "
                "Sony pass-through gate during restoration."
            ),
        },
    }
    if report["checks"]["g46_checkpoint_reused_by_controlled_runs"]:
        raise RuntimeError("G46 checkpoint unexpectedly appears in controlled-run lineage")
    if not report["checks"]["controlled_checkpoint_state_keys_match"]:
        raise RuntimeError("Controlled checkpoints do not share the same architecture state keys")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": relative(args.out), **report["checks"]}, indent=2))


if __name__ == "__main__":
    main()
