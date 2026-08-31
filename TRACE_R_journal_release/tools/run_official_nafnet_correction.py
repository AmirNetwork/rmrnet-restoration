#!/usr/bin/env python3
"""Correct the controlled benchmark with the official width-32 NAFNet.

The earlier controlled ledger labelled a compact NAF-style diagnostic as
NAFNet. This runner performs a provenance-preserving correction:

1. read the validation-only NAFNet checkpoint selection;
2. freeze checkpoint, upstream source, detector, and test-split hashes;
3. evaluate that checkpoint once on the existing controlled test protocol;
4. merge only the corrected NAFNet rows into the immutable controlled ledger.

No TRACE-R or comparator result is recomputed, and no test metric is used for
checkpoint or residual-policy selection.

Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_tracer_locked_confirmatory as locked


BASE = Path(r"E:\TRACE_R_experiments\trace_locked_confirmatory_v65_20260828")
SELECTION = Path(
    r"E:\TRACE_R_experiments\official_nafnet_matched_v68_20260828\validation"
    r"\best_by_val_map.json"
)
OUT = Path(r"E:\TRACE_R_experiments\trace_locked_confirmatory_v72_20260828")
UPSTREAM_SOURCE = ROOT / "third_party" / "NAFNet-main" / "basicsr" / "models" / "archs" / "NAFNet_arch.py"
UPSTREAM_WEIGHT = ROOT / "weights" / "nafnet" / "NAFNet-GoPro-width32.pth"
NAFNET_WRAPPER = ROOT / "baselines" / "nafnet_road.py"
RESTORE_RUNNER = ROOT / "tools" / "restore_yolo_split.py"
STABILITY_AUDIT = (
    Path(r"E:\TRACE_R_experiments\official_nafnet_matched_v68_20260828")
    / "validation"
    / "stability_audit_v1"
    / "checkpoint_stability.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def freeze_record(args: argparse.Namespace) -> dict[str, Any]:
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("test_split_used") is not False:
        raise RuntimeError("NAFNet selection ledger is not validation-only")
    selected = selection.get("best_by_val_map50", {}).get("nafnet")
    if not isinstance(selected, dict):
        raise KeyError("best_by_val_map50.nafnet is missing")
    checkpoint = Path(selected["checkpoint"]).resolve()
    for required in (
        checkpoint,
        UPSTREAM_SOURCE,
        UPSTREAM_WEIGHT,
        NAFNET_WRAPPER,
        RESTORE_RUNNER,
        STABILITY_AUDIT,
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    stability = json.loads(STABILITY_AUDIT.read_text(encoding="utf-8"))
    checkpoint_stability = next(
        (
            row
            for row in stability.get("checkpoints", [])
            if row.get("checkpoint") == checkpoint.name
        ),
        None,
    )
    if not checkpoint_stability or not checkpoint_stability.get("stability_valid"):
        raise RuntimeError(
            "Validation-selected NAFNet checkpoint failed the validation-only "
            f"output-stability gate: {checkpoint}"
        )

    record: dict[str, Any] = {
        "status": "official_nafnet_selection_frozen_before_test",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_split": "validation only",
        "test_used_for_selection": False,
        "selection_criterion": selection.get("criterion"),
        "selection_ledger": str(args.selection.resolve()),
        "selection_ledger_sha256": locked.sha256(args.selection),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": locked.sha256(checkpoint),
        "epoch": int(selected["epoch"]),
        "validation_joint_map50": float(selected["joint_mean_map50"]),
        "official_architecture": "NAFNet width 32; enc_blk_nums [1,1,1,28]; middle_blk_num 1; dec_blk_nums [1,1,1,1]",
        "official_source": str(UPSTREAM_SOURCE.resolve()),
        "official_source_sha256": locked.sha256(UPSTREAM_SOURCE),
        "nafnet_wrapper": str(NAFNET_WRAPPER.resolve()),
        "nafnet_wrapper_sha256": locked.sha256(NAFNET_WRAPPER),
        "restore_runner": str(RESTORE_RUNNER.resolve()),
        "restore_runner_sha256": locked.sha256(RESTORE_RUNNER),
        "released_initialization": str(UPSTREAM_WEIGHT.resolve()),
        "released_initialization_sha256": locked.sha256(UPSTREAM_WEIGHT),
        "validation_stability_audit": str(STABILITY_AUDIT.resolve()),
        "validation_stability_audit_sha256": locked.sha256(STABILITY_AUDIT),
        "validation_stability_gate": checkpoint_stability,
        "single_restored_image": True,
        "detector_output_fusion": False,
        "inference_information": "image only",
        "detectors": {
            dataset: {
                "path": str((ROOT / info["detector"]).resolve()),
                "sha256": locked.sha256(ROOT / info["detector"]),
            }
            for dataset, info in locked.DATASETS.items()
        },
        "test_source_yaml_sha256": {
            f"{dataset}:{scenario}": locked.sha256(locked.source_yaml(dataset, scenario))
            for dataset in locked.DATASETS
            for scenario in locked.SCENARIOS
        },
    }
    return record


def freeze_once(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        immutable = (
            "selection_ledger_sha256",
            "checkpoint_sha256",
            "epoch",
            "official_source_sha256",
            "nafnet_wrapper_sha256",
            "restore_runner_sha256",
            "released_initialization_sha256",
            "validation_stability_audit_sha256",
            "detectors",
            "test_source_yaml_sha256",
        )
        for key in immutable:
            if previous.get(key) != record.get(key):
                raise RuntimeError(f"Frozen NAFNet correction changed: {key}")
        return previous
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def evaluate(record: dict[str, Any], out: Path, device: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    policy = {
        "checkpoint": record["checkpoint"],
        "model": "nafnet",
        "epoch": record["epoch"],
    }
    restored: dict[str, dict[str, Path]] = {
        dataset: {} for dataset in locked.DATASETS
    }
    for dataset in locked.DATASETS:
        for scenario in locked.SCENARIOS:
            source = locked.source_yaml(dataset, scenario)
            output = out / "restored" / "nafnet" / dataset / scenario
            if not locked.restoration_complete(source, output):
                if output.exists():
                    shutil.rmtree(output)
                locked.run(
                    locked.restore_command(
                        "nafnet", policy, source, scenario, output, device
                    ),
                    out / "logs" / "restore_official_nafnet.log",
                )
            restored[dataset][scenario] = output / "data.yaml"
    detection = locked.evaluate_detector("nafnet", restored, out)
    fidelity = locked.paired_fidelity("nafnet", restored, out)
    return detection, fidelity


def merge(
    base: Path,
    out: Path,
    frozen: dict[str, Any],
    detection: list[dict[str, Any]],
    fidelity: list[dict[str, Any]],
) -> dict[str, Any]:
    old_detection = locked.read_csv(base / "detection" / "all_condition_metrics.csv")
    merged_detection: list[dict[str, Any]] = [
        row for row in old_detection if row["method"] not in {"nafnet", "nafnet_meta"}
    ]
    merged_detection.extend(detection)
    locked.write_csv(out / "detection" / "all_condition_metrics.csv", merged_detection)
    locked.write_csv(
        out / "detection" / "aggregate_metrics.csv",
        locked.aggregate_detection(merged_detection),
    )
    clean_source = base / "detection" / "clean_ceiling.csv"
    clean_target = out / "detection" / "clean_ceiling.csv"
    clean_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(clean_source, clean_target)

    old_fidelity = locked.read_csv(base / "fidelity" / "all_summary.csv")
    merged_fidelity: list[dict[str, Any]] = [
        row for row in old_fidelity if row["method"] not in {"nafnet", "nafnet_meta"}
    ]
    merged_fidelity.extend(fidelity)
    locked.write_csv(out / "fidelity" / "all_summary.csv", merged_fidelity)

    ledger = json.loads((base / "final_provenance_ledger.json").read_text(encoding="utf-8"))
    ledger["status"] = "confirmatory_test_complete_with_official_nafnet_correction"
    ledger["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    ledger["policies"].pop("nafnet_meta", None)
    payload_budget: dict[str, dict[str, Any]] = {}
    for name in ("instructir", "dfpir", "demoe_auto", "demoe_oracle"):
        policy = ledger["policies"][name]
        checkpoint = Path(policy["checkpoint"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        args = payload.get("args", {})
        if hasattr(args, "__dict__"):
            args = vars(args)
        metrics = payload.get("metrics", {})
        declared_epoch = int(payload.get("epoch", metrics.get("epoch", -1)))
        updates = int(metrics.get("optimizer_step", -1))
        if declared_epoch != 32 or updates != 4096:
            raise RuntimeError(
                f"Unexpected matched budget in {checkpoint}: "
                f"epoch={declared_epoch}, updates={updates}"
            )
        payload_budget[name] = {
            "checkpoint_filename": checkpoint.name,
            "checkpoint_payload_epoch": declared_epoch,
            "optimizer_updates": updates,
            "samples_per_epoch": int(args.get("samples_per_epoch", -1)),
            "effective_batch_size": int(args.get("effective_batch_size", -1)),
        }
        policy["checkpoint_index_label"] = policy["epoch"]
        policy["epoch"] = declared_epoch
        policy["optimizer_updates"] = updates
    ledger["policies"]["nafnet"] = {
        "model": "nafnet",
        "architecture": frozen["official_architecture"],
        "epoch": frozen["epoch"],
        "checkpoint": frozen["checkpoint"],
        "validation_joint_map50": frozen["validation_joint_map50"],
        "inference_information": "image only",
        "checkpoint_sha256": frozen["checkpoint_sha256"],
        "scenario_labels_at_inference": False,
        "detector_output_fusion": False,
        "single_restored_image": True,
    }
    ledger["policies"]["instructir"]["inference_information"] = (
        "scenario text instruction required by the published interface"
    )
    ledger["policies"]["dfpir"]["inference_information"] = (
        "scenario CLIP prompt required by the published interface"
    )
    ledger["policies"]["dfpir"]["scenario_labels_at_inference"] = True
    ledger["official_nafnet_correction"] = {
        **frozen,
        "reason": (
            "The inherited ledger used a compact NAF-style diagnostic. The "
            "reported NAFNet row now uses the authors' released width-32 architecture."
        ),
        "inherited_controlled_run": str(base.resolve()),
        "inherited_controlled_ledger_sha256": locked.sha256(
            base / "final_provenance_ledger.json"
        ),
        "only_corrected_method": "nafnet",
        "test_metric_used_for_selection": False,
    }
    ledger["matched_budget_payload_audit"] = payload_budget
    ledger["artifacts"] = {
        "detection_conditions": str(out / "detection" / "all_condition_metrics.csv"),
        "detection_aggregate": str(out / "detection" / "aggregate_metrics.csv"),
        "clean_detection_ceiling": str(clean_target),
        "fidelity_summary": str(out / "fidelity" / "all_summary.csv"),
    }
    target = out / "final_provenance_ledger.json"
    target.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    return ledger


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    record = freeze_once(
        args.out / "frozen_official_nafnet_before_test.json",
        freeze_record(args),
    )
    detection, fidelity = evaluate(record, args.out, args.device)
    ledger = merge(args.base, args.out, record, detection, fidelity)
    print(
        json.dumps(
            {
                "status": ledger["status"],
                "official_nafnet": ledger["policies"]["nafnet"],
                "aggregate": locked.read_csv(
                    args.out / "detection" / "aggregate_metrics.csv"
                ),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
