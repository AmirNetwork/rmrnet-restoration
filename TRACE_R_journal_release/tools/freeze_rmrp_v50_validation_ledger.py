#!/usr/bin/env python3
# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
"""Freeze the executed RMR-P expert-policy validation into one ledger.

This script performs no training, inference, model selection, or metric
calculation. It verifies and copies the completed validation artifacts so the
paper builders consume one immutable source of truth.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXECUTED = Path(r"E:\RMRP_experiments\rmrp_expert_fusion_v50_validation_20260822")
BASELINES = ROOT / "experiments/matched_baselines_v28_epoch70_validation_20260821"
OUT = ROOT / "experiments/final_rmrp_v50_validation_ledger_20260824"
METHODS = ("demoe", "dfpir", "instructir", "nafnet", "nafnet_meta")
DATASETS = ("ivcnz", "pcm")
CAUSES = ("motion", "defocus", "lowlight", "mixed")
CONTROLS = ("correct", "unavailable", "cross_condition_shuffled")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scenario_metrics(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_csv(path):
        dataset, cause = row["name"].split("_", 1)
        if dataset not in DATASETS or cause not in CAUSES:
            raise RuntimeError(f"Unexpected metric row {row['name']!r} in {path}")
        result[f"{dataset}_{cause}"] = {
            key: float(row[key])
            for key in ("map50", "map50_95", "precision", "recall")
        }
        result[f"{dataset}_{cause}"]["class_names"] = json.loads(row["class_names"])
        result[f"{dataset}_{cause}"]["class_map50"] = json.loads(row["class_map50"])
    expected = {f"{dataset}_{cause}" for dataset in DATASETS for cause in CAUSES}
    if set(result) != expected:
        raise RuntimeError(f"Incomplete metric set in {path}: {sorted(result)}")
    return result


def mean_for(metrics: dict[str, dict[str, Any]], dataset: str) -> float:
    return sum(float(metrics[f"{dataset}_{cause}"]["map50"]) for cause in CAUSES) / len(CAUSES)


def copy_evidence() -> dict[str, str]:
    evidence = OUT / "executed_evidence"
    hashes: dict[str, str] = {}
    for control in CONTROLS:
        for name in ("summary.json", "metrics_ivcnz.csv", "metrics_pcm.csv"):
            source = EXECUTED / control / name
            if not source.is_file():
                raise FileNotFoundError(source)
            target = evidence / control / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[target.relative_to(ROOT).as_posix()] = sha256(target)
    provenance = EXECUTED / "validation_provenance.json"
    target = evidence / "validation_provenance.json"
    shutil.copy2(provenance, target)
    hashes[target.relative_to(ROOT).as_posix()] = sha256(target)
    return hashes


def main() -> None:
    executed_ledger = read_json(EXECUTED / "validation_provenance.json")
    if executed_ledger.get("test_split_used") is not False:
        raise RuntimeError("Executed RMR-P ledger is not validation-only")
    if set(CONTROLS) - set(executed_ledger.get("results", {})):
        raise RuntimeError("Executed ledger does not contain every required metadata control")

    selected = read_json(BASELINES / "best_by_val_map.json")
    if selected.get("test_split_used") is not False:
        raise RuntimeError("Baseline selection used a test split")

    rmrp_metrics: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        rows = read_csv(EXECUTED / "correct" / f"metrics_{dataset}.csv")
        for row in rows:
            name = row["name"]
            rmrp_metrics[name] = {
                key: float(row[key])
                for key in ("map50", "map50_95", "precision", "recall")
            }
            rmrp_metrics[name]["class_names"] = json.loads(row["class_names"])
            rmrp_metrics[name]["class_map50"] = json.loads(row["class_map50"])
    expected = {f"{dataset}_{cause}" for dataset in DATASETS for cause in CAUSES}
    if set(rmrp_metrics) != expected:
        raise RuntimeError("RMR-P metric rows are incomplete")

    baseline_payload: dict[str, Any] = {}
    for method in METHODS:
        record = selected["best_by_val_map50"][method]
        if record.get("test_split_used") is not False:
            raise RuntimeError(f"Baseline {method} is not validation-only")
        epoch = int(record["epoch"])
        checkpoint = Path(record["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        metrics = scenario_metrics(
            BASELINES / "full_validation" / method / f"epoch_{epoch:03d}" / "metrics.csv"
        )
        observed = {dataset: mean_for(metrics, dataset) for dataset in DATASETS}
        for dataset in DATASETS:
            if abs(observed[dataset] - float(record[f"{dataset}_mean_map50"])) > 1e-12:
                raise RuntimeError(f"Baseline aggregate mismatch for {method}/{dataset}")
        baseline_payload[method] = {
            "epoch": epoch,
            "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": sha256(checkpoint),
            "mean_map50": observed,
            "joint_mean_map50": sum(observed.values()) / len(observed),
            "conditions": metrics,
        }

    rmrp_means = {dataset: mean_for(rmrp_metrics, dataset) for dataset in DATASETS}
    strongest_name, strongest = max(
        baseline_payload.items(), key=lambda item: item[1]["joint_mean_map50"]
    )
    controls = {
        name: read_json(EXECUTED / name / "summary.json")
        for name in CONTROLS
    }
    condition_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for cause in CAUSES:
            key = f"{dataset}_{cause}"
            demoe = baseline_payload["demoe"]["conditions"][key]["map50"]
            condition_rows.append(
                {
                    "dataset": dataset,
                    "condition": cause,
                    "rmrp_map50": rmrp_metrics[key]["map50"],
                    "demoe_map50": demoe,
                    "delta": rmrp_metrics[key]["map50"] - demoe,
                }
            )

    code_paths = (
        ROOT / "models/rmrp_expert_fusion.py",
        ROOT / "tools/restore_yolo_split.py",
        ROOT / "tools/validate_rmrp_expert_fusion.py",
        ROOT / "tools/evaluate_rmrp_v50_validation_fidelity.py",
        Path(__file__),
    )
    fidelity_rows = read_csv(OUT / "fidelity_summary.csv")
    fidelity: dict[str, dict[str, dict[str, float]]] = {}
    for row in fidelity_rows:
        fidelity.setdefault(row["model"], {}).setdefault(row["dataset"], {})[
            row["condition"]
        ] = {
            "psnr": float(row["psnr"]),
            "ssim": float(row["ssim"]),
        }
    for model, datasets in fidelity.items():
        for dataset, conditions in datasets.items():
            if set(conditions) != set(CAUSES):
                raise RuntimeError(f"Incomplete fidelity rows for {model}/{dataset}")
            conditions["mean"] = {
                "psnr": sum(conditions[cause]["psnr"] for cause in CAUSES) / len(CAUSES),
                "ssim": sum(conditions[cause]["ssim"] for cause in CAUSES) / len(CAUSES),
            }
    payload = {
        "status": "FROZEN_VALIDATION_ONLY",
        "method": "RMR-P",
        "method_expansion": "Road Metadata-aware Restoration for Pavement Inspection",
        "created_from_executed_outputs_only": True,
        "test_split_used": False,
        "evidence_boundary": (
            "No newly untouched labelled IVCNZ/PCM holdout remains. These are "
            "sequence/source-disjoint validation results; previously opened test "
            "and purge-holdout outputs are excluded from the new claim."
        ),
        "selection": {
            "metric": "unweighted joint mean validation mAP50 over IVCNZ and PCM",
            "split": "sequence/source-disjoint validation",
            "scenario_family_is_model_input": False,
            "policy": executed_ledger["policy"],
        },
        "rmrp": {
            "mean_map50": rmrp_means,
            "joint_mean_map50": sum(rmrp_means.values()) / len(rmrp_means),
            "conditions": rmrp_metrics,
        },
        "matched_baselines": baseline_payload,
        "strongest_matched_baseline": {
            "model": strongest_name,
            "ivcnz_delta": rmrp_means["ivcnz"] - strongest["mean_map50"]["ivcnz"],
            "pcm_delta": rmrp_means["pcm"] - strongest["mean_map50"]["pcm"],
            "joint_delta": (
                sum(rmrp_means.values()) / len(rmrp_means)
                - strongest["joint_mean_map50"]
            ),
        },
        "metadata_controls": controls,
        "fidelity": fidelity,
        "condition_wins_over_demoe": sum(row["delta"] > 0 for row in condition_rows),
        "condition_count": len(condition_rows),
        "code_sha256": {
            path.relative_to(ROOT).as_posix(): sha256(path) for path in code_paths
        },
        "executed_source": str(EXECUTED),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    payload["copied_evidence_sha256"] = copy_evidence()
    for name in ("fidelity_summary.csv", "fidelity_per_image.csv"):
        path = OUT / name
        payload["copied_evidence_sha256"][path.relative_to(ROOT).as_posix()] = sha256(path)
    (OUT / "provenance_ledger.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    write_csv(OUT / "per_condition.csv", condition_rows)
    write_csv(
        OUT / "matched_validation_table.csv",
        [
            {
                "method": "RMR-P",
                "ivcnz_mean_map50": rmrp_means["ivcnz"],
                "pcm_mean_map50": rmrp_means["pcm"],
                "joint_mean_map50": sum(rmrp_means.values()) / len(rmrp_means),
            }
        ]
        + [
            {
                "method": method,
                "ivcnz_mean_map50": row["mean_map50"]["ivcnz"],
                "pcm_mean_map50": row["mean_map50"]["pcm"],
                "joint_mean_map50": row["joint_mean_map50"],
            }
            for method, row in baseline_payload.items()
        ],
    )
    write_csv(
        OUT / "metadata_controls.csv",
        [
            {
                "control": name,
                "ivcnz_mean_map50": row["mean_map50"]["ivcnz"],
                "pcm_mean_map50": row["mean_map50"]["pcm"],
                "joint_mean_map50": row["joint_mean_map50"],
            }
            for name, row in controls.items()
        ],
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
