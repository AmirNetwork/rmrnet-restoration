#!/usr/bin/env python3
"""Evaluate TRACE-R packet interventions on validation data only.

The selected checkpoint is fixed before this diagnostic. Aligned, unavailable,
and wrong-condition packets are applied to identical validation images. The
test split is never opened and no detector predictions are combined.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CAUSES = ("motion", "defocus", "lowlight", "mixed")
SCENARIOS = {
    "motion": "motion_horizontal_medium",
    "defocus": "defocus_medium",
    "lowlight": "lowlight_medium",
    "mixed": "mixed_motion_lowlight",
}
WRONG_CAUSE = {"motion": "defocus", "defocus": "lowlight", "lowlight": "mixed", "mixed": "motion"}
DATASETS = {
    "ivcnz": {
        "prefix": "pothole_yolo_sequence_disjoint_practical_sensor_calibrated_v2",
        "detector": "runs/detect/runs/yolo11s_sequence_disjoint_v1_20260716/pothole_clean_80ep/weights/best.pt",
    },
    "pcm": {
        "prefix": "road_damage_pcm_yolo_sequence_disjoint_practical_sensor_calibrated_v2",
        "detector": "runs/detect/runs/yolo11s_sequence_disjoint_v1_20260716/pcm_clean_80ep/weights/best.pt",
    },
}
TRACE_ROOT = Path(r"E:\TRACE_R_experiments\matched_budget_trace_v53_20260827")
TRACE_SELECTION = TRACE_ROOT / "validation" / "best_by_val_map.json"
TRACE_CORRECT_METRICS = (
    TRACE_ROOT / "validation" / "full_validation" / "rmrp" / "epoch_008" / "metrics.csv"
)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(r"E:\TRACE_R_experiments\trace_metadata_controls_v66_20260828"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--control",
        action="append",
        choices=("unavailable", "wrong_condition"),
        help="Defaults to both controls; aligned metrics are always imported from the frozen run.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def source_yaml(dataset: str, cause: str) -> Path:
    return ROOT / "datasets" / f"{DATASETS[dataset]['prefix']}_{cause}_val" / "data.yaml"


def yaml_directory(data_yaml: Path, folder: str, split: str = "val") -> Path:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(payload["path"])
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    relative = str(payload[split]).replace("images", folder, 1)
    path = Path(relative)
    return path if path.is_absolute() else root / path


def image_count(data_yaml: Path) -> int:
    return sum(
        path.suffix.lower() in IMAGE_SUFFIXES
        for path in yaml_directory(data_yaml, "images").rglob("*")
        if path.is_file()
    )


def complete(source: Path, output: Path) -> bool:
    data_yaml = output / "data.yaml"
    return data_yaml.exists() and image_count(source) == image_count(data_yaml) > 0


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"command": command, "log": str(log)}), flush=True)
    with log.open("a", encoding="utf-8", errors="replace") as handle:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode:
        tail = log.read_text(encoding="utf-8", errors="replace")[-6000:]
        raise RuntimeError(f"Command failed ({result.returncode})\n{tail}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(control: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    values: dict[str, list[float]] = {dataset: [] for dataset in DATASETS}
    conditions: dict[str, float] = {}
    for row in rows:
        dataset, cause = row["name"].split("_", 1)
        score = float(row["map50"])
        values[dataset].append(score)
        conditions[row["name"]] = score
    means = {dataset: sum(scores) / len(scores) for dataset, scores in values.items()}
    return {
        "control": control,
        "ivcnz_mean_map50": means["ivcnz"],
        "pcm_mean_map50": means["pcm"],
        "joint_mean_map50": (means["ivcnz"] + means["pcm"]) / 2.0,
        "conditions": conditions,
    }


def evaluate_control(
    control: str,
    checkpoint: Path,
    out: Path,
    device: str,
) -> list[dict[str, str]]:
    entries: dict[str, list[tuple[str, Path]]] = {dataset: [] for dataset in DATASETS}
    for dataset in DATASETS:
        for cause in CAUSES:
            source = source_yaml(dataset, cause)
            restored = out / control / "restored" / dataset / cause
            if not complete(source, restored):
                if restored.exists():
                    shutil.rmtree(restored)
                command = [
                    sys.executable,
                    "tools/restore_yolo_split.py",
                    "--data",
                    str(source),
                    "--split",
                    "val",
                    "--model",
                    "rcadnet",
                    "--scenario",
                    SCENARIOS[cause],
                    "--out",
                    str(restored),
                    "--device",
                    device,
                    "--rcadnet-weights",
                    str(checkpoint),
                    "--rcadnet-code-source",
                    "metadata",
                    "--require-metadata",
                    "--gate-threshold",
                    "-1",
                    "--rcadnet-output-stage",
                    "restored",
                    "--residual-strength",
                    "1.0",
                    "--rcadnet-metadata-control",
                    "unavailable" if control == "unavailable" else "correct",
                ]
                if control == "wrong_condition":
                    wrong_metadata = yaml_directory(
                        source_yaml(dataset, WRONG_CAUSE[cause]), "metadata"
                    )
                    command += ["--metadata-dir-override", str(wrong_metadata)]
                run(command, out / control / "restore.log")
            entries[dataset].append((cause, restored / "data.yaml"))

    rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        metrics = out / control / f"metrics_{dataset}.csv"
        if not metrics.exists():
            command = [
                sys.executable,
                "tools/eval_yolo_suite.py",
                "--weights",
                str(ROOT / DATASETS[dataset]["detector"]),
                "--split",
                "val",
                "--imgsz",
                "640",
                "--batch",
                "8",
                "--device",
                "0",
                "--workers",
                "0",
                "--conf",
                "0.001",
                "--iou",
                "0.7",
                "--project",
                str(out / control / "yolo_runs"),
                "--out",
                str(metrics),
            ]
            for cause, data_yaml in entries[dataset]:
                command += ["--item", f"{dataset}_{cause}={data_yaml}"]
            run(command, out / control / "eval.log")
        rows.extend(read_rows(metrics))
    return rows


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    selection = json.loads(TRACE_SELECTION.read_text(encoding="utf-8"))
    if selection.get("test_split_used") is not False:
        raise RuntimeError("TRACE-R checkpoint was not selected on validation only")
    row = selection["best_by_val_map50"]["rmrp"]
    checkpoint = Path(row["checkpoint"])
    controls = tuple(dict.fromkeys(args.control or ["unavailable", "wrong_condition"]))
    summaries = [summarize("aligned", read_rows(TRACE_CORRECT_METRICS))]
    for control in controls:
        summaries.append(summarize(control, evaluate_control(control, checkpoint, out, args.device)))
    write_rows(out / "metadata_control_summary.csv", summaries)
    provenance = {
        "status": "complete",
        "scope": "validation-only packet intervention",
        "test_split_used": False,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "scenario_labels_at_inference": False,
        "detector_output_fusion": False,
        "controls": summaries,
        "detectors": {
            dataset: {
                "path": str((ROOT / info["detector"]).resolve()),
                "sha256": sha256(ROOT / info["detector"]),
            }
            for dataset, info in DATASETS.items()
        },
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
