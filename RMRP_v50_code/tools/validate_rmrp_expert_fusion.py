#!/usr/bin/env python3
"""Validate the sensor-routed RMR-P expert policy without reading test data.

The script is restartable. A restoration block is reused only when its source
and output image counts agree and runtime provenance is present. Checkpoints,
policy code, detector weights, and metric files are hashed in the final ledger.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CAUSES = ("motion", "defocus", "lowlight", "mixed")
CROSS_CONDITION_SOURCE = {
    "motion": "defocus",
    "defocus": "lowlight",
    "lowlight": "mixed",
    "mixed": "motion",
}
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
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(r"E:\RMRP_experiments\rmrp_expert_fusion_v50_validation_20260822"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--control",
        action="append",
        choices=("correct", "unavailable", "shuffled", "cross_condition_shuffled"),
        help="Repeat to run several controls; defaults to correct metadata only.",
    )
    parser.add_argument(
        "--demoe-weights",
        type=Path,
        default=ROOT / "experiments/matched_final_candidate_index_v28_epoch70_20260821/demoe/demoe_epoch_070.pth",
    )
    parser.add_argument(
        "--dfpir-weights",
        type=Path,
        default=ROOT / "experiments/matched_final_candidate_index_v28_epoch70_20260821/dfpir/dfpir_epoch_070.pth",
    )
    parser.add_argument(
        "--instructir-weights",
        type=Path,
        default=ROOT / "experiments/matched_final_candidate_index_v28_epoch70_20260821/instructir/instructir_epoch_070.pth",
    )
    parser.add_argument(
        "--instructir-lm-weights",
        type=Path,
        default=ROOT / "weights/instructir/lm_instructir-7d.pt",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_yaml(dataset: str, cause: str) -> Path:
    prefix = DATASETS[dataset]["prefix"]
    return ROOT / "datasets" / f"{prefix}_{cause}_val" / "data.yaml"


def image_directory(data_yaml: Path, split: str = "val") -> Path:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(payload["path"])
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    value = Path(payload[split])
    return value if value.is_absolute() else root / value


def image_count(directory: Path) -> int:
    return sum(
        path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        for path in directory.rglob("*")
    )


def metadata_directory(data_yaml: Path, split: str = "val") -> Path:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(payload["path"])
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    value = Path(str(payload[split]).replace("images", "metadata", 1))
    return value if value.is_absolute() else root / value


def complete(source: Path, output: Path) -> bool:
    output_yaml = output / "data.yaml"
    runtime = output / "runtime.json"
    if not output_yaml.exists() or not runtime.exists():
        return False
    return image_count(image_directory(source)) == image_count(image_directory(output_yaml))


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + subprocess.list2cmdline(command) + "\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            console_encoding = sys.stdout.encoding or "utf-8"
            console_line = line.encode(
                console_encoding, errors="replace"
            ).decode(console_encoding)
            print(console_line, end="", flush=True)
            handle.write(line)
            handle.flush()
        code = process.wait()
    if code:
        raise RuntimeError(f"Command failed with exit code {code}; inspect {log}")


def restore_command(
    args: argparse.Namespace,
    dataset: str,
    cause: str,
    source: Path,
    output: Path,
    control: str,
) -> list[str]:
    effective_control = "correct" if control == "cross_condition_shuffled" else control
    command = [
        sys.executable,
        "tools/restore_yolo_split.py",
        "--data",
        str(source),
        "--split",
        "val",
        "--model",
        "rmrp_fusion",
        "--scenario",
        "not_used_by_sensor_router",
        "--out",
        str(output),
        "--device",
        args.device,
        "--demoe-weights",
        str(args.demoe_weights),
        "--dfpir-weights",
        str(args.dfpir_weights),
        "--instructir-image-weights",
        str(args.instructir_weights),
        "--instructir-lm-weights",
        str(args.instructir_lm_weights),
        "--dfpir-clip",
        "--require-metadata",
        "--rcadnet-metadata-control",
        effective_control,
        "--metadata-control-seed",
        "2026",
    ]
    if control == "cross_condition_shuffled":
        wrong_cause = CROSS_CONDITION_SOURCE[cause]
        wrong_metadata = metadata_directory(source_yaml(dataset, wrong_cause))
        command += ["--metadata-dir-override", str(wrong_metadata)]
    return command


def evaluate_control(args: argparse.Namespace, out: Path, control: str) -> list[dict[str, str]]:
    control_root = out / control
    entries: dict[str, list[tuple[str, Path]]] = {dataset: [] for dataset in DATASETS}
    for dataset in DATASETS:
        for cause in CAUSES:
            source = source_yaml(dataset, cause)
            restored = control_root / "restored" / dataset / cause
            if complete(source, restored):
                print(f"[resume] {control} {dataset}/{cause}", flush=True)
            else:
                run(
                    restore_command(args, dataset, cause, source, restored, control),
                    control_root / "restore.log",
                )
            entries[dataset].append((cause, restored / "data.yaml"))

    rows: list[dict[str, str]] = []
    for dataset, items in entries.items():
        metrics = control_root / f"metrics_{dataset}.csv"
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
                str(control_root / "yolo_runs"),
                "--out",
                str(metrics),
            ]
            for cause, data_yaml in items:
                command += ["--item", f"{dataset}_{cause}={data_yaml}"]
            run(command, control_root / "eval.log")
        rows.extend(csv.DictReader(metrics.open(encoding="utf-8")))
    return rows


def summarize(rows: list[dict[str, str]]) -> dict:
    per_dataset: dict[str, list[float]] = {dataset: [] for dataset in DATASETS}
    per_cause: dict[str, dict[str, float]] = {dataset: {} for dataset in DATASETS}
    for row in rows:
        dataset, cause = row["name"].split("_", 1)
        value = float(row["map50"])
        per_dataset[dataset].append(value)
        per_cause[dataset][cause] = value
    means = {
        dataset: sum(values) / len(values)
        for dataset, values in per_dataset.items()
    }
    return {
        "per_cause_map50": per_cause,
        "mean_map50": means,
        "joint_mean_map50": sum(means.values()) / len(means),
    }


def main() -> None:
    args = parse_args()
    args.out = args.out.resolve()
    controls = tuple(dict.fromkeys(args.control or ["correct"]))
    for path in (
        args.demoe_weights,
        args.dfpir_weights,
        args.instructir_weights,
        args.instructir_lm_weights,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    args.out.mkdir(parents=True, exist_ok=True)

    results = {}
    for control in controls:
        rows = evaluate_control(args, args.out, control)
        summary = summarize(rows)
        results[control] = summary
        (args.out / control / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    # A restart may intentionally request only the controls that are still
    # missing. Merge previously completed summaries so the provenance ledger
    # always describes the complete executed study rather than only this
    # invocation.
    for control in ("correct", "unavailable", "shuffled", "cross_condition_shuffled"):
        summary_path = args.out / control / "summary.json"
        if control not in results and summary_path.exists():
            results[control] = json.loads(summary_path.read_text(encoding="utf-8"))

    baseline_metrics = (
        Path(r"E:\RMRP_experiments\heterogeneous_expert_fusion_val_v1_20260822")
        / "full_validation/demoe/epoch_070/metrics.csv"
    )
    baseline = summarize(list(csv.DictReader(baseline_metrics.open(encoding="utf-8"))))
    correct = results.get("correct")
    ledger = {
        "status": "complete",
        "scope": "validation only",
        "test_split_used": False,
        "selection_metric": "joint mean validation mAP50 over IVCNZ and PCM",
        "scenario_family_is_model_input": False,
        "policy": {
            "motion_threshold": 0.18,
            "defocus_threshold": 0.20,
            "lowlight_threshold": 0.385,
            "support_threshold": 0.50,
            "lowlight_dfpir_weight": 0.40,
            "mixed_dfpir_weight": 0.075,
            "gyro_full_scale": 4.0,
        },
        "results": results,
        "matched_demoe": baseline,
        "correct_minus_matched_demoe": (
            {
                dataset: correct["mean_map50"][dataset] - baseline["mean_map50"][dataset]
                for dataset in DATASETS
            }
            | {
                "joint": correct["joint_mean_map50"] - baseline["joint_mean_map50"]
            }
            if correct is not None
            else None
        ),
        "artifacts": {
            str(path.resolve()): sha256(path)
            for path in (
                args.demoe_weights,
                args.dfpir_weights,
                args.instructir_weights,
                args.instructir_lm_weights,
                Path(__file__),
                ROOT / "models/rmrp_expert_fusion.py",
                ROOT / "tools/restore_yolo_split.py",
                baseline_metrics,
                *(ROOT / DATASETS[dataset]["detector"] for dataset in DATASETS),
            )
        },
    }
    (args.out / "validation_provenance.json").write_text(
        json.dumps(ledger, indent=2), encoding="utf-8"
    )
    print(json.dumps(ledger, indent=2), flush=True)


if __name__ == "__main__":
    main()
