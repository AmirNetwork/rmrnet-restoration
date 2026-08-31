#!/usr/bin/env python3
# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
"""Run the frozen TRACE-R confirmatory test exactly once.

The script is deliberately separate from checkpoint selection. It reads the
validation-only ledgers, freezes checkpoint and detector hashes, and then
evaluates the already selected policies on the sequence-disjoint test splits.
An interrupted run is restartable, but a frozen checkpoint cannot be changed.

Each method emits one restored image. Detector predictions are never fused.
DeMoE-auto is the deployable comparison; DeMoE-oracle is reported separately
because it receives the benchmark corruption family at inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = {
    "motion": "motion_horizontal_medium",
    "defocus": "defocus_medium",
    "lowlight": "lowlight_medium",
    "mixed": "mixed_motion_lowlight",
}
DATASETS = {
    "ivcnz": {
        "prefix": "pothole_yolo_sequence_disjoint_practical_sensor_calibrated_v2",
        "clean": "datasets/pothole_yolo_sequence_disjoint_v1/data.yaml",
        "detector": "runs/detect/runs/yolo11s_sequence_disjoint_v1_20260716/pothole_clean_80ep/weights/best.pt",
    },
    "pcm": {
        "prefix": "road_damage_pcm_yolo_sequence_disjoint_practical_sensor_calibrated_v2",
        "clean": "datasets/road_damage_pcm_yolo_sequence_disjoint_v1/data.yaml",
        "detector": "runs/detect/runs/yolo11s_sequence_disjoint_v1_20260716/pcm_clean_80ep/weights/best.pt",
    },
}
BASELINE_ROOT = ROOT / "experiments" / "matched_final_candidate_index_v28_epoch70_20260821"
BASELINE_VALIDATION = (
    ROOT
    / "experiments"
    / "matched_baselines_v28_epoch70_validation_20260821"
    / "full_validation_selection_summary.csv"
)
DEMOE_AUTO_VALIDATION = (
    Path(r"E:\TRACE_R_experiments\matched_demoe_auto_validation_v64_20260828")
    / "full_validation_selection_summary.csv"
)
TRACE_VALIDATION = (
    Path(r"E:\TRACE_R_experiments\matched_budget_trace_v53_20260827")
    / "validation"
    / "best_by_val_map.json"
)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(r"E:\TRACE_R_experiments\trace_locked_confirmatory_v65_20260828"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "raw",
            "nafnet",
            "nafnet_meta",
            "instructir",
            "dfpir",
            "demoe_auto",
            "demoe_oracle",
            "trace_r",
        ],
        choices=[
            "raw",
            "nafnet",
            "nafnet_meta",
            "instructir",
            "dfpir",
            "demoe_auto",
            "demoe_oracle",
            "trace_r",
        ],
    )
    parser.add_argument(
        "--discard-restored",
        action="store_true",
        help="Remove restored images after all detector and fidelity metrics are complete.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"command": command, "log": str(log)}), flush=True)
    with log.open("a", encoding="utf-8", errors="replace") as handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode:
        tail = log.read_text(encoding="utf-8", errors="replace")[-6000:]
        raise RuntimeError(f"Command failed ({completed.returncode})\n{tail}")


def best_row(rows: Iterable[dict[str, str]], model: str) -> dict[str, str]:
    candidates = [row for row in rows if row["model"] == model]
    if not candidates:
        raise ValueError(f"No validation row for {model}")
    if any(row.get("test_split_used", "False") != "False" for row in candidates):
        raise RuntimeError(f"Validation ledger for {model} reports test access")
    return max(candidates, key=lambda row: (float(row["joint_mean_map50"]), -int(row["epoch"])))


def selected_policies() -> dict[str, dict[str, Any]]:
    baseline_rows = read_csv(BASELINE_VALIDATION)
    auto_rows = read_csv(DEMOE_AUTO_VALIDATION)
    trace_record = json.loads(TRACE_VALIDATION.read_text(encoding="utf-8"))
    if trace_record.get("test_split_used") is not False:
        raise RuntimeError("TRACE-R selection ledger is not validation-only")
    trace_row = trace_record["best_by_val_map50"]["rmrp"]
    policies: dict[str, dict[str, Any]] = {}
    for name in ("nafnet", "nafnet_meta", "instructir", "dfpir"):
        row = best_row(baseline_rows, name)
        policies[name] = {
            "model": name,
            "epoch": int(row["epoch"]),
            "checkpoint": Path(row["checkpoint"]),
            "validation_joint_map50": float(row["joint_mean_map50"]),
            "inference_information": "image only" if name != "nafnet_meta" else "same 82-field packet",
        }
    auto = best_row(auto_rows, "demoe")
    policies["demoe_auto"] = {
        "model": "demoe",
        "epoch": int(auto["epoch"]),
        "checkpoint": Path(auto["checkpoint"]),
        "validation_joint_map50": float(auto["joint_mean_map50"]),
        "demoe_task": "auto",
        "inference_information": "image router",
    }
    oracle = best_row(baseline_rows, "demoe")
    policies["demoe_oracle"] = {
        "model": "demoe",
        "epoch": int(oracle["epoch"]),
        "checkpoint": Path(oracle["checkpoint"]),
        "validation_joint_map50": float(oracle["joint_mean_map50"]),
        "demoe_task": "scenario",
        "inference_information": "oracle benchmark condition",
    }
    policies["trace_r"] = {
        "model": "rmrp",
        "epoch": int(trace_row["epoch"]),
        "checkpoint": Path(trace_row["checkpoint"]),
        "validation_joint_map50": float(trace_row["joint_mean_map50"]),
        "inference_information": "observable 82-field packet",
    }
    for name, policy in policies.items():
        checkpoint = policy["checkpoint"]
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        policy["checkpoint"] = str(checkpoint.resolve())
        policy["checkpoint_sha256"] = sha256(checkpoint)
        policy["scenario_labels_at_inference"] = name in {"demoe_oracle", "instructir"}
        policy["detector_output_fusion"] = False
        policy["single_restored_image"] = True
    return policies


def source_yaml(dataset: str, scenario: str) -> Path:
    prefix = DATASETS[dataset]["prefix"]
    return ROOT / "datasets" / f"{prefix}_{scenario}_test" / "data.yaml"


def yaml_image_directory(data_yaml: Path, split: str = "test") -> Path:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(payload.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    images = Path(payload[split])
    return images if images.is_absolute() else root / images


def image_count(data_yaml: Path, split: str = "test") -> int:
    return sum(
        path.suffix.lower() in IMAGE_SUFFIXES
        for path in yaml_image_directory(data_yaml, split).rglob("*")
        if path.is_file()
    )


def restoration_complete(source: Path, output: Path) -> bool:
    data_yaml = output / "data.yaml"
    return (
        data_yaml.exists()
        and image_count(source) > 0
        and image_count(source) == image_count(data_yaml)
    )


def restore_command(
    policy_name: str,
    policy: dict[str, Any],
    data_yaml: Path,
    scenario: str,
    output: Path,
    device: str,
) -> list[str]:
    command = [
        sys.executable,
        "tools/restore_yolo_split.py",
        "--data",
        str(data_yaml),
        "--split",
        "test",
        "--scenario",
        SCENARIOS[scenario],
        "--out",
        str(output),
        "--device",
        device,
    ]
    checkpoint = policy["checkpoint"]
    if policy_name == "trace_r":
        command += [
            "--model",
            "rcadnet",
            "--rcadnet-weights",
            checkpoint,
            "--rcadnet-code-source",
            "metadata",
            "--require-metadata",
            "--gate-threshold",
            "-1",
            "--rcadnet-output-stage",
            "restored",
            "--residual-strength",
            "1.0",
        ]
    elif policy_name in {"demoe_auto", "demoe_oracle"}:
        command += [
            "--model",
            "demoe",
            "--demoe-weights",
            checkpoint,
            "--demoe-task",
            policy["demoe_task"],
        ]
    elif policy_name == "dfpir":
        command += ["--model", "dfpir", "--dfpir-weights", checkpoint, "--dfpir-clip"]
    elif policy_name in {"nafnet", "nafnet_meta"}:
        command += ["--model", policy_name, "--nafnet-weights", checkpoint]
    elif policy_name == "instructir":
        command += [
            "--model",
            "instructir",
            "--instructir-image-weights",
            checkpoint,
            "--instructir-lm-weights",
            str(ROOT / "weights/instructir/lm_instructir-7d.pt"),
            "--instructir-prompt-mode",
            "scenario",
        ]
    else:
        raise ValueError(policy_name)
    return command


def evaluate_detector(
    method: str,
    restored: dict[str, dict[str, Path]],
    out: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        metrics = out / "detection" / method / f"metrics_{dataset}.csv"
        if not metrics.exists():
            command = [
                sys.executable,
                "tools/eval_yolo_suite.py",
                "--weights",
                str(ROOT / DATASETS[dataset]["detector"]),
                "--split",
                "test",
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
                str(out / "detection" / method / "yolo_runs"),
                "--out",
                str(metrics),
            ]
            for scenario, data_yaml in restored[dataset].items():
                command += ["--item", f"{dataset}_{scenario}={data_yaml}"]
            run(command, out / "logs" / f"detect_{method}_{dataset}.log")
        for row in read_csv(metrics):
            rows.append({"method": method, **row})
    return rows


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def load_rgb_aligned(path: Path, size: tuple[int, int]) -> np.ndarray:
    """Load a clean reference on the restored image canvas.

    The calibrated synthetic sets preserve aspect ratio but store their
    degraded frames on a smaller canvas than the original clean sequence.
    Detection consumes each stored image through the fixed YOLO letterbox
    path.  Fidelity is therefore measured at the restoration canvas by
    applying one deterministic antialiased resize to the clean reference.
    """
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if rgb.size != size:
            rgb = rgb.resize(size, resample=Image.Resampling.LANCZOS)
        return np.asarray(rgb, dtype=np.float32) / 255.0


def paired_fidelity(
    method: str,
    restored: dict[str, dict[str, Path]],
    out: Path,
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        clean_yaml = ROOT / DATASETS[dataset]["clean"]
        clean_dir = yaml_image_directory(clean_yaml)
        clean_by_stem: dict[str, Path] = {}
        for clean_path in clean_dir.rglob("*"):
            if not clean_path.is_file() or clean_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if clean_path.stem in clean_by_stem:
                raise ValueError(f"Duplicate clean image stem: {clean_path.stem}")
            clean_by_stem[clean_path.stem] = clean_path
        for scenario, data_yaml in restored[dataset].items():
            output_dir = yaml_image_directory(data_yaml)
            values: list[tuple[float, float]] = []
            for prediction_path in sorted(output_dir.iterdir()):
                if prediction_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                target_path = clean_by_stem.get(prediction_path.stem)
                if target_path is None:
                    raise FileNotFoundError(
                        f"No clean reference with stem {prediction_path.stem!r} in {clean_dir}"
                    )
                prediction = load_rgb(prediction_path)
                height, width = prediction.shape[:2]
                target = load_rgb_aligned(target_path, (width, height))
                mse = float(np.mean((prediction - target) ** 2))
                psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-12)))
                ssim = float(
                    structural_similarity(target, prediction, channel_axis=2, data_range=1.0)
                )
                values.append((psnr, ssim))
                paired_rows.append(
                    {
                        "method": method,
                        "dataset": dataset,
                        "scenario": scenario,
                        "image": prediction_path.name,
                        "alignment_policy": "clean reference resized to restoration canvas (Lanczos)",
                        "psnr": psnr,
                        "ssim": ssim,
                    }
                )
            summary.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "scenario": scenario,
                    "images": len(values),
                    "alignment_policy": "clean reference resized to restoration canvas (Lanczos)",
                    "psnr": float(np.mean([value[0] for value in values])),
                    "ssim": float(np.mean([value[1] for value in values])),
                }
            )
    write_csv(out / "fidelity" / f"paired_{method}.csv", paired_rows)
    write_csv(out / "fidelity" / f"summary_{method}.csv", summary)
    return summary


def aggregate_detection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ("map50", "map50_95", "precision", "recall")
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        dataset, _scenario = row["name"].split("_", 1)
        for metric in metrics:
            grouped.setdefault((row["method"], dataset, metric), []).append(
                float(row[metric])
            )
    output: list[dict[str, Any]] = []
    methods = sorted({key[0] for key in grouped})
    for method in methods:
        row: dict[str, Any] = {"method": method}
        for metric in metrics:
            ivcnz = float(np.mean(grouped[(method, "ivcnz", metric)]))
            pcm = float(np.mean(grouped[(method, "pcm", metric)]))
            row[f"ivcnz_mean_{metric}"] = ivcnz
            row[f"pcm_mean_{metric}"] = pcm
            row[f"joint_mean_{metric}"] = (ivcnz + pcm) / 2.0
        output.append(row)
    return output


def evaluate_clean_ceilings(out: Path) -> list[dict[str, Any]]:
    """Score each dataset's native clean test split with its frozen detector."""

    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        metrics = out / "detection" / "clean_ceiling" / f"metrics_{dataset}.csv"
        if not metrics.exists():
            run(
                [
                    sys.executable,
                    "tools/eval_yolo_suite.py",
                    "--weights",
                    str(ROOT / DATASETS[dataset]["detector"]),
                    "--split",
                    "test",
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
                    str(out / "detection" / "clean_ceiling" / "yolo_runs"),
                    "--out",
                    str(metrics),
                    "--item",
                    f"{dataset}_clean={ROOT / DATASETS[dataset]['clean']}",
                ],
                out / "logs" / f"detect_clean_{dataset}.log",
            )
        rows.extend({"dataset": dataset, **row} for row in read_csv(metrics))
    write_csv(out / "detection" / "clean_ceiling.csv", rows)
    return rows


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    policies = selected_policies()
    detectors = {
        dataset: {
            "path": str((ROOT / info["detector"]).resolve()),
            "sha256": sha256(ROOT / info["detector"]),
        }
        for dataset, info in DATASETS.items()
    }
    source_hashes = {
        f"{dataset}:{scenario}": sha256(source_yaml(dataset, scenario))
        for dataset in DATASETS
        for scenario in SCENARIOS
    }
    frozen = {
        "status": "selection_frozen_before_test",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_split": "validation only",
        "test_used_for_selection": False,
        "single_restored_image_per_method": True,
        "detector_output_fusion": False,
        "policies": policies,
        "detectors": detectors,
        "test_source_yaml_sha256": source_hashes,
    }
    frozen_path = out / "frozen_selection_before_test.json"
    if frozen_path.exists():
        previous = json.loads(frozen_path.read_text(encoding="utf-8"))
        for key in ("policies", "detectors", "test_source_yaml_sha256"):
            if previous[key] != frozen[key]:
                raise RuntimeError(f"Frozen confirmatory configuration changed: {key}")
        frozen = previous
    else:
        frozen_path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")

    all_detection: list[dict[str, Any]] = []
    all_fidelity: list[dict[str, Any]] = []
    for method in args.methods:
        print(json.dumps({"method": method, "status": "running"}), flush=True)
        restored: dict[str, dict[str, Path]] = {dataset: {} for dataset in DATASETS}
        for dataset in DATASETS:
            for scenario in SCENARIOS:
                source = source_yaml(dataset, scenario)
                if method == "raw":
                    restored[dataset][scenario] = source
                    continue
                output = out / "restored" / method / dataset / scenario
                if not restoration_complete(source, output):
                    if output.exists():
                        shutil.rmtree(output)
                    run(
                        restore_command(
                            method,
                            policies[method],
                            source,
                            scenario,
                            output,
                            args.device,
                        ),
                        out / "logs" / f"restore_{method}.log",
                    )
                restored[dataset][scenario] = output / "data.yaml"
        all_detection.extend(evaluate_detector(method, restored, out))
        all_fidelity.extend(paired_fidelity(method, restored, out))

    write_csv(out / "detection" / "all_condition_metrics.csv", all_detection)
    aggregate = aggregate_detection(all_detection)
    write_csv(out / "detection" / "aggregate_metrics.csv", aggregate)
    evaluate_clean_ceilings(out)
    write_csv(out / "fidelity" / "all_summary.csv", all_fidelity)
    ledger = {
        **frozen,
        "status": "confirmatory_test_complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_split_opened_after_selection_freeze": True,
        "artifacts": {
            "detection_conditions": str((out / "detection" / "all_condition_metrics.csv")),
            "detection_aggregate": str((out / "detection" / "aggregate_metrics.csv")),
            "clean_detection_ceiling": str((out / "detection" / "clean_ceiling.csv")),
            "fidelity_summary": str((out / "fidelity" / "all_summary.csv")),
        },
    }
    (out / "final_provenance_ledger.json").write_text(
        json.dumps(ledger, indent=2), encoding="utf-8"
    )
    if args.discard_restored:
        restored_root = out / "restored"
        if restored_root.exists() and restored_root.resolve().is_relative_to(out):
            shutil.rmtree(restored_root)
    print(json.dumps({"status": ledger["status"], "aggregate": aggregate}, indent=2), flush=True)


if __name__ == "__main__":
    main()
