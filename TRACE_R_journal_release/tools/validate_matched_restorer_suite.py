#!/usr/bin/env python3
"""Select matched-restorer checkpoints using validation detection only.

The selector first ranks epoch 0/5/.../30 on deterministic validation subsets,
then evaluates the two strongest checkpoints per method on the complete IVCNZ
and PCM validation splits. The final score is the unweighted mean mAP50 over
both datasets and four corruption families. Test data are never read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import yaml


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
        "detector": "runs/detect/runs/yolo11s_sequence_disjoint_v1_20260716/pothole_clean_80ep/weights/best.pt",
    },
    "pcm": {
        "prefix": "road_damage_pcm_yolo_sequence_disjoint_practical_sensor_calibrated_v2",
        "detector": "runs/detect/runs/yolo11s_sequence_disjoint_v1_20260716/pcm_clean_80ep/weights/best.pt",
    },
}
MODELS = ("rmrp", "nafnet", "nafnet_meta", "demoe", "dfpir", "instructir")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=tuple(SCENARIOS),
        help=(
            "Restrict an architecture pilot to one or more corruption causes. "
            "Omit this option for final four-cause checkpoint selection."
        ),
    )
    parser.add_argument("--subset-size", type=int, default=64)
    parser.add_argument(
        "--subset-only",
        action="store_true",
        help="Stop after deterministic validation-subset ranking; never open full validation.",
    )
    parser.add_argument(
        "--full-only",
        action="store_true",
        help=(
            "Evaluate every requested checkpoint on the complete validation "
            "sets. Use this for final policy selection when a small subset is "
            "not representative. Test data remain sealed."
        ),
    )
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--candidate-epochs", nargs="+", type=int, default=[0, 5, 10, 15, 20, 25, 30])
    parser.add_argument(
        "--model-dir",
        action="append",
        default=[],
        metavar="MODEL=PATH",
        help=(
            "Optional checkpoint-directory override for one model. This keeps "
            "immutable baseline runs and separately trained RMR-P variants in "
            "their original provenance directories."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--demoe-task",
        choices=("auto", "scenario"),
        default="scenario",
        help=(
            "DeMoE inference policy. 'auto' uses its image router; 'scenario' "
            "is an oracle-task upper bound that receives the benchmark cause."
        ),
    )
    parser.add_argument(
        "--discard-restored-after-scoring",
        action="store_true",
        help=(
            "Retain complete metrics and logs but remove temporary restored "
            "validation images after the frozen detector has scored them."
        ),
    )
    parser.add_argument(
        "--skip-residual-stage",
        action="store_true",
        help="Skip the legacy post-selection residual diagnostic.",
    )
    parser.add_argument(
        "--rmrp-output-stage",
        choices=("restored", "neural_restored", "physics_restoration_input"),
        default="restored",
        help="Validation-only RMR-P architecture output-stage audit.",
    )
    parser.add_argument(
        "--rmrp-residual-strength",
        type=float,
        default=1.0,
        help=(
            "Validation-only strength eta for input + eta*(restored-input). "
            "The selected value must later be stored as an automatic model policy."
        ),
    )
    parser.add_argument(
        "--rmrp-prompt-router-override",
        choices=("hard", "sparse_blend"),
        default=None,
        help="Validation-only RMR-P prompt-router audit.",
    )
    parser.add_argument(
        "--rmrp-prompt-delta-scale-override",
        type=float,
        default=None,
        help="Validation-only RMR-P prompt-residual scale audit.",
    )
    parser.add_argument(
        "--rmrp-sensor-route-mode-override",
        choices=("posterior", "physical_fused"),
        default=None,
        help="Validation-only RMR-P sensor-routing-state audit.",
    )
    parser.add_argument(
        "--rmrp-compound-motion-blend-override",
        type=float,
        default=None,
        help="Validation-only fixed compound-expert blend in [0, 1].",
    )
    parser.add_argument(
        "--rmrp-demoe-route-acceptance",
        nargs=3,
        type=float,
        metavar=("MOTION", "DEFOCUS", "LOWLIGHT"),
        default=None,
        help=(
            "Validation-only cause compatibility for DeMoE-based RMR-P. "
            "Each value must lie in [0, 1]."
        ),
    )
    parser.add_argument(
        "--rmrp-backbone-route-mode-override",
        choices=("metadata", "image", "sensor_task"),
        default=None,
    )
    parser.add_argument(
        "--rmrp-semantic-adapter-gain-override",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--rmrp-semantic-adapter-acceptance",
        nargs=4,
        type=float,
        metavar=("MOTION", "DEFOCUS", "LOWLIGHT", "COMPOUND"),
        default=None,
    )
    parser.add_argument(
        "--rmrp-compound-metadata-acceptance-override",
        type=float,
        default=None,
    )
    return parser.parse_args()


def parse_model_dirs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected MODEL=PATH for --model-dir, received {value!r}")
        model, raw_path = value.split("=", 1)
        if model not in MODELS:
            raise ValueError(f"Unsupported model in --model-dir: {model!r}")
        path = Path(raw_path)
        result[model] = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    return result


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def source_yaml(dataset: str, scenario: str) -> Path:
    prefix = DATASETS[dataset]["prefix"]
    return ROOT / "datasets" / f"{prefix}_{scenario}_val" / "data.yaml"


def source_root(yaml_path: Path) -> tuple[Path, dict]:
    document = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = Path(document["path"])
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()
    return root, document


def stable_subset(names: Iterable[str], size: int, salt: str) -> list[str]:
    ranked = sorted(
        names,
        key=lambda name: hashlib.sha256(f"{salt}:{name}".encode("utf-8")).hexdigest(),
    )
    return ranked[: min(size, len(ranked))]


def prepare_subset(
    dataset: str,
    out: Path,
    size: int,
    scenarios: Iterable[str],
) -> tuple[dict[str, Path], list[str]]:
    roots: dict[str, tuple[Path, dict]] = {}
    shared: set[str] | None = None
    for scenario in scenarios:
        root, document = source_root(source_yaml(dataset, scenario))
        roots[scenario] = (root, document)
        names = {path.name for path in (root / "images" / "val").glob("*.*")}
        shared = names if shared is None else shared & names
    assert shared is not None
    selected = stable_subset(shared, size, f"matched-v1:{dataset}")
    result: dict[str, Path] = {}
    for scenario, (root, document) in roots.items():
        target = out / "validation_subset_sources" / dataset / scenario
        for name in selected:
            stem = Path(name).stem
            hardlink_or_copy(root / "images" / "val" / name, target / "images" / "val" / name)
            for folder, suffix in (("labels", ".txt"), ("metadata", ".json")):
                source = root / folder / "val" / f"{stem}{suffix}"
                if source.exists():
                    hardlink_or_copy(source, target / folder / "val" / source.name)
        subset_yaml = {
            "path": str(target.resolve()),
            "train": "images/val",
            "val": "images/val",
            "test": "images/val",
            "names": document["names"],
            "nc": document.get("nc", len(document["names"])),
        }
        target.mkdir(parents=True, exist_ok=True)
        (target / "data.yaml").write_text(yaml.safe_dump(subset_yaml, sort_keys=False), encoding="utf-8")
        result[scenario] = target / "data.yaml"
    return result, selected


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
            # Ultralytics may emit Unicode progress glyphs. Preserve the full
            # UTF-8 log while replacing only characters unsupported by the
            # active Windows console code page.
            console_encoding = sys.stdout.encoding or "utf-8"
            console_line = line.encode(console_encoding, errors="replace").decode(console_encoding)
            print(console_line, end="", flush=True)
            handle.write(line)
            handle.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed ({return_code}); inspect {log}")


def checkpoint(
    training_root: Path,
    model_dirs: dict[str, Path],
    model: str,
    epoch: int,
) -> Path:
    model_root = model_dirs.get(model, training_root / model)
    path = model_root / f"{model}_epoch_{epoch:03d}.pth"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def restore_command(
    model: str,
    weights: Path,
    data: Path,
    scenario: str,
    out: Path,
    device: str,
    rmrp_output_stage: str,
    rmrp_residual_strength: float,
    rmrp_prompt_router_override: str | None,
    rmrp_prompt_delta_scale_override: float | None,
    rmrp_sensor_route_mode_override: str | None,
    rmrp_compound_motion_blend_override: float | None,
    rmrp_demoe_route_acceptance: list[float] | None,
    rmrp_backbone_route_mode_override: str | None,
    rmrp_semantic_adapter_gain_override: float | None,
    rmrp_semantic_adapter_acceptance: list[float] | None,
    rmrp_compound_metadata_acceptance_override: float | None,
    demoe_task: str,
) -> list[str]:
    command = [
        sys.executable, "tools/restore_yolo_split.py",
        "--data", str(data), "--split", "val", "--scenario", SCENARIOS[scenario],
        "--out", str(out), "--device", device,
    ]
    if model == "rmrp":
        command += [
            "--model", "rcadnet", "--rcadnet-weights", str(weights),
            "--rcadnet-code-source", "metadata", "--require-metadata",
            "--gate-threshold", "-1",
            "--rcadnet-output-stage", rmrp_output_stage,
            "--residual-strength", str(rmrp_residual_strength),
        ]
        if rmrp_prompt_router_override is not None:
            command += [
                "--rcadnet-prompt-router-override",
                rmrp_prompt_router_override,
            ]
        if rmrp_prompt_delta_scale_override is not None:
            command += [
                "--rcadnet-prompt-delta-scale-override",
                str(rmrp_prompt_delta_scale_override),
            ]
        if rmrp_sensor_route_mode_override is not None:
            command += [
                "--rcadnet-sensor-route-mode-override",
                rmrp_sensor_route_mode_override,
            ]
        if rmrp_compound_motion_blend_override is not None:
            command += [
                "--rcadnet-compound-motion-blend-override",
                str(rmrp_compound_motion_blend_override),
            ]
        if rmrp_demoe_route_acceptance is not None:
            command += [
                "--rcadnet-demoe-route-acceptance",
                *(str(value) for value in rmrp_demoe_route_acceptance),
            ]
        if rmrp_backbone_route_mode_override is not None:
            command += [
                "--rcadnet-backbone-route-mode-override",
                rmrp_backbone_route_mode_override,
            ]
        if rmrp_semantic_adapter_gain_override is not None:
            command += [
                "--rcadnet-semantic-adapter-gain-override",
                str(rmrp_semantic_adapter_gain_override),
            ]
        if rmrp_semantic_adapter_acceptance is not None:
            command += [
                "--rcadnet-semantic-adapter-acceptance",
                *(str(value) for value in rmrp_semantic_adapter_acceptance),
            ]
        if rmrp_compound_metadata_acceptance_override is not None:
            command += [
                "--rcadnet-compound-metadata-acceptance-override",
                str(rmrp_compound_metadata_acceptance_override),
            ]
    elif model == "nafnet":
        command += ["--model", "nafnet", "--nafnet-weights", str(weights)]
    elif model == "nafnet_meta":
        command += ["--model", "nafnet_meta", "--nafnet-weights", str(weights)]
    elif model == "demoe":
        command += [
            "--model", "demoe", "--demoe-weights", str(weights),
            "--demoe-task", demoe_task,
        ]
    elif model == "dfpir":
        command += ["--model", "dfpir", "--dfpir-weights", str(weights), "--dfpir-clip"]
    elif model == "instructir":
        command += [
            # Adapted checkpoints store the image-restoration state under
            # payload["model"]. InstructIRAdapter unwraps that state while
            # retaining support for the official raw image-only checkpoint.
            "--model", "instructir", "--instructir-image-weights", str(weights),
            "--instructir-lm-weights", str(ROOT / "weights/instructir/lm_instructir-7d.pt"),
            "--instructir-prompt-mode", "scenario",
        ]
    else:
        raise ValueError(model)
    return command


def yaml_split_directory(data_yaml: Path, split: str = "val") -> Path:
    """Resolve an Ultralytics split directory without opening image labels."""
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    split_value = payload.get(split)
    if not isinstance(split_value, str):
        raise ValueError(f"{data_yaml} has no scalar {split!r} image split")
    root_value = payload.get("path", ".")
    root = Path(root_value)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    images = Path(split_value)
    return images if images.is_absolute() else (root / images).resolve()


def image_count(directory: Path) -> int:
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def restored_scenario_complete(source_yaml: Path, restored: Path) -> bool:
    """Reuse output only when its manifest exists and image count is exact."""
    output_yaml = restored / "data.yaml"
    if not output_yaml.exists():
        return False
    source_images = yaml_split_directory(source_yaml)
    restored_images = yaml_split_directory(output_yaml)
    return image_count(source_images) > 0 and image_count(source_images) == image_count(restored_images)


def evaluate_checkpoint(
    model: str,
    epoch: int,
    weights: Path,
    sources: dict[str, dict[str, Path]],
    out: Path,
    stage: str,
    device: str,
    keep_restored: bool,
    rmrp_output_stage: str,
    rmrp_residual_strength: float,
    rmrp_prompt_router_override: str | None,
    rmrp_prompt_delta_scale_override: float | None,
    rmrp_sensor_route_mode_override: str | None,
    rmrp_compound_motion_blend_override: float | None,
    rmrp_demoe_route_acceptance: list[float] | None,
    rmrp_backbone_route_mode_override: str | None,
    rmrp_semantic_adapter_gain_override: float | None,
    rmrp_semantic_adapter_acceptance: list[float] | None,
    rmrp_compound_metadata_acceptance_override: float | None,
    demoe_task: str,
) -> dict:
    candidate = out / stage / model / f"epoch_{epoch:03d}"
    metrics_path = candidate / "metrics.csv"
    if metrics_path.exists():
        rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
    else:
        items: dict[str, list[tuple[str, Path]]] = {dataset: [] for dataset in DATASETS}
        for dataset in DATASETS:
            for scenario in sources[dataset]:
                restored = candidate / "restored" / dataset / scenario
                source_yaml = sources[dataset][scenario]
                if restored_scenario_complete(source_yaml, restored):
                    print(f"Reusing complete validation restoration: {dataset}/{scenario}", flush=True)
                else:
                    if restored.exists():
                        shutil.rmtree(restored)
                    run(
                        restore_command(
                            model,
                            weights,
                            source_yaml,
                            scenario,
                            restored,
                            device,
                            rmrp_output_stage,
                            rmrp_residual_strength,
                            rmrp_prompt_router_override,
                            rmrp_prompt_delta_scale_override,
                            rmrp_sensor_route_mode_override,
                            rmrp_compound_motion_blend_override,
                            rmrp_demoe_route_acceptance,
                            rmrp_backbone_route_mode_override,
                            rmrp_semantic_adapter_gain_override,
                            rmrp_semantic_adapter_acceptance,
                            rmrp_compound_metadata_acceptance_override,
                            demoe_task,
                        ),
                        candidate / "restore.log",
                    )
                items[dataset].append((scenario, restored / "data.yaml"))
        rows = []
        for dataset, entries in items.items():
            dataset_metrics = candidate / f"metrics_{dataset}.csv"
            command = [
                sys.executable, "tools/eval_yolo_suite.py",
                "--weights", str(ROOT / DATASETS[dataset]["detector"]),
                "--split", "val", "--imgsz", "640", "--batch", "8",
                "--device", "0", "--workers", "0", "--conf", "0.001", "--iou", "0.7",
                "--project", str(candidate / "yolo_runs"), "--out", str(dataset_metrics),
            ]
            for scenario, data_yaml in entries:
                command += ["--item", f"{dataset}_{scenario}={data_yaml}"]
            run(command, candidate / "eval.log")
            dataset_rows = list(csv.DictReader(dataset_metrics.open(encoding="utf-8")))
            rows.extend(dataset_rows)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if not keep_restored:
            restored_root = candidate / "restored"
            if restored_root.resolve().is_relative_to(out.resolve()) and restored_root.exists():
                shutil.rmtree(restored_root)

    by_dataset: dict[str, list[float]] = {dataset: [] for dataset in DATASETS}
    for row in rows:
        dataset = row["name"].split("_", 1)[0]
        by_dataset[dataset].append(float(row["map50"]))
    means = {dataset: sum(values) / len(values) for dataset, values in by_dataset.items()}
    return {
        "model": model,
        "epoch": epoch,
        "checkpoint": str(weights.resolve()),
        "ivcnz_mean_map50": means["ivcnz"],
        "pcm_mean_map50": means["pcm"],
        "joint_mean_map50": (means["ivcnz"] + means["pcm"]) / 2.0,
        "stage": stage,
        "test_split_used": False,
        "demoe_task": demoe_task if model == "demoe" else "not_applicable",
        "rmrp_output_stage": rmrp_output_stage if model == "rmrp" else "not_applicable",
        "rmrp_residual_strength": rmrp_residual_strength if model == "rmrp" else None,
        "rmrp_prompt_router": (
            (rmrp_prompt_router_override or "checkpoint_default")
            if model == "rmrp"
            else "not_applicable"
        ),
        "rmrp_prompt_delta_scale_override": (
            rmrp_prompt_delta_scale_override
            if model == "rmrp"
            else None
        ),
        "rmrp_sensor_route_mode": (
            (rmrp_sensor_route_mode_override or "checkpoint_default")
            if model == "rmrp"
            else "not_applicable"
        ),
        "rmrp_compound_motion_blend": (
            rmrp_compound_motion_blend_override if model == "rmrp" else None
        ),
        "rmrp_demoe_route_acceptance": (
            json.dumps(rmrp_demoe_route_acceptance)
            if model == "rmrp" and rmrp_demoe_route_acceptance is not None
            else "checkpoint_default" if model == "rmrp" else "not_applicable"
        ),
        "rmrp_backbone_route_mode": (
            (rmrp_backbone_route_mode_override or "checkpoint_default")
            if model == "rmrp"
            else "not_applicable"
        ),
        "rmrp_semantic_adapter_gain": (
            rmrp_semantic_adapter_gain_override if model == "rmrp" else None
        ),
        "rmrp_semantic_adapter_acceptance": (
            json.dumps(rmrp_semantic_adapter_acceptance)
            if model == "rmrp" and rmrp_semantic_adapter_acceptance is not None
            else "checkpoint_default" if model == "rmrp" else "not_applicable"
        ),
        "rmrp_compound_metadata_acceptance": (
            rmrp_compound_metadata_acceptance_override
            if model == "rmrp"
            else None
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_completed_rows(path: Path, stage: str) -> list[dict]:
    """Load only auditable rows that can be reused after an interrupted run."""
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            if raw.get("stage") != stage or raw.get("test_split_used") != "False":
                continue
            row = dict(raw)
            row["epoch"] = int(row["epoch"])
            for key in ("ivcnz_mean_map50", "pcm_mean_map50", "joint_mean_map50"):
                row[key] = float(row[key])
            row["test_split_used"] = False
            rows.append(row)
    return rows


def find_completed_row(
    rows: list[dict], model: str, epoch: int, weights: Path
) -> dict | None:
    resolved = str(weights.resolve())
    for row in rows:
        if (
            row["model"] == model
            and int(row["epoch"]) == epoch
            and str(Path(row["checkpoint"]).resolve()) == resolved
            and row["test_split_used"] is False
        ):
            return row
    return None


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.rmrp_residual_strength <= 1.0:
        raise ValueError("--rmrp-residual-strength must be in [0, 1]")
    selected_scenarios = tuple(dict.fromkeys(args.scenario or SCENARIOS))
    training_root = (ROOT / args.training_root).resolve() if not args.training_root.is_absolute() else args.training_root
    model_dirs = parse_model_dirs(args.model_dir)
    out = (ROOT / args.out).resolve() if not args.out.is_absolute() else args.out
    out.mkdir(parents=True, exist_ok=True)

    subset_sources: dict[str, dict[str, Path]] = {}
    subset_manifest = {
        "selection_split": "validation only",
        "test_split_used": False,
        "selection_rule": (
            "joint mean mAP50 across IVCNZ and PCM for causes: "
            + ", ".join(selected_scenarios)
        ),
        "scenarios": list(selected_scenarios),
        "rmrp_output_stage": args.rmrp_output_stage,
        "rmrp_residual_strength": args.rmrp_residual_strength,
        "rmrp_prompt_router_override": args.rmrp_prompt_router_override,
        "rmrp_prompt_delta_scale_override": args.rmrp_prompt_delta_scale_override,
        "rmrp_sensor_route_mode_override": args.rmrp_sensor_route_mode_override,
        "rmrp_compound_motion_blend_override": args.rmrp_compound_motion_blend_override,
        "rmrp_demoe_route_acceptance": args.rmrp_demoe_route_acceptance,
        "rmrp_backbone_route_mode_override": args.rmrp_backbone_route_mode_override,
        "rmrp_semantic_adapter_gain_override": args.rmrp_semantic_adapter_gain_override,
        "rmrp_semantic_adapter_acceptance": args.rmrp_semantic_adapter_acceptance,
        "rmrp_compound_metadata_acceptance_override": (
            args.rmrp_compound_metadata_acceptance_override
        ),
        "demoe_task": args.demoe_task,
        "checkpoint_roots": {
            model: str(model_dirs.get(model, training_root / model))
            for model in args.models
        },
        "datasets": {},
    }
    for dataset in DATASETS:
        sources, selected = prepare_subset(
            dataset, out, args.subset_size, selected_scenarios
        )
        subset_sources[dataset] = sources
        subset_manifest["datasets"][dataset] = {
            "n": len(selected),
            "identities": selected,
            "identity_sha256": hashlib.sha256("\n".join(selected).encode("utf-8")).hexdigest(),
        }
    (out / "validation_subset_manifest.json").write_text(
        json.dumps(subset_manifest, indent=2), encoding="utf-8"
    )

    subset_csv = out / "subset_selection_summary.csv"
    subset_rows = read_completed_rows(subset_csv, "subset")
    top_epochs: dict[str, list[int]] = {}
    if args.full_only:
        top_epochs = {
            model: list(dict.fromkeys(args.candidate_epochs))
            for model in args.models
        }
    else:
        for model in args.models:
            model_rows = []
            for epoch in args.candidate_epochs:
                weights = checkpoint(training_root, model_dirs, model, epoch)
                row = find_completed_row(subset_rows, model, epoch, weights)
                if row is None:
                    row = evaluate_checkpoint(
                        model,
                        epoch,
                        weights,
                        subset_sources,
                        out,
                        "subset",
                        args.device,
                        not args.discard_restored_after_scoring,
                        args.rmrp_output_stage,
                        args.rmrp_residual_strength,
                        args.rmrp_prompt_router_override,
                        args.rmrp_prompt_delta_scale_override,
                        args.rmrp_sensor_route_mode_override,
                        args.rmrp_compound_motion_blend_override,
                        args.rmrp_demoe_route_acceptance,
                        args.rmrp_backbone_route_mode_override,
                        args.rmrp_semantic_adapter_gain_override,
                        args.rmrp_semantic_adapter_acceptance,
                        args.rmrp_compound_metadata_acceptance_override,
                        args.demoe_task,
                    )
                    subset_rows.append(row)
                else:
                    print(f"[resume] subset model={model} epoch={epoch}", flush=True)
                model_rows.append(row)
                write_csv(subset_csv, subset_rows)
            ranked = sorted(
                model_rows,
                key=lambda row: (-row["joint_mean_map50"], row["epoch"]),
            )
            top_epochs[model] = [
                int(row["epoch"]) for row in ranked[: args.top_k]
            ]

    if args.subset_only:
        record = {
            "status": "subset_complete",
            "criterion": "joint mean validation-subset mAP50; lower epoch breaks ties",
            "test_split_used": False,
            "top_epochs": top_epochs,
        }
        (out / "subset_selection.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        print(json.dumps(record, indent=2), flush=True)
        return

    full_sources = {
        dataset: {
            scenario: source_yaml(dataset, scenario)
            for scenario in selected_scenarios
        }
        for dataset in DATASETS
    }
    full_csv = out / "full_validation_selection_summary.csv"
    full_rows = read_completed_rows(full_csv, "full_validation")
    for model in args.models:
        for epoch in top_epochs[model]:
            weights = checkpoint(training_root, model_dirs, model, epoch)
            row = find_completed_row(full_rows, model, epoch, weights)
            if row is None:
                row = evaluate_checkpoint(
                    model,
                    epoch,
                    weights,
                    full_sources,
                    out,
                    "full_validation",
                    args.device,
                    not args.discard_restored_after_scoring,
                    args.rmrp_output_stage,
                    args.rmrp_residual_strength,
                    args.rmrp_prompt_router_override,
                    args.rmrp_prompt_delta_scale_override,
                    args.rmrp_sensor_route_mode_override,
                    args.rmrp_compound_motion_blend_override,
                    args.rmrp_demoe_route_acceptance,
                    args.rmrp_backbone_route_mode_override,
                    args.rmrp_semantic_adapter_gain_override,
                    args.rmrp_semantic_adapter_acceptance,
                    args.rmrp_compound_metadata_acceptance_override,
                    args.demoe_task,
                )
                full_rows.append(row)
            else:
                print(f"[resume] full model={model} epoch={epoch}", flush=True)
            write_csv(full_csv, full_rows)

    best: dict[str, dict] = {}
    for model in args.models:
        candidates = [row for row in full_rows if row["model"] == model]
        selected = sorted(candidates, key=lambda row: (-row["joint_mean_map50"], row["epoch"]))[0]
        best[model] = selected
        for row in candidates:
            if row is selected:
                continue
            restored = out / "full_validation" / model / f"epoch_{row['epoch']:03d}" / "restored"
            if restored.exists() and restored.resolve().is_relative_to(out.resolve()):
                shutil.rmtree(restored)

    # Retain one common, deterministic validation subset for the subsequent
    # residual-strength sweep. The same candidate grid is evaluated for every
    # method, and no test pixels or labels are opened by this stage.
    residual_rows: list[dict] = []
    if not args.skip_residual_stage:
        for model in args.models:
            epoch = int(best[model]["epoch"])
            row = evaluate_checkpoint(
                model,
                epoch,
                checkpoint(training_root, model_dirs, model, epoch),
                subset_sources,
                out,
                "residual_validation",
                args.device,
                not args.discard_restored_after_scoring,
                args.rmrp_output_stage,
                args.rmrp_residual_strength,
                args.rmrp_prompt_router_override,
                args.rmrp_prompt_delta_scale_override,
                args.rmrp_sensor_route_mode_override,
                args.rmrp_compound_motion_blend_override,
                args.rmrp_demoe_route_acceptance,
                args.rmrp_backbone_route_mode_override,
                args.rmrp_semantic_adapter_gain_override,
                args.rmrp_semantic_adapter_acceptance,
                args.rmrp_compound_metadata_acceptance_override,
                args.demoe_task,
            )
            residual_rows.append(row)
        write_csv(out / "residual_validation_summary.csv", residual_rows)
    record = {
        "status": "complete",
        "criterion": "maximum joint mean validation mAP50; lower epoch breaks ties",
        "test_split_used": False,
        "best_by_val_map50": best,
        "residual_policy_source": (
            "skipped"
            if args.skip_residual_stage
            else "deterministic validation subset; test not read"
        ),
        "restored_validation_images_retained": not args.discard_restored_after_scoring,
    }
    (out / "best_by_val_map.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
