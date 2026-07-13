from __future__ import annotations

"""Evaluate the revised identity-aware RMR-Net checkpoints on controlled splits.

This runner is deliberately evaluation-only. The revised checkpoints are produced
by train_rmrnet.py from the 30-epoch controlled models plus held-out-safe native
identity frames. Here we restore the held-out test splits for all controlled
scenarios and evaluate the same frozen YOLO11s detectors used by the paper.
"""

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "trc_revised_identity_final"

SCENARIOS = {
    "motion": "motion_horizontal_medium",
    "defocus": "defocus_medium",
    "lowlight": "lowlight_medium",
    "mixed": "mixed_motion_lowlight",
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    test_root: Path
    checkpoint: Path
    detector_weights: Path
    yolo_prefix: str
    mixed_test_dir: str
    bench_out: Path


DATASETS = [
    DatasetSpec(
        key="pothole",
        test_root=ROOT / "data" / "pothole_restoration_test",
        checkpoint=ROOT / "runs" / "rmrnet_v31_native_identity_pothole3ep" / "rcadnet_best.pth",
        detector_weights=ROOT / "runs" / "detect" / "runs" / "yolo11s_v26" / "pothole_clean_80ep" / "weights" / "best.pt",
        yolo_prefix="pothole_yolo",
        mixed_test_dir="pothole_yolo_mixed_test",
        bench_out=ROOT / "runs" / "bench_trc_revised_identity_pothole",
    ),
    DatasetSpec(
        key="pcm",
        test_root=ROOT / "data" / "pcm_restoration_test",
        checkpoint=ROOT / "runs" / "rmrnet_v31_native_identity_pcm3ep" / "rcadnet_best.pth",
        detector_weights=ROOT / "runs" / "detect" / "runs" / "yolo11s_v26" / "pcm_clean_80ep" / "weights" / "best.pt",
        yolo_prefix="pcm_yolo",
        mixed_test_dir="pcm_yolo_mixed_test",
        bench_out=ROOT / "runs" / "bench_trc_revised_identity_pcm",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=["pothole", "pcm"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--force-restore", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], log_path: Path) -> None:
    print(json.dumps({"cmd": cmd}), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            safe = line.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
            try:
                print(safe, end="")
            except UnicodeEncodeError:
                print(safe.encode("ascii", errors="replace").decode("ascii"), end="")
            handle.write(line)
        code = proc.wait()
    if code:
        raise subprocess.CalledProcessError(code, cmd)


def source_data_yaml(spec: DatasetSpec, scenario_key: str) -> Path:
    directory = spec.mixed_test_dir if scenario_key == "mixed" else f"{spec.yolo_prefix}_{scenario_key}_test"
    path = ROOT / "datasets" / directory / "data.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def restored_dir(spec: DatasetSpec, scenario_key: str) -> Path:
    return ROOT / "datasets" / f"{spec.yolo_prefix}_{scenario_key}_test_rmrnet_revised_identity"


def restore_split(args: argparse.Namespace, spec: DatasetSpec, scenario_key: str) -> Path:
    out = restored_dir(spec, scenario_key)
    yaml_path = out / "data.yaml"
    if yaml_path.exists() and not args.force_restore:
        return yaml_path
    run(
        [
            sys.executable,
            "tools/restore_yolo_split.py",
            "--data",
            str(source_data_yaml(spec, scenario_key)),
            "--split",
            "test",
            "--model",
            "rcadnet",
            "--scenario",
            SCENARIOS[scenario_key],
            "--out",
            str(out),
            "--device",
            args.device,
            "--rcadnet-weights",
            str(spec.checkpoint),
            "--rcadnet-code-source",
            "metadata",
            "--gate-threshold",
            "-1",
            "--residual-strength",
            "1.0",
        ],
        EXP / "logs" / f"restore_{spec.key}_{scenario_key}.log",
    )
    return yaml_path


def eval_suite(args: argparse.Namespace, spec: DatasetSpec, items: list[tuple[str, Path]]) -> Path:
    out_stem = EXP / f"{spec.key}_test_rmrnet_revised_identity"
    cmd = [
        sys.executable,
        "tools/eval_yolo_suite.py",
        "--weights",
        str(spec.detector_weights),
        "--imgsz",
        "640",
        "--batch",
        str(args.eval_batch),
        "--device",
        "0" if args.device.startswith("cuda") else "cpu",
        "--workers",
        "0",
        "--split",
        "test",
        "--project",
        str(ROOT / "runs" / "detection_eval_trc_revised_identity"),
        "--out",
        str(out_stem),
    ]
    for name, data_yaml in items:
        cmd.extend(["--item", f"{name}={data_yaml}"])
    run(cmd, EXP / "logs" / f"eval_{spec.key}.log")
    return out_stem.with_suffix(".csv")


def eval_per_class(args: argparse.Namespace, spec: DatasetSpec, items: list[tuple[str, Path]]) -> Path:
    out_stem = EXP / f"{spec.key}_test_rmrnet_revised_identity_per_class"
    cmd = [
        sys.executable,
        "tools/eval_yolo_per_class_suite.py",
        "--weights",
        str(spec.detector_weights),
        "--imgsz",
        "640",
        "--batch",
        str(args.eval_batch),
        "--device",
        "0" if args.device.startswith("cuda") else "cpu",
        "--workers",
        "0",
        "--split",
        "test",
        "--project",
        str(ROOT / "runs" / "detection_eval_trc_revised_identity_per_class"),
        "--out",
        str(out_stem),
    ]
    for name, data_yaml in items:
        cmd.extend(["--item", f"{name}={data_yaml}"])
    run(cmd, EXP / "logs" / f"per_class_{spec.key}.log")
    return out_stem.with_suffix(".csv")


def benchmark_restoration(args: argparse.Namespace, spec: DatasetSpec) -> Path:
    cmd = [
        sys.executable,
        "benchmark_unified_restoration.py",
        "--data-root",
        str(spec.test_root),
        "--model",
        "rcadnet",
        "--rcadnet-weights",
        str(spec.checkpoint),
        "--rcadnet-code-source",
        "metadata",
        "--device",
        args.device,
        "--max-side",
        "0",
        "--warmup",
        "2",
        "--out",
        str(spec.bench_out),
    ]
    for scenario in ("motion_horizontal_medium", "defocus_medium", "lowlight_medium"):
        cmd.extend(["--scenario", scenario])
    run(cmd, EXP / "logs" / f"bench_{spec.key}.log")
    return spec.bench_out / "metrics.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def selected_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    if not args.dataset:
        return DATASETS
    wanted = set(args.dataset)
    return [spec for spec in DATASETS if spec.key in wanted]


def main() -> None:
    args = parse_args()
    EXP.mkdir(parents=True, exist_ok=True)
    manifest = []
    for spec in selected_specs(args):
        if not spec.checkpoint.exists():
            raise FileNotFoundError(spec.checkpoint)
        items = [(f"{scenario}_rmr_revised", restore_split(args, spec, scenario)) for scenario in SCENARIOS]
        detection_csv = eval_suite(args, spec, items)
        per_class_csv = eval_per_class(args, spec, items)
        restoration_csv = benchmark_restoration(args, spec)
        manifest.append(
            {
                "dataset": spec.key,
                "checkpoint": str(spec.checkpoint),
                "detection_csv": str(detection_csv),
                "per_class_csv": str(per_class_csv),
                "restoration_csv": str(restoration_csv),
                "scenarios": SCENARIOS,
                "notes": "Revised identity-aware checkpoint evaluated on held-out controlled test splits.",
            }
        )
    (EXP / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": manifest}, indent=2), flush=True)


if __name__ == "__main__":
    main()
