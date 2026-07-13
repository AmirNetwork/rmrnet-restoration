from __future__ import annotations

"""Run the final 30-epoch RMR-Net sweep for the TRC manuscript.

This script is intentionally boring and auditable. It does not touch test data
until after validation mAP50 selects a checkpoint.

Training objective implemented by train_rmrnet.py:

    L = L_base + lambda_s ||r||_1 + lambda_TDP L_TDP + lambda_J L_J
        + lambda_A L_anchor + lambda_E L_evidence + lambda_D L_detail

with lambda_AC = 0. Active contours are evaluated after detection as a boundary
measurement stage, not optimized as a train-time loss.
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "trc_final_30ep"
RUN_ROOT = ROOT / "runs"


SCENARIOS = {
    "motion": "motion_horizontal_medium",
    "defocus": "defocus_medium",
    "lowlight": "lowlight_medium",
    "mixed": "mixed_motion_lowlight",
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    train_root: Path
    val_root: Path
    test_root: Path
    run_dir: Path
    init_weights: Path
    detector_weights: Path
    yolo_prefix: str
    mixed_val_dir: str
    mixed_test_dir: str


DATASETS = [
    DatasetSpec(
        key="pothole",
        train_root=ROOT / "data" / "pothole_restoration",
        val_root=ROOT / "data" / "pothole_restoration_val",
        test_root=ROOT / "data" / "pothole_restoration_test",
        run_dir=RUN_ROOT / "trc_final_rmrnet_pothole_30ep",
        init_weights=RUN_ROOT / "fresh_final_rmr_base_pothole_mixed" / "rcadnet_best_loss.pth",
        detector_weights=RUN_ROOT / "detect" / "runs" / "yolo11s_v26" / "pothole_clean_80ep" / "weights" / "best.pt",
        yolo_prefix="pothole_yolo",
        mixed_val_dir="pothole_yolo_mixed_test_val",
        mixed_test_dir="pothole_yolo_mixed_test",
    ),
    DatasetSpec(
        key="pcm",
        train_root=ROOT / "data" / "pcm_restoration_train",
        val_root=ROOT / "data" / "pcm_restoration_val",
        test_root=ROOT / "data" / "pcm_restoration_test",
        run_dir=RUN_ROOT / "trc_final_rmrnet_pcm_30ep",
        init_weights=RUN_ROOT / "fresh_final_rmr_base_pcm_mixed" / "rcadnet_best_loss.pth",
        detector_weights=RUN_ROOT / "detect" / "runs" / "yolo11s_v26" / "pcm_clean_80ep" / "weights" / "best.pt",
        yolo_prefix="pcm_yolo",
        mixed_val_dir="pcm_yolo_mixed_val",
        mixed_test_dir="pcm_yolo_mixed_test",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dataset", action="append", choices=["pothole", "pcm"], help="Repeat to limit datasets.")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--force-restore", action="store_true", help="Restore splits even if data.yaml already exists.")
    parser.add_argument("--eval-batch", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def selected_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    if not args.dataset:
        return DATASETS
    wanted = set(args.dataset)
    return [spec for spec in DATASETS if spec.key in wanted]


def run(cmd: list[str], *, log_path: Path | None = None) -> None:
    printable = {"cmd": cmd}
    print(json.dumps(printable), flush=True)
    if log_path is None:
        subprocess.run(cmd, cwd=ROOT, check=True)
        return
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


def source_data_yaml(spec: DatasetSpec, scenario_key: str, split: str) -> Path:
    if scenario_key == "mixed":
        directory = spec.mixed_val_dir if split == "val" else spec.mixed_test_dir
    else:
        directory = f"{spec.yolo_prefix}_{scenario_key}_{split}"
    path = ROOT / "datasets" / directory / "data.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def restored_dir(spec: DatasetSpec, scenario_key: str, split: str, epoch: int) -> Path:
    return ROOT / "datasets" / f"{spec.yolo_prefix}_{scenario_key}_{split}_rmrnet_trc30_ep{epoch:03d}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_dataset(args: argparse.Namespace, spec: DatasetSpec) -> None:
    final_ckpt = spec.run_dir / f"rcadnet_epoch_{args.epochs:03d}.pth"
    if args.skip_train and final_ckpt.exists():
        print(json.dumps({"skip_train": spec.key, "checkpoint": str(final_ckpt)}), flush=True)
        return
    highest = highest_epoch(spec.run_dir)
    if final_ckpt.exists():
        print(json.dumps({"train_complete": spec.key, "checkpoint": str(final_ckpt)}), flush=True)
        return
    if highest > 0:
        init_weights = spec.run_dir / f"rcadnet_epoch_{highest:03d}.pth"
        out_dir = spec.run_dir.parent / f"{spec.run_dir.name}_resume_from{highest:03d}_to{args.epochs:03d}"
        epochs_to_run = args.epochs - highest
        log_name = f"train_{spec.key}_resume_{highest:03d}_to_{args.epochs:03d}.log"
    else:
        init_weights = spec.init_weights
        out_dir = spec.run_dir
        epochs_to_run = args.epochs
        log_name = f"train_{spec.key}_30ep.log"
    if not init_weights.exists():
        raise FileNotFoundError(init_weights)
    cmd = [
        sys.executable,
        "train_rmrnet.py",
        "--data-root",
        str(spec.train_root),
        "--epochs",
        str(epochs_to_run),
        "--batch-size",
        str(args.batch_size),
        "--patch-size",
        str(args.patch_size),
        "--lr",
        str(args.lr),
        "--width",
        str(args.width),
        "--device",
        args.device,
        "--out",
        str(out_dir),
        "--num-workers",
        "0",
        "--val-data-root",
        str(spec.val_root),
        "--val-every",
        "1",
        "--save-every-epoch",
        "--seed",
        str(args.seed),
        "--init-weights",
        str(init_weights),
        "--code-source",
        "metadata_fused",
        "--block-type",
        "evidence",
        "--attention-type",
        "task",
        "--conditioning",
        "gated_basis",
        "--basis-sparsity-weight",
        "1.0",
        "--aux-code-weight",
        "0.05",
        "--metadata-dropout",
        "0.10",
        "--metadata-noise",
        "0.01",
        "--detail-preserve",
        "--detail-gain",
        "0.12",
        "--edge-weight",
        "0.15",
        "--freq-weight",
        "0.05",
        "--defect-weight",
        "0.10",
        "--visibility-weight",
        "0.08",
        "--use-task-losses",
        "--tdp-yolo-weights",
        str(spec.detector_weights),
        "--tdp-layers",
        "2,4",
        "--tdp-layer-weights",
        "0.5,1",
        "--detector-input-size",
        "256",
        "--lambda-tdp",
        "0.001",
        "--tdp-defect-mask-weight",
        "4.0",
        "--tdp-defect-mask-power",
        "1.5",
        "--lambda-jacobian",
        "0.00002",
        "--jacobian-probes",
        "1",
        "--lambda-active-contour",
        "0",
        "--lambda-detector-input-anchor",
        "0.0005",
        "--lambda-evidence-nonregression",
        "0.02",
        "--lambda-detail-copy",
        "0.002",
        "--cqmix-prob",
        "0.5",
        "--task-loss-warmup-epochs",
        "3",
        "--grad-clip",
        "1.0",
        "--amp",
        "--debug-first-batches",
        "0",
    ]
    for scenario in SCENARIOS.values():
        cmd.extend(["--scenario", scenario, "--val-scenario", scenario])
    run(cmd, log_path=EXP / "logs" / log_name)
    if highest > 0:
        merge_resume_run(spec, resume_dir=out_dir, start_epoch=highest, final_epoch=args.epochs)


def highest_epoch(run_dir: Path) -> int:
    if not run_dir.exists():
        return 0
    epochs = []
    for path in run_dir.glob("rcadnet_epoch_*.pth"):
        try:
            epochs.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return max(epochs, default=0)


def merge_resume_run(spec: DatasetSpec, *, resume_dir: Path, start_epoch: int, final_epoch: int) -> None:
    """Copy resumed checkpoints into the main run with continuous epoch names."""

    copied: list[dict[str, object]] = []
    for local_epoch in range(1, final_epoch - start_epoch + 1):
        src = resume_dir / f"rcadnet_epoch_{local_epoch:03d}.pth"
        dst_epoch = start_epoch + local_epoch
        dst = spec.run_dir / f"rcadnet_epoch_{dst_epoch:03d}.pth"
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, dst)
        copied.append({"source": str(src), "destination": str(dst), "epoch": dst_epoch})
    for name in ["rcadnet_last.pth", "rcadnet_best.pth", "rcadnet_best_loss.pth", "rcadnet_best_psnr.pth"]:
        src = resume_dir / name
        if src.exists():
            shutil.copy2(src, spec.run_dir / name)
    append_adjusted_selection_history(spec.run_dir, resume_dir, start_epoch)
    manifest = {
        "dataset": spec.key,
        "resume_dir": str(resume_dir),
        "main_run_dir": str(spec.run_dir),
        "start_epoch": start_epoch,
        "final_epoch": final_epoch,
        "copied_checkpoints": copied,
        "note": "Optimizer state was restarted from the saved model checkpoint after an interrupted run.",
    }
    (spec.run_dir / f"resume_from_{start_epoch:03d}_to_{final_epoch:03d}.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def append_adjusted_selection_history(main_dir: Path, resume_dir: Path, start_epoch: int) -> None:
    src = resume_dir / "selection_history.csv"
    dst = main_dir / "selection_history.csv"
    if not src.exists() or not dst.exists():
        return
    src_rows = read_csv(src)
    if not src_rows:
        return
    adjusted: list[dict[str, object]] = []
    for row in src_rows:
        new_row: dict[str, object] = dict(row)
        if "epoch" in new_row:
            new_row["epoch"] = int(float(str(new_row["epoch"]))) + start_epoch
        adjusted.append(new_row)
    existing = read_csv(dst)
    fieldnames = list(existing[0].keys()) if existing else list(adjusted[0].keys())
    with dst.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        for row in adjusted:
            writer.writerow(row)


def restore_split(args: argparse.Namespace, spec: DatasetSpec, scenario_key: str, split: str, epoch: int) -> Path:
    out = restored_dir(spec, scenario_key, split, epoch)
    yaml_path = out / "data.yaml"
    if yaml_path.exists() and not args.force_restore:
        return yaml_path
    ckpt = spec.run_dir / f"rcadnet_epoch_{epoch:03d}.pth"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    run(
        [
            sys.executable,
            "tools/restore_yolo_split.py",
            "--data",
            str(source_data_yaml(spec, scenario_key, split)),
            "--split",
            split,
            "--model",
            "rcadnet",
            "--scenario",
            SCENARIOS[scenario_key],
            "--out",
            str(out),
            "--device",
            args.device,
            "--rcadnet-weights",
            str(ckpt),
            "--rcadnet-code-source",
            "metadata",
            "--gate-threshold",
            "-1",
            "--residual-strength",
            "1.0",
        ],
        log_path=EXP / "logs" / f"restore_{spec.key}_{scenario_key}_{split}_ep{epoch:03d}.log",
    )
    return yaml_path


def eval_items(args: argparse.Namespace, spec: DatasetSpec, items: list[tuple[str, Path]], out_stem: Path, split: str) -> Path:
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
        split,
        "--project",
        str(ROOT / "runs" / "detection_eval_trc_final_30ep"),
        "--out",
        str(out_stem),
    ]
    for name, path in items:
        cmd.extend(["--item", f"{name}={path}"])
    run(cmd, log_path=EXP / "logs" / f"eval_{out_stem.name}.log")
    return out_stem.with_suffix(".csv")


def eval_per_class(args: argparse.Namespace, spec: DatasetSpec, items: list[tuple[str, Path]], out_stem: Path, split: str) -> Path:
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
        split,
        "--project",
        str(ROOT / "runs" / "detection_eval_trc_final_30ep_per_class"),
        "--out",
        str(out_stem),
    ]
    for name, path in items:
        cmd.extend(["--item", f"{name}={path}"])
    run(cmd, log_path=EXP / "logs" / f"per_class_{out_stem.name}.log")
    return out_stem.with_suffix(".csv")


def select_epoch(spec: DatasetSpec, val_csv: Path) -> int:
    rows = read_csv(val_csv)
    per_epoch: dict[int, list[float]] = {}
    detail_rows: list[dict[str, object]] = []
    for row in rows:
        name = row["name"]
        scenario = name.split("_rmr_ep")[0]
        epoch = int(name.split("ep")[-1])
        score = float(row["map50"])
        per_epoch.setdefault(epoch, []).append(score)
        detail_rows.append({"dataset": spec.key, "epoch": epoch, "scenario": scenario, "val_map50": score})
    summary_rows: list[dict[str, object]] = []
    for epoch in sorted(per_epoch):
        scores = per_epoch[epoch]
        summary_rows.append(
            {
                "dataset": spec.key,
                "epoch": epoch,
                "mean_val_map50": sum(scores) / len(scores),
                "num_scenarios": len(scores),
            }
        )
    best = max(summary_rows, key=lambda row: (float(row["mean_val_map50"]), -int(row["epoch"])))
    write_csv(EXP / f"{spec.key}_val_selection_detail.csv", detail_rows)
    write_csv(EXP / f"{spec.key}_val_selection_summary.csv", summary_rows)
    (EXP / f"{spec.key}_best_by_val_map.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    return int(best["epoch"])


def evaluate_dataset(args: argparse.Namespace, spec: DatasetSpec) -> dict[str, object]:
    val_items: list[tuple[str, Path]] = []
    for epoch in range(1, args.epochs + 1):
        for scenario_key in SCENARIOS:
            val_items.append(
                (
                    f"{scenario_key}_rmr_ep{epoch:03d}",
                    restore_split(args, spec, scenario_key, "val", epoch),
                )
            )
    val_csv = eval_items(args, spec, val_items, EXP / f"{spec.key}_val_rmrnet_epochs", "val")
    selected_epoch = select_epoch(spec, val_csv)

    test_items: list[tuple[str, Path]] = []
    for scenario_key in SCENARIOS:
        test_items.append(
            (
                f"{scenario_key}_rmr_selected",
                restore_split(args, spec, scenario_key, "test", selected_epoch),
            )
        )
    test_csv = eval_items(args, spec, test_items, EXP / f"{spec.key}_test_rmrnet_selected", "test")
    per_class_csv = eval_per_class(args, spec, test_items, EXP / f"{spec.key}_test_rmrnet_selected_per_class", "test")
    return {
        "dataset": spec.key,
        "epochs": args.epochs,
        "selected_epoch": selected_epoch,
        "checkpoint": str(spec.run_dir / f"rcadnet_epoch_{selected_epoch:03d}.pth"),
        "validation_csv": str(val_csv),
        "test_csv": str(test_csv),
        "per_class_csv": str(per_class_csv),
        "active_contour_loss": 0.0,
        "selection_rule": "mean validation mAP50 across motion, defocus, low light, and mixed splits",
    }


def benchmark_restoration(args: argparse.Namespace, spec: DatasetSpec, selected_epoch: int) -> Path:
    ckpt = spec.run_dir / f"rcadnet_epoch_{selected_epoch:03d}.pth"
    out = ROOT / "runs" / f"bench_trc_final_rmrnet_{spec.key}_30ep"
    cmd = [
        sys.executable,
        "benchmark_unified_restoration.py",
        "--data-root",
        str(spec.test_root),
        "--model",
        "rcadnet",
        "--rcadnet-weights",
        str(ckpt),
        "--rcadnet-code-source",
        "metadata",
        "--device",
        args.device,
        "--max-side",
        "0",
        "--warmup",
        "2",
        "--out",
        str(out),
    ]
    for scenario in ("motion_horizontal_medium", "defocus_medium", "lowlight_medium"):
        cmd.extend(["--scenario", scenario])
    run(cmd, log_path=EXP / "logs" / f"bench_restoration_{spec.key}.log")
    return out / "metrics.csv"


def main() -> None:
    args = parse_args()
    EXP.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for spec in selected_specs(args):
        if not args.eval_only and not args.train_only:
            train_dataset(args, spec)
        elif not args.eval_only:
            train_dataset(args, spec)
        if args.train_only:
            continue
        result = evaluate_dataset(args, spec)
        result["restoration_metrics_csv"] = str(benchmark_restoration(args, spec, int(result["selected_epoch"])))
        manifest.append(result)
    (EXP / "trc_final_30ep_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": manifest}, indent=2), flush=True)


if __name__ == "__main__":
    main()
