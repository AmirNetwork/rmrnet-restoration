# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "fresh_final_selection"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    run_dir: Path
    yolo_weights: Path
    yolo_prefix: str
    mixed_val_dir: str
    mixed_test_dir: str
    epochs: tuple[int, ...] = (1, 2, 3)


SCENARIOS = {
    "motion": "motion_horizontal_medium",
    "defocus": "defocus_medium",
    "lowlight": "lowlight_medium",
    "mixed": "mixed_motion_lowlight",
}


DATASETS = [
    DatasetSpec(
        key="pothole",
        run_dir=ROOT / "runs" / "fresh_final_rmr_task_pothole_mixed",
        yolo_weights=ROOT / "runs" / "detect" / "runs" / "yolo11s_v26" / "pothole_clean_80ep" / "weights" / "best.pt",
        yolo_prefix="pothole_yolo",
        mixed_val_dir="pothole_yolo_mixed_test_val",
        mixed_test_dir="pothole_yolo_mixed_test",
    ),
    DatasetSpec(
        key="pcm",
        run_dir=ROOT / "runs" / "fresh_final_rmr_task_pcm_mixed",
        yolo_weights=ROOT / "runs" / "detect" / "runs" / "yolo11s_v26" / "pcm_clean_80ep" / "weights" / "best.pt",
        yolo_prefix="pcm_yolo",
        mixed_val_dir="pcm_yolo_mixed_val",
        mixed_test_dir="pcm_yolo_mixed_test",
    ),
]


def run(cmd: list[str]) -> None:
    print(json.dumps({"cmd": cmd}), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def source_data_yaml(spec: DatasetSpec, scenario_key: str, split: str) -> Path:
    if scenario_key == "mixed":
        directory = spec.mixed_val_dir if split == "val" else spec.mixed_test_dir
    else:
        directory = f"{spec.yolo_prefix}_{scenario_key}_{split}"
    return ROOT / "datasets" / directory / "data.yaml"


def restored_dir(spec: DatasetSpec, scenario_key: str, split: str, epoch: int) -> Path:
    return ROOT / "datasets" / f"{spec.yolo_prefix}_{scenario_key}_{split}_rmrnet_fresh_final_ep{epoch:03d}"


def restore_split(spec: DatasetSpec, scenario_key: str, split: str, epoch: int) -> Path:
    out = restored_dir(spec, scenario_key, split, epoch)
    yaml_path = out / "data.yaml"
    if yaml_path.exists():
        return yaml_path

    checkpoint = spec.run_dir / f"rcadnet_epoch_{epoch:03d}.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

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
            "cuda",
            "--rcadnet-weights",
            str(checkpoint),
            "--rcadnet-code-source",
            "metadata",
            "--gate-threshold",
            "-1",
            "--residual-strength",
            "1.0",
        ]
    )
    return yaml_path


def eval_items(spec: DatasetSpec, items: list[tuple[str, Path]], out_stem: Path, split: str) -> Path:
    cmd = [
        sys.executable,
        "tools/eval_yolo_suite.py",
        "--weights",
        str(spec.yolo_weights),
        "--imgsz",
        "640",
        "--batch",
        "8",
        "--device",
        "0",
        "--workers",
        "0",
        "--split",
        split,
        "--project",
        str(ROOT / "runs" / "detection_eval_fresh_final_selection"),
        "--out",
        str(out_stem),
    ]
    for name, path in items:
        cmd.extend(["--item", f"{name}={path}"])
    run(cmd)
    return out_stem.with_suffix(".csv")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def select_epoch(spec: DatasetSpec, val_csv: Path) -> int:
    rows = read_rows(val_csv)
    per_epoch: dict[int, list[float]] = {}
    detail_rows: list[dict[str, object]] = []
    for row in rows:
        parts = row["name"].split("_")
        scenario = parts[0]
        epoch = int(parts[-1].replace("ep", ""))
        map50 = float(row["map50"])
        per_epoch.setdefault(epoch, []).append(map50)
        detail_rows.append({"dataset": spec.key, "epoch": epoch, "scenario": scenario, "val_map50": map50})

    summary_rows = []
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
    write_rows(EXP / f"{spec.key}_val_selection_detail.csv", detail_rows)
    write_rows(EXP / f"{spec.key}_val_selection_summary.csv", summary_rows)
    (EXP / f"{spec.key}_best_by_val_map.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    return int(best["epoch"])


def main() -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    manifest = []
    for spec in DATASETS:
        val_items = []
        for epoch in spec.epochs:
            for scenario_key in SCENARIOS:
                yaml_path = restore_split(spec, scenario_key, "val", epoch)
                val_items.append((f"{scenario_key}_rmr_ep{epoch:03d}", yaml_path))
        val_csv = eval_items(spec, val_items, EXP / f"{spec.key}_val_rmrnet_epochs", "val")
        selected_epoch = select_epoch(spec, val_csv)

        test_items = []
        for scenario_key in SCENARIOS:
            yaml_path = restore_split(spec, scenario_key, "test", selected_epoch)
            test_items.append((f"{scenario_key}_rmr_selected", yaml_path))
        test_csv = eval_items(spec, test_items, EXP / f"{spec.key}_test_rmrnet_selected", "test")
        manifest.append(
            {
                "dataset": spec.key,
                "selected_epoch": selected_epoch,
                "checkpoint": str(spec.run_dir / f"rcadnet_epoch_{selected_epoch:03d}.pth"),
                "validation_csv": str(val_csv),
                "test_csv": str(test_csv),
            }
        )
    (EXP / "selection_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": manifest}, indent=2), flush=True)


if __name__ == "__main__":
    main()
