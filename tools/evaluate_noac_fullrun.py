from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "no_active_contour_fullrun"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    run_dir: Path
    yolo_weights: Path
    split_prefix: str
    epochs: tuple[int, ...]


SCENARIOS = {
    "motion": "motion_horizontal_medium",
    "defocus": "defocus_medium",
    "lowlight": "lowlight_medium",
    "mixed": "mixed_motion_lowlight",
}

DATASETS = [
    DatasetSpec(
        key="pothole",
        run_dir=ROOT / "runs" / "rmrnet_noac_pothole_yolo11s",
        yolo_weights=ROOT / "runs" / "detect" / "runs" / "yolo11s_v26" / "pothole_clean_80ep" / "weights" / "best.pt",
        split_prefix="pothole_yolo",
        epochs=(1, 2, 3, 4),
    ),
    DatasetSpec(
        key="pcm",
        run_dir=ROOT / "runs" / "rmrnet_noac_pcm_yolo11s",
        yolo_weights=ROOT / "runs" / "detect" / "runs" / "yolo11s_v26" / "pcm_clean_80ep" / "weights" / "best.pt",
        split_prefix="pcm_yolo",
        epochs=(1, 2, 3, 4),
    ),
]


SPLIT_OVERRIDES = {
    # The mixed pothole validation split was generated earlier from the test
    # split family, so its directory name keeps the extra "_test" token.
    ("pothole", "mixed", "val"): "pothole_yolo_mixed_test_val",
}


def run(cmd: list[str]) -> None:
    print(json.dumps({"cmd": cmd}), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def data_yaml(spec: DatasetSpec, scenario_key: str, split: str) -> Path:
    dataset_name = SPLIT_OVERRIDES.get(
        (spec.key, scenario_key, split),
        f"{spec.split_prefix}_{scenario_key}_{split}",
    )
    path = ROOT / "datasets" / dataset_name / "data.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def restored_dir(spec: DatasetSpec, scenario_key: str, split: str, epoch: int) -> Path:
    return ROOT / "datasets" / f"{spec.split_prefix}_{scenario_key}_{split}_rmrnet_noac_ep{epoch:03d}"


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
            str(data_yaml(spec, scenario_key, split)),
            "--split",
            "val" if split == "val" else "test",
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
        "test" if split == "test" else "val",
        "--project",
        str(ROOT / "runs" / "detection_eval_noac_fullrun"),
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def select_epoch(spec: DatasetSpec, val_csv: Path) -> int:
    rows = read_rows(val_csv)
    by_epoch: dict[int, list[float]] = {}
    detail = []
    for row in rows:
        epoch = int(row["name"].split("ep")[-1])
        scenario = row["name"].split("_noac")[0]
        value = float(row["map50"])
        by_epoch.setdefault(epoch, []).append(value)
        detail.append({"dataset": spec.key, "scenario": scenario, "epoch": epoch, "val_map50": value})
    summary = [
        {"dataset": spec.key, "epoch": epoch, "mean_val_map50": sum(values) / len(values), "num_scenarios": len(values)}
        for epoch, values in sorted(by_epoch.items())
    ]
    best = max(summary, key=lambda row: (float(row["mean_val_map50"]), -int(row["epoch"])))
    write_rows(EXP / f"{spec.key}_selection_detail.csv", detail)
    write_rows(EXP / f"{spec.key}_selection_summary.csv", summary)
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
                val_items.append((f"{scenario_key}_noac_ep{epoch:03d}", yaml_path))
        val_csv = eval_items(spec, val_items, EXP / f"{spec.key}_val_noac_epochs", "val")
        best_epoch = select_epoch(spec, val_csv)

        test_items = []
        for scenario_key in SCENARIOS:
            yaml_path = restore_split(spec, scenario_key, "test", best_epoch)
            test_items.append((f"{scenario_key}_rmrnet_noac", yaml_path))
        test_csv = eval_items(spec, test_items, EXP / f"{spec.key}_test_noac_selected", "test")
        manifest.append(
            {
                "dataset": spec.key,
                "selected_epoch": best_epoch,
                "checkpoint": str(spec.run_dir / f"rcadnet_epoch_{best_epoch:03d}.pth"),
                "validation_csv": str(val_csv),
                "test_csv": str(test_csv),
                "active_contour_loss": 0.0,
            }
        )
    (EXP / "NOAC_FULLRUN_SELECTION_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": manifest}, indent=2), flush=True)


if __name__ == "__main__":
    main()
