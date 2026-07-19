# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

"""Profile the exact ILX-RD46 detector and assemble end-to-end field latency.

The detector is timed with the same single-full-frame 1280-pixel protocol used
for the field tables. Restoration time is read from the native-resolution tiled
restore manifest produced by ``run_ilx_latest_rmr30_audit.py``. The resulting
table separates one-view and two-view detector costs; no Jetson claim is made.
"""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "experiments" / "gt46_yolo26_coordinate_revised" / "gt46_native_images"
WEIGHTS = ROOT / "Yolo26_coordinate" / "YOLO26s_RDD_FRDC_Distilled_v2.pt"
RESTORE_SUMMARY = (
    ROOT
    / "experiments"
    / "roboflow_geotagged_v5_native_real"
    / "v47_rmr30_pcm_ep028_eval_sets"
    / "restore_summary.csv"
)
OUT = ROOT / "experiments" / "major_revision_complexity_20260715" / "ilx_field_runtime.json"
TABLE = ROOT / "paper_automation_in_construction_rmrnet" / "tables" / "table_field_system_runtime.tex"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    return parser.parse_args()


def synchronize(device: str) -> None:
    if torch.cuda.is_available() and str(device).lower() != "cpu":
        torch.cuda.synchronize()


def restoration_means() -> dict[str, float]:
    with RESTORE_SUMMARY.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    means: dict[str, float] = {}
    for model in ("rmr_blind", "rmr_metadata"):
        values = [float(row["runtime_s"]) * 1000.0 for row in rows if row["model"] == model and float(row["runtime_s"]) > 0]
        if not values:
            raise RuntimeError(f"No measured native restoration rows for {model}")
        means[model] = statistics.mean(values)
    return means


def detector_profile(device: str, imgsz: int) -> dict[str, float]:
    paths = sorted(IMAGES.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(IMAGES)
    model = YOLO(str(WEIGHTS))
    for path in paths[:2]:
        model.predict(source=str(path), imgsz=imgsz, conf=0.10, device=device, verbose=False)
    synchronize(device)
    if torch.cuda.is_available() and str(device).lower() != "cpu":
        torch.cuda.reset_peak_memory_stats()

    wall_ms: list[float] = []
    engine_ms: list[float] = []
    for path in paths:
        synchronize(device)
        started = time.perf_counter()
        result = model.predict(source=str(path), imgsz=imgsz, conf=0.10, device=device, verbose=False)[0]
        synchronize(device)
        wall_ms.append((time.perf_counter() - started) * 1000.0)
        speed = result.speed
        engine_ms.append(float(speed.get("preprocess", 0.0)) + float(speed.get("inference", 0.0)) + float(speed.get("postprocess", 0.0)))
    peak = torch.cuda.max_memory_allocated() / (1024.0**2) if torch.cuda.is_available() and str(device).lower() != "cpu" else 0.0
    return {
        "images": len(paths),
        "wall_mean_ms": statistics.mean(wall_ms),
        "wall_median_ms": statistics.median(wall_ms),
        "engine_mean_ms": statistics.mean(engine_ms),
        "peak_gpu_memory_mb": peak,
    }


def write_table(restoration: dict[str, float], detector: dict[str, float]) -> None:
    rest = restoration["rmr_metadata"]
    det = detector["wall_mean_ms"]
    rows = [
        ("Raw detector", 0.0, 1, det),
        ("RMR-Net single view", rest, 1, rest + det),
        ("Raw + RMR-Net two view", rest, 2, rest + 2.0 * det),
    ]
    body = [f"{name} & {restore:.1f} & {passes} & {total:.1f} \\\\" for name, restore, passes, total in rows]
    TABLE.write_text(
        "\n".join(
            [
                r"\begin{table}[!t]",
                r"\centering",
                r"\caption{ILX-RD46 field-system latency on the RTX 3050 workstation. Restoration preserves 4752$\times$3168 pixels using 768-pixel overlapping tiles; detection uses one full-frame input with internal resizing to 1280 pixels. Times include image loading and post-processing.}",
                r"\label{tab:field_system_runtime}",
                r"\scriptsize",
                r"\begin{tabular}{lrrr}",
                r"\toprule",
                r"Policy & Restoration ms & Detector passes & Total ms/image \\",
                r"\midrule",
                *body,
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    restoration = restoration_means()
    detector = detector_profile(args.device, args.imgsz)
    payload = {
        "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "detector_weights": str(WEIGHTS.relative_to(ROOT)),
        "detector_imgsz": args.imgsz,
        "detector_protocol": "single full frame; end-to-end one-to-one head; no NMS, crop, mask, or detector tiling",
        "restoration_tiling": "768-pixel tiles, 96-pixel overlap, native-size output",
        "restoration_mean_ms": restoration,
        "detector": detector,
        "dual_evidence_total_mean_ms": restoration["rmr_metadata"] + 2.0 * detector["wall_mean_ms"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_table(restoration, detector)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
