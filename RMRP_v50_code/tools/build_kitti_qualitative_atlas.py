from __future__ import annotations

"""Build a metadata-selected KITTI restoration atlas.

Frames are selected from raw OXTS motion magnitude only, before restoration
outputs or clean targets are scored. This makes the qualitative selection
independent of model error while still showing informative high-motion cases.
"""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
import torch
from torchvision.transforms import functional as TF


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.demoe_adapter import DeMoEAdapter
from benchmark_unified_restoration import (
    load_image,
    load_rcadnet,
    pad_to_multiple,
    run_rcadnet,
    unpad,
)
from rcadnet.dataset import metadata_for_mode
from rcadnet.practical_metadata import GYRO_END, sensor_packet_from_mapping


SCENARIO = "kitti_oxts_physical_motion"
DATA = ROOT / "data" / "kitti_oxts_sensor_fused_test"
RMR_WEIGHTS = (
    ROOT
    / "runs"
    / "kitti_sensor_fused_rmr_20260730_raw_telemetry_30ep"
    / "rcadnet_best_psnr.pth"
)
DEMOE_WEIGHTS = ROOT / "weights" / "demoe" / "DeMoE.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--rmr-weights", type=Path, default=RMR_WEIGHTS)
    parser.add_argument("--demoe-weights", type=Path, default=DEMOE_WEIGHTS)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "paper_automation_in_construction_rmrnet" / "figures" / "fig_kitti_qualitative_atlas.png",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quantiles", default="0.50,0.75,0.95")
    return parser.parse_args()


def raw_motion_score(metadata_path: Path) -> float:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    filtered = metadata_for_mode(metadata, "raw_telemetry")
    packet = sensor_packet_from_mapping(filtered).numpy()
    gyro = packet[:GYRO_END].reshape(-1, 3)
    return float(np.sqrt(np.square(gyro[:, :2]).sum(axis=1)).mean())


def select_frames(metadata_dir: Path, quantiles: list[float]) -> list[dict[str, object]]:
    candidates = [
        {"stem": path.stem, "metadata": path, "motion_score": raw_motion_score(path)}
        for path in sorted(metadata_dir.glob("*.json"))
    ]
    if not candidates:
        raise FileNotFoundError(metadata_dir)
    ordered = sorted(candidates, key=lambda row: float(row["motion_score"]))
    selected: list[dict[str, object]] = []
    used: set[str] = set()
    for quantile in quantiles:
        index = int(round(float(quantile) * (len(ordered) - 1)))
        while index < len(ordered) and str(ordered[index]["stem"]) in used:
            index += 1
        if index >= len(ordered):
            index = next(
                candidate_index
                for candidate_index in range(len(ordered) - 1, -1, -1)
                if str(ordered[candidate_index]["stem"]) not in used
            )
        row = dict(ordered[index])
        row["requested_motion_quantile"] = quantile
        row["rank"] = index + 1
        row["population"] = len(ordered)
        selected.append(row)
        used.add(str(row["stem"]))
    return selected


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.squeeze(0).detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scenario_root = args.data_root.resolve() / "scenarios" / SCENARIO
    input_dir = scenario_root / "input"
    gt_dir = scenario_root / "gt"
    metadata_dir = scenario_root / "metadata"
    quantiles = [float(value) for value in args.quantiles.split(",")]
    selected = select_frames(metadata_dir, quantiles)

    rmr = load_rcadnet(str(args.rmr_weights.resolve()), device)
    demoe = DeMoEAdapter(
        args.demoe_weights.resolve(),
        device=device,
        task="auto",
        smoke=False,
        strict=True,
    )
    rows: list[dict[str, object]] = []
    panels: list[list[np.ndarray]] = []
    with torch.inference_mode():
        for selection in selected:
            stem = str(selection["stem"])
            input_path = next(input_dir.glob(f"{stem}.*"))
            gt_path = next(gt_dir.glob(f"{stem}.*"))
            degraded = load_image(input_path, 0)
            clean = load_image(gt_path, 0)
            padded, size = pad_to_multiple(degraded)
            clean_padded, _ = pad_to_multiple(clean)
            restored_rmr = run_rcadnet(
                rmr,
                padded,
                SCENARIO,
                device,
                "metadata",
                "raw_telemetry",
                input_path,
            )
            restored_demoe = demoe(
                padded.to(device),
                scenario=SCENARIO,
                task="auto",
            )
            degraded = unpad(padded.cpu(), size)
            clean = unpad(clean_padded.cpu(), size)
            restored_rmr = unpad(restored_rmr.cpu(), size)
            restored_demoe = unpad(restored_demoe.cpu(), size)
            arrays = [
                to_numpy(degraded),
                to_numpy(restored_demoe),
                to_numpy(restored_rmr),
                to_numpy(clean),
            ]
            psnr = [
                peak_signal_noise_ratio(arrays[-1], image, data_range=1.0)
                for image in arrays[:-1]
            ]
            panels.append(arrays)
            rows.append(
                {
                    **{key: value for key, value in selection.items() if key != "metadata"},
                    "input_file": str(input_path.relative_to(ROOT)),
                    "clean_file": str(gt_path.relative_to(ROOT)),
                    "selection_rule": "nearest requested quantile of raw OXTS gyro magnitude",
                    "degraded_psnr": float(psnr[0]),
                    "demoe_psnr": float(psnr[1]),
                    "rmr_raw_oxts_psnr": float(psnr[2]),
                }
            )

    titles = ["Degraded", "DeMoE", "RMR-P + raw OXTS", "Clean target"]
    # KITTI frames are wide. A compact canvas keeps each motion-quantile row
    # visually adjacent instead of leaving misleading bands of white space.
    figure, axes = plt.subplots(len(rows), 4, figsize=(11.4, 3.45), squeeze=False)
    for row_index, (row, images) in enumerate(zip(rows, panels)):
        for column_index, image in enumerate(images):
            axis = axes[row_index, column_index]
            axis.imshow(image)
            axis.axis("off")
            if row_index == 0:
                axis.set_title(titles[column_index], fontsize=10, fontweight="bold")
            if column_index < 3:
                key = ("degraded_psnr", "demoe_psnr", "rmr_raw_oxts_psnr")[column_index]
                axis.text(
                    0.02,
                    0.96,
                    f"{float(row[key]):.2f} dB",
                    transform=axis.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                    color="white",
                    bbox={"facecolor": "black", "alpha": 0.70, "pad": 2, "edgecolor": "none"},
                )
        axes[row_index, 0].text(
            -0.03,
            0.5,
            f"OXTS q={float(row['requested_motion_quantile']):.2f}",
            transform=axes[row_index, 0].transAxes,
            va="center",
            ha="right",
            rotation=90,
            fontsize=8,
        )
    figure.subplots_adjust(left=0.065, right=0.995, top=0.90, bottom=0.01, wspace=0.02, hspace=0.015)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    manifest = {
        "figure": str(args.out.resolve()),
        "selection_before_output_scoring": True,
        "selection_uses_ground_truth_or_restoration_error": False,
        "selection_rule": "nearest frames to raw OXTS motion quantiles",
        "quantiles": quantiles,
        "rmr_checkpoint": str(args.rmr_weights.resolve()),
        "demoe_checkpoint": str(args.demoe_weights.resolve()),
        "rows": rows,
    }
    manifest_path = args.out.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
