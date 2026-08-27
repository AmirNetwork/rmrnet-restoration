from __future__ import annotations

"""Run the sequence-disjoint CRID-46 detector/restorer comparison.

This evaluator supports a validation-first temporal CRID comparison. Detector,
metadata, split, and output roots are explicit command-line inputs. It freezes
the TRACE-R residual strength before the evaluation partition is opened:

1. Run without ``--run-test`` to evaluate candidate residual strengths on the
   validation sequence and write ``frozen_selection_before_test.json``.
2. Run with ``--run-test`` to apply that frozen policy once to the held-out
   sequence and compare every restoration method under the same detector.

Only compact native-coordinate CSV predictions are written.  No input image is
resized or copied, and test annotations are not loaded during validation.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.eval_yolo26_coordinate_gt46 import (  # noqa: E402
    corpus_ap,
    evaluate_one,
    primary,
)
from rcadnet.practical_metadata import (  # noqa: E402
    ACCEL_END,
    CONTEXT_START,
)
from tools.restore_native_yolo_split import (  # noqa: E402
    load_rcadnet,
    native_partial_sensor_packet,
    restore_tiled,
)


SPLIT_ROOT = ROOT / "datasets" / "gt46_sony_external_replay_20260803"
SPLIT_LABEL_ROOT = ROOT / "datasets" / "gt46_sony_classbalanced_20260801"
NATIVE = SPLIT_LABEL_ROOT / "images"
DETECTOR = (
    ROOT
    / "runs"
    / "detect"
    / "runs"
    / "detect"
    / "yolo26_sony_replay_v3_20260803"
    / "stage_b_full"
    / "weights"
    / "best.pt"
)
CURRENT_RMR = ROOT / "experiments" / "ilx46_ins_synchronized_rmr_20260801" / "views"
BASELINES = (
    ROOT
    / "experiments"
    / "final20260810_crid46_native_restoration"
)
DEFAULT_OUT = ROOT / "experiments" / "crid46_sequence_disjoint_system_20260803"
DEFAULT_CURRENT_CHECKPOINT = (
    ROOT
    / "runs"
    / "rmrnet_pcm_calibrated_v2_lowlight_mixed_full_v2_20260728"
    / "rcadnet_epoch_001.pth"
)
METADATA_ROOT = (
    ROOT
    / "experiments"
    / "ilx46_ins_synchronized_rmr_20260801"
    / "native_exact"
    / "metadata"
    / "test"
)

# I_o = I_d + eta (I_r - I_d), with eta selected on validation only.
# The fine grid is intentionally one-dimensional: it calibrates the bounded
# residual already defined by TRACE-R instead of adding field-specific image
# processing or another trainable component.
RMR_CANDIDATES = {
    f"rmr_fine_eta{str(eta).replace('.', 'p')}": eta
    for eta in (0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75, 1.00)
}
COMPARATORS = {
    "raw": NATIVE,
    "nafnet": BASELINES / "nafnet" / "images" / "test",
    "dfpir": BASELINES / "dfpir" / "images" / "test",
    "demoe_auto": BASELINES / "demoe_auto" / "images" / "test",
    "instructir": BASELINES / "instructir_generic" / "images" / "test",
}
CLASS_NAMES = {
    0: "Longitudinal Crack (D00)",
    1: "Transverse Crack (D10)",
    2: "Alligator Crack (D20)",
    3: "Pothole (D40)",
}
PRED_TO_GT = {
    "Longitudinal Crack (D00)": "Longitudinal Crack",
    "Transverse Crack (D10)": "Transverse Crack",
    "Alligator Crack (D20)": "Alligator Crack",
    "Pothole (D40)": "Potholes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--detector", type=Path, default=DETECTOR)
    parser.add_argument(
        "--split-root",
        type=Path,
        default=SPLIT_ROOT,
        help="Directory containing val.txt and test.txt split manifests.",
    )
    parser.add_argument(
        "--label-root",
        type=Path,
        default=SPLIT_LABEL_ROOT,
        help="Native image/YOLO-label root used only for the active split.",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=BASELINES,
        help="Root containing the native-resolution comparator outputs.",
    )
    parser.add_argument(
        "--detector-provenance",
        default="Sony detector selected on the sequence-disjoint validation split",
        help="Human-readable detector provenance saved in the frozen manifest.",
    )
    parser.add_argument(
        "--native-root",
        type=Path,
        default=NATIVE,
        help="Authoritative native-resolution CRID image directory.",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--rmr-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=METADATA_ROOT,
        help="Per-frame practical sensor sidecars used by TRACE-R.",
    )
    parser.add_argument(
        "--rmr-metadata-mode",
        choices=("provided", "camera_only", "inertial_only", "unavailable"),
        default="provided",
        help=(
            "Use the full audited packet, a declared modality subset, or zero "
            "every channel for the metadata-unavailable validation control."
        ),
    )
    parser.add_argument(
        "--rmr-full-dir",
        type=Path,
        default=None,
        help="Existing native-resolution full-strength RMR outputs for the active split.",
    )
    parser.add_argument(
        "--generate-rmr",
        action="store_true",
        help="Restore only the active split with --rmr-checkpoint before detector evaluation.",
    )
    parser.add_argument("--tile", type=int, default=768)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument(
        "--eta",
        action="append",
        type=float,
        help=(
            "Validation-only residual strength. Repeat to screen a compact "
            "grid; the default evaluates the full preregistered grid."
        ),
    )
    parser.add_argument(
        "--rmr-only",
        action="store_true",
        help="Screen only RMR residual views; final selection should omit this flag.",
    )
    parser.add_argument("--run-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance_path(path: Path) -> str:
    """Prefer repository-relative paths, retaining valid external audit roots."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def split_names(split: str, split_root: Path = SPLIT_ROOT) -> list[str]:
    rows = (split_root / f"{split}.txt").read_text(encoding="utf-8").splitlines()
    return [Path(row.strip()).name for row in rows if row.strip()]


def resolve_sources(folder: Path, names: list[str]) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(folder)
    by_stem = {
        path.stem: path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    missing = [name for name in names if Path(name).stem not in by_stem]
    if missing:
        raise FileNotFoundError(f"{folder} is missing {missing}")
    return [by_stem[Path(name).stem] for name in names]


@torch.inference_mode()
def restore_current_rmr(
    names: list[str],
    checkpoint: Path,
    output: Path,
    *,
    device_name: str,
    tile: int,
    overlap: int,
    metadata_root: Path,
    metadata_mode: str,
    native_root: Path,
    force: bool,
) -> None:
    """Restore the active split with the current practical-sensor model."""

    output.mkdir(parents=True, exist_ok=True)
    expected = {Path(name).stem for name in names}
    present = {path.stem for path in output.glob("*.jpg")}
    if present == expected and not force:
        return
    device = torch.device(
        f"cuda:{device_name}"
        if torch.cuda.is_available() and str(device_name).isdigit()
        else device_name
    )
    model = load_rcadnet(checkpoint, device)
    for name in names:
        source = resolve_sources(native_root, [name])[0]
        target = output / f"{Path(name).stem}.jpg"
        if target.exists() and not force:
            continue
        metadata_path = metadata_root / f"{Path(name).stem}.json"
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        packet = native_partial_sensor_packet(metadata, device).unsqueeze(0)
        if metadata_mode == "unavailable":
            packet = torch.zeros_like(packet)
        elif metadata_mode == "camera_only":
            packet[:, :ACCEL_END] = 0.0
            context = packet[:, CONTEXT_START:]
            context[:, 8:12] = 0.0
            context[:, 13:15] = 0.0
            context[:, 15] = context[:, 12]
        elif metadata_mode == "inertial_only":
            context = packet[:, CONTEXT_START:]
            context[:, :8] = 0.0
            context[:, 12] = 0.0
            context[:, 15] = torch.maximum(context[:, 13], context[:, 14])
        with Image.open(source) as image:
            tensor = TF.to_tensor(image.convert("RGB")).unsqueeze(0).to(device)
        restored = restore_tiled(
            tensor,
            lambda patch: model(patch, packet),
            tile=tile,
            overlap=overlap,
        ).clamp(0.0, 1.0)
        TF.to_pil_image(restored[0].cpu()).save(target, quality=98, subsampling=0)
        print(f"restored current RMR: {name}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def subset_gt(
    names: list[str], label_root: Path | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Load only labels belonging to the active split.

    The normalized YOLO labels were exported from the revised COCO annotation
    file without changing the native 4752 x 3168 image geometry.  Reading one
    label file per requested image keeps the confirmatory test annotations
    physically unopened while validation policies are selected.
    """
    label_root = SPLIT_LABEL_ROOT if label_root is None else label_root
    gt: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []
    for name in names:
        image_candidates = [
            label_root / "images" / name,
            label_root / "images" / "test" / name,
        ]
        label_candidates = [
            label_root / "labels" / f"{Path(name).stem}.txt",
            label_root / "labels" / "test" / f"{Path(name).stem}.txt",
        ]
        image_path = next((path for path in image_candidates if path.exists()), image_candidates[0])
        label_path = next((path for path in label_candidates if path.exists()), label_candidates[0])
        if not image_path.exists() or not label_path.exists():
            missing.append(name)
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        items: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            fields = line.split()
            if len(fields) != 5:
                raise RuntimeError(
                    f"Expected a YOLO box at {label_path}:{line_number}; "
                    f"received {len(fields)} fields"
                )
            class_id_f, x_center, y_center, box_width, box_height = map(float, fields)
            class_id = int(class_id_f)
            if class_id not in CLASS_NAMES or class_id_f != class_id:
                raise RuntimeError(f"Unknown class id {class_id_f} at {label_path}:{line_number}")
            label = PRED_TO_GT[CLASS_NAMES[class_id]]
            x1 = (x_center - box_width / 2.0) * width
            y1 = (y_center - box_height / 2.0) * height
            x2 = (x_center + box_width / 2.0) * width
            y2 = (y_center + box_height / 2.0) * height
            items.append(
                {
                    "image": name,
                    "label": label,
                    "primary": primary(label),
                    "box": (x1, y1, x2, y2),
                }
            )
        gt[name] = items
    if missing:
        raise RuntimeError(f"Missing split-isolated image/label files for {missing}")
    return gt


def prediction_complete(path: Path, names: list[str]) -> bool:
    if not path.exists():
        return False
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    present = {Path(row["image"]).stem for row in rows}
    # Images with zero detections do not appear in the CSV.  The sidecar records
    # the exact inference corpus and is therefore the completion authority.
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return False
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    return record.get("images") == names and present <= {Path(name).stem for name in names}


def predict(
    model: YOLO,
    method: str,
    source: Path | float,
    names: list[str],
    output: Path,
    args: argparse.Namespace,
    rmr_full: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if prediction_complete(output, names) and not args.force:
        print(f"[SKIP] {method}: {output}", flush=True)
        return
    if isinstance(source, float):
        raw_paths = resolve_sources(args.native_root, names)
        restored_paths = resolve_sources(rmr_full, names)
        sources = []
        for raw_path, restored_path in zip(raw_paths, restored_paths):
            raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
            restored = cv2.imread(str(restored_path), cv2.IMREAD_COLOR)
            if raw is None or restored is None:
                raise RuntimeError(f"Could not read {raw_path} or {restored_path}")
            if raw.shape != restored.shape:
                raise AssertionError(f"Native dimensions changed for {raw_path.name}")
            sources.append(cv2.addWeighted(raw, 1.0 - source, restored, source, 0.0))
        source_record = {
            "raw": provenance_path(args.native_root),
            "restored": provenance_path(rmr_full),
            "residual_strength_eta": source,
        }
    else:
        sources = [str(path) for path in resolve_sources(source, names)]
        source_record = provenance_path(source)
    results = model.predict(
        source=sources,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        max_det=300,
        save=False,
        verbose=False,
        stream=False,
    )
    fields = ["image", "class_name", "confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for native_name, result in zip(names, results):
            boxes = result.boxes
            if boxes is None:
                continue
            xyxy = boxes.xyxy.detach().cpu().tolist()
            confidence = boxes.conf.detach().cpu().tolist()
            classes = boxes.cls.detach().cpu().tolist()
            for box, score, class_id in zip(xyxy, confidence, classes):
                class_id = int(class_id)
                if class_id not in CLASS_NAMES:
                    continue
                writer.writerow(
                    {
                        "image": native_name,
                        "class_name": CLASS_NAMES[class_id],
                        "confidence": float(score),
                        "bbox_x1": float(box[0]),
                        "bbox_y1": float(box[1]),
                        "bbox_x2": float(box[2]),
                        "bbox_y2": float(box[3]),
                    }
                )
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "method": method,
                "source": source_record,
                "images": names,
                "native_resolution_preserved": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_compact_predictions(path: Path) -> dict[str, list[dict[str, Any]]]:
    predictions: dict[str, list[dict[str, Any]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = PRED_TO_GT[row["class_name"]]
            item = {
                "image": row["image"],
                "label": label,
                "primary": primary(label),
                "conf": float(row["confidence"]),
                "box": tuple(float(row[key]) for key in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2")),
            }
            predictions.setdefault(row["image"], []).append(item)
    return predictions


def metrics(method: str, gt: dict[str, list[dict]], predictions: dict[str, list[dict]]) -> dict[str, Any]:
    summary, classes = evaluate_one(method, gt, predictions)
    row10 = next(row for row in summary if row["mode"] == "primary" and row["iou"] == 0.10)
    row50 = next(row for row in summary if row["mode"] == "primary" and row["iou"] == 0.50)
    coverage = next(row for row in summary if row["mode"] == "primary_success")
    return {
        "method": method,
        "images": int(row10["images"]),
        "gt": int(row10["gt"]),
        "pred": int(row10["pred"]),
        "ap10_primary": corpus_ap(gt, predictions, 0.10, "primary"),
        "ap50_primary": corpus_ap(gt, predictions, 0.50, "primary"),
        "f1_iou10": float(row10["f1"]),
        "f1_iou50": float(row50["f1"]),
        "coverage": float(coverage["recall"]),
        "class_metrics": classes,
    }


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["method", "images", "gt", "pred", "ap10_primary", "ap50_primary", "f1_iou10", "f1_iou50", "coverage"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)


def main() -> None:
    args = parse_args()
    args.out = args.out.resolve()
    args.detector = args.detector.resolve()
    args.split_root = args.split_root.resolve()
    args.label_root = args.label_root.resolve()
    args.baseline_root = args.baseline_root.resolve()
    args.native_root = args.native_root.resolve()
    args.metadata_root = args.metadata_root.resolve()
    if not args.detector.exists():
        raise FileNotFoundError(args.detector)
    if not args.split_root.exists():
        raise FileNotFoundError(args.split_root)
    if not args.label_root.exists():
        raise FileNotFoundError(args.label_root)
    if not args.baseline_root.exists():
        raise FileNotFoundError(args.baseline_root)
    if not args.native_root.exists():
        raise FileNotFoundError(args.native_root)
    if not args.metadata_root.exists():
        raise FileNotFoundError(args.metadata_root)
    args.out.mkdir(parents=True, exist_ok=True)
    detector_hash = sha256(args.detector)
    frozen = args.out / "frozen_selection_before_test.json"

    candidate_grid = (
        {
            f"rmr_fine_eta{str(eta).replace('.', 'p')}": float(eta)
            for eta in args.eta
        }
        if args.eta
        else RMR_CANDIDATES
    )
    if any(not 0.0 < eta <= 1.0 for eta in candidate_grid.values()):
        raise ValueError("Every --eta must be in (0, 1]")
    if args.run_test and (args.rmr_only or args.eta):
        raise ValueError("--rmr-only/--eta are validation-screening options only")

    if args.run_test:
        if not frozen.exists():
            raise RuntimeError("Run validation first; frozen_selection_before_test.json is missing")
        selection = json.loads(frozen.read_text(encoding="utf-8"))
        if selection["detector_sha256"] != detector_hash:
            raise RuntimeError("Detector checkpoint changed after validation selection")
        frozen_metadata_root = (ROOT / selection["rmr_metadata_root"]).resolve()
        if args.metadata_root != frozen_metadata_root:
            raise RuntimeError(
                "RMR metadata root changed after validation selection: "
                f"expected {frozen_metadata_root}, received {args.metadata_root}"
            )
        frozen_native_root = (ROOT / selection["native_root"]).resolve()
        if args.native_root != frozen_native_root:
            raise RuntimeError(
                "Native image root changed after validation selection: "
                f"expected {frozen_native_root}, received {args.native_root}"
            )
        frozen_split_root = (ROOT / selection["split_root"]).resolve()
        frozen_label_root = (ROOT / selection["label_root"]).resolve()
        frozen_baseline_root = (ROOT / selection["baseline_root"]).resolve()
        if args.split_root != frozen_split_root:
            raise RuntimeError("Split root changed after validation selection")
        if args.label_root != frozen_label_root:
            raise RuntimeError("Label root changed after validation selection")
        if args.baseline_root != frozen_baseline_root:
            raise RuntimeError("Baseline root changed after validation selection")
        if args.rmr_metadata_mode != selection.get("rmr_metadata_mode", "provided"):
            raise RuntimeError("RMR metadata mode changed after validation selection")
        split = "test"
        chosen = selection["selected_rmr_view"]
        methods = {
            "raw": args.native_root,
            chosen: RMR_CANDIDATES[chosen],
            "nafnet": args.baseline_root / "nafnet" / "images" / "test",
            "dfpir": args.baseline_root / "dfpir" / "images" / "test",
            "demoe_auto": args.baseline_root / "demoe_auto" / "images" / "test",
            "instructir": args.baseline_root / "instructir_generic" / "images" / "test",
        }
    else:
        split = "val"
        methods = (
            dict(candidate_grid)
            if args.rmr_only
            else {
                "raw": args.native_root,
                "nafnet": args.baseline_root / "nafnet" / "images" / "test",
                "dfpir": args.baseline_root / "dfpir" / "images" / "test",
                "demoe_auto": args.baseline_root / "demoe_auto" / "images" / "test",
                "instructir": args.baseline_root / "instructir_generic" / "images" / "test",
                **candidate_grid,
            }
        )

    names = split_names(split, args.split_root)
    gt = subset_gt(names, args.label_root)
    checkpoint = args.rmr_checkpoint.resolve() if args.rmr_checkpoint is not None else None
    rmr_full = (
        args.rmr_full_dir.resolve()
        if args.rmr_full_dir is not None
        else CURRENT_RMR / "current_metadata_eta1p00"
    )
    if args.generate_rmr:
        if checkpoint is None:
            checkpoint = DEFAULT_CURRENT_CHECKPOINT.resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        rmr_full = args.out / "current_rmr_restored" / split
        restore_current_rmr(
            names,
            checkpoint,
            rmr_full,
            device_name=args.device,
            tile=args.tile,
            overlap=args.overlap,
            metadata_root=args.metadata_root,
            metadata_mode=args.rmr_metadata_mode,
            native_root=args.native_root,
            force=args.force,
        )
    if args.run_test:
        selected_checkpoint_hash = selection.get("rmr_checkpoint_sha256")
        active_checkpoint_hash = sha256(checkpoint) if checkpoint is not None else None
        if selected_checkpoint_hash != active_checkpoint_hash:
            raise RuntimeError("RMR checkpoint changed after validation selection")
    model = YOLO(str(args.detector))
    prediction_root = args.out / f"{split}_predictions"
    rows = []
    for method, folder in methods.items():
        prediction_path = prediction_root / f"{method}.csv"
        predict(model, method, folder, names, prediction_path, args, rmr_full)
        rows.append(metrics(method, gt, load_compact_predictions(prediction_path)))
        print(f"{split}: {method}: AP@.10={rows[-1]['ap10_primary']:.4f}, F1@.10={rows[-1]['f1_iou10']:.4f}", flush=True)

    rows.sort(key=lambda row: (row["ap10_primary"], row["f1_iou10"], row["coverage"]), reverse=True)
    write_summary(args.out / f"{split}_summary.csv", rows)
    (args.out / f"{split}_details.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if not args.run_test:
        selected = max(
            (row for row in rows if row["method"] in candidate_grid),
            key=lambda row: (row["ap10_primary"], row["f1_iou10"], row["coverage"]),
        )
        frozen.write_text(
            json.dumps(
                {
                    "status": "FROZEN_BEFORE_TEST",
                    "selected_rmr_view": selected["method"],
                    "selection_metric": "validation primary AP@0.10, then F1@0.10, then relaxed coverage",
                    "validation_metrics": {key: selected[key] for key in ("ap10_primary", "ap50_primary", "f1_iou10", "f1_iou50", "coverage")},
                    "detector": provenance_path(args.detector),
                    "detector_sha256": detector_hash,
                    "detector_selection": args.detector_provenance,
                    "detector_training_audit": (
                        provenance_path(args.split_root / "audit.json")
                        if (args.split_root / "audit.json").exists()
                        else None
                    ),
                    "rmr_checkpoint": provenance_path(checkpoint) if checkpoint is not None else None,
                    "rmr_checkpoint_sha256": sha256(checkpoint) if checkpoint is not None else None,
                    "rmr_full_output": provenance_path(rmr_full),
                    "rmr_metadata": (
                        "unavailable control (all 82 packet channels zeroed)"
                        if args.rmr_metadata_mode == "unavailable"
                        else "camera/EXIF channels only; inertial and vehicle channels masked"
                        if args.rmr_metadata_mode == "camera_only"
                        else "inertial/vehicle channels only; camera channels masked"
                        if args.rmr_metadata_mode == "inertial_only"
                        else (
                            "native nested Sony EXIF plus complete 200 Hz SBG INS: "
                            "measured body acceleration and attitude-derived angular "
                            "rate calibrated against the overlapping direct-rate log"
                        )
                    ),
                    "rmr_metadata_mode": args.rmr_metadata_mode,
                    "rmr_metadata_root": provenance_path(args.metadata_root),
                    "rmr_metadata_mode": args.rmr_metadata_mode,
                    "native_root": provenance_path(args.native_root),
                    "split_root": provenance_path(args.split_root),
                    "label_root": provenance_path(args.label_root),
                    "baseline_root": provenance_path(args.baseline_root),
                    "rmr_metadata_sha256": {
                        Path(name).stem: sha256(
                            args.metadata_root / f"{Path(name).stem}.json"
                        )
                        for name in names
                    },
                    "validation_images": names,
                    "test_images_or_annotations_read": False,
                    "screening_only": bool(args.rmr_only or args.eta),
                    "imgsz": args.imgsz,
                    "confidence_floor": args.conf,
                    "nms_iou": args.iou,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Frozen selection: {selected['method']}", flush=True)
    else:
        (args.out / "test_execution_manifest.json").write_text(
            json.dumps(
                {
                    "status": "HELD_OUT_TEST_EXECUTED_ONCE",
                    "frozen_selection": provenance_path(frozen),
                    "detector_sha256": detector_hash,
                    "test_images": names,
                    "rmr_metadata_root": provenance_path(args.metadata_root),
                    "native_root": provenance_path(args.native_root),
                    "split_root": provenance_path(args.split_root),
                    "label_root": provenance_path(args.label_root),
                    "baseline_root": provenance_path(args.baseline_root),
                    "rmr_metadata_sha256": {
                        Path(name).stem: sha256(
                            args.metadata_root / f"{Path(name).stem}.json"
                        )
                        for name in names
                    },
                    "methods": list(methods),
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
