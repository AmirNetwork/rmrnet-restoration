#!/usr/bin/env python3
"""Run the one-time CRID-320 native-resolution confirmatory test.

The runner accepts no checkpoint or residual-strength search arguments. Every
operating point is read from its validation-selection record, and TRACE-R must
carry a field-policy freeze created before this program opens ``test.txt``.
Interrupted runs may be resumed from per-method records; a completed run cannot
be repeated in the same output directory.

Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.evaluate_crid320_validation import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_EXPORT,
    DEFAULT_METADATA,
    hardlink_or_copy,
    label_for_image,
    metric_payload,
    restore_validation,
    sha256,
)


DEFAULT_DETECTOR = Path(
    r"E:\TRACE_R_experiments\crid320_detector_v1_20260829\weights\best.pt"
)
DEFAULT_OUT = Path(r"E:\TRACE_R_experiments\crid320_sealed_test_20260831")
TRACE_CHECKPOINT = Path(
    r"E:\TRACE_R_experiments\crid320_trace_field_policy_v8_20260831"
    r"\trace_crid_field_policy_frozen.pth"
)
TRACE_VALIDATION = Path(
    r"E:\TRACE_R_experiments\crid320_trace_field_policy_v8_validation_20260831"
    r"\rmrp\trace_crid_field_policy_frozen\validation_selection.json"
)
BASELINE_VALIDATIONS = {
    "nafnet": Path(
        r"E:\TRACE_R_experiments\crid320_restorer_validation_20260829"
        r"\nafnet\nafnet_field_epoch_020\validation_selection.json"
    ),
    "instructir": Path(
        r"E:\TRACE_R_experiments\crid320_restorer_validation_20260829"
        r"\instructir\instructir_field_epoch_008\validation_selection.json"
    ),
    "dfpir": Path(
        r"E:\TRACE_R_experiments\crid320_restorer_validation_20260829"
        r"\dfpir\dfpir_field_epoch_008\validation_selection.json"
    ),
    "demoe": Path(
        r"E:\TRACE_R_experiments\crid320_restorer_validation_20260829"
        r"\demoe\demoe_field_epoch_008\validation_selection.json"
    ),
}
LM_HEAD = ROOT / "weights" / "instructir" / "lm_instructir-7d.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def locked_test_paths(export_root: Path) -> list[Path]:
    manifest = export_root / "test.txt"
    rows = [
        Path(line.strip()).resolve()
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 80 or len({path.stem for path in rows}) != 80:
        raise RuntimeError("CRID-320 sealed test must contain 80 unique frames")
    return rows


def load_locked_spec(model: str, report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "VALIDATION_SELECTION_ONLY_TEST_UNOPENED":
        raise RuntimeError(f"Invalid validation status for {model}: {report_path}")
    if report.get("test_images_or_labels_read") is not False:
        raise RuntimeError(f"Validation report opened test data: {report_path}")
    checkpoint = Path(report["checkpoint"]).resolve()
    if sha256(checkpoint) != report["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint changed after validation: {checkpoint}")
    selected = report["selected"]
    return {
        "model": model,
        "checkpoint": checkpoint,
        "checkpoint_sha256": report["checkpoint_sha256"],
        "eta": float(selected["eta"]),
        "validation_map50": float(selected["map50"]),
        "validation_map50_95": float(selected["map50_95"]),
        "validation_report": report_path.resolve(),
        "validation_report_sha256": sha256(report_path),
        "lm_head": LM_HEAD.resolve() if model == "instructir" else None,
    }


def locked_specs() -> list[dict[str, Any]]:
    specs = [load_locked_spec(model, path) for model, path in BASELINE_VALIDATIONS.items()]
    trace = load_locked_spec("rmrp", TRACE_VALIDATION)
    if trace["checkpoint"] != TRACE_CHECKPOINT.resolve():
        raise RuntimeError("TRACE validation report does not identify the frozen policy")
    payload = torch.load(TRACE_CHECKPOINT, map_location="cpu", weights_only=False)
    freeze = payload.get("field_policy_freeze", {})
    if freeze.get("status") != "FROZEN_BEFORE_SEALED_TEST":
        raise RuntimeError("TRACE checkpoint lacks a pre-test policy freeze")
    if freeze.get("test_images_or_labels_read") is not False:
        raise RuntimeError("TRACE checkpoint freeze does not certify test isolation")
    del payload
    specs.append(trace)
    return specs


def build_test_view(
    paths: list[Path],
    full_dir: Path | None,
    view_root: Path,
    eta: float,
) -> Path:
    image_dir = view_root / "images" / "test"
    label_dir = view_root / "labels" / "test"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for source in paths:
        output = image_dir / f"{source.stem}.jpg"
        hardlink_or_copy(label_for_image(source), label_dir / f"{source.stem}.txt")
        if not output.exists():
            if full_dir is None:
                hardlink_or_copy(source, output)
            else:
                with Image.open(source) as native_image, Image.open(
                    full_dir / f"{source.stem}.jpg"
                ) as restored_image:
                    native = native_image.convert("RGB")
                    restored = restored_image.convert("RGB")
                    if native.size != restored.size:
                        raise AssertionError(f"Native geometry changed for {source.name}")
                    Image.blend(native, restored, eta).save(
                        output, quality=98, subsampling=0
                    )
        outputs.append(output.resolve())
    manifest = view_root / "test.txt"
    manifest.write_text("\n".join(str(path) for path in outputs) + "\n", encoding="utf-8")
    data = {
        "path": str(view_root.resolve()),
        "train": str(manifest.resolve()),
        "val": str(manifest.resolve()),
        "test": str(manifest.resolve()),
        "names": CLASS_NAMES,
    }
    yaml_path = view_root / "data_test_only.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return yaml_path


def read_ground_truth(image_path: Path) -> list[tuple[int, tuple[int, int, int, int]]]:
    width, height = Image.open(image_path).size
    boxes: list[tuple[int, tuple[int, int, int, int]]] = []
    for line in label_for_image(image_path).read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        cls, cx, cy, bw, bh = int(values[0]), *map(float, values[1:5])
        x1 = int(round((cx - bw / 2.0) * width))
        y1 = int(round((cy - bh / 2.0) * height))
        x2 = int(round((cx + bw / 2.0) * width))
        y2 = int(round((cy + bh / 2.0) * height))
        boxes.append((cls, (x1, y1, x2, y2)))
    return boxes


def write_overlays(
    detector: YOLO,
    original_paths: list[Path],
    evaluated_paths: list[Path],
    out_dir: Path,
    *,
    imgsz: int,
    device: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=24)
    for index, (source, evaluated) in enumerate(
        zip(original_paths, evaluated_paths, strict=True), 1
    ):
        target = out_dir / f"{source.stem}.jpg"
        if target.exists():
            continue
        # Passing the complete native-resolution list to Ultralytics can retain
        # all decoded frames and exhaust a 6 GB device. One-frame prediction is
        # equivalent and makes overlay generation restartable.
        result = detector.predict(
            source=str(evaluated),
            imgsz=imgsz,
            conf=0.03,
            iou=0.70,
            max_det=300,
            batch=1,
            device=device,
            verbose=False,
        )[0]
        with Image.open(evaluated) as opened:
            image = opened.convert("RGB")
        draw = ImageDraw.Draw(image)
        line_width = max(3, int(round(min(image.size) / 900)))
        for cls, box in read_ground_truth(source):
            draw.rectangle(box, outline=(255, 200, 0), width=line_width)
            draw.text((box[0], max(0, box[1] - 28)), f"GT {CLASS_NAMES[cls]}", fill=(255, 200, 0), font=font)
        if result.boxes is not None:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            confidences = result.boxes.conf.detach().cpu().numpy()
            for box, cls, confidence in zip(boxes, classes, confidences, strict=True):
                coords = tuple(int(round(value)) for value in box.tolist())
                draw.rectangle(coords, outline=(0, 220, 110), width=line_width)
                draw.text(
                    (coords[0], max(0, coords[1] - 28)),
                    f"{CLASS_NAMES[int(cls)]} {float(confidence):.2f}",
                    fill=(0, 220, 110),
                    font=font,
                )
        image.save(target, quality=94, subsampling=0)
        del result
        if torch.cuda.is_available() and index % 8 == 0:
            torch.cuda.empty_cache()


def evaluate_method(
    detector: YOLO,
    spec: dict[str, Any] | None,
    test_paths: list[Path],
    metadata_root: Path,
    root: Path,
    *,
    imgsz: int,
    device_arg: str,
    device: torch.device,
) -> dict[str, Any]:
    method = "native" if spec is None else str(spec["model"])
    method_root = root / method
    record_path = method_root / "sealed_test_result.json"
    if record_path.exists():
        return json.loads(record_path.read_text(encoding="utf-8"))
    if spec is None:
        full_dir = None
        eta = 0.0
    else:
        full_dir = method_root / "full_restoration"
        restore_validation(
            str(spec["model"]),
            Path(spec["checkpoint"]),
            Path(spec["lm_head"]) if spec["lm_head"] else None,
            test_paths,
            metadata_root,
            full_dir,
            device,
            tile=768,
            overlap=96,
            nafnet_global_long_side=1536,
            force=False,
            audit_status="SEALED_TEST_EXECUTION_AFTER_POLICY_FREEZE",
            test_images_or_labels_read=True,
        )
        eta = float(spec["eta"])
    view_root = method_root / "selected_view"
    yaml_path = build_test_view(test_paths, full_dir, view_root, eta)
    pending_path = method_root / "metrics_before_overlays.json"
    if pending_path.exists():
        pending = json.loads(pending_path.read_text(encoding="utf-8"))
        metrics_payload = pending["metrics"]
    else:
        metrics = detector.val(
            data=str(yaml_path),
            split="test",
            imgsz=imgsz,
            batch=1,
            device=device_arg,
            conf=0.001,
            iou=0.70,
            max_det=300,
            plots=False,
            save_json=False,
            save_txt=True,
            save_conf=True,
            project=str(method_root / "detector_run"),
            name="test",
            exist_ok=True,
            verbose=False,
        )
        metrics_payload = metric_payload(metrics)
        atomic_json(
            pending_path,
            {
                "status": "SEALED_TEST_METRICS_FIXED_BEFORE_VISUALIZATION",
                "method": method,
                "metrics": metrics_payload,
                "recorded_unix": time.time(),
            },
        )
        del metrics
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    evaluated_paths = [
        (view_root / "images" / "test" / f"{path.stem}.jpg").resolve()
        for path in test_paths
    ]
    write_overlays(
        detector,
        test_paths,
        evaluated_paths,
        method_root / "overlays",
        imgsz=imgsz,
        device=device_arg,
    )
    record = {
        "status": "ONE_TIME_SEALED_TEST_COMPLETE",
        "method": method,
        "eta": eta,
        "checkpoint": str(spec["checkpoint"]) if spec else None,
        "checkpoint_sha256": spec["checkpoint_sha256"] if spec else None,
        "validation_report": str(spec["validation_report"]) if spec else None,
        "validation_report_sha256": spec["validation_report_sha256"] if spec else None,
        "validation_map50": spec["validation_map50"] if spec else None,
        "validation_map50_95": spec["validation_map50_95"] if spec else None,
        **metrics_payload,
        "images": len(test_paths),
        "completed_unix": time.time(),
    }
    atomic_json(record_path, record)
    return record


def main() -> None:
    args = parse_args()
    export_root = args.export_root.resolve()
    metadata_root = args.metadata_root.resolve()
    detector_path = args.detector.resolve()
    out = args.out.resolve()
    completion = out / "SEALED_TEST_COMPLETE.json"
    if completion.exists():
        print(completion.read_text(encoding="utf-8"))
        return
    for path in (export_root, metadata_root, detector_path, LM_HEAD):
        if not path.exists():
            raise FileNotFoundError(path)
    specs = locked_specs()
    detector_freeze = detector_path.parent.parent / "detector_selection_freeze.json"
    detector_audit = json.loads(detector_freeze.read_text(encoding="utf-8"))
    if detector_audit.get("status") != "FROZEN_BEFORE_CRID320_TEST":
        raise RuntimeError("Detector was not frozen before the CRID-320 test")
    if sha256(detector_path) != detector_audit["selected_checkpoint_sha256"]:
        raise RuntimeError("Detector changed after its validation freeze")

    # This is the single deliberate point at which the sealed manifest is read.
    test_paths = locked_test_paths(export_root)
    opened_path = out / "SEALED_TEST_OPENED.json"
    if opened_path.exists():
        opened = json.loads(opened_path.read_text(encoding="utf-8"))
        if opened.get("test_manifest_sha256") != sha256(export_root / "test.txt"):
            raise RuntimeError("Sealed test manifest changed during a resumed run")
        if opened.get("detector_sha256") != sha256(detector_path):
            raise RuntimeError("Detector changed during a resumed sealed-test run")
    else:
        opened = {
            "status": "SEALED_TEST_OPENED_AFTER_ALL_SELECTIONS_FROZEN",
            "opened_unix": time.time(),
            "test_manifest": str((export_root / "test.txt").resolve()),
            "test_manifest_sha256": sha256(export_root / "test.txt"),
            "detector": str(detector_path),
            "detector_sha256": sha256(detector_path),
            "methods": [
                {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in spec.items()
                }
                for spec in specs
            ],
            "test_images": len(test_paths),
        }
        atomic_json(opened_path, opened)

    device = torch.device(
        f"cuda:{args.device}"
        if torch.cuda.is_available() and str(args.device).isdigit()
        else args.device
    )
    detector = YOLO(str(detector_path))
    rows = [
        evaluate_method(
            detector,
            None,
            test_paths,
            metadata_root,
            out,
            imgsz=args.imgsz,
            device_arg=args.device,
            device=device,
        )
    ]
    for spec in specs:
        rows.append(
            evaluate_method(
                detector,
                spec,
                test_paths,
                metadata_root,
                out,
                imgsz=args.imgsz,
                device_arg=args.device,
                device=device,
            )
        )
    final = {
        "status": "ONE_TIME_CRID320_SEALED_TEST_COMPLETE",
        "opened_record": str((out / "SEALED_TEST_OPENED.json").resolve()),
        "opened_record_sha256": sha256(out / "SEALED_TEST_OPENED.json"),
        "test_manifest_sha256": opened["test_manifest_sha256"],
        "detector_sha256": opened["detector_sha256"],
        "results": rows,
        "completed_unix": time.time(),
    }
    atomic_json(completion, final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
