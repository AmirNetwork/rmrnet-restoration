#!/usr/bin/env python3
"""Validation-only checkpoint and output-strength selection on CRID-320.

This program cannot evaluate the sealed test block.  It restores the locked
60-frame temporal validation sequence at native resolution, evaluates each
predeclared residual strength with one frozen CRID detector, and writes the
selected operating point plus complete file hashes.  Test execution is kept in
a separate program so an accidental command-line switch cannot expose labels.

Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcadnet.practical_metadata import CONTEXT_START, sensor_packet_from_mapping
from tools.restore_native_yolo_split import restore_global_residual, restore_tiled
from tools.train_crid320_restorer import field_scenarios
from train_matched_restorer import TrainableRestorer


DEFAULT_EXPORT = ROOT / "datasets" / "crid320_annotation_20260829" / "exports" / "latest"
DEFAULT_METADATA = (
    ROOT
    / "experiments"
    / "geotagged_cam1_complete_sbg_ins_metadata_v4_20260811"
    / "metadata"
)
DEFAULT_OUT = Path(r"E:\TRACE_R_experiments\crid320_restorer_validation_20260829")
CLASS_NAMES = {
    0: "longitudinal_crack",
    1: "transverse_crack",
    2: "alligator_crack",
    3: "pothole",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=("rmrp", "demoe", "nafnet", "dfpir", "instructir"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lm-head-weights", type=Path)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--eta", type=float, action="append")
    parser.add_argument("--tile", type=int, default=768)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--nafnet-global-long-side", type=int, default=1536)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--device", default="0")
    parser.add_argument("--force-restore", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hardlink_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def validation_paths(export_root: Path) -> list[Path]:
    manifest = export_root / "validation.txt"
    rows = [
        Path(line.strip()).resolve()
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 60:
        raise RuntimeError("CRID-320 validation must contain exactly 60 locked frames")
    if any("test" in {part.lower() for part in path.parts} for path in rows):
        raise RuntimeError("Validation manifest unexpectedly references the sealed test tree")
    return rows


def label_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    index = next(
        (i for i, part in enumerate(parts) if part.lower() == "images"),
        None,
    )
    if index is None:
        raise ValueError(f"Image path has no images component: {image_path}")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def save_rgb(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(tensor.detach().cpu().clamp(0.0, 1.0)).save(
        path,
        quality=98,
        subsampling=0,
    )


def apply_native_output_filter(
    path: Path,
    packet: torch.Tensor,
    policy: dict[str, Any] | None,
) -> dict[str, float | bool] | None:
    """Apply the checkpoint-declared native anti-alias policy automatically."""

    if not policy:
        return None
    if policy.get("kind") != "gaussian_detail":
        raise ValueError(f"Unsupported native output filter: {policy}")
    camera_reliability = float(packet[CONTEXT_START + 12].detach().cpu())
    imu_reliability = float(packet[CONTEXT_START + 13].detach().cpu())
    metadata_available = float(packet[CONTEXT_START + 15].detach().cpu())
    enabled = bool(
        metadata_available >= float(policy.get("metadata_threshold", 0.5))
        and camera_reliability >= float(policy.get("camera_reliability_threshold", 0.5))
        and imu_reliability >= float(policy.get("imu_reliability_threshold", 0.5))
    )
    if enabled:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not decode restored image {path}")
        floating = image.astype(np.float32) / 255.0
        sigma = float(policy["sigma_native_px"])
        alpha = float(policy["detail_gain"])
        low = cv2.GaussianBlur(floating, (0, 0), sigmaX=sigma, sigmaY=sigma)
        output = np.clip(
            np.rint((floating + alpha * (floating - low)) * 255.0), 0, 255
        ).astype(np.uint8)
        if not cv2.imwrite(str(path), output, [cv2.IMWRITE_JPEG_QUALITY, 98]):
            raise RuntimeError(f"Could not write filtered image {path}")
    return {
        "enabled": enabled,
        "camera_reliability": camera_reliability,
        "imu_reliability": imu_reliability,
        "metadata_available": metadata_available,
    }


@torch.inference_mode()
def restore_validation(
    model_name: str,
    checkpoint: Path,
    lm_head_weights: Path | None,
    paths: list[Path],
    metadata_root: Path,
    full_dir: Path,
    device: torch.device,
    *,
    tile: int,
    overlap: int,
    nafnet_global_long_side: int,
    force: bool,
    audit_status: str = "VALIDATION_DIAGNOSTIC_ONLY_TEST_UNOPENED",
    test_images_or_labels_read: bool = False,
) -> None:
    full_dir.mkdir(parents=True, exist_ok=True)
    expected = {path.stem for path in paths}
    present = {path.stem for path in full_dir.glob("*.jpg")}
    if expected == present and not force:
        return
    restorer = TrainableRestorer(
        model_name,
        [checkpoint],
        device,
        lm_head_weights,
    ).to(device).eval()
    uses_dfpir_backbone = (
        model_name == "dfpir"
        or restorer.arch.get("backbone") == "dfpir_sensor_prompt"
    )
    use_amp = bool(
        device.type == "cuda"
        and (uses_dfpir_backbone or model_name == "instructir")
    )
    amp_dtype = torch.bfloat16 if uses_dfpir_backbone else torch.float16
    gate_audit: list[dict[str, Any]] = []
    output_filter = restorer.arch.get("native_output_filter")
    for index, source in enumerate(paths, 1):
        target = full_dir / f"{source.stem}.jpg"
        if target.exists() and not force:
            continue
        with Image.open(source) as image:
            native = TF.to_tensor(image.convert("RGB")).unsqueeze(0).to(device)
        metadata_path = metadata_root / f"{source.stem}.json"
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
        packet = sensor_packet_from_mapping(
            json.loads(metadata_path.read_text(encoding="utf-8")),
            device=device,
        ).unsqueeze(0)

        def restore_patch(patch: torch.Tensor) -> torch.Tensor:
            with torch.amp.autocast(
                "cuda", enabled=use_amp, dtype=amp_dtype
            ):
                output, auxiliary = restorer(
                    patch,
                    packet.expand(patch.shape[0], -1),
                    field_scenarios(model_name, patch.shape[0]),
                )
            native_gate = auxiliary.get("native_gate")
            if native_gate is not None:
                gate_audit.append(
                    {
                        "image": source.name,
                        "native_gate": [
                            float(value) for value in native_gate.detach().float().cpu().flatten()
                        ],
                    }
                )
            output = output.float()
            if not torch.isfinite(output).all():
                raise FloatingPointError(
                    f"Non-finite {model_name} validation output for {source.name}"
                )
            return output

        if model_name in {"nafnet", "dfpir", "instructir"} or uses_dfpir_backbone:
            restored = restore_global_residual(
                native,
                restore_patch,
                long_side=nafnet_global_long_side,
                multiple=16,
            )
        else:
            restored = restore_tiled(
                native,
                restore_patch,
                tile=tile,
                overlap=overlap,
            )
        if tuple(restored.shape[-2:]) != tuple(native.shape[-2:]):
            raise AssertionError(f"Native geometry changed for {source.name}")
        save_rgb(restored[0], target)
        filter_audit = apply_native_output_filter(target, packet[0], output_filter)
        if filter_audit is not None:
            gate_audit.append(
                {"image": source.name, "native_output_filter": filter_audit}
            )
        print(
            f"restored {model_name} {index:03d}/{len(paths):03d}: {source.name}",
            flush=True,
        )
    if gate_audit:
        (full_dir.parent / "native_gate_audit.json").write_text(
            json.dumps(
                {
                    "status": audit_status,
                    "checkpoint": str(checkpoint),
                    "records": gate_audit,
                    "test_images_or_labels_read": test_images_or_labels_read,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    del restorer
    if device.type == "cuda":
        torch.cuda.empty_cache()


def build_view(
    paths: list[Path],
    full_dir: Path,
    view_root: Path,
    eta: float,
) -> Path:
    image_dir = view_root / "images" / "val"
    label_dir = view_root / "labels" / "val"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for source in paths:
        output = image_dir / f"{source.stem}.jpg"
        label = label_for_image(source)
        hardlink_or_copy(label, label_dir / f"{source.stem}.txt")
        if not output.exists():
            if eta == 0.0:
                hardlink_or_copy(source, output)
            else:
                with Image.open(source) as native_image, Image.open(
                    full_dir / f"{source.stem}.jpg"
                ) as restored_image:
                    native = native_image.convert("RGB")
                    restored = restored_image.convert("RGB")
                    if native.size != restored.size:
                        raise AssertionError(f"Native geometry changed for {source.name}")
                    blended = Image.blend(native, restored, eta)
                    blended.save(output, quality=98, subsampling=0)
        output_paths.append(output.resolve())
    manifest = view_root / "val.txt"
    manifest.write_text(
        "\n".join(str(path) for path in output_paths) + "\n",
        encoding="utf-8",
    )
    data = {
        "path": str(view_root.resolve()),
        # Ultralytics' detection schema requires both keys even when val() is
        # the only requested operation. Pointing train to the same locked
        # validation manifest satisfies parsing without opening training or
        # sealed-test data during selection.
        "train": str(manifest.resolve()),
        "val": str(manifest.resolve()),
        "names": CLASS_NAMES,
    }
    yaml_path = view_root / "data_val_only.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return yaml_path


def metric_payload(metrics: Any) -> dict[str, Any]:
    box = metrics.box
    all_ap = getattr(box, "all_ap", None)
    class_map50: list[float | None] = [None] * len(CLASS_NAMES)
    if all_ap is not None and getattr(all_ap, "ndim", 0) == 2 and all_ap.shape[1]:
        class_indices = getattr(box, "ap_class_index", range(len(all_ap)))
        for row_index, class_index in enumerate(class_indices):
            class_map50[int(class_index)] = float(all_ap[row_index, 0])
    return {
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "class_map50": class_map50,
        "class_map50_95": [float(value) for value in box.maps],
        "class_names": CLASS_NAMES,
        "speed_ms": {key: float(value) for key, value in metrics.speed.items()},
    }


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.detector = args.detector.resolve()
    args.export_root = args.export_root.resolve()
    args.metadata_root = args.metadata_root.resolve()
    args.out = args.out.resolve()
    args.lm_head_weights = args.lm_head_weights.resolve() if args.lm_head_weights else None
    for path in (args.checkpoint, args.detector, args.export_root, args.metadata_root):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.model == "instructir" and args.lm_head_weights is None:
        raise ValueError("InstructIR requires --lm-head-weights")
    etas = sorted(set(args.eta or (0.25, 0.50, 0.75, 1.00)))
    if any(not 0.0 < eta <= 1.0 for eta in etas):
        raise ValueError("Validation residual strengths must be in (0, 1]")
    device = torch.device(
        f"cuda:{args.device}"
        if torch.cuda.is_available() and str(args.device).isdigit()
        else args.device
    )
    paths = validation_paths(args.export_root)
    checkpoint_payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint_arch = checkpoint_payload.get("arch", {})
    uses_dfpir_backbone = (
        args.model == "dfpir"
        or checkpoint_arch.get("backbone") == "dfpir_sensor_prompt"
    )
    del checkpoint_payload
    checkpoint_root = args.out / args.model / args.checkpoint.stem
    full_dir = checkpoint_root / "full_restoration"
    restore_validation(
        args.model,
        args.checkpoint,
        args.lm_head_weights,
        paths,
        args.metadata_root,
        full_dir,
        device,
        tile=args.tile,
        overlap=args.overlap,
        nafnet_global_long_side=args.nafnet_global_long_side,
        force=args.force_restore,
    )

    detector = YOLO(str(args.detector))
    rows: list[dict[str, Any]] = []
    for eta in etas:
        eta_tag = str(eta).replace(".", "p")
        view_root = checkpoint_root / f"eta_{eta_tag}"
        yaml_path = build_view(paths, full_dir, view_root, eta)
        metrics = detector.val(
            data=str(yaml_path),
            split="val",
            imgsz=args.imgsz,
            batch=1,
            device=args.device,
            conf=0.001,
            iou=0.70,
            max_det=300,
            plots=False,
            save_json=False,
            save_txt=True,
            save_conf=True,
            project=str(checkpoint_root / "detector_runs"),
            name=f"eta_{eta_tag}",
            exist_ok=True,
            verbose=False,
        )
        row = {"eta": eta, **metric_payload(metrics)}
        rows.append(row)
        print(json.dumps(row), flush=True)

    selected = max(rows, key=lambda row: (row["map50"], row["map50_95"], -row["eta"]))
    audit = {
        "status": "VALIDATION_SELECTION_ONLY_TEST_UNOPENED",
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "detector": str(args.detector),
        "detector_sha256": sha256(args.detector),
        "validation_manifest": str(args.export_root / "validation.txt"),
        "validation_manifest_sha256": sha256(args.export_root / "validation.txt"),
        "metadata_root": str(args.metadata_root) if args.model == "rmrp" else None,
        "metadata_sha256": {
            path.stem: sha256(args.metadata_root / f"{path.stem}.json")
            for path in paths
        }
        if args.model == "rmrp"
        else None,
        "native_resolution": [4752, 3168],
        "inference": {
            "tile": None
            if args.model in {"nafnet", "dfpir", "instructir"} or uses_dfpir_backbone
            else args.tile,
            "overlap": None
            if args.model in {"nafnet", "dfpir", "instructir"} or uses_dfpir_backbone
            else args.overlap,
            "global_residual_long_side": (
                args.nafnet_global_long_side
                if args.model in {"nafnet", "dfpir", "instructir"}
                or uses_dfpir_backbone
                else None
            ),
            "single_output_no_detection_fusion": True,
        },
        "selection_metric": "validation mAP50; mAP50-95 then lower eta break ties",
        "candidates": rows,
        "selected": selected,
        "test_images_or_labels_read": False,
        "completed_unix": time.time(),
    }
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    (checkpoint_root / "validation_selection.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
