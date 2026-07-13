from __future__ import annotations

"""Build GT49 per-method evaluation folders in YOLO26/RDD four-class order.

The restored/native method images are hardlinked from the existing GT49
restoration output tree.  Ground-truth labels are remapped from the Roboflow
six-class GT49 taxonomy into the same four-class order used by the downloaded
YOLO26s RDD-FRDC checkpoint:

    0 Longitudinal Crack (D00)
    1 Transverse Crack (D10)
    2 Alligator Crack (D20)
    3 Pothole (D40)

Non-defect labels (`others`, `road_intersection`) are ignored.  This is an
evaluation-format conversion only; it does not train on GT49 labels.
"""

import argparse
import shutil
from pathlib import Path

import yaml


CLASS_MAP = {
    0: 2,  # alligator -> D20
    1: 0,  # longitudinal -> D00
    3: 3,  # pothole -> D40
    5: 1,  # transverse -> D10
}

TARGET_NAMES = [
    "Longitudinal Crack (D00)",
    "Transverse Crack (D10)",
    "Alligator Crack (D20)",
    "Pothole (D40)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--restored-root",
        type=Path,
        default=Path("experiments/roboflow_geotagged_v5_native_real/v32_final_gt49_allmethods/restored/native_real"),
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=Path("experiments/roboflow_geotagged_v5_native_real/native_real_yolo_newroad6"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/roboflow_geotagged_v5_native_real/v36_yolo26_rdd4_eval_sets"),
    )
    parser.add_argument(
        "--methods",
        default="raw,rmr_blind,rmr_metadata,rmr_metadata_gated,nafnet,dfpir,demoe_auto,demoe_scenario,instructir_generic,instructir_metadata",
    )
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def convert_label(src: Path, dst: Path) -> int:
    rows: list[str] = []
    if src.exists():
        for line in src.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            old_cls = int(float(parts[0]))
            if old_cls not in CLASS_MAP:
                continue
            parts[0] = str(CLASS_MAP[old_cls])
            rows.append(" ".join(parts[:5]))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def main() -> None:
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    gt_labels = args.gt_root / "labels" / "test"
    args.out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for method in methods:
        src_images = args.restored_root / method / "images" / "test"
        if not src_images.exists():
            continue
        out_method = args.out / method
        out_images = out_method / "images" / "test"
        out_labels = out_method / "labels" / "test"
        n_images = 0
        n_boxes = 0
        for image in sorted(src_images.iterdir()):
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                continue
            link_or_copy(image, out_images / image.name)
            n_boxes += convert_label(gt_labels / f"{image.stem}.txt", out_labels / f"{image.stem}.txt")
            n_images += 1
        data = {
            "path": str(out_method.resolve()).replace("\\", "/"),
            "train": "images/test",
            "val": "images/test",
            "test": "images/test",
            "nc": 4,
            "names": TARGET_NAMES,
        }
        (out_method / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        manifest.append({"method": method, "images": n_images, "boxes": n_boxes, "data": str(out_method / "data.yaml")})
    (args.out / "manifest.yaml").write_text(yaml.safe_dump({"methods": manifest}, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump({"out": str(args.out), "methods": manifest}, sort_keys=False))


if __name__ == "__main__":
    main()
