from __future__ import annotations

"""Build GT49 per-method evaluation folders for a two-class detector.

The image files are hardlinked from each restored/native method folder. Labels
are remapped from GT49's six-class taxonomy into:

    0 crack   (alligator, longitudinal, transverse)
    1 pothole

Non-defect labels (`others`, `road_intersection`) are ignored. This keeps the
native field test focused on common defect detection, with type evaluated
separately using the six-class detector outputs.
"""

import argparse
import shutil
from pathlib import Path

import yaml


CRACK_CLASSES = {0, 1, 5}
POTHOLE_CLASS = 3


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
        default=Path("experiments/roboflow_geotagged_v5_native_real/v34_primary_eval_sets"),
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
    rows = []
    if src.exists():
        for line in src.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            if cls in CRACK_CLASSES:
                parts[0] = "0"
            elif cls == POTHOLE_CLASS:
                parts[0] = "1"
            else:
                continue
            rows.append(" ".join(parts[:5]))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def main() -> None:
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    gt_labels = args.gt_root / "labels" / "test"
    manifest = []
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
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            link_or_copy(image, out_images / image.name)
            n_boxes += convert_label(gt_labels / f"{image.stem}.txt", out_labels / f"{image.stem}.txt")
            n_images += 1
        data = {
            "path": str(out_method.resolve()).replace("\\", "/"),
            "train": "images/test",
            "val": "images/test",
            "test": "images/test",
            "nc": 2,
            "names": ["crack", "pothole"],
        }
        (out_method / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        manifest.append({"method": method, "images": n_images, "boxes": n_boxes, "data": str(out_method / "data.yaml")})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.yaml").write_text(yaml.safe_dump({"methods": manifest}, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump({"out": str(args.out), "methods": manifest}, sort_keys=False))


if __name__ == "__main__":
    main()
