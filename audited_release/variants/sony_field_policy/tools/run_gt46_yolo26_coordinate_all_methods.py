# RMR-Net project author and integration: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
# Third-party methods retain the original authorship and licenses cited in the paper.

from __future__ import annotations

"""Run the revised GT46 native-field detector/evaluator for all restoration methods.

This script is intentionally narrow: it uses the user-provided
Yolo26_coordinate.py detector exactly as the native-coordinate detector for the
46-image revised Roboflow export. It does not use the older GT49 crop/mask/tile
pipeline.
"""

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = ROOT / "experiments" / "gt46_yolo26_coordinate_revised"
NATIVE_IMAGES = EXP_ROOT / "gt46_native_images"
ANNOTATIONS = ROOT / "road-defect-seg-9junedata.coco-segmentation" / "train" / "_annotations.coco.json"
CAM1_METADATA = ROOT / "geotagged" / "precise_cam1_coords.csv"
DETECTOR_SCRIPT = ROOT / "Yolo26_coordinate" / "Yolo26_coordinate.py"
DETECTOR_WEIGHTS = ROOT / "Yolo26_coordinate" / "YOLO26s_RDD_FRDC_Distilled_v2.pt"
EVAL_SCRIPT = ROOT / "tools" / "eval_yolo26_coordinate_gt46.py"

EVAL_SET_ROOT = ROOT / "experiments" / "roboflow_geotagged_v5_native_real" / "v36_yolo26_rdd4_eval_sets"
ETA_SET_ROOT = ROOT / "experiments" / "roboflow_geotagged_v5_native_real" / "v36_yolo26_rdd4_eta_sets"
DETECTION_HEAVY_SET_ROOT = ROOT / "experiments" / "roboflow_geotagged_v5_native_real" / "v44_detection_heavy_yolo26_eval_sets"
NATIVE_GATE_GAMMA085_ROOT = ROOT / "experiments" / "roboflow_geotagged_v5_native_real" / "v45_native_gate_gamma085_eval_sets"
DETECTOR_DOMINANT_2EP_ROOT = ROOT / "experiments" / "roboflow_geotagged_v5_native_real" / "v46_detector_dominant_2ep_eval_sets"
DETECTOR_DOMINANT_2EP_ETA_ROOT = ROOT / "experiments" / "roboflow_geotagged_v5_native_real" / "v46_detector_dominant_2ep_eta_sweep"


METHODS = [
    ("raw", "Raw native", NATIVE_IMAGES),
    ("rmr_blind", "RMR-Net image-only", EVAL_SET_ROOT / "rmr_blind" / "images" / "test"),
    ("rmr_metadata", "RMR-Net metadata full", EVAL_SET_ROOT / "rmr_metadata" / "images" / "test"),
    ("rmr_metadata_gated", "RMR-Net metadata-gated", EVAL_SET_ROOT / "rmr_metadata_gated" / "images" / "test"),
    ("rmr_eta_0p1", r"RMR-Net weak residual ($\eta=0.1$)", ETA_SET_ROOT / "rmr_eta_0p1" / "images" / "test"),
    ("rmr_detheavy_blind", "RMR-Net detector-heavy image-only", DETECTION_HEAVY_SET_ROOT / "rmr_blind" / "images" / "test"),
    ("rmr_detheavy_metadata", "RMR-Net detector-heavy metadata", DETECTION_HEAVY_SET_ROOT / "rmr_metadata" / "images" / "test"),
    ("rmr_detheavy_gated", "RMR-Net detector-heavy gated", DETECTION_HEAVY_SET_ROOT / "rmr_metadata_gated" / "images" / "test"),
    ("rmr_detdom2ep_blind", "RMR-Net detector-dominant image-only", DETECTOR_DOMINANT_2EP_ROOT / "rmr_blind" / "images" / "test"),
    ("rmr_detdom2ep_metadata", "RMR-Net detector-dominant metadata", DETECTOR_DOMINANT_2EP_ROOT / "rmr_metadata" / "images" / "test"),
    ("rmr_detdom2ep_metadata_eta0p05", r"RMR-Net detector-dominant metadata ($\eta=0.05$)", DETECTOR_DOMINANT_2EP_ETA_ROOT / "rmr_metadata_eta0p05" / "images" / "test"),
    ("rmr_native_gate_gamma085", r"RMR-Net native gate ($\gamma=0.85$)", NATIVE_GATE_GAMMA085_ROOT / "rmr_metadata_gated" / "images" / "test"),
    ("nafnet", "NAFNet-road", EVAL_SET_ROOT / "nafnet" / "images" / "test"),
    ("dfpir", "DFPIR", EVAL_SET_ROOT / "dfpir" / "images" / "test"),
    ("demoe_auto", "DeMoE-auto", EVAL_SET_ROOT / "demoe_auto" / "images" / "test"),
    ("demoe_scenario", "DeMoE-scenario", EVAL_SET_ROOT / "demoe_scenario" / "images" / "test"),
    ("instructir_generic", "InstructIR-generic", EVAL_SET_ROOT / "instructir_generic" / "images" / "test"),
    ("instructir_metadata", "InstructIR-metadata", EVAL_SET_ROOT / "instructir_metadata" / "images" / "test"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu.")
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument(
        "--detector-script",
        type=Path,
        default=DETECTOR_SCRIPT,
        help="YOLO26 coordinate detector script. Use Yolo26_coordinate_revised.py for class-specific thresholds.",
    )
    parser.add_argument(
        "--class-conf",
        default="",
        help="Optional class-specific thresholds forwarded to revised detector, e.g. D00=0.15,D10=0.25,D20=0.25.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional human-readable run tag. If omitted a tag is derived from detector settings.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--atlas-max", type=int, default=12)
    parser.add_argument("--force-detect", action="store_true", help="Delete and rerun detector outputs.")
    parser.add_argument("--skip-detect", action="store_true", help="Only rerun evaluation/table generation.")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def gt46_names() -> list[str]:
    names = sorted(p.name for p in NATIVE_IMAGES.glob("*.jpg"))
    if len(names) != 46:
        raise RuntimeError(f"Expected 46 native GT images in {NATIVE_IMAGES}, found {len(names)}")
    return names


def copy_exact_subset(src: Path, dst: Path, names: list[str]) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.mkdir(parents=True, exist_ok=True)
    missing = []
    for name in names:
        s = src / name
        if not s.exists():
            missing.append(name)
            continue
        d = dst / name
        if not d.exists() or s.stat().st_size != d.stat().st_size:
            shutil.copy2(s, d)
    if missing:
        raise FileNotFoundError(f"{src} is missing {len(missing)} GT46 files, e.g. {missing[:5]}")


def prepare_method_images(names: list[str]) -> list[tuple[str, str, Path]]:
    staged_root = EXP_ROOT / "method_images"
    staged = []
    for key, label, src in METHODS:
        dst = staged_root / key
        copy_exact_subset(src, dst, names)
        staged.append((key, label, dst))
    return staged


def detector_complete(out_dir: Path) -> bool:
    return (
        (out_dir / "detections.geojson").exists()
        and any(out_dir.glob("*.csv"))
    )


def slug(text: str) -> str:
    text = text.strip().replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)
    text = text.strip("_")
    return text or "none"


def detector_run_tag(args: argparse.Namespace) -> str:
    if args.tag:
        return slug(args.tag)
    script_stem = Path(args.detector_script).stem
    base = f"{script_stem}_imgsz{args.imgsz}_conf{str(args.conf).replace('.', 'p')}"
    if args.class_conf.strip():
        base += f"_cls_{slug(args.class_conf)}"
    return slug(base)


def run_detector_for_methods(args: argparse.Namespace, staged: list[tuple[str, str, Path]]) -> list[tuple[str, str, Path]]:
    pred_root = EXP_ROOT / f"method_detections_{detector_run_tag(args)}"
    pred_root.mkdir(parents=True, exist_ok=True)
    out = []
    for key, label, image_dir in staged:
        method_out = pred_root / key
        if args.force_detect and method_out.exists():
            shutil.rmtree(method_out)
        if args.skip_detect or detector_complete(method_out):
            print(f"[SKIP] detector already available for {key}: {method_out}")
        else:
            method_out.mkdir(parents=True, exist_ok=True)
            run(
                [
                    sys.executable,
                    str(args.detector_script),
                    "--images",
                    str(image_dir),
                    "--out",
                    str(method_out),
                    "--csv",
                    str(CAM1_METADATA),
                    "--model",
                    str(DETECTOR_WEIGHTS),
                    "--device",
                    args.device,
                    "--workers",
                    str(args.workers),
                    "--imgsz",
                    str(args.imgsz),
                    "--conf",
                    str(args.conf),
                ]
                + (["--class_conf", args.class_conf] if args.class_conf.strip() else [])
            )
        out.append((key, label, method_out))
    return out


def run_evaluation(args: argparse.Namespace, detections: list[tuple[str, str, Path]]) -> Path:
    eval_out = EXP_ROOT / f"evaluation_all_methods_{detector_run_tag(args)}"
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--annotations",
        str(ANNOTATIONS),
        "--images",
        str(NATIVE_IMAGES),
        "--out",
        str(eval_out),
        "--atlas-max",
        str(args.atlas_max),
        "--pred-root",
    ]
    cmd.extend(str(path) for _key, _label, path in detections)
    cmd.append("--pred-name")
    cmd.extend(key for key, _label, _path in detections)
    run(cmd)
    return eval_out


def read_summary(eval_out: Path) -> dict[tuple[str, str, float], dict[str, str]]:
    path = eval_out / "summary_metrics.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {(r["run"], r["mode"], float(r["iou"])): r for r in rows}


def read_class(eval_out: Path) -> dict[str, dict[str, dict[str, str]]]:
    path = eval_out / "class_metrics_iou10.csv"
    out: dict[str, dict[str, dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out.setdefault(row["run"], {})[row["label"]] = row
    return out


def fmt(value: str | float) -> str:
    return f"{float(value):.3f}"


def write_latex_table(eval_out: Path, args: argparse.Namespace) -> Path:
    summary = read_summary(eval_out)
    class_rows = read_class(eval_out)
    labels = {key: label for key, label, _src in METHODS}
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        (
            r"\caption{High-quality Sony native-image field test on the revised 46-frame cam1 "
            r"set using the exact YOLO26-coordinate detector supplied with the experiment. "
            r"Images remain at native 4752$\times$3168 resolution; the detector runs full-image "
            rf"inference with YOLO26s RDD-FRDC weights, \texttt{{imgsz={args.imgsz}}}, "
            rf"\texttt{{conf={args.conf:.2f}}}, and real cam1 pose/EXIF metadata for geospatial output. "
            r"No crop, road mask, tiling, or GT46 fine-tuning is used. Metrics are computed on "
            r"the common defect taxonomy shared by the annotations and detector. GT success is "
            r"recall-oriented: a labeled defect is recovered when a same-primary-class prediction "
            r"has IoU$\geq$0.10, covers at least 25\% of the GT area, or contains the GT center.}"
        ),
        r"\label{tab:geotagged_native_pilot}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Method & N & Pred & GT success & P@.10 & R@.10 & F1@.10 & F1@.50 & Crack F1@.10 \\",
        r"\midrule",
    ]
    for key, _label, _src in METHODS:
        r10 = summary[(key, "primary", 0.10)]
        r50 = summary[(key, "primary", 0.50)]
        succ = summary[(key, "primary_success", -1.0)]
        crack_components = []
        for crack_label in ["Alligator Crack", "Longitudinal Crack", "Transverse Crack"]:
            if crack_label in class_rows.get(key, {}):
                crack_components.append(float(class_rows[key][crack_label]["f1_iou10"]))
        crack_f1 = sum(crack_components) / len(crack_components) if crack_components else 0.0
        lines.append(
            f"{labels[key]} & {r10['images']} & {r10['pred']} & {fmt(succ['recall'])} & "
            f"{fmt(r10['precision'])} & {fmt(r10['recall'])} & {fmt(r10['f1'])} & "
            f"{fmt(r50['f1'])} & {crack_f1:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    tex_path = eval_out / "table_geotagged_native_pilot.tex"
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    return tex_path


def main() -> None:
    args = parse_args()
    names = gt46_names()
    staged = prepare_method_images(names)
    detections = run_detector_for_methods(args, staged)
    eval_out = run_evaluation(args, detections)
    tex = write_latex_table(eval_out, args)
    print(f"\n[OK] evaluation: {eval_out}")
    print(f"[OK] latex table: {tex}")


if __name__ == "__main__":
    main()
