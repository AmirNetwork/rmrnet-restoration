from __future__ import annotations

"""Create paper tables and figures for the revised GT46 native-coordinate test."""

import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "gt46_yolo26_coordinate_revised"
RUN_TAG = "yolo26rev_imgsz1280_conf010_clsD00-015_D10-025_D20-025_gamma085"
MAIN_EVAL = EXP / f"evaluation_all_methods_{RUN_TAG}"
MAIN_DETECTIONS = EXP / f"method_detections_{RUN_TAG}"
METHOD_IMAGES = EXP / "method_images"
NATIVE_POLICY_SWEEP = EXP / "native_evidence_policy_sweep_yolo26_coordinate"
PAPER = ROOT / "paper_trc_rmrnet"
FIG_DIR = PAPER / "figures"
TABLE_DIR = PAPER / "tables"

sys.path.insert(0, str(ROOT / "tools"))
from eval_yolo26_coordinate_gt46 import COLORS, load_gt, load_predictions  # noqa: E402


METHOD_LABELS = {
    "raw": "Raw native",
    "rmr_blind": "RMR image-only",
    "rmr_metadata": "RMR metadata",
    "rmr_metadata_gated": "RMR gated",
    "rmr_native_gate_gamma085": "RMR native gate",
    "rmr_eta_0p1": "RMR weak residual",
    "nafnet": "NAFNet",
    "dfpir": "DFPIR",
    "demoe_auto": "DeMoE-auto",
    "demoe_scenario": "DeMoE-scenario",
    "instructir_generic": "InstructIR",
    "instructir_metadata": "InstructIR-meta",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric(rows: list[dict[str, str]], run: str, mode: str, iou: float, key: str) -> float:
    for row in rows:
        if row["run"] == run and row["mode"] == mode and math.isclose(float(row["iou"]), iou):
            return float(row[key])
    raise KeyError((run, mode, iou, key))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def write_main_table() -> None:
    src = MAIN_EVAL / "table_geotagged_native_pilot.tex"
    dst = TABLE_DIR / "table_geotagged_native_pilot.tex"
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def write_protocol_table() -> None:
    text = r"""\begin{table*}[!t]
\centering
\caption{Protocol lock for the revised GT46 native-coordinate field test. The revised Roboflow labels are used only for final scoring; the detector and restoration outputs are fixed before evaluation.}
\label{tab:native_protocol_lock}
\small
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{lp{0.58\textwidth}p{0.22\textwidth}}
\toprule
Item & Locked setting & Label use \\
\midrule
Dataset & Revised high-quality Sony cam1 field subset with 46 native 4752$\times$3168 frames. Forty-four frames contain 167 common-class GT boxes: longitudinal crack, transverse crack, alligator crack, and pothole. & Final scoring only. \\
Detector & Frozen YOLO26s RDD-FRDC distilled checkpoint, run by \texttt{Yolo26\_coordinate.py}. No GT46 detector fine-tuning is used. & No GT46 labels. \\
High-resolution inference & Full-image inference only with \texttt{imgsz=1280}, \texttt{conf=0.10}, and class thresholds \texttt{D00=0.15,D10=0.25,D20=0.25}. No crop, road mask, tile fusion, or hand-designed ROI is used in this revised result. & No GT46 labels. \\
Metadata handling & Real cam1 pose/geolocation metadata from \texttt{precise\_cam1\_coords.csv} is matched by camera filename for geospatial detector output. Detection boxes are still produced from pixels by the frozen detector. & No GT46 labels. \\
Primary field score & Same-primary-class crack-versus-pothole recovery at IoU 0.10 and IoU 0.50. Exact crack subtype F1 is reported separately because the detector often confuses longitudinal, transverse, and alligator cracks. & Final scoring only. \\
Tolerant recovery & A GT defect is counted as recovered when a same-primary-class prediction has IoU$\geq$0.10, covers at least 25\% of the GT area, or contains the GT-box center. & Final scoring only. \\
\bottomrule
\end{tabular}
\end{table*}
"""
    (TABLE_DIR / "table_native_protocol_lock.tex").write_text(text, encoding="utf-8")


def write_native_gate_sweep_table() -> None:
    rows = read_rows(NATIVE_POLICY_SWEEP / "policy_sweep_summary.csv")
    labels = [
        ("identity_q100", "Raw native"),
        ("gamma090_sharp120", "$\\gamma=0.90$, sharpness 1.20"),
        ("contrast110_gamma095_sharp110", "contrast 1.10, $\\gamma=0.95$, sharpness 1.10"),
        ("gamma085_sharp110", "$\\gamma=0.85$, sharpness 1.10"),
        ("gamma090_sharp110", "$\\gamma=0.90$, sharpness 1.10"),
        ("autocontrast02_sharp110", "autocontrast 0.02, sharpness 1.10"),
    ]
    by_tag = {row["tag"]: row for row in rows}
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Native evidence-gate calibration on the revised GT46 coordinate test. Policies are deterministic, preserve the native image size, and are much weaker than full restoration. The selected deployment gate, $\gamma=0.85$ with sharpness 1.10, maximizes tolerant GT recovery while keeping F1 close to raw native inference.}",
        r"\label{tab:native_gate_sweep}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Policy & GT success & P@.10 & R@.10 & F1@.10 \\",
        r"\midrule",
    ]
    for tag, label in labels:
        row = by_tag[tag]
        succ = float(row["gt_success"])
        p = float(row["precision_iou10"])
        r = float(row["recall_iou10"])
        f = float(row["f1_iou10"])
        lines.append(f"{label} & {succ:.3f} & {p:.3f} & {r:.3f} & {f:.3f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (TABLE_DIR / "table_native_gate_sweep.tex").write_text("\n".join(lines), encoding="utf-8")


def write_uncertainty_table() -> None:
    rows = read_rows(MAIN_EVAL / "summary_metrics.csv")
    selected = [
        ("raw", "Raw native"),
        ("rmr_metadata_gated", "RMR-Net metadata-gated"),
        ("rmr_native_gate_gamma085", "RMR-Net native gate"),
        ("demoe_auto", "DeMoE-auto"),
    ]
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{GT46 native-coordinate uncertainty audit. Wilson intervals are shown for tolerant known-GT success over 167 common-class annotations. Intervals overlap, so the native field result is interpreted as a safety and deployment-policy test rather than a large restoration win.}",
        r"\label{tab:native_uncertainty}",
        r"\small",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & GT success & 95\% Wilson CI & F1@.10 \\",
        r"\midrule",
    ]
    for run, label in selected:
        gt = int(metric(rows, run, "primary_success", -1.0, "gt"))
        tp = int(metric(rows, run, "primary_success", -1.0, "tp"))
        succ = tp / gt
        lo, hi = wilson(tp, gt)
        f = metric(rows, run, "primary", 0.10, "f1")
        lines.append(f"{label} & {succ:.3f} & [{lo:.3f}, {hi:.3f}] & {f:.3f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (TABLE_DIR / "table_native_uncertainty.tex").write_text("\n".join(lines), encoding="utf-8")


def make_main_metric_figure() -> None:
    rows = read_rows(MAIN_EVAL / "summary_metrics.csv")
    runs = ["raw", "rmr_native_gate_gamma085", "rmr_metadata_gated", "demoe_auto", "instructir_generic", "nafnet", "dfpir"]
    labels = [METHOD_LABELS[r] for r in runs]
    f10 = [metric(rows, r, "primary", 0.10, "f1") for r in runs]
    success = [metric(rows, r, "primary_success", -1.0, "recall") for r in runs]
    f50 = [metric(rows, r, "primary", 0.50, "f1") for r in runs]
    x = range(len(runs))
    width = 0.24
    plt.figure(figsize=(9.4, 4.4))
    plt.bar([i - width for i in x], success, width, label="GT success", color="#4477AA")
    plt.bar(list(x), f10, width, label="F1@0.10", color="#66AA55")
    plt.bar([i + width for i in x], f50, width, label="F1@0.50", color="#CC6677")
    plt.xticks(list(x), labels, rotation=25, ha="right")
    plt.ylabel("score")
    plt.ylim(0, 0.52)
    plt.grid(axis="y", alpha=0.25)
    plt.legend(ncol=3, frameon=False, loc="upper left")
    plt.tight_layout()
    for target in [FIG_DIR / "fig_geotagged_gt46_coordinate_prf.png", MAIN_EVAL / "fig_gt46_coordinate_prf.png"]:
        plt.savefig(target, dpi=220)
    plt.close()


def make_native_gate_figure() -> None:
    rows = read_rows(NATIVE_POLICY_SWEEP / "policy_sweep_summary.csv")
    keep = ["identity_q100", "gamma090_sharp120", "contrast110_gamma095_sharp110", "gamma085_sharp110", "gamma090_sharp110", "autocontrast02_sharp110"]
    labels = ["raw", "g.90+s1.20", "c1.10+g.95+s1.10", "g.85+s1.10", "g.90+s1.10", "auto+s1.10"]
    by_tag = {row["tag"]: row for row in rows}
    f1 = [float(by_tag[tag]["f1_iou10"]) for tag in keep]
    success = [float(by_tag[tag]["gt_success"]) for tag in keep]
    plt.figure(figsize=(7.4, 3.8))
    plt.plot(labels, f1, marker="o", label="F1@0.10", color="#117733")
    plt.plot(labels, success, marker="s", label="GT success", color="#332288")
    plt.xlabel("native gate policy")
    plt.ylabel("score")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0.28, 0.40)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    for target in [FIG_DIR / "fig_geotagged_gate_sweep.png", NATIVE_POLICY_SWEEP / "fig_gt46_native_gate_sweep.png"]:
        plt.savefig(target, dpi=220)
    plt.close()


def make_example_figure() -> None:
    gt, _counts = load_gt(ROOT / "road-defect-seg-9junedata.coco-segmentation" / "train" / "_annotations.coco.json")
    methods = [
        ("Raw", "raw"),
        ("RMR native gate", "rmr_native_gate_gamma085"),
        ("RMR metadata full", "rmr_metadata"),
        ("DeMoE-auto", "demoe_auto"),
    ]
    preds = {title: load_predictions(MAIN_DETECTIONS / key) for title, key in methods}
    chosen = sorted(gt, key=lambda n: len(gt[n]), reverse=True)[:4]
    font = ImageFont.load_default()
    cell_w, cell_h = 520, 346
    rows = []
    for image in chosen:
        panels = []
        for title, key in methods:
            pred = preds[title]
            base_path = METHOD_IMAGES / key / image
            if not base_path.exists():
                base_path = EXP / "gt46_native_images" / image
            base = Image.open(base_path).convert("RGB")
            base.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
            sx, sy = base.width / 4752.0, base.height / 3168.0
            panel = Image.new("RGB", (cell_w, cell_h + 26), (255, 255, 255))
            panel.paste(base, ((cell_w - base.width) // 2, 26))
            d = ImageDraw.Draw(panel)
            d.text((6, 7), f"{title}: GT={len(gt[image])}, pred={len(pred.get(image, []))}", fill=(0, 0, 0), font=font)
            xoff = (cell_w - base.width) // 2
            yoff = 26
            for g in gt[image]:
                x1, y1, x2, y2 = g["box"]
                d.rectangle((xoff + x1 * sx, yoff + y1 * sy, xoff + x2 * sx, yoff + y2 * sy), outline=(255, 214, 10), width=2)
            for p in pred.get(image, []):
                x1, y1, x2, y2 = p["box"]
                color = COLORS[p["label"]]
                d.rectangle((xoff + x1 * sx, yoff + y1 * sy, xoff + x2 * sx, yoff + y2 * sy), outline=color, width=2)
            panels.append(panel)
        row = Image.new("RGB", (cell_w * len(panels), cell_h + 26), (255, 255, 255))
        for i, panel in enumerate(panels):
            row.paste(panel, (i * cell_w, 0))
        rows.append(row)
    canvas = Image.new("RGB", (cell_w * 4, (cell_h + 26) * len(rows)), (255, 255, 255))
    for i, row in enumerate(rows):
        canvas.paste(row, (0, i * (cell_h + 26)))
    for target in [FIG_DIR / "fig_gt46_coordinate_examples.jpg", MAIN_EVAL / "fig_gt46_coordinate_examples.jpg"]:
        canvas.save(target, quality=92)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    write_main_table()
    write_protocol_table()
    write_native_gate_sweep_table()
    write_uncertainty_table()
    make_main_metric_figure()
    make_native_gate_figure()
    make_example_figure()
    print(f"[OK] wrote GT46 paper assets to {PAPER}")


if __name__ == "__main__":
    main()
