from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments" / "geotagged_cam1_native"
PAPER = ROOT / "paper_ieee_tits_rmrnet"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"


DISPLAY = {
    "raw": "Raw native",
    "rmr_blind": "RMR image-only",
    "rmr_metadata": "RMR metadata",
    "nafnet": "NAFNet-road",
    "dfpir": "DFPIR",
    "demoe_auto": "DeMoE-auto",
    "demoe_scenario": "DeMoE-scenario",
    "instructir_generic": "InstructIR-generic",
    "instructir_metadata": "InstructIR-metadata",
}


def read_runtime() -> dict[str, float]:
    path = EXP / "restored" / "restore_summary.csv"
    rows: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.setdefault(row["model"], []).append(float(row["runtime_s"]))
    return {k: sum(v) / len(v) for k, v in rows.items() if v}


def read_snake(method: str) -> dict:
    path = EXP / "snake" / method / "snake_boundary_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_rows() -> list[dict[str, object]]:
    det = read_csv_rows(EXP / "detection" / "geotagged_native_detection_summary.csv")
    runtimes = read_runtime()
    rows: list[dict[str, object]] = []
    for row in det:
        method = row["method"]
        snake = read_snake(method)
        images = int(row["images"])
        rows.append(
            {
                "method": method,
                "display": DISPLAY.get(method, method),
                "metadata": "yes" if "metadata" in method else ("generic prompt" if "generic" in method else "no"),
                "images": images,
                "detections_per_image": float(row["detections_per_image"]),
                "accepted_per_image": float(snake["successes"]) / max(images, 1),
                "contour_success_rate": float(snake["success_rate"]),
                "mean_confidence": float(row["mean_confidence"]),
                "runtime_s": runtimes.get(method, 0.0),
            }
        )
    return rows


def write_table(rows: list[dict[str, object]]) -> Path:
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Native-resolution geotagged cam1 field pilot. The eight-image pilot uses 4752$\times$3168 outputs, no resized saved images, and native-tile YOLO scanning. The folder has no ground-truth defect labels, so the table reports detection yield and accepted-contour yield rather than mAP accuracy. Metadata rows use either RMR's numeric code path or InstructIR's metadata-derived natural-language prompt.}",
        r"\label{tab:geotagged_native_pilot}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabularx}{\textwidth}{lccrrrr}",
        r"\toprule",
        r"Method & Metadata & Images & Det./image & Accepted contours/image & Contour success & Runtime s/image \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['display']} & {row['metadata']} & {int(row['images'])} & "
            f"{fmt(float(row['detections_per_image']))} & {fmt(float(row['accepted_per_image']))} & "
            f"{fmt(100.0 * float(row['contour_success_rate']), 1)}\\% & {fmt(float(row['runtime_s']), 2)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table*}"]
    out = TABLES / "table_geotagged_native_pilot.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def write_figure(rows: list[dict[str, object]]) -> Path:
    colors = {
        "Raw native": "#8d8d8d",
        "RMR image-only": "#59a14f",
        "RMR metadata": "#2f7f4f",
        "NAFNet-road": "#4e79a7",
        "DFPIR": "#f28e2b",
        "DeMoE-auto": "#7f6dba",
        "DeMoE-scenario": "#b279a2",
        "InstructIR-generic": "#e15759",
        "InstructIR-metadata": "#c43c39",
    }
    labels = [str(row["display"]) for row in rows]
    x = range(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), dpi=190)
    axes[0].bar(x, [float(row["detections_per_image"]) for row in rows], color=[colors.get(l, "#999") for l in labels])
    axes[0].set_ylabel("detections / image")
    axes[0].set_title("Native-tile YOLO yield")
    axes[1].bar(x, [float(row["accepted_per_image"]) for row in rows], color=[colors.get(l, "#999") for l in labels])
    axes[1].set_ylabel("accepted contours / image")
    axes[1].set_title("Detector-to-contour yield")
    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=38, ha="right")
        ax.grid(axis="y", color="#d8dee6", alpha=0.75)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Geotagged high-resolution field pilot: yield metrics, not accuracy", y=1.04)
    fig.tight_layout()
    out = FIGURES / "fig_geotagged_native_pilot.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    table = write_table(rows)
    fig = write_figure(rows)
    summary = {
        "table": str(table),
        "figure": str(fig),
        "rows": rows,
    }
    (EXP / "paper_asset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"table": str(table), "figure": str(fig)}, indent=2))


if __name__ == "__main__":
    main()
