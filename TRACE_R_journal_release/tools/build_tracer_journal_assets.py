#!/usr/bin/env python3
"""Build IEEE-journal tables and vector figures for TRACE-R.

All controlled-study values are read from the frozen validation-only ledger.
The script performs no training, model selection, or metric optimization.
Photographic panels are embedded in PDF containers; headings, annotations,
legends, and borders are vector graphics using a Times-compatible font.
"""

from __future__ import annotations

# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>

import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PAPER = ROOT / "paper_ieee_tits_trace_r"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"
LEDGER_PATH = (
    ROOT
    / "experiments"
    / "final_rmrp_v50_validation_ledger_20260824"
    / "provenance_ledger.json"
)

METHODS = ("nafnet", "nafnet_meta", "instructir", "dfpir", "demoe", "trace_r")
DISPLAY = {
    "raw": "Degraded input",
    "nafnet": "NAFNet",
    "nafnet_meta": "FiLM-NAFNet",
    "instructir": "InstructIR",
    "dfpir": "DFPIR",
    "demoe": "DeMoE",
    "rmrp": "TRACE-R",
    "trace_r": "TRACE-R",
}
CRID_POLICY = ROOT / "experiments" / "crid46_tilefix_guarded_policy_20260812"
CAUSES = ("motion", "defocus", "lowlight", "mixed")
CAUSE_LABELS = ("Motion", "Defocus", "Low light", "Mixed")


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.65,
        }
    )


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def indexed_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["method"]: row for row in csv.DictReader(handle)}


def record(ledger: dict[str, Any], method: str) -> dict[str, Any]:
    if method == "trace_r":
        return ledger["rmrp"]
    return ledger["matched_baselines"][method]


def metric(ledger: dict[str, Any], method: str, dataset: str, cause: str) -> float:
    return float(record(ledger, method)["conditions"][f"{dataset}_{cause}"]["map50"])


def bold(value: float, peers: list[float]) -> str:
    rendered = f"{value:.3f}"
    return rf"\textbf{{{rendered}}}" if value >= max(peers) - 1e-12 else rendered


def table(
    caption: str,
    label: str,
    columns: str,
    header: str,
    rows: list[str],
    *,
    wide: bool = True,
    note: str | None = None,
) -> str:
    environment = "table*" if wide else "table"
    lines = [
        rf"\begin{{{environment}}}[!t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.7pt}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    if note:
        lines.extend([r"\vspace{1mm}", rf"\parbox{{0.96\linewidth}}{{\scriptsize {note}}}"])
    lines.append(rf"\end{{{environment}}}")
    return "\n".join(lines)


def build_tables(ledger: dict[str, Any]) -> None:
    rows: list[str] = []
    for method in METHODS:
        item = record(ledger, method)
        ivcnz = float(item["mean_map50"]["ivcnz"])
        pcm = float(item["mean_map50"]["pcm"])
        joint = float(item["joint_mean_map50"])
        rows.append(
            f"{DISPLAY[method]} & "
            f"{bold(ivcnz, [float(record(ledger, m)['mean_map50']['ivcnz']) for m in METHODS])} & "
            f"{bold(pcm, [float(record(ledger, m)['mean_map50']['pcm']) for m in METHODS])} & "
            f"{bold(joint, [float(record(ledger, m)['joint_mean_map50']) for m in METHODS])} "
            + r"\\"
        )
    write(
        TABLES / "table_controlled_summary.tex",
        table(
            "Matched controlled-study detection on disjoint validation partitions.",
            "tab:controlled_summary",
            "lrrr",
            r"Method & IVCNZ mAP50 & PCM mAP50 & Joint mean",
            rows,
            wide=False,
        ),
    )

    condition_rows = []
    for method in METHODS:
        values = [metric(ledger, method, dataset, cause) for dataset in ("ivcnz", "pcm") for cause in CAUSES]
        rendered = []
        for dataset in ("ivcnz", "pcm"):
            for cause in CAUSES:
                value = metric(ledger, method, dataset, cause)
                rendered.append(bold(value, [metric(ledger, peer, dataset, cause) for peer in METHODS]))
        condition_rows.append(f"{DISPLAY[method]} & " + " & ".join(rendered) + r" \\")
    write(
        TABLES / "table_condition_results.tex",
        table(
            "Condition-level mAP50 on IVCNZ and PCM.",
            "tab:condition_results",
            "lrrrrrrrr",
            "Method & \\multicolumn{4}{c}{IVCNZ} & \\multicolumn{4}{c}{PCM} \\\\\n"
            "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
            "& Mot. & Def. & Low & Mix. & Mot. & Def. & Low & Mix.",
            condition_rows,
        ),
    )

    controls = ledger["metadata_controls"]
    control_rows = []
    for key, name in (
        ("correct", "Aligned packet"),
        ("unavailable", "All sensor groups unavailable"),
        ("cross_condition_shuffled", "Wrong-condition packet"),
    ):
        item = controls[key]
        control_rows.append(
            f"{name} & {item['mean_map50']['ivcnz']:.3f} & "
            f"{item['mean_map50']['pcm']:.3f} & {item['joint_mean_map50']:.3f} "
            + r"\\"
        )
    write(
        TABLES / "table_metadata_controls.tex",
        table(
            "TRACE-R packet interventions.",
            "tab:metadata_controls",
            "lrrr",
            r"Packet supplied at inference & IVCNZ & PCM & Joint",
            control_rows,
            wide=False,
        ),
    )

    fidelity_rows = []
    for method in ("raw", "instructir", "dfpir", "demoe", "rmrp"):
        item = ledger["fidelity"][method]
        fidelity_rows.append(
            f"{DISPLAY[method]} & {item['ivcnz']['mean']['psnr']:.2f} & "
            f"{item['ivcnz']['mean']['ssim']:.3f} & {item['pcm']['mean']['psnr']:.2f} & "
            f"{item['pcm']['mean']['ssim']:.3f} " + r"\\"
        )
    write(
        TABLES / "table_fidelity.tex",
        table(
            "Paired fidelity over all four controlled conditions.",
            "tab:fidelity",
            "lrrrr",
            "Method & \\multicolumn{2}{c}{IVCNZ} & \\multicolumn{2}{c}{PCM} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
            "& PSNR & SSIM & PSNR & SSIM",
            fidelity_rows,
        ),
    )

    kitti_rows = [
        r"Degraded input & none & 26.06 & 0.845 & 0.00 \\",
        r"NAFNet & none & 26.03 & 0.848 & -0.03 \\",
        r"FiLM-NAFNet & raw OXTS & 25.96 & 0.846 & -0.10 \\",
        r"TRACE-R & unavailable packet & 26.82 & 0.865 & +0.76 \\",
        r"TRACE-R & raw OXTS & \textbf{26.86} & \textbf{0.865} & \textbf{+0.79} \\",
    ]
    write(
        TABLES / "table_kitti.tex",
        table(
            "Drive-disjoint KITTI transfer with measured OXTS telemetry.",
            "tab:kitti",
            "llrrr",
            r"Method & Capture information & PSNR & SSIM & $\Delta$PSNR",
            kitti_rows,
            wide=False,
        ),
    )

    crid_val = indexed_csv(CRID_POLICY / "validation_summary.csv")
    crid_later = indexed_csv(CRID_POLICY / "supportive_summary.csv")
    crid_methods = (
        ("raw", "Raw image"),
        ("nafnet", "NAFNet"),
        ("dfpir", "DFPIR"),
        ("demoe_auto", "DeMoE"),
        ("instructir", "InstructIR"),
        ("rmr_guarded_dual_view", "TRACE-R"),
    )
    columns: dict[str, list[float]] = {key: [] for key in ("v10", "v50", "vm", "s10", "s50", "sm")}
    crid_values: dict[str, dict[str, float]] = {}
    for method, _ in crid_methods:
        v10 = float(crid_val[method]["ap10_primary"])
        v50 = float(crid_val[method]["ap50_primary"])
        s10 = float(crid_later[method]["ap10_primary"])
        s50 = float(crid_later[method]["ap50_primary"])
        values = {"v10": v10, "v50": v50, "vm": (v10 + v50) / 2, "s10": s10, "s50": s50, "sm": (s10 + s50) / 2}
        crid_values[method] = values
        for key, value in values.items():
            columns[key].append(value)
    crid_rows = []
    for method, name in crid_methods:
        values = crid_values[method]
        rendered = [bold(values[key], columns[key]) for key in ("v10", "v50", "vm", "s10", "s50", "sm")]
        crid_rows.append(f"{name} & " + " & ".join(rendered) + r" \\")
    write(
        TABLES / "table_crid.tex",
        table(
            "CRID native-image detection using a frozen detector.",
            "tab:crid",
            "lrrrrrr",
            "Input & \\multicolumn{3}{c}{Policy block (12)} & \\multicolumn{3}{c}{Later block (13)} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
            "& AP@.10 & AP@.50 & Mean & AP@.10 & AP@.50 & Mean",
            crid_rows,
        ),
    )

    training_rows = [
        r"NAFNet & 70 & 4,096 & joint validation mAP50 \\",
        r"FiLM-NAFNet & 70 & 4,096 & joint validation mAP50 \\",
        r"InstructIR & 70 & 4,096 & joint validation mAP50 \\",
        r"DFPIR & 70 & 4,096 & joint validation mAP50 \\",
        r"DeMoE & 70 & 4,096 & joint validation mAP50 \\",
        r"TRACE-R router & matched experts & 0 & joint validation mAP50 \\",
    ]
    write(
        TABLES / "table_training_audit.tex",
        table(
            "Matched target-domain adaptation and selection budget.",
            "tab:training_audit",
            "lrrl",
            r"Method & Epochs & Optimizer steps & Selection endpoint",
            training_rows,
        ),
    )

    checkpoint_rows = []
    for method in ("demoe", "dfpir", "instructir"):
        item = ledger["matched_baselines"][method]
        checkpoint_rows.append(
            f"{DISPLAY[method]} & 70 & \\texttt{{{item['checkpoint_sha256'][:16]}}} " + r"\\"
        )
    write(
        TABLES / "table_checkpoints.tex",
        table(
            "TRACE-R expert checkpoints.",
            "tab:checkpoints",
            "lrl",
            r"Expert & Adaptation epochs & SHA-256 prefix",
            checkpoint_rows,
            wide=False,
        ),
    )


def add_border(ax: Any) -> None:
    ax.add_patch(
        Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            fill=False,
            edgecolor="black",
            linewidth=0.75,
            clip_on=False,
            zorder=100,
        )
    )


def node(ax: Any, xy: tuple[float, float], wh: tuple[float, float], text: str, face: str) -> None:
    x, y = xy
    width, height = wh
    ax.add_patch(Rectangle((x, y), width, height, facecolor=face, edgecolor="black", linewidth=0.65))
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=6.8)


def connector(ax: Any, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=7, linewidth=0.65, color="black")
    )


def build_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.16, 2.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_border(ax)
    ax.text(0.018, 0.95, "TRACE-R inference", fontweight="bold", va="top", fontsize=8)

    node(ax, (0.025, 0.60), (0.13, 0.16), "Degraded image\n$ I_d $", "#eef6f7")
    node(ax, (0.025, 0.25), (0.13, 0.22), "Camera / IMU / vehicle\npacket $m\\in\\mathbb{R}^{82}$", "#fff2e8")
    node(ax, (0.20, 0.24), (0.18, 0.26), "Physical state $h(m)$\n\nmotion and exposure\nfocus and illumination\nsupport and availability", "#fff2e8")
    node(ax, (0.43, 0.27), (0.13, 0.20), "Cause router\n$M,D,L,X,F$", "#eef0f8")

    node(ax, (0.61, 0.58), (0.14, 0.12), "DFPIR", "#edf6ef")
    node(ax, (0.61, 0.42), (0.14, 0.12), "InstructIR", "#edf6ef")
    node(ax, (0.61, 0.26), (0.14, 0.12), "DeMoE", "#edf6ef")
    node(ax, (0.81, 0.36), (0.16, 0.26), "Restored output\n\nselected expert or\nvalidation-fixed blend\n\n$ I_o\\in[0,1] $", "#eef0f8")

    # Image evidence reaches every expert; telemetry controls only the route.
    connector(ax, (0.155, 0.68), (0.59, 0.68))
    ax.plot([0.59, 0.59], [0.32, 0.68], color="black", linewidth=0.65)
    for y in (0.64, 0.48, 0.32):
        connector(ax, (0.59, y), (0.61, y))
    connector(ax, (0.155, 0.36), (0.20, 0.36))
    connector(ax, (0.38, 0.37), (0.43, 0.37))
    for y in (0.64, 0.48, 0.32):
        connector(ax, (0.56, 0.37), (0.61, y))
    for y in (0.64, 0.48, 0.32):
        connector(ax, (0.75, y), (0.81, 0.49))
    ax.text(0.20, 0.19, "Unavailable sensor groups are masked independently", fontsize=5.9)
    ax.text(0.43, 0.21, "No dataset or corruption label", fontsize=5.9)

    ax.plot([0.02, 0.98], [0.11, 0.11], color="black", linewidth=0.5)
    ax.text(0.025, 0.055, "Matched adaptation:", fontweight="bold", va="center")
    ax.text(
        0.18,
        0.055,
        r"$\mathcal{L}=\mathcal{L}_{char}+\lambda_g\mathcal{L}_{grad}+\lambda_T\mathcal{L}_{TDP}+\lambda_D\mathcal{L}_{det}$",
        va="center",
    )
    ax.text(0.72, 0.055, "Frozen detector; equal budget", va="center")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    fig.savefig(FIGURES / "fig_trace_architecture.pdf")
    plt.close(fig)


def build_controlled_results(ledger: dict[str, Any]) -> None:
    labels = [DISPLAY[method] for method in METHODS]
    ivcnz = [float(record(ledger, method)["mean_map50"]["ivcnz"]) for method in METHODS]
    pcm = [float(record(ledger, method)["mean_map50"]["pcm"]) for method in METHODS]
    controls = ledger["metadata_controls"]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.35), gridspec_kw={"width_ratios": [1.45, 1.0]})
    x = np.arange(len(METHODS))
    width = 0.36
    axes[0].bar(x - width / 2, ivcnz, width, label="IVCNZ", color="#31688e", edgecolor="black", linewidth=0.35)
    axes[0].bar(x + width / 2, pcm, width, label="PCM", color="#d9824b", edgecolor="black", linewidth=0.35)
    axes[0].set_xticks(x, labels, rotation=23, ha="right")
    axes[0].set_ylabel("Mean mAP50")
    axes[0].set_ylim(0, 0.60)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    axes[0].set_title("(a) Matched restoration comparison", loc="left", fontweight="bold")

    keys = ("correct", "unavailable", "cross_condition_shuffled")
    names = ("Aligned", "Unavailable", "Wrong\ncondition")
    values = [float(controls[key]["joint_mean_map50"]) for key in keys]
    bars = axes[1].bar(names, values, color=("#278b74", "#9ba7af", "#bb6258"), edgecolor="black", linewidth=0.4)
    axes[1].set_ylim(0, 0.45)
    axes[1].set_ylabel("Joint mean mAP50")
    axes[1].set_title("(b) Packet-content intervention", loc="left", fontweight="bold")
    for bar, value in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.007, f"{value:.3f}", ha="center", va="bottom")

    for ax in axes:
        ax.grid(axis="y", color="#d7dde1", linewidth=0.45)
        ax.set_axisbelow(True)
        add_border(ax)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.27, wspace=0.34)
    fig.savefig(FIGURES / "fig_trace_controlled_results.pdf")
    plt.close(fig)


def build_controlled_qualitative() -> None:
    import tools.build_rmrp_v50_validation_qualitative as qualitative

    qualitative.PAPER = PAPER
    qualitative.FIGURES = FIGURES
    qualitative.OUT = ROOT / "experiments" / "final_trace_r_qualitative_20260824"
    qualitative.DISPLAY["rmrp"] = "TRACE-R"
    qualitative.DATASETS["ivcnz"]["figure"] = "fig_trace_ivcnz_qualitative.pdf"
    qualitative.DATASETS["pcm"]["figure"] = "fig_trace_pcm_qualitative.pdf"
    qualitative.main()


def build_crid_results() -> None:
    methods = ("Raw", "NAFNet", "DFPIR", "DeMoE", "InstructIR", "TRACE-R")
    keys = ("raw", "nafnet", "dfpir", "demoe_auto", "instructir", "rmr_guarded_dual_view")
    validation = indexed_csv(CRID_POLICY / "validation_summary.csv")
    supportive = indexed_csv(CRID_POLICY / "supportive_summary.csv")
    val_mean = [
        (float(validation[key]["ap10_primary"]) + float(validation[key]["ap50_primary"])) / 2
        for key in keys
    ]
    later_mean = [
        (float(supportive[key]["ap10_primary"]) + float(supportive[key]["ap50_primary"])) / 2
        for key in keys
    ]
    x = np.arange(len(methods))
    width = 0.36
    fig, ax = plt.subplots(figsize=(3.5, 2.25))
    colors_a = ["#aeb6bc"] * 5 + ["#278b74"]
    colors_b = ["#d7dcdf"] * 5 + ["#67b8a6"]
    ax.bar(x - width / 2, val_mean, width, color=colors_a, edgecolor="black", linewidth=0.4, label="Policy block")
    ax.bar(x + width / 2, later_mean, width, color=colors_b, edgecolor="black", linewidth=0.4, label="Later block")
    ax.set_xticks(x, methods, rotation=25, ha="right")
    ax.set_ylim(0.25, 0.54)
    ax.set_ylabel("Mean of AP@.10 and AP@.50")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(axis="y", color="#d7dde1", linewidth=0.45)
    ax.set_axisbelow(True)
    add_border(ax)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.96, bottom=0.29)
    fig.savefig(FIGURES / "fig_trace_crid_ap.pdf")
    plt.close(fig)


def wrap_raster(source: Path, destination: Path, *, figsize: tuple[float, float]) -> None:
    image = Image.open(source).convert("RGB")
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    ax.axis("off")
    add_border(ax)
    fig.subplots_adjust(left=0.008, right=0.992, top=0.992, bottom=0.008)
    fig.savefig(destination, dpi=300)
    plt.close(fig)


def build_wrapped_field_assets() -> None:
    source = ROOT / "paper_automation_in_construction_rmrnet" / "figures"
    wrap_raster(source / "fig_sony_collection_system.png", FIGURES / "fig_trace_crid_collection.pdf", figsize=(7.16, 2.45))


def build_crid_overlays() -> None:
    """Rebuild field overlays with the journal name and frozen field policy."""

    import tools.build_crid_direct_sbg_overlays as crid

    crid.PAPER_FIGURE = FIGURES / "fig_trace_crid_policy_atlas.png"
    crid.SUPPLEMENT_FIGURE = FIGURES / "fig_trace_crid_all13.png"
    crid.METHODS = tuple(
        (key, "TRACE-R" if key == "rmr_guarded_dual_view" else name, folder)
        for key, name, folder in crid.METHODS
    )
    crid.main()
    wrap_raster(
        crid.PAPER_FIGURE,
        FIGURES / "fig_trace_crid_policy_atlas.pdf",
        figsize=(7.16, 3.0),
    )
    wrap_raster(
        crid.SUPPLEMENT_FIGURE,
        FIGURES / "fig_trace_crid_all13.pdf",
        figsize=(7.16, 8.0),
    )


def build_crid_validation_example() -> None:
    """Show one CRID validation frame with vector labels and box overlays."""

    import tools.build_crid_direct_sbg_overlays as crid
    from tools.run_crid46_sequence_disjoint_comparison import (
        load_compact_predictions,
        subset_gt,
    )

    selected = ["Cam1_2026-06-09_14-56-15_capt0465_466.jpg"]
    ground_truth = subset_gt(selected, crid.LABEL_ROOT)
    operating_rows = json.loads(
        (crid.OPERATING / "frozen_operating_points_before_test.json").read_text(
            encoding="utf-8"
        )
    )["operating_points"]
    thresholds = {row["method"]: float(row["threshold"]) for row in operating_rows}
    guard = json.loads(
        (crid.GUARDED / "frozen_policy_before_supportive_test.json").read_text(
            encoding="utf-8"
        )
    )
    thresholds["rmr_guarded_dual_view"] = float(
        guard["detector_operating_point"]["threshold"]
    )
    methods = tuple(
        (key, "TRACE-R" if key == "rmr_guarded_dual_view" else name, folder)
        for key, name, folder in crid.METHODS
    )
    prediction_maps = {}
    for method, _, _ in methods:
        path = (
            crid.GUARDED / "val_predictions/rmr_guarded_dual_view.csv"
            if method == "rmr_guarded_dual_view"
            else crid.EXPERIMENT / "val_predictions" / f"{method}.csv"
        )
        prediction_maps[method] = load_compact_predictions(path)
    decision_rows = json.loads(
        (crid.GUARDED / "val_predictions/rmr_guarded_dual_view.json").read_text(
            encoding="utf-8"
        )
    )["decisions"]
    decisions = {row["image"]: row for row in decision_rows}
    native_images = crid.by_stem(crid.NATIVE)
    restored_images = crid.by_stem(crid.EXPERIMENT / "current_rmr_restored/val")
    image_roots = {}
    for method, _, folder in methods:
        if method != "rmr_guarded_dual_view":
            image_roots[method] = crid.by_stem(folder)
        else:
            image_roots[method] = {
                Path(name).stem: (
                    native_images[Path(name).stem]
                    if decisions[name]["policy_output"] == "native"
                    else restored_images[Path(name).stem]
                )
                for name in selected
            }
    name = selected[0]
    native = Image.open(image_roots["raw"][Path(name).stem]).convert("RGB")
    crop = crid.crop_for(ground_truth[name], native.size, aspect=1.48)
    left, top, _, _ = crop

    panels: list[tuple[str, str | None]] = [("Ground truth", None)] + [
        (display, method) for method, display, _ in methods
    ]
    fig = plt.figure(figsize=(7.16, 2.62))
    grid = fig.add_gridspec(2, 8, hspace=0.16, wspace=0.08)
    slots = [(0, 0), (0, 2), (0, 4), (0, 6), (1, 1), (1, 3), (1, 5)]

    for (title, method), (row, column) in zip(panels, slots):
        ax = fig.add_subplot(grid[row, column : column + 2])
        source = native if method is None else Image.open(
            image_roots[method][Path(name).stem]
        ).convert("RGB")
        ax.imshow(np.asarray(source.crop(crop)))
        ax.set_title(title, fontweight="bold", pad=2.0)
        ax.set_xticks([])
        ax.set_yticks([])

        # Yellow rectangles are manual annotations. Green rectangles are
        # detector predictions retained at validation-selected thresholds.
        for item in ground_truth[name]:
            x1, y1, x2, y2 = item["box"]
            ax.add_patch(
                Rectangle(
                    (x1 - left, y1 - top),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="#f5be2d",
                    linewidth=0.85,
                )
            )
        if method is not None:
            for item in prediction_maps[method].get(name, []):
                if float(item["conf"]) < thresholds[method]:
                    continue
                x1, y1, x2, y2 = item["box"]
                ax.add_patch(
                    Rectangle(
                        (x1 - left, y1 - top),
                        x2 - x1,
                        y2 - y1,
                        fill=False,
                        edgecolor="#14b07d",
                        linewidth=0.75,
                    )
                )
        if method == "rmr_guarded_dual_view":
            note = crid.decision_note(method, name, decisions)
            ax.text(
                0.02,
                0.03,
                note,
                transform=ax.transAxes,
                fontsize=5.2,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "black",
                    "linewidth": 0.3,
                    "pad": 1.5,
                },
            )
        add_border(ax)

    fig.text(
        0.5,
        0.018,
        "Yellow: manual annotation    Green: prediction at the validation-selected operating threshold",
        ha="center",
        fontsize=6.2,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.965, bottom=0.075)
    fig.savefig(FIGURES / "fig_trace_crid_validation_example.pdf")
    plt.close(fig)


def main() -> None:
    configure_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    if ledger.get("status") != "FROZEN_VALIDATION_ONLY" or ledger.get("test_split_used") is not False:
        raise RuntimeError("TRACE-R paper assets require the frozen validation-only ledger")
    build_tables(ledger)
    build_architecture()
    build_controlled_results(ledger)
    build_controlled_qualitative()
    build_crid_results()
    build_wrapped_field_assets()
    build_crid_overlays()
    build_crid_validation_example()
    manifest = {
        "status": "COMPLETE",
        "method": "TRACE-R",
        "source_ledger": LEDGER_PATH.relative_to(ROOT).as_posix(),
        "test_split_used": False,
        "figure_format": "PDF with vector labels and borders",
        "figures": [
            "fig_trace_architecture.pdf",
            "fig_trace_controlled_results.pdf",
            "fig_trace_pcm_qualitative.pdf",
            "fig_trace_ivcnz_qualitative.pdf",
            "fig_trace_crid_ap.pdf",
            "fig_trace_crid_validation_example.pdf",
            "fig_trace_crid_all13.pdf",
        ],
        "tables": [
            "table_controlled_summary.tex",
            "table_condition_results.tex",
            "table_metadata_controls.tex",
            "table_fidelity.tex",
            "table_crid.tex",
            "table_training_audit.tex",
            "table_checkpoints.tex",
        ],
    }
    write(ROOT / "experiments" / "trace_r_journal_asset_manifest_20260824.json", json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
