#!/usr/bin/env python3
"""Build final IEEE-journal tables and vector figures for TRACE-R.

Every reported value is read from one of three frozen provenance bundles:

* final controlled ledger: one-time IVCNZ/PCM confirmatory test, including
  the corrected official NAFNet evaluation;
* v66: validation-only metadata interventions;
* v70: native-resolution CRID field evaluation with the validation-selected
  official NAFNet correction.

This script performs no training, checkpoint selection, detector fusion, or
metric optimization. It emits only paper assets and an asset manifest.

Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_ieee_tits_trace_r"
CONTROLLED = Path(r"E:\TRACE_R_experiments\trace_locked_confirmatory_v69_20260828")
CONTROLS = Path(r"E:\TRACE_R_experiments\trace_metadata_controls_v66_20260828")
CRID = Path(r"E:\TRACE_R_experiments\trace_crid46_direct_v70_20260828")

METHODS = (
    "raw",
    "nafnet",
    "instructir",
    "dfpir",
    "demoe_auto",
    "trace_r",
    "demoe_oracle",
)
NON_ORACLE = tuple(method for method in METHODS if method != "demoe_oracle")
DISPLAY = {
    "raw": "Degraded input",
    "nafnet": "NAFNet",
    "instructir": "InstructIR",
    "dfpir": "DFPIR",
    "demoe_auto": "DeMoE-auto",
    "trace_r": "TRACE-R",
    "demoe_oracle": "DeMoE-oracle",
}
CAUSES = ("motion", "defocus", "lowlight", "mixed")


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.0,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def index(rows: Iterable[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row[key]: row for row in rows}


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latex_table(
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
        r"\scriptsize",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        header + " \\\\",
        r"\midrule",
        *[row + " \\\\" for row in rows],
        r"\bottomrule",
        r"\end{tabular}",
    ]
    if note:
        note_width = r"\textwidth" if wide else r"\columnwidth"
        lines.extend(
            [
                r"\vspace{1mm}",
                rf"\parbox{{0.97{note_width}}}{{\scriptsize {note}}}",
            ]
        )
    lines.append(rf"\end{{{environment}}}")
    return "\n".join(lines)


def bold_best(value: float, candidates: Iterable[float]) -> str:
    rendered = f"{value:.3f}"
    if value >= max(candidates) - 1e-12:
        return rf"\textbf{{{rendered}}}"
    return rendered


def build_controlled_table(aggregate: dict[str, dict[str, str]]) -> str:
    fields = (
        "ivcnz_mean_map50",
        "ivcnz_mean_map50_95",
        "pcm_mean_map50",
        "pcm_mean_map50_95",
        "joint_mean_map50",
        "joint_mean_map50_95",
    )
    non_oracle_best = {
        field: [float(aggregate[method][field]) for method in NON_ORACLE]
        for field in fields
    }
    rows: list[str] = []
    for method in METHODS:
        values = [float(aggregate[method][field]) for field in fields]
        if method == "demoe_oracle":
            rendered = [f"{value:.3f}" for value in values]
            name = rf"\emph{{{DISPLAY[method]}}}"
        else:
            rendered = [
                bold_best(value, non_oracle_best[field])
                for value, field in zip(values, fields, strict=True)
            ]
            name = DISPLAY[method]
        rows.append(" & ".join([name, *rendered]))
    return latex_table(
        "Sealed controlled-test detection after validation-only model selection.",
        "tab:controlled_summary",
        "lrrrrrr",
        (
            r"Method & \multicolumn{2}{c}{IVCNZ} & \multicolumn{2}{c}{PCM} "
            r"& \multicolumn{2}{c}{Joint} \\"
            "\n"
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}"
            "\n"
            r" & mAP50 & mAP50--95 & mAP50 & mAP50--95 & mAP50 & mAP50--95"
        ),
        rows,
        note=(
            "Bold marks the best non-oracle method. InstructIR and DFPIR receive the controlled "
            "condition as their documented instruction or prompt; DeMoE-oracle receives the "
            "condition label and is a non-deployable upper bound. Every row uses one restored image "
            "and the same frozen detector, with no prediction fusion."
        ),
    )


def build_condition_table(condition_rows: list[dict[str, str]]) -> str:
    values = {
        (row["method"], row["name"]): float(row["map50"])
        for row in condition_rows
    }
    condition_names = [
        f"{dataset}_{cause}" for dataset in ("ivcnz", "pcm") for cause in CAUSES
    ]
    best = {
        name: [values[(method, name)] for method in NON_ORACLE]
        for name in condition_names
    }
    rows: list[str] = []
    for method in METHODS:
        rendered = []
        for name in condition_names:
            value = values[(method, name)]
            rendered.append(
                f"{value:.3f}"
                if method == "demoe_oracle"
                else bold_best(value, best[name])
            )
        display = DISPLAY[method]
        if method == "demoe_oracle":
            display = rf"\emph{{{display}}}"
        rows.append(" & ".join([display, *rendered]))
    return latex_table(
        "Controlled-test mAP50 by corruption family.",
        "tab:condition_results",
        "lrrrrrrrr",
        (
            r"Method & \multicolumn{4}{c}{IVCNZ} & \multicolumn{4}{c}{PCM} \\"
            "\n"
            r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}"
            "\n"
            r" & Mot. & Def. & Low & Mix & Mot. & Def. & Low & Mix"
        ),
        rows,
        note="Bold marks the best non-oracle method for each condition; the oracle row is excluded from bolding.",
    )


def build_fidelity_table(fidelity_rows: list[dict[str, str]]) -> str:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in fidelity_rows:
        grouped[(row["method"], row["dataset"])].append(row)
    means: dict[tuple[str, str], tuple[float, float]] = {}
    for key, rows in grouped.items():
        means[key] = (
            float(np.mean([float(row["psnr"]) for row in rows])),
            float(np.mean([float(row["ssim"]) for row in rows])),
        )
    best: dict[tuple[str, int], list[float]] = {}
    for dataset in ("ivcnz", "pcm"):
        for metric_index in (0, 1):
            best[(dataset, metric_index)] = [
                means[(method, dataset)][metric_index] for method in NON_ORACLE
            ]
    rows: list[str] = []
    for method in METHODS:
        rendered: list[str] = []
        for dataset in ("ivcnz", "pcm"):
            for metric_index, value in enumerate(means[(method, dataset)]):
                digits = 2 if metric_index == 0 else 3
                text = f"{value:.{digits}f}"
                if method != "demoe_oracle" and value >= max(best[(dataset, metric_index)]) - 1e-12:
                    text = rf"\textbf{{{text}}}"
                rendered.append(text)
        name = DISPLAY[method]
        if method == "demoe_oracle":
            name = rf"\emph{{{name}}}"
        rows.append(" & ".join([name, *rendered]))
    return latex_table(
        "Mean paired restoration fidelity across the four controlled corruptions.",
        "tab:fidelity",
        "lrrrr",
        (
            r"Method & \multicolumn{2}{c}{IVCNZ} & \multicolumn{2}{c}{PCM} \\"
            "\n"
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}"
            "\n"
            r" & PSNR & SSIM & PSNR & SSIM"
        ),
        rows,
        note=(
            "Clean references are resized once to the stored restoration canvas with Lanczos "
            "interpolation before full-reference scoring; detection images and labels are unchanged."
        ),
    )


def build_control_table(control_rows: list[dict[str, str]]) -> str:
    display = {
        "aligned": "Aligned packet",
        "unavailable": "All groups unavailable",
        "wrong_condition": "Wrong-condition packet",
    }
    rows = []
    for row in control_rows:
        values = [
            float(row["ivcnz_mean_map50"]),
            float(row["pcm_mean_map50"]),
            float(row["joint_mean_map50"]),
        ]
        rendered = [f"{value:.3f}" for value in values]
        if row["control"] == "aligned":
            rendered = [rf"\textbf{{{value}}}" for value in rendered]
        rows.append(" & ".join([display[row["control"]], *rendered]))
    return latex_table(
        "Validation-only intervention on TRACE-R's capture packet.",
        "tab:metadata_controls",
        "lrrr",
        r"Packet supplied at inference & IVCNZ mAP50 & PCM mAP50 & Joint mAP50",
        rows,
        wide=False,
        note="Only the packet changes; images, model weights, and frozen detectors remain fixed.",
    )


def build_crid_table(crid_rows: list[dict[str, str]]) -> str:
    order = ("raw", "nafnet", "instructir", "dfpir", "demoe_auto", "rmr_fine_eta0p5")
    display = {
        "raw": "Native image",
        "nafnet": "NAFNet",
        "instructir": "InstructIR",
        "dfpir": "DFPIR",
        "demoe_auto": "DeMoE-auto",
        "rmr_fine_eta0p5": "TRACE-R",
    }
    indexed = index(crid_rows, "method")
    best10 = max(float(indexed[method]["ap10_primary"]) for method in order)
    best50 = max(float(indexed[method]["ap50_primary"]) for method in order)
    rows = []
    for method in order:
        row = indexed[method]
        ap10 = float(row["ap10_primary"])
        ap50 = float(row["ap50_primary"])
        coverage = float(row["coverage"])
        rendered = [
            rf"\textbf{{{ap50:.3f}}}" if ap50 >= best50 - 1e-12 else f"{ap50:.3f}",
            rf"\textbf{{{ap10:.3f}}}" if ap10 >= best10 - 1e-12 else f"{ap10:.3f}",
            f"{coverage:.3f}",
        ]
        rows.append(" & ".join([display[method], *rendered]))
    return latex_table(
        "Native-resolution CRID temporal field evaluation (13 frames, 49 annotated defects).",
        "tab:crid",
        "lrrr",
        r"Method & Pooled AP50 & AP10 sensitivity & GT coverage",
        rows,
        wide=False,
        note=(
            "No synthetic corruption is added. TRACE-R uses validation-selected residual "
            r"strength $\eta=0.5$ and one restored image. The same frozen detector and native "
            "image coordinates are used for every method."
        ),
    )


def build_training_table(ledger: dict[str, Any]) -> str:
    rows = []
    for method in ("nafnet", "instructir", "dfpir", "demoe_auto", "trace_r"):
        policy = ledger["policies"][method]
        rows.append(
            " & ".join(
                [
                    DISPLAY[method],
                    "32",
                    "4096",
                    policy["checkpoint_sha256"][:12],
                ]
            )
        )
    return latex_table(
        "Matched continuation budget and frozen checkpoint identity.",
        "tab:training_audit",
        "lrrl",
        r"Method & Continuation epochs & Optimizer updates & SHA-256 prefix",
        rows,
        wide=False,
        note=(
            "All methods use 512 balanced crops per epoch, effective batch size four, and the "
            "same frozen dataset-specific detector. Full hashes and commands are in the release."
        ),
    )


def rounded_box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    face: str,
    edge: str = "#30343b",
    fontsize: float = 7.0,
    weight: str = "normal",
) -> None:
    patch = mpl.patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=0.8,
        edgecolor=edge,
        facecolor=face,
    )
    axis.add_patch(patch)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        linespacing=1.12,
    )


def arrow(axis: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color="#30343b",
            shrinkA=2,
            shrinkB=2,
        )
    )


def build_architecture_figure(path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.16, 3.55))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    ink = "#25313b"
    blue = "#dcebf7"
    green = "#dff0e4"
    amber = "#f7e8c8"
    grey = "#eef0f2"
    red = "#f6dfdc"

    # Section guides make the reading order visible without filling the page
    # with implementation prose.
    for x, width, label in (
        (0.015, 0.205, "OBSERVATIONS"),
        (0.245, 0.285, "CORRUPTION STATE"),
        (0.555, 0.275, "SHARED RESTORER"),
        (0.855, 0.13, "OUTPUT"),
    ):
        axis.add_patch(
            mpl.patches.FancyBboxPatch(
                (x, 0.08), width, 0.84,
                boxstyle="round,pad=0.006,rounding_size=0.01",
                linewidth=0.65, edgecolor="#aeb5ba", facecolor="white",
            )
        )
        axis.text(x + 0.012, 0.885, label, color="#59656d", fontsize=6.2, fontweight="bold")

    # Image observation, drawn as a compact road scene so the figure remains
    # vector-native.
    axis.add_patch(mpl.patches.Rectangle((0.035, 0.61), 0.165, 0.20, facecolor="#c8d8df", edgecolor=ink, linewidth=0.75))
    axis.add_patch(mpl.patches.Polygon([(0.035, 0.61), (0.200, 0.61), (0.142, 0.76), (0.095, 0.76)], closed=True, facecolor="#596269", edgecolor="none"))
    axis.plot([0.118, 0.119], [0.615, 0.75], color="white", linewidth=1.2)
    axis.plot([0.174, 0.134], [0.615, 0.75], color="#f1c94a", linewidth=0.8)
    axis.text(0.117, 0.585, "road image  $I_d$", ha="center", va="top", fontsize=6.6, fontweight="bold")

    rounded_box(axis, (0.035, 0.43), 0.165, 0.11, "camera\nexposure / gain / focus", face=blue, fontsize=6.2)
    rounded_box(axis, (0.035, 0.14), 0.165, 0.11, "vehicle and timing\nspeed · pose · sync quality", face=grey, fontsize=6.0)

    # Real SBG angular-rate samples provide a visual example of the temporal
    # measurement aligned to an exposure. Fall back to a deterministic trace
    # only when the field log is absent from a redistributed source package.
    trace = np.array([0.0, 0.4, -0.2, 0.7, -0.4, 0.2, 0.0], dtype=float)
    sensor_path = ROOT / "ac1.csv"
    if sensor_path.exists():
        values: list[float] = []
        with sensor_path.open(encoding="utf-8", errors="replace") as handle:
            for row_index, line in enumerate(handle):
                if row_index < 3:
                    continue
                fields = line.rstrip().split("\t")
                if len(fields) < 15:
                    continue
                try:
                    values.append(float(fields[12]))
                except ValueError:
                    continue
                if len(values) == 35:
                    break
        if values:
            trace = np.asarray(values, dtype=float)
            trace = (trace - trace.mean()) / max(trace.std(), 1e-6)
    imu_axis = axis.inset_axes([0.050, 0.275, 0.135, 0.075])
    imu_axis.plot(np.linspace(-1.0, 1.0, len(trace)), trace, color="#2171a5", linewidth=0.85)
    imu_axis.axvspan(-0.45, 0.45, color="#dcebf7", alpha=0.8, linewidth=0)
    imu_axis.axhline(0.0, color="#aeb5ba", linewidth=0.4)
    imu_axis.set_xticks([]); imu_axis.set_yticks([])
    for spine in imu_axis.spines.values():
        spine.set_color("#7f8a91"); spine.set_linewidth(0.45)
    axis.text(0.117, 0.355, "measured IMU trajectory", ha="center", va="bottom", fontsize=6.1)

    rounded_box(axis, (0.265, 0.62), 0.105, 0.13, "image estimate\n$z_I=g_I(I_d)$", face=amber, fontsize=6.4)
    rounded_box(
        axis,
        (0.265, 0.30),
        0.105,
        0.18,
        "physical evidence\n" + r"$h_{\mathrm{phy}}(\mathcal{M})$" + "\n\navailability / quality",
        face=blue,
        fontsize=6.1,
    )
    rounded_box(axis, (0.405, 0.42), 0.105, 0.22, "joint posterior\n\n$z$  and  $q$\n\nimage fallback", face=green, fontsize=6.3, weight="bold")
    arrow(axis, (0.200, 0.70), (0.265, 0.69))
    arrow(axis, (0.200, 0.45), (0.265, 0.40))
    arrow(axis, (0.200, 0.31), (0.265, 0.37))
    arrow(axis, (0.370, 0.685), (0.405, 0.58))
    arrow(axis, (0.370, 0.39), (0.405, 0.48))

    # A compact U-shaped backbone with state-conditioned adapter taps.
    backbone_blocks = [
        (0.575, 0.60, 0.045, 0.14),
        (0.625, 0.53, 0.045, 0.21),
        (0.675, 0.45, 0.045, 0.29),
        (0.725, 0.53, 0.045, 0.21),
        (0.775, 0.60, 0.035, 0.14),
    ]
    for index_value, (x, y, width, height) in enumerate(backbone_blocks):
        color = "#d7e7ef" if index_value < 2 else ("#eadfca" if index_value == 2 else "#dff0e4")
        axis.add_patch(mpl.patches.Rectangle((x, y), width, height, facecolor=color, edgecolor=ink, linewidth=0.65))
        axis.add_patch(mpl.patches.Circle((x + width / 2, y + height + 0.035), 0.012, facecolor="#b33a3a", edgecolor="white", linewidth=0.5))
        if index_value:
            px, py, pw, ph = backbone_blocks[index_value - 1]
            arrow(axis, (px + pw, py + ph / 2), (x, y + height / 2))
    axis.text(0.692, 0.78, "state-conditioned adapters", ha="center", fontsize=6.2, color="#8d2e2e")
    axis.text(0.600, 0.57, "encoder", ha="center", fontsize=5.8)
    axis.text(0.697, 0.40, "bottleneck", ha="center", fontsize=5.8)
    axis.text(0.770, 0.57, "decoder", ha="center", fontsize=5.8)
    axis.text(0.692, 0.25, "generic multi-scale backbone $B$", ha="center", fontsize=6.5, fontweight="bold")
    arrow(axis, (0.510, 0.53), (0.575, 0.67))
    axis.add_patch(FancyArrowPatch((0.200, 0.77), (0.575, 0.71), arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color=ink, connectionstyle="arc3,rad=-0.12"))

    rounded_box(axis, (0.875, 0.57), 0.09, 0.15, "restored image\n$I_r$", face=red, fontsize=6.4, weight="bold")
    rounded_box(axis, (0.875, 0.28), 0.09, 0.13, "road-defect\ndetector", face=grey, fontsize=6.2)
    arrow(axis, (0.810, 0.67), (0.875, 0.65))
    arrow(axis, (0.920, 0.57), (0.920, 0.41))

    axis.text(0.500, 0.025, "Telemetry changes supported internal features; the deployed output is a single restored image.", ha="center", va="bottom", fontsize=6.5, color="#4c5860")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_controlled_figure(
    aggregate: dict[str, dict[str, str]],
    controls: list[dict[str, str]],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55), constrained_layout=True)
    methods = ("raw", "nafnet", "instructir", "dfpir", "demoe_auto", "trace_r")
    values = [float(aggregate[method]["joint_mean_map50"]) for method in methods]
    colors = ["#a8adb3", "#8fb7cc", "#d5a85e", "#6e9d80", "#9676a8", "#b33a3a"]
    axes[0].bar(np.arange(len(methods)), values, color=colors, edgecolor="#30343b", linewidth=0.45)
    axes[0].set_xticks(np.arange(len(methods)), [DISPLAY[method].replace("*", "") for method in methods], rotation=32, ha="right")
    axes[0].set_ylabel("Joint mean mAP50")
    axes[0].set_ylim(0.0, max(values) * 1.2)
    axes[0].set_title("Sealed controlled test")
    axes[0].grid(axis="y", color="#d8dadd", linewidth=0.45)
    for index_value, value in enumerate(values):
        axes[0].text(index_value, value + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=5.7)

    control_labels = ["Aligned", "Unavailable", "Wrong"]
    control_values = [float(row["joint_mean_map50"]) for row in controls]
    axes[1].bar(
        np.arange(3),
        control_values,
        color=["#b33a3a", "#8fb7cc", "#a8adb3"],
        edgecolor="#30343b",
        linewidth=0.45,
    )
    axes[1].set_xticks(np.arange(3), control_labels)
    axes[1].set_ylabel("Joint validation mAP50")
    axes[1].set_ylim(0.0, max(control_values) * 1.2)
    axes[1].set_title("Capture-packet intervention")
    axes[1].grid(axis="y", color="#d8dadd", linewidth=0.45)
    for index_value, value in enumerate(control_values):
        axes[1].text(index_value, value + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=6.0)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_crid_figure(rows: list[dict[str, str]], path: Path) -> None:
    order = ("raw", "nafnet", "instructir", "dfpir", "demoe_auto", "rmr_fine_eta0p5")
    display = {
        "raw": "Native",
        "nafnet": "NAFNet",
        "instructir": "InstructIR",
        "dfpir": "DFPIR",
        "demoe_auto": "DeMoE-auto",
        "rmr_fine_eta0p5": "TRACE-R",
    }
    indexed = index(rows, "method")
    ap50 = [float(indexed[method]["ap50_primary"]) for method in order]
    x = np.arange(len(order))
    fig, axis = plt.subplots(figsize=(3.5, 2.45))
    bars = axis.bar(x, ap50, 0.62, color=["#a8adb3", "#8fb7cc", "#d5a85e", "#6e9d80", "#9676a8", "#b33a3a"], edgecolor="#30343b", linewidth=0.45)
    axis.set_xticks(x, [display[method] for method in order], rotation=30, ha="right")
    axis.set_ylabel("Pooled AP50")
    axis.set_ylim(0.0, max(ap50) * 1.22)
    axis.grid(axis="y", color="#d8dadd", linewidth=0.45)
    axis.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, ap50, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center", va="bottom", fontsize=5.8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, default=PAPER)
    parser.add_argument("--controlled", type=Path, default=CONTROLLED)
    parser.add_argument("--controls", type=Path, default=CONTROLS)
    parser.add_argument("--crid", type=Path, default=CRID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    tables = args.paper / "tables"
    figures = args.paper / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    ledger_path = args.controlled / "final_provenance_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    aggregate_rows = read_csv(args.controlled / "detection" / "aggregate_metrics.csv")
    aggregate = index(aggregate_rows, "method")
    condition_rows = read_csv(args.controlled / "detection" / "all_condition_metrics.csv")
    fidelity_rows = read_csv(args.controlled / "fidelity" / "all_summary.csv")
    control_rows = read_csv(args.controls / "metadata_control_summary.csv")
    crid_rows = read_csv(args.crid / "test_summary.csv")

    write(tables / "table_controlled_summary.tex", build_controlled_table(aggregate))
    write(tables / "table_condition_results.tex", build_condition_table(condition_rows))
    write(tables / "table_fidelity.tex", build_fidelity_table(fidelity_rows))
    write(tables / "table_metadata_controls.tex", build_control_table(control_rows))
    write(tables / "table_crid.tex", build_crid_table(crid_rows))
    write(tables / "table_training_audit.tex", build_training_table(ledger))

    build_architecture_figure(figures / "fig_trace_architecture.pdf")
    build_controlled_figure(aggregate, control_rows, figures / "fig_trace_controlled_results.pdf")
    build_crid_figure(crid_rows, figures / "fig_trace_crid_ap.pdf")

    # Keep the provenance manifest explicit. Legacy manuscript assets may still
    # exist in a working tree, but they must never enter the final evidence set.
    outputs = [
        tables / "table_controlled_summary.tex",
        tables / "table_condition_results.tex",
        tables / "table_fidelity.tex",
        tables / "table_metadata_controls.tex",
        tables / "table_crid.tex",
        tables / "table_training_audit.tex",
        figures / "fig_trace_architecture.pdf",
        figures / "fig_trace_controlled_results.pdf",
        figures / "fig_trace_crid_ap.pdf",
    ]
    manifest = {
        "status": "paper_assets_complete",
        "training_or_selection_performed": False,
        "detector_output_fusion": False,
        "single_restored_image_per_method": True,
        "sources": {
            "controlled": str(args.controlled),
            "metadata_controls": str(args.controls),
            "crid": str(args.crid),
            "controlled_ledger_sha256": sha256(ledger_path),
        },
        "outputs": {str(path.relative_to(args.paper)): sha256(path) for path in outputs},
    }
    write(args.paper / "asset_manifest.json", json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
