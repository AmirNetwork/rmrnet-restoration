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
CONTROLLED = Path(r"E:\TRACE_R_experiments\trace_locked_confirmatory_v72_20260828")
CONTROLS = Path(r"E:\TRACE_R_experiments\trace_metadata_controls_v66_20260828")
CRID = Path(r"E:\TRACE_R_experiments\crid320_sealed_test_20260831")

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
        "Sealed controlled-test detection. Dataset scores average four corruption families; Joint averages the two datasets.",
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
        "aligned": "Aligned record",
        "unavailable": "All groups unavailable",
        "wrong_condition": "Inconsistent record",
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
        "Validation-only intervention on TRACE-R's capture record.",
        "tab:metadata_controls",
        "lrrr",
        r"Capture record at inference & IVCNZ mAP50 & PCM mAP50 & Joint mAP50",
        rows,
        wide=False,
        note="Only the capture record changes; images, model weights, and frozen detectors remain fixed.",
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
    """Draw a monochrome, equation-led summary of TRACE-R."""
    fig, axis = plt.subplots(figsize=(7.16, 4.55))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    ink = "#111111"
    fill = "#ffffff"
    pale = "#f3f3f3"

    axis.text(0.50, 0.965, "Inference path", ha="center", va="top", fontsize=8.0, fontweight="bold")
    axis.plot([0.02, 0.98], [0.93, 0.93], color=ink, linewidth=0.65)

    rounded_box(axis, (0.025, 0.655), 0.145, 0.115, "Degraded road image\n" + r"$I_d$", face=fill, fontsize=6.3, weight="bold")
    rounded_box(axis, (0.025, 0.405), 0.145, 0.155, "Exposure-aligned record\n" + r"$\mathcal{M}=\{m_C,M_I,m_V,a,r\}$" + "\ncamera · IMU · vehicle", face=fill, fontsize=5.6)
    rounded_box(axis, (0.205, 0.675), 0.145, 0.095, "Image estimate\n" + r"$z_I=g_I(I_d)$" + "  (3)", face=fill, fontsize=6.0)
    rounded_box(axis, (0.205, 0.410), 0.145, 0.150, "Measured estimate\n" + r"$z_{\rm phy}=h_{\rm phy}(\mathcal{M})$" + "\n" + r"$z_m=\mathrm{clip}(z_{\rm phy}+\Delta z_m)$" + "\n(4)--(5)", face=fill, fontsize=5.25)
    rounded_box(axis, (0.390, 0.525), 0.175, 0.175, "Reliability-aware fusion\n" + r"$\bar z=q\odot z_m+(1-q)\odot z_I$" + "\n" + r"$z=\mathrm{clip}(\bar z+\Delta z_\psi)$" + "\n(6)", face=fill, fontsize=5.55, weight="bold")
    rounded_box(axis, (0.410, 0.350), 0.135, 0.095, "Adapter evidence\n" + r"$s=[z_I,z_m,|z_m-z_I|,q]$", face=fill, fontsize=5.4)

    arrow(axis, (0.170, 0.715), (0.205, 0.722))
    arrow(axis, (0.170, 0.480), (0.205, 0.485))
    arrow(axis, (0.350, 0.722), (0.390, 0.635))
    arrow(axis, (0.350, 0.485), (0.390, 0.575))
    arrow(axis, (0.478, 0.525), (0.478, 0.445))

    rounded_box(axis, (0.615, 0.535), 0.205, 0.190, "Restoration backbone  " + r"$B$" + "\n\nencoder  " + r"$\rightarrow$" + "  bottleneck  " + r"$\rightarrow$" + "  decoder\n\n" + r"$x_{l+1}=B_l(x'_l)$" + "  (7)", face=fill, fontsize=5.7, weight="bold")
    rounded_box(axis, (0.620, 0.340), 0.195, 0.115, "Sensor-conditioned adapters\n" + r"$x'_l=x_l+c_l(s,q)P_l\sigma(D_l(Q_l\mathcal{N}(x_l);s))$" + "\n(8)--(9)", face=pale, fontsize=4.9)
    axis.text(0.718, 0.755, "expert route from " + r"$z$" + " when supported by " + r"$B$", ha="center", va="bottom", fontsize=5.1)
    arrow(axis, (0.565, 0.615), (0.615, 0.640))
    arrow(axis, (0.545, 0.395), (0.620, 0.395))
    arrow(axis, (0.718, 0.455), (0.718, 0.535))
    axis.plot([0.170, 0.185, 0.585], [0.745, 0.825, 0.825], color=ink, linewidth=0.75)
    arrow(axis, (0.585, 0.825), (0.645, 0.725))

    rounded_box(axis, (0.865, 0.555), 0.110, 0.145, "Restored image\n" + r"$I_r=f_\theta(I_d,\mathcal{M})$", face=fill, fontsize=6.0, weight="bold")
    arrow(axis, (0.820, 0.630), (0.865, 0.630))

    axis.plot([0.02, 0.98], [0.285, 0.285], color=ink, linewidth=0.65, linestyle="--")
    axis.text(0.50, 0.265, "Training objectives", ha="center", va="top", fontsize=7.6, fontweight="bold")
    rounded_box(
        axis,
        (0.055, 0.075),
        0.405,
        0.125,
        "Paired road data\n"
        + r"$\mathcal{L}=\mathcal{L}_{\rm fid}+\lambda_t\mathcal{L}_{\rm TDP}+\lambda_d\mathcal{L}_{\rm det}+\lambda_s\mathcal{L}_{\rm state}+\lambda_p\mathcal{L}_{\rm phy}$"
        + "  (15)",
        face=fill,
        fontsize=5.35,
    )
    rounded_box(
        axis,
        (0.540, 0.075),
        0.405,
        0.125,
        "Labelled field calibration\n"
        + r"$\mathcal{L}_{\rm field}=\lambda_d\mathcal{L}_{\rm det}+\beta_0\mathcal{L}_{\rm id}+\beta_1\mathcal{L}_{\rm edge}+\beta_2\mathcal{L}_{\rm TV}$"
        + "  (16)",
        face=fill,
        fontsize=5.35,
    )
    axis.add_patch(FancyArrowPatch((0.745, 0.200), (0.745, 0.335), arrowstyle="-|>", mutation_scale=7, linewidth=0.65, linestyle="--", color=ink))
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

    write(tables / "table_controlled_summary.tex", build_controlled_table(aggregate))
    write(tables / "table_condition_results.tex", build_condition_table(condition_rows))
    write(tables / "table_fidelity.tex", build_fidelity_table(fidelity_rows))
    write(tables / "table_metadata_controls.tex", build_control_table(control_rows))
    write(tables / "table_training_audit.tex", build_training_table(ledger))

    build_architecture_figure(figures / "fig_trace_architecture.pdf")
    build_controlled_figure(aggregate, control_rows, figures / "fig_trace_controlled_results.pdf")

    # Keep the provenance manifest explicit. Legacy manuscript assets may still
    # exist in a working tree, but they must never enter the final evidence set.
    outputs = [
        tables / "table_controlled_summary.tex",
        tables / "table_inference_inputs.tex",
        tables / "table_condition_results.tex",
        tables / "table_fidelity.tex",
        tables / "table_metadata_controls.tex",
        tables / "table_crid.tex",
        tables / "table_training_audit.tex",
        tables / "table_architecture_audit.tex",
        figures / "fig_trace_architecture.pdf",
        figures / "fig_trace_controlled_results.pdf",
        figures / "fig_trace_crid_ap.pdf",
        figures / "fig_trace_ivcnz_qualitative.pdf",
        figures / "fig_trace_pcm_qualitative.pdf",
        figures / "fig_trace_crid_qualitative.pdf",
        figures / "fig_trace_crid_collection.pdf",
        figures / "qualitative_selection_manifest.json",
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
