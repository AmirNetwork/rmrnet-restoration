#!/usr/bin/env python3
"""Build the publication figures used by the Automation in Construction paper.

The script is deliberately presentation-only: it reads frozen CSV/JSON results
and archived qualitative atlases, and never changes experiment outputs.  This
keeps the plotted values tied to the manuscript's provenance ledger while
giving every figure one consistent visual language.
"""

# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_automation_in_construction_rmrnet"
FIG = PAPER / "figures"
SOURCE = ROOT / "artifacts" / "autcon_figure_sources_20260719"
ATLAS_SOURCE = ROOT / "artifacts" / "autcon_controlled_atlases_20260720"
EVIDENCE = ROOT / "experiments" / "major_revision_evidence_20260715"

INK = "#173042"
MUTED = "#60717D"
TEAL = "#0B7A75"
BLUE = "#326A9A"
GOLD = "#D49A24"
CORAL = "#C95D4B"
LIGHT = "#E8EEF1"
PALE_TEAL = "#DCEFED"
PALE_BLUE = "#E2EBF3"
PALE_GOLD = "#F7EDCF"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 320,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def first_metric(path: Path, model: str | None = None) -> dict[str, str]:
    rows = read_csv(path)
    if model is None:
        return rows[0]
    return next(row for row in rows if row.get("model") == model)


def finish_axis(ax: plt.Axes, *, grid: str | None = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=LIGHT, linewidth=0.8)
        ax.set_axisbelow(True)


def value_label(ax: plt.Axes, x: float, y: float, text: str, *, color: str = INK) -> None:
    ax.annotate(text, (x, y), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=8, color=color)


def architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 5.2))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, detail: str, color: str) -> None:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.035,rounding_size=0.06",
            facecolor=color, edgecolor=INK, linewidth=1.15,
        )
        ax.add_patch(patch)
        ax.text(x + 0.12, y + h - 0.22, title, weight="bold", fontsize=10, va="top")
        ax.text(x + 0.12, y + h - 0.55, detail, fontsize=8.2, color=MUTED, va="top", linespacing=1.25)

    def arrow(x1: float, y1: float, x2: float, y2: float, color: str = INK) -> None:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color=color))

    ax.text(0.1, 5.02, "DEPLOYED RESTORATION PATH", fontsize=9.5, color=TEAL, weight="bold")
    box(0.1, 3.15, 1.35, 1.35, "Road image", "$I_d$", PALE_BLUE)
    box(1.85, 3.15, 1.65, 1.35, "Image state", r"8 degradation coordinates" + "\n" + r"$c_b=g_\phi(I_d)$", PALE_TEAL)
    box(1.85, 1.55, 1.65, 1.10, "Capture record", "exposure / motion /\nscenario fields $c_m$", PALE_GOLD)
    box(3.95, 2.55, 1.85, 1.65, "Reliable fusion", "sparse monotone basis\navailability-aware gate\n" + r"$z=\alpha e_m+(1-\alpha)e_b$", PALE_TEAL)
    box(6.25, 2.55, 2.05, 1.65, "Conditioned restorer", "3-scale encoder--decoder\nFiLM + task-evidence gates", PALE_BLUE)
    box(8.75, 2.55, 1.65, 1.65, "Bounded detail", "degradation-aware residual\n" + r"cap $\eta_d=0.12$", PALE_GOLD)
    box(10.80, 2.85, 1.25, 1.05, "Output", "$I_r$ or raw\nsafety path", PALE_TEAL)
    arrow(1.45, 3.82, 1.85, 3.82)
    arrow(3.50, 3.82, 3.95, 3.55)
    arrow(3.50, 2.10, 3.95, 2.95)
    arrow(5.80, 3.38, 6.25, 3.38)
    arrow(8.30, 3.38, 8.75, 3.38)
    arrow(10.40, 3.38, 10.80, 3.38)
    arrow(1.45, 3.42, 6.25, 3.05, color=MUTED)

    ax.text(0.1, 1.12, "TRAINING-ONLY GUIDANCE", fontsize=9.5, color=CORAL, weight="bold")
    train_items = [
        (0.1, "Fidelity", "$L_1$ + gradient +\nfrequency + visibility"),
        (2.45, "Degradation state", "supervised code +\nsparse basis"),
        (4.80, "Detector evidence", "frozen YOLO features\nTDP + CQMix"),
        (7.15, "Stability", "Jacobian estimate +\nfeature anchor"),
        (9.50, "Non-regression", "evidence and detail\ncopy safeguards"),
    ]
    for x, title, detail in train_items:
        box(x, 0.12, 1.95, 0.78, title, detail, "#F8ECE8")
    for i in range(len(train_items) - 1):
        arrow(train_items[i][0] + 1.95, 0.51, train_items[i + 1][0], 0.51, color=CORAL)
    ax.text(11.62, 1.02, "Detector discarded\nafter training", ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(pad=0.25)
    fig.savefig(FIG / "fig_rmrnet_architecture.png", bbox_inches="tight")
    fig.savefig(FIG / "fig_rmrnet_architecture.pdf", bbox_inches="tight")
    plt.close(fig)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ["arialbd.ttf" if bold else "arial.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def rebuild_atlas(source_name: str, output_name: str, rows: list[int], row_names: list[str]) -> None:
    src = Image.open(ATLAS_SOURCE / source_name).convert("RGB")
    # Six method columns produced by build_detection_qualitative_atlas.py.
    x_ranges = [(168, 418), (428, 678), (688, 938), (948, 1198), (1208, 1458), (1468, 1718)]
    headers = ["Clean", "Degraded", "NAFNet", "DFPIR", "DeMoE-scenario", "RMR-Net + metadata"]
    y_starts = [98, 283, 468, 652, 835, 1019]
    cell_w, cell_h = 400, 278
    left, top, gap_x, gap_y = 170, 78, 12, 66
    canvas = Image.new("RGB", (left + 6 * cell_w + 5 * gap_x + 18, top + 2 * cell_h + gap_y + 28), "white")
    draw = ImageDraw.Draw(canvas)
    for c, header in enumerate(headers):
        x = left + c * (cell_w + gap_x)
        draw.text((x + 8, 20), header, fill=INK, font=font(25, bold=True))
        if c == 5:
            draw.rectangle((x, 56, x + cell_w, 61), fill=TEAL)
    for r, (row_index, row_name) in enumerate(zip(rows, row_names)):
        y = top + r * (cell_h + gap_y)
        source_y = y_starts[row_index]
        draw.text((12, y + 16), row_name, fill=INK, font=font(25, bold=True))
        draw.text((12, y + 54), "matched detector\nevidence", fill=MUTED, font=font(19))
        for c, (x1, x2) in enumerate(x_ranges):
            crop = src.crop((x1, source_y, x2, min(source_y + 164, src.height)))
            crop = crop.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            x = left + c * (cell_w + gap_x)
            canvas.paste(crop, (x, y))
            draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline="#BAC5CB", width=2)
        if r == 0:
            draw.line((12, y + cell_h + 30, canvas.width - 12, y + cell_h + 30), fill=LIGHT, width=3)
    canvas.save(FIG / output_name, dpi=(320, 320), quality=96)


def kitti_main_figure() -> None:
    rows = [
        ("Degraded", first_metric(ROOT / "runs/bench_kitti_realmeta_longexp_splitB_degraded/metrics.csv")),
        ("RMR image-only", first_metric(ROOT / "runs/bench_kitti_realmeta_longexp_splitB_rmr60_blind/metrics.csv")),
        ("NAFNet", first_metric(ROOT / "runs/bench_kitti_realmeta_longexp_splitB_nafnet_matched/metrics.csv")),
        ("DFPIR", first_metric(ROOT / "runs/bench_kitti_realmeta_longexp_splitB_metadata_naf_dfpir/metrics.csv", "DFPIR-CVPR2025")),
        ("RMR + full prior", first_metric(ROOT / "runs/bench_kitti_realmeta_longexp_splitB_rmr60_metadata/metrics.csv")),
    ]
    labels = [r[0] for r in rows]
    psnr = [float(r[1]["psnr"]) for r in rows]
    ssim = [float(r[1]["ssim"]) for r in rows]
    colors = [MUTED, BLUE, "#7895B2", GOLD, TEAL]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.15), constrained_layout=True)
    for ax, values, ylabel, limits in (
        (axes[0], psnr, "PSNR (dB)", (min(psnr) - 0.25, max(psnr) + 0.28)),
        (axes[1], ssim, "SSIM", (min(ssim) - 0.012, max(ssim) + 0.014)),
    ):
        x = np.arange(len(labels))
        ax.plot(x, values, color="#AAB6BC", linewidth=1.2, zorder=1)
        ax.scatter(x, values, s=70, c=colors, edgecolor="white", linewidth=0.8, zorder=2)
        for xi, yi in zip(x, values):
            value_label(ax, xi, yi, f"{yi:.2f}" if ylabel.startswith("PSNR") else f"{yi:.3f}")
        ax.set_xticks(x, labels, rotation=23, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*limits)
        ax.axhline(values[0], color=MUTED, linestyle=(0, (3, 2)), linewidth=0.9)
        finish_axis(ax)
    axes[0].set_title("Reconstruction fidelity")
    axes[1].set_title("Structural fidelity")
    fig.savefig(FIG / "fig_kitti_realmeta_results.png", bbox_inches="tight")
    plt.close(fig)


def kitti_control_figure() -> None:
    raw = json.loads((ROOT / "runs/kitti_realmeta_robustness_rawtelemetry_trained_30ep/summary.json").read_text(encoding="utf-8"))
    full = json.loads((ROOT / "runs/kitti_realmeta_robustness_raw_audit_existing/summary.json").read_text(encoding="utf-8"))
    labels = ["Degraded", "Image-only", "Scalar", "Raw OXTS", "Full prior"]
    psnr = [raw["mean"][k]["psnr"] for k in ("degraded", "blind", "raw_scalar", "raw_telemetry")] + [full["mean"]["true"]["psnr"]]
    ssim = [raw["mean"][k]["ssim"] for k in ("degraded", "blind", "raw_scalar", "raw_telemetry")] + [full["mean"]["true"]["ssim"]]
    colors = [MUTED, BLUE, "#84B7A7", TEAL, INK]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), constrained_layout=True)
    for ax, values, ylabel, digits in ((axes[0], psnr, "PSNR (dB)", 2), (axes[1], ssim, "SSIM", 3)):
        y = np.arange(len(labels))
        ax.hlines(y, min(values) - 0.08, values, color=LIGHT, linewidth=5)
        ax.scatter(values, y, s=70, c=colors, edgecolor="white", linewidth=0.8, zorder=2)
        for yi, value in zip(y, values):
            ax.annotate(f"{value:.{digits}f}", (value, yi), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
        ax.set_xlim(min(values) - 0.12, max(values) + (0.22 if digits == 2 else 0.015))
        finish_axis(ax, grid="x")
    axes[0].set_title("Real telemetry, no derived blur fields")
    axes[1].set_title("Generator-aligned prior as upper bound")
    fig.savefig(FIG / "fig_kitti_rawtelemetry_audit.png", bbox_inches="tight")
    plt.close(fig)


def field_summary_figure() -> None:
    direct = read_csv(EVIDENCE / "ilx_direct_single_view_metrics.csv")
    direct_order = ["raw", "rmr_blind", "rmr_metadata", "demoe_auto", "instructir_generic"]
    direct = [next(r for r in direct if r["run"] == key) for key in direct_order]
    budget = read_csv(EVIDENCE / "ilx_temporal_holdout_fixed_budget.csv")
    budget_order = ["raw", "raw_plus_rmr", "raw_plus_nafnet", "raw_plus_dfpir", "raw_plus_demoe_auto", "raw_plus_instructir"]
    budget = [next(r for r in budget if r["run"] == key) for key in budget_order]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.25), constrained_layout=True)
    left_names = ["Raw", "RMR image-only", "RMR + context", "DeMoE", "InstructIR"]
    left_values = [float(r["f1_iou10"]) for r in direct]
    right_names = ["Raw", "Raw + RMR", "Raw + NAFNet", "Raw + DFPIR", "Raw + DeMoE", "Raw + InstructIR"]
    right_values = [float(r["f1_iou10"]) for r in budget]
    right_coverage = [float(r["coverage"]) for r in budget]
    for ax, names, values, title in (
        (axes[0], left_names, left_values, "Direct restored views (44 frames)"),
        (axes[1], right_names, right_values, "Raw-preserving holdout policy (22 frames)"),
    ):
        y = np.arange(len(names))
        colors = [MUTED] + [TEAL if "RMR" in name else BLUE for name in names[1:]]
        ax.barh(y, values, color=colors, height=0.58)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlabel("F1 at IoU 0.10")
        ax.set_xlim(0.24, 0.40)
        ax.axvline(values[0], color=MUTED, linestyle=(0, (3, 2)), linewidth=1)
        for yi, value in zip(y, values):
            ax.text(value + 0.003, yi, f"{value:.3f}", va="center", fontsize=8)
        ax.set_title(title)
        finish_axis(ax, grid="x")
    y = np.arange(len(right_names))
    axes[1].scatter(right_coverage, y, marker="D", s=25, color=GOLD, label="relaxed coverage", zorder=3)
    axes[1].legend(frameon=False, loc="lower right", fontsize=7.5)
    fig.savefig(FIG / "fig_ilx_audited_system_results.png", bbox_inches="tight")
    plt.close(fig)


def field_pr_figure() -> None:
    rows = read_csv(EVIDENCE / "ilx_temporal_pr_curve.csv")
    systems = ["Raw native", "Raw + RMR-Net", "Raw + NAFNet-road", "Raw + DeMoE-auto"]
    colors = [INK, TEAL, BLUE, CORAL]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for system, color in zip(systems, colors):
        selected = sorted((r for r in rows if r["system"] == system), key=lambda r: float(r["recall_iou10"]))
        ax.plot([float(r["recall_iou10"]) for r in selected], [float(r["precision_iou10"]) for r in selected], color=color, linewidth=2.0, label=system)
    ax.set_xlabel("Recall at IoU 0.10")
    ax.set_ylabel("Precision at IoU 0.10")
    ax.set_xlim(0.0, 0.27)
    ax.set_ylim(0.32, 1.02)
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=8)
    finish_axis(ax, grid="both")
    fig.tight_layout()
    fig.savefig(FIG / "fig_ilx_temporal_pr_curve.png", bbox_inches="tight")
    plt.close(fig)


def correlation_figure() -> None:
    rows = read_csv(ROOT / "experiments/v31_demoe_integration/fidelity_detection_pairs_with_demoe.csv")
    methods = ["RMR-Net", "NAFNet-road", "DFPIR", "DeMoE-auto", "DeMoE-scenario"]
    colors = dict(zip(methods, [TEAL, BLUE, GOLD, CORAL, "#8B6BA8"]))
    markers = {"motion": "o", "defocus": "s", "low light": "^"}
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    for method in methods:
        selected = [r for r in rows if r["method"] == method]
        for row in selected:
            ax.scatter(float(row["psnr"]), float(row["map50"]), s=65, marker=markers[row["scenario"]], color=colors[method], edgecolor="white", linewidth=0.8)
        ax.scatter([], [], s=55, color=colors[method], label=method)
    for scenario, marker in markers.items():
        ax.scatter([], [], s=55, facecolor="none", edgecolor=INK, marker=marker, label=scenario)
    ax.set_xlabel("PSNR (dB)")
    ax.set_ylabel("Frozen-detector mAP50")
    finish_axis(ax, grid="both")
    ax.legend(frameon=False, ncol=2, fontsize=7.5, loc="best")
    ax.text(
        0.02,
        0.97,
        r"Spearman $\rho=-0.058$; Kendall $\tau=-0.041$",
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        color=MUTED,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 2.0, "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(FIG / "fig_fidelity_detection_correlation.png", bbox_inches="tight")
    plt.close(fig)


def rebuild_failure_figure() -> None:
    src = Image.open(SOURCE / "fig_native_failure_modes.png").convert("RGB")
    panel_w, panel_h = 720, 480
    selected_rows = [1, 2]
    scale_w, scale_h = 820, 546
    left, top, gap = 110, 80, 14
    canvas = Image.new("RGB", (left + 2 * scale_w + gap + 18, top + 2 * scale_h + 62), "white")
    draw = ImageDraw.Draw(canvas)
    for c, title in enumerate(["Raw Sony ILX-LR1 frame", "Context-conditioned RMR-Net"]):
        x = left + c * (scale_w + gap)
        draw.text((x + 8, 20), title, fill=INK, font=font(25, bold=True))
    for r, source_row in enumerate(selected_rows):
        y = top + r * (scale_h + 30)
        draw.text((12, y + 18), f"Case {chr(65+r)}", fill=INK, font=font(24, bold=True))
        for c in range(2):
            crop = src.crop((c * panel_w, source_row * 516 + 36, (c + 1) * panel_w, source_row * 516 + 516))
            crop = crop.resize((scale_w, scale_h), Image.Resampling.LANCZOS)
            x = left + c * (scale_w + gap)
            canvas.paste(crop, (x, y))
            draw.rectangle((x, y, x + scale_w - 1, y + scale_h - 1), outline="#BAC5CB", width=2)
    canvas.save(FIG / "fig_native_failure_modes.png", dpi=(320, 320), quality=96)


def rebuild_boundary_figure() -> None:
    src = Image.open(SOURCE / "fig_snake_boundary_cross_model.png").convert("RGB")
    columns = [("Degraded", 0, (0, 0)), ("DFPIR", 1, (1, 0)), ("DeMoE-scenario", 3, (7, 1)), ("RMR-Net", 5, (6, 3))]
    panel_w, target_w = 270, 390
    left, top, gap_x, gap_y = 150, 72, 14, 66
    target_h = 205
    canvas = Image.new("RGB", (left + 4 * target_w + 3 * gap_x + 18, top + 2 * target_h + gap_y + 24), "white")
    draw = ImageDraw.Draw(canvas)
    for c, (name, _, _) in enumerate(columns):
        x = left + c * (target_w + gap_x)
        draw.text((x + 5, 16), name, fill=INK, font=font(24, bold=True))
        if name == "RMR-Net":
            draw.rectangle((x, 50, x + target_w, 55), fill=TEAL)
    for r, row_name in enumerate(["Pothole / motion", "Crack / defocus"]):
        y = top + r * (target_h + gap_y)
        draw.text((10, y + 12), row_name, fill=INK, font=font(22, bold=True))
        for c, (_name, source_col, counts) in enumerate(columns):
            source_x = source_col * 282
            if r == 0:
                crop = src.crop((source_x, 78, source_x + panel_w, 204))
            else:
                crop = src.crop((source_x, 280, source_x + panel_w, 372))
            crop = crop.resize((target_w, target_h), Image.Resampling.LANCZOS)
            x = left + c * (target_w + gap_x)
            canvas.paste(crop, (x, y))
            draw.rectangle((x, y, x + target_w - 1, y + target_h - 1), outline="#BAC5CB", width=2)
            draw.text((x + 8, y + target_h + 7), f"accepted contours: {counts[r]}", fill=MUTED, font=font(18))
    canvas.save(FIG / "fig_snake_boundary_cross_model.png", dpi=(320, 320), quality=96)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    architecture_figure()
    rebuild_atlas(
        "fig_detection_candidate_atlas_pothole_zoom.png",
        "fig_detection_candidate_atlas_pothole_zoom.png",
        rows=[0, 3],
        row_names=["Motion blur", "Defocus"],
    )
    rebuild_atlas(
        "fig_detection_candidate_atlas_pcm_crack_zoom.png",
        "fig_detection_candidate_atlas_pcm_crack_zoom.png",
        rows=[1, 2],
        row_names=["Motion crack", "Defocus crack"],
    )
    kitti_main_figure()
    kitti_control_figure()
    field_summary_figure()
    field_pr_figure()
    correlation_figure()
    rebuild_failure_figure()
    rebuild_boundary_figure()
    print(json.dumps({"figures": 10, "output": str(FIG)}, indent=2))


if __name__ == "__main__":
    main()
