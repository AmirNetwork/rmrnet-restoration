from __future__ import annotations

"""Build GT49 defect-first paper assets from executed detector outputs.

The native GT49 section now uses the stronger YOLO26s RDD-FRDC detector.  The
paper reports the field test in the same order a road agency would use it:

1. common-defect localization (crack or pothole),
2. duplicate-suppressed defect reporting, and
3. subtype accuracy as a secondary audit.

No GT49 labels are used to train or tune models here; this script only converts
the already executed prediction CSVs into paper tables and figures.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
GT49 = ROOT / "experiments" / "roboflow_geotagged_v5_native_real" / "v38_yolo26_cropmask_defect_eval"
ETA = ROOT / "experiments" / "roboflow_geotagged_v5_native_real" / "v38_yolo26_cropmask_eta_defect_eval"
TAU = ROOT / "experiments" / "roboflow_geotagged_v5_native_real" / "v38_yolo26_cropmask_tau_defect_eval"
DUAL = ROOT / "experiments" / "roboflow_geotagged_v5_native_real" / "v41_dual_evidence_final_defect_eval"
SAFETY = ROOT / "experiments" / "roboflow_geotagged_v5_native_real" / "v43_safety_gate_final_defect_eval"
TRC = ROOT / "paper_trc_rmrnet"
IEEE = ROOT / "paper_ieee_tits_rmrnet"

METHOD_ORDER = [
    "raw",
    "rmr_safety_gate",
    "rmr_blind",
    "rmr_metadata",
    "rmr_eta_0p25",
    "rmr_metadata_gated",
    "nafnet",
    "dfpir",
    "demoe_auto",
    "demoe_scenario",
    "instructir_generic",
    "instructir_metadata",
]

METHOD_LABELS = {
    "raw": "Raw native",
    "rmr_safety_gate": "RMR-Net safety gate",
    "rmr_blind": "RMR-Net image-only",
    "rmr_metadata": "RMR-Net metadata full",
    "rmr_eta_0p25": "RMR-Net recovery policy",
    "rmr_metadata_gated": "RMR-Net metadata-gated",
    "nafnet": "NAFNet-road",
    "dfpir": "DFPIR",
    "demoe_auto": "DeMoE-auto",
    "demoe_scenario": "DeMoE-scenario",
    "instructir_generic": "InstructIR-generic",
    "instructir_metadata": "InstructIR-metadata",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else 0.0


def fmt(value: float, digits: int = 3, *, bold: bool = False) -> str:
    text = f"{value:.{digits}f}"
    return rf"\textbf{{{text}}}" if bold else text


def load_summary() -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(GT49 / "defect_protocol_summary.csv")
    if (ETA / "defect_protocol_summary.csv").exists():
        rows.extend(read_csv(ETA / "defect_protocol_summary.csv"))
    if (DUAL / "defect_protocol_summary.csv").exists():
        rows.extend(read_csv(DUAL / "defect_protocol_summary.csv"))
    if (SAFETY / "defect_protocol_summary.csv").exists():
        rows.extend(read_csv(SAFETY / "defect_protocol_summary.csv"))
    return {(row["method"], row["stage"]): row for row in rows}


def load_stage_summary(root: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv(root / "defect_protocol_summary.csv")
    return {(row["method"], row["stage"]): row for row in rows}


def write_table(target_dir: Path) -> None:
    summary = load_summary()
    gt_success_values = [f(summary[(m, "dedup")], "primary_gt_success_recall") for m in METHOD_ORDER]
    best_gt_success = max(gt_success_values)
    dedup_f1_values = [f(summary[(m, "dedup")], "primary_f1_iou10") for m in METHOD_ORDER]
    best_dedup_f1 = max(dedup_f1_values)
    f1_50_values = [f(summary[(m, "dedup")], "primary_f1_iou50") for m in METHOD_ORDER]
    best_f1_50 = max(f1_50_values)

    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{High-quality Sony native-image field test on all 49 annotated cam1 frames using the modified YOLO26s RDD-FRDC crop/mask detector and the defect-first GT49 protocol. Images remain in native coordinates; detection is restricted to the lower-road region scanned by the detector, giving 83 ROI defects. The detector is frozen and not trained on GT49. Detection uses a lower-half road crop, full-frame inference on that crop, 2048-pixel center-safe tiles, class-aware fusion NMS, and a fixed road-surface gate applied identically to every method. Because the manual labels may omit visible defects, GT success is recall-oriented: a labeled defect is counted as recovered when a same-primary-class prediction has IoU$\geq$0.10, covers at least 25\% of the GT box, or contains the GT-box center. Duplicate-suppressed F1 remains reported for comparison, while subtype accuracy is a secondary diagnostic. The RMR-Net safety gate passes high-quality native frames through unchanged; this is the correct deployment behavior when full restoration would introduce artifacts or suppress detector evidence.}",
        r"\label{tab:geotagged_native_pilot}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Method & N & GT success & Report P@.10 & Report R@.10 & Report F1@.10 & F1@.50 & Subtype F1 & Type acc. \\",
        r"\midrule",
    ]
    for method in METHOD_ORDER:
        filtered = summary[(method, "filtered")]
        dedup = summary[(method, "dedup")]
        label = METHOD_LABELS[method]
        report_f1 = f(dedup, "primary_f1_iou10")
        values = [
            filtered["images"],
            fmt(f(dedup, "primary_gt_success_recall"), bold=abs(f(dedup, "primary_gt_success_recall") - best_gt_success) < 1e-12),
            fmt(f(dedup, "primary_precision_iou10")),
            fmt(f(dedup, "primary_recall_iou10")),
            fmt(report_f1, bold=abs(report_f1 - best_dedup_f1) < 1e-12),
            fmt(f(dedup, "primary_f1_iou50"), bold=abs(f(dedup, "primary_f1_iou50") - best_f1_50) < 1e-12),
            fmt(f(dedup, "subtype_f1_iou10")),
            fmt(f(dedup, "type_accuracy_iou10_primary_matches")),
        ]
        lines.append(f"{label} & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (target_dir / "tables").mkdir(parents=True, exist_ok=True)
    (target_dir / "tables" / "table_geotagged_native_pilot.tex").write_text("\n".join(lines), encoding="utf-8")


def write_eta_table(target_dir: Path) -> None:
    summary = load_stage_summary(ETA)
    methods = ["rmr_eta_0p0", "rmr_eta_0p1", "rmr_eta_0p25", "rmr_eta_0p5", "rmr_eta_0p75", "rmr_eta_1p0"]
    labels = {
        "rmr_eta_0p0": r"$\eta=0$ pass-through",
        "rmr_eta_0p1": r"$\eta=0.1$",
        "rmr_eta_0p25": r"$\eta=0.25$",
        "rmr_eta_0p5": r"$\eta=0.5$",
        "rmr_eta_0p75": r"$\eta=0.75$",
        "rmr_eta_1p0": r"$\eta=1$ full residual",
    }
    best = max(f(summary[(m, "dedup")], "primary_f1_iou10") for m in methods)
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Residual-strength sweep on the high-quality Sony native-image field test with the modified YOLO26s crop/mask detector. $\eta=0$ is pass-through and $\eta=1$ is full RMR-Net residual restoration. The sweep is reported as a deployment-policy audit; future route-level deployments should fix $\eta$ on calibration data before field testing.}",
        r"\label{tab:geotagged_eta_sweep}",
        r"\scriptsize",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Policy & P@.10 & R@.10 & F1@.10 & F1@.50 & Subtype F1 & Pred/img \\",
        r"\midrule",
    ]
    for method in methods:
        row = summary[(method, "dedup")]
        f1 = f(row, "primary_f1_iou10")
        pred_per_img = f(row, "pred_defects") / max(f(row, "images"), 1.0)
        lines.append(
            labels[method]
            + " & "
            + " & ".join(
                [
                    fmt(f(row, "primary_precision_iou10")),
                    fmt(f(row, "primary_recall_iou10")),
                    fmt(f1, bold=abs(f1 - best) < 1e-12),
                    fmt(f(row, "primary_f1_iou50")),
                    fmt(f(row, "subtype_f1_iou10")),
                    fmt(pred_per_img, digits=2),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (target_dir / "tables" / "table_geotagged_eta_sweep.tex").write_text("\n".join(lines), encoding="utf-8")


def write_tau_table(target_dir: Path) -> None:
    summary = load_stage_summary(TAU)
    methods = ["rmr_tau_00", "rmr_tau_01", "rmr_tau_02", "rmr_tau_03", "rmr_tau_04", "rmr_tau_05", "rmr_tau_06"]
    labels = {
        "rmr_tau_00": r"$\tau=0.00$",
        "rmr_tau_01": r"$\tau=0.01$",
        "rmr_tau_02": r"$\tau=0.02$",
        "rmr_tau_03": r"$\tau=0.03$",
        "rmr_tau_04": r"$\tau=0.04$",
        "rmr_tau_05": r"$\tau=0.05$",
        "rmr_tau_06": r"$\tau=0.06$ / raw-like",
    }
    best = max(f(summary[(m, "dedup")], "primary_f1_iou10") for m in methods)
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{No-reference gate-threshold sweep on the Sony native-image field test with the modified YOLO26s crop/mask detector. The gate compares road-evidence scores before and after restoration; higher thresholds pass more high-quality native frames through unchanged.}",
        r"\label{tab:native_gate_sweep}",
        r"\scriptsize",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Policy & P@.10 & R@.10 & F1@.10 & F1@.50 & Subtype F1 & Pred/img \\",
        r"\midrule",
    ]
    for method in methods:
        row = summary[(method, "dedup")]
        f1 = f(row, "primary_f1_iou10")
        pred_per_img = f(row, "pred_defects") / max(f(row, "images"), 1.0)
        lines.append(
            labels[method]
            + " & "
            + " & ".join(
                [
                    fmt(f(row, "primary_precision_iou10")),
                    fmt(f(row, "primary_recall_iou10")),
                    fmt(f1, bold=abs(f1 - best) < 1e-12),
                    fmt(f(row, "primary_f1_iou50")),
                    fmt(f(row, "subtype_f1_iou10")),
                    fmt(pred_per_img, digits=2),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (target_dir / "tables" / "table_native_gate_sweep.tex").write_text("\n".join(lines), encoding="utf-8")


def write_figure(target_dir: Path) -> None:
    summary = load_summary()
    methods = METHOD_ORDER
    labels = [
        METHOD_LABELS[m]
        .replace("RMR-Net ", "RMR-")
        .replace("safety gate", "safe")
        .replace("recovery policy", "eta=0.25")
        .replace("-generic", "")
        .replace("-metadata", "-meta")
        for m in methods
    ]
    proposal = [f(summary[(m, "filtered")], "primary_f1_iou10") for m in methods]
    dedup = [f(summary[(m, "dedup")], "primary_f1_iou10") for m in methods]
    gt_success = [f(summary[(m, "dedup")], "primary_gt_success_recall") for m in methods]
    f1_50 = [f(summary[(m, "dedup")], "primary_f1_iou50") for m in methods]
    type_acc = [f(summary[(m, "dedup")], "type_accuracy_iou10_primary_matches") for m in methods]
    colors = ["#8c939c" if m == "raw" else "#0f8b8d" if m.startswith("rmr") else "#4e79a7" for m in methods]

    fig, axes = plt.subplots(1, 4, figsize=(15.8, 3.4))
    x = list(range(len(methods)))
    axes[0].bar(x, gt_success, width=0.70, color=colors)
    axes[0].set_ylabel("known-GT success")
    axes[0].set_ylim(0, max(0.42, max(gt_success) * 1.18))
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar([i - 0.18 for i in x], proposal, width=0.36, color="#b8c0cc", label="proposal")
    axes[1].bar([i + 0.18 for i in x], dedup, width=0.36, color=colors, label="duplicate-suppressed")
    axes[1].set_ylabel("primary F1 @ IoU 0.10")
    axes[1].set_ylim(0, max(0.56, max(proposal + dedup) * 1.12))
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(x, f1_50, color=colors, width=0.7)
    axes[2].set_ylabel("primary F1 @ IoU 0.50")
    axes[2].set_ylim(0, max(0.28, max(f1_50) * 1.18))
    axes[2].grid(axis="y", alpha=0.25)

    axes[3].bar(x, type_acc, color=colors, width=0.7)
    axes[3].set_ylabel("subtype accuracy")
    axes[3].set_ylim(0, max(0.75, max(type_acc) * 1.12))
    axes[3].grid(axis="y", alpha=0.25)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    fig.suptitle("GT49 native field test with YOLO26s crop/mask: known-GT recovery and reporting quality", fontsize=11, y=1.04)
    fig.tight_layout()
    (target_dir / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(target_dir / "figures" / "fig_geotagged_defect_first.png", dpi=240, bbox_inches="tight")
    # Keep the legacy figure name in sync because the manuscript already uses it.
    fig.savefig(target_dir / "figures" / "fig_geotagged_all49_prf.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_policy_figures(target_dir: Path) -> None:
    eta_summary = load_stage_summary(ETA)
    eta_methods = ["rmr_eta_0p0", "rmr_eta_0p1", "rmr_eta_0p25", "rmr_eta_0p5", "rmr_eta_0p75", "rmr_eta_1p0"]
    eta_labels = ["0", "0.1", "0.25", "0.5", "0.75", "1"]
    eta_f1 = [f(eta_summary[(m, "dedup")], "primary_f1_iou10") for m in eta_methods]
    eta_f150 = [f(eta_summary[(m, "dedup")], "primary_f1_iou50") for m in eta_methods]

    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    ax.plot(eta_labels, eta_f1, marker="o", color="#0f8b8d", label="F1 @ IoU 0.10")
    ax.plot(eta_labels, eta_f150, marker="s", color="#4e79a7", label="F1 @ IoU 0.50")
    ax.set_xlabel("RMR residual strength eta")
    ax.set_ylabel("duplicate-suppressed primary F1")
    ax.set_ylim(0, max(0.56, max(eta_f1 + eta_f150) * 1.12))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(target_dir / "figures" / "fig_geotagged_eta_sweep.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    tau_summary = load_stage_summary(TAU)
    tau_methods = ["rmr_tau_00", "rmr_tau_01", "rmr_tau_02", "rmr_tau_03", "rmr_tau_04", "rmr_tau_05", "rmr_tau_06"]
    tau_labels = ["0.00", "0.01", "0.02", "0.03", "0.04", "0.05", "0.06"]
    tau_f1 = [f(tau_summary[(m, "dedup")], "primary_f1_iou10") for m in tau_methods]
    tau_f150 = [f(tau_summary[(m, "dedup")], "primary_f1_iou50") for m in tau_methods]

    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    ax.plot(tau_labels, tau_f1, marker="o", color="#0f8b8d", label="F1 @ IoU 0.10")
    ax.plot(tau_labels, tau_f150, marker="s", color="#4e79a7", label="F1 @ IoU 0.50")
    ax.set_xlabel("gate threshold tau")
    ax.set_ylabel("duplicate-suppressed primary F1")
    ax.set_ylim(0, max(0.56, max(tau_f1 + tau_f150) * 1.12))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(target_dir / "figures" / "fig_geotagged_tau_sweep.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not (GT49 / "defect_protocol_summary.csv").exists():
        raise FileNotFoundError(GT49 / "defect_protocol_summary.csv")
    for target in [TRC, IEEE]:
        write_table(target)
        write_eta_table(target)
        write_tau_table(target)
        write_figure(target)
        write_policy_figures(target)
    print(f"Updated GT49 defect-first assets from {GT49}")


if __name__ == "__main__":
    main()
