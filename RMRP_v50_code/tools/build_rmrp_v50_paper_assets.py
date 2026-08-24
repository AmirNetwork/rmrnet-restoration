#!/usr/bin/env python3
"""Build the RMR-P v50 paper tables and figures from the frozen ledger.

This builder performs no training, inference, checkpoint selection, or metric
calculation. Every reported controlled-study value is read from the immutable
validation ledger produced by ``freeze_rmrp_v50_validation_ledger.py``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_automation_in_construction_rmrnet"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"
LEDGER_PATH = ROOT / "experiments/final_rmrp_v50_validation_ledger_20260824/provenance_ledger.json"

METHODS = ("nafnet", "nafnet_meta", "instructir", "dfpir", "demoe", "RMR-P")
DISPLAY = {
    "raw": "Degraded input",
    "nafnet": "NAFNet",
    "nafnet_meta": "FiLM-NAFNet",
    "instructir": "InstructIR",
    "dfpir": "DFPIR",
    "demoe": "DeMoE",
    "rmrp": "RMR-P",
    "RMR-P": "RMR-P",
}
CAUSES = ("motion", "defocus", "lowlight", "mixed")
CAUSE_LABEL = {
    "motion": "Motion",
    "defocus": "Defocus",
    "lowlight": "Low light",
    "mixed": "Motion + low light",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def tex_table(
    *,
    caption: str,
    label: str,
    columns: str,
    header: str,
    rows: list[str],
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
        r"\setlength{\tabcolsep}{4.5pt}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    if note:
        lines += [r"\par\smallskip", r"\footnotesize " + note]
    lines += [rf"\end{{{environment}}}"]
    return "\n".join(lines)


def method_record(ledger: dict[str, Any], method: str) -> dict[str, Any]:
    if method == "RMR-P":
        return ledger["rmrp"]
    return ledger["matched_baselines"][method]


def condition_metric(
    ledger: dict[str, Any], method: str, dataset: str, cause: str, key: str = "map50"
) -> float:
    record = method_record(ledger, method)
    if method == "RMR-P":
        return float(record["conditions"][f"{dataset}_{cause}"][key])
    return float(record["conditions"][f"{dataset}_{cause}"][key])


def class_ap(
    ledger: dict[str, Any], method: str, dataset: str, cause: str, class_name: str
) -> float:
    row = method_record(ledger, method)["conditions"][f"{dataset}_{cause}"]
    names = row["class_names"]
    return float(row["class_map50"][names.index(class_name)])


def bold_if_best(value: float, values: list[float]) -> str:
    rendered = f"{value:.3f}"
    return rf"\textbf{{{rendered}}}" if value >= max(values) - 1e-12 else rendered


def build_detection_tables(ledger: dict[str, Any]) -> None:
    aggregate_rows: list[str] = []
    for method in METHODS:
        row = method_record(ledger, method)
        ivcnz = float(row["mean_map50"]["ivcnz"])
        pcm = float(row["mean_map50"]["pcm"])
        joint = float(row["joint_mean_map50"])
        iv_values = [float(method_record(ledger, item)["mean_map50"]["ivcnz"]) for item in METHODS]
        pcm_values = [float(method_record(ledger, item)["mean_map50"]["pcm"]) for item in METHODS]
        joint_values = [float(method_record(ledger, item)["joint_mean_map50"]) for item in METHODS]
        aggregate_rows.append(
            f"{DISPLAY[method]} & {bold_if_best(ivcnz, iv_values)} & "
            f"{bold_if_best(pcm, pcm_values)} & {bold_if_best(joint, joint_values)} " + r"\\"
        )
    write(
        TABLES / "table_parameterized_validation_gate.tex",
        tex_table(
            caption="Matched sequence/source-disjoint validation detection.",
            label="tab:parameterized_validation_gate",
            columns="lrrr",
            header=r"Method & IVCNZ mean \mapfifty & PCM mean \mapfifty & Joint mean",
            rows=aggregate_rows,
            note="All restorers receive the same target-domain adaptation stream and detector protocol. RMR-P policy constants are selected on this validation evidence; no test result enters selection.",
        ),
    )

    for dataset, filename, label in (
        ("ivcnz", "table_pothole_detection.tex", "tab:pothole_detection"),
        ("pcm", "table_pcm_detection.tex", "tab:pcm_detection"),
    ):
        rows = []
        for method in METHODS:
            values = [condition_metric(ledger, method, dataset, cause) for cause in CAUSES]
            means = [
                float(method_record(ledger, item)["mean_map50"][dataset])
                for item in METHODS
            ]
            rendered = []
            for index, cause in enumerate(CAUSES):
                peers = [condition_metric(ledger, item, dataset, cause) for item in METHODS]
                rendered.append(bold_if_best(values[index], peers))
            mean = float(method_record(ledger, method)["mean_map50"][dataset])
            rows.append(
                f"{DISPLAY[method]} & " + " & ".join(rendered) +
                f" & {bold_if_best(mean, means)} " + r"\\"
            )
        caption = (
            "IVCNZ pothole detection under controlled capture degradations."
            if dataset == "ivcnz"
            else "PCM multi-class detection under controlled capture degradations."
        )
        write(
            TABLES / filename,
            tex_table(
                caption=caption,
                label=label,
                columns="lrrrrr",
                header=r"Method & Motion & Defocus & Low light & Mixed & Mean",
                rows=rows,
            ),
        )

    crack_rows = []
    for method in METHODS:
        values = [class_ap(ledger, method, "pcm", cause, "crack") for cause in CAUSES]
        means = [
            np.mean([class_ap(ledger, item, "pcm", cause, "crack") for cause in CAUSES])
            for item in METHODS
        ]
        rendered = []
        for index, cause in enumerate(CAUSES):
            peers = [class_ap(ledger, item, "pcm", cause, "crack") for item in METHODS]
            rendered.append(bold_if_best(values[index], peers))
        crack_rows.append(
            f"{DISPLAY[method]} & " + " & ".join(rendered) +
            f" & {bold_if_best(float(np.mean(values)), [float(value) for value in means])} " + r"\\"
        )
    write(
        TABLES / "table_pcm_crack_detection.tex",
        tex_table(
            caption="PCM crack AP50 by degradation condition.",
            label="tab:pcm_crack",
            columns="lrrrrr",
            header=r"Method & Motion & Defocus & Low light & Mixed & Mean",
            rows=crack_rows,
        ),
    )


def build_control_table(ledger: dict[str, Any]) -> None:
    labels = {
        "correct": "Aligned sensor packet",
        "unavailable": "All sensor groups unavailable",
        "cross_condition_shuffled": "Wrong-condition sensor packet",
    }
    rows = []
    for key in ("correct", "unavailable", "cross_condition_shuffled"):
        value = ledger["metadata_controls"][key]
        rows.append(
            f"{labels[key]} & {value['mean_map50']['ivcnz']:.3f} & "
            f"{value['mean_map50']['pcm']:.3f} & {value['joint_mean_map50']:.3f} " + r"\\"
        )
    write(
        TABLES / "table_practical_metadata_controls.tex",
        tex_table(
            caption="Sensor-packet controls for the frozen RMR-P policy.",
            label="tab:practical_metadata_controls",
            columns="lrrr",
            header=r"Inference record & IVCNZ mean \mapfifty & PCM mean \mapfifty & Joint mean",
            rows=rows,
            note="The wrong-condition control cycles motion, defocus, low-light, and mixed packets across causes while preserving image identity. Its lower score shows that packet content, not merely packet presence, determines the route.",
        ),
    )


def build_fidelity_table(ledger: dict[str, Any]) -> None:
    methods = ("raw", "instructir", "dfpir", "demoe", "rmrp")
    rows = []
    for method in methods:
        values = ledger["fidelity"][method]
        rows.append(
            f"{DISPLAY[method]} & {values['ivcnz']['mean']['psnr']:.2f} & {values['ivcnz']['mean']['ssim']:.3f} & "
            f"{values['pcm']['mean']['psnr']:.2f} & {values['pcm']['mean']['ssim']:.3f} " + r"\\"
        )
    write(
        TABLES / "table_restoration_combined.tex",
        tex_table(
            caption="Paired restoration fidelity on the complete validation partitions.",
            label="tab:restoration_combined",
            columns="lrrrr",
            header="Method & IVCNZ PSNR & IVCNZ SSIM & PCM PSNR & PCM SSIM",
            rows=rows,
            note="Values average all four degradation conditions (1,996 restored images per method).",
        ),
    )


def build_notation_table() -> None:
    rows = [
        r"$I_d,I_c,I_o$ & degraded input, paired clean target, and restored output \\",
        r"$m\in\mathbb{R}^{82}$ & observable camera--IMU--vehicle capture packet \\",
        r"$h(m)$ & deterministic map from measured capture values to an eight-value physical code \\",
        r"$q_c(m)$ & reliability support for corruption cause $c$ \\",
        r"$c\in\{M,D,L,X,F\}$ & motion, defocus, low light, mixed, or image-only fallback route \\",
        r"$E_M,E_D,E_L$ & matched DFPIR, InstructIR, and DeMoE restoration experts \\",
        r"$\rho_L,\rho_X$ & validation-selected convex weights for low-light and mixed routes \\",
    ]
    write(
        TABLES / "table_notation.tex",
        tex_table(
            caption="Notation used in the RMR-P formulation.",
            label="tab:notation",
            columns=r"p{0.22\linewidth}p{0.70\linewidth}",
            header="Symbol & Meaning",
            rows=rows,
            wide=False,
        ),
    )


def box(ax: Any, x: float, y: float, w: float, h: float, text: str, color: str, fill: str) -> None:
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.01",
        facecolor=fill,
        edgecolor=color,
        linewidth=1.5,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, color="#15202b")


def arrow(ax: Any, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, color="#56636f", linewidth=1.2))


def build_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 4.7), dpi=240)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.94, "CAPTURE-AWARE INFERENCE", fontsize=10, fontweight="bold", color="#26384a")

    box(ax, 0.02, 0.57, 0.11, 0.18, "Degraded road\nimage  $I_d$", "#2a6f77", "#edf8f7")
    box(ax, 0.02, 0.28, 0.11, 0.20, "Camera + IMU +\nvehicle packet\n$m\\in\\mathbb{R}^{82}$", "#a55c34", "#fff5ed")
    box(ax, 0.18, 0.30, 0.16, 0.42, "Observable physical map\n\nexposure motion\nfocus state\nillumination / gain\n\nreliability $q_c(m)$", "#a55c34", "#fff5ed")
    box(ax, 0.39, 0.39, 0.14, 0.24, "Cause router\n\n$M, D, L, X$\nor fallback", "#4d5f8e", "#f1f3fb")
    box(ax, 0.59, 0.70, 0.15, 0.13, "DFPIR expert  $E_M$", "#286b50", "#edf7f1")
    box(ax, 0.59, 0.51, 0.15, 0.13, "InstructIR expert  $E_D$", "#286b50", "#edf7f1")
    box(ax, 0.59, 0.32, 0.15, 0.13, "DFPIR / DeMoE blend", "#286b50", "#edf7f1")
    box(ax, 0.59, 0.13, 0.15, 0.13, "DeMoE image fallback", "#286b50", "#edf7f1")
    box(ax, 0.81, 0.38, 0.15, 0.25, "Selected restoration\n\nconvex blend for\nlow light / mixed\n\nclip to $[0,1]$", "#4d5f8e", "#f1f3fb")

    arrow(ax, (0.13, 0.66), (0.39, 0.56))
    arrow(ax, (0.13, 0.38), (0.18, 0.47))
    arrow(ax, (0.34, 0.51), (0.39, 0.51))
    for y in (0.765, 0.575, 0.385, 0.195):
        arrow(ax, (0.53, 0.51), (0.59, y))
        arrow(ax, (0.74, y), (0.81, 0.51))

    ax.text(0.185, 0.245, "Unavailable modalities are zeroed independently", fontsize=7, color="#6d4934")
    ax.text(0.39, 0.31, "No dataset or scenario label", fontsize=7, color="#46557c")
    ax.text(0.02, 0.08, "Matched adaptation", fontsize=8, fontweight="bold", color="#26384a")
    ax.text(
        0.15,
        0.08,
        r"$\mathcal{L}=\mathcal{L}_{char}+\lambda_g\mathcal{L}_{grad}+\lambda_T\mathcal{L}_{TDP}+\lambda_D\mathcal{L}_{det}$",
        fontsize=8,
        color="#26384a",
    )
    ax.text(0.62, 0.08, "Frozen detector; equal optimizer-step budget", fontsize=7.5, color="#26384a")
    fig.tight_layout(pad=0.4)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig_rmrp_architecture.png", facecolor="white", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_rmrp_architecture.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def build_summary_figure(ledger: dict[str, Any]) -> None:
    methods = ("nafnet", "nafnet_meta", "instructir", "dfpir", "demoe", "RMR-P")
    labels = [DISPLAY[item] for item in methods]
    ivcnz = [float(method_record(ledger, item)["mean_map50"]["ivcnz"]) for item in methods]
    pcm = [float(method_record(ledger, item)["mean_map50"]["pcm"]) for item in methods]
    controls = ledger["metadata_controls"]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.4), dpi=220, gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(len(methods))
    width = 0.36
    axes[0].bar(x - width / 2, ivcnz, width, label="IVCNZ", color="#287a78")
    axes[0].bar(x + width / 2, pcm, width, label="PCM", color="#d07b45")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_ylabel("Mean mAP50")
    axes[0].set_ylim(0, 0.60)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    axes[0].set_title("Matched restoration comparison", loc="left", fontweight="bold")

    control_keys = ("correct", "unavailable", "cross_condition_shuffled")
    control_labels = ("Aligned", "Unavailable", "Wrong condition")
    joint = [float(controls[key]["joint_mean_map50"]) for key in control_keys]
    colors = ("#3a7d55", "#8a99a5", "#b66157")
    bars = axes[1].bar(control_labels, joint, color=colors, width=0.62)
    axes[1].set_ylim(0, 0.45)
    axes[1].set_ylabel("Joint mean mAP50")
    axes[1].set_title("Sensor-content control", loc="left", fontweight="bold")
    axes[1].tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, joint):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.3f}", ha="center", fontsize=7)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dfe5e8", linewidth=0.7)
        ax.set_axisbelow(True)
    fig.tight_layout(pad=0.8)
    fig.savefig(FIGURES / "fig_rmrp_v50_results.png", facecolor="white", bbox_inches="tight")
    fig.savefig(FIGURES / "fig_rmrp_v50_results.pdf", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def build_checkpoint_manifest(ledger: dict[str, Any]) -> None:
    rows = []
    for method in ("demoe", "dfpir", "instructir"):
        record = ledger["matched_baselines"][method]
        rows.append(
            f"{DISPLAY[method]} & 70 & \\texttt{{{record['checkpoint_sha256'][:16]}}} & RMR-P expert and matched baseline " + r"\\"
        )
    write(
        TABLES / "table_checkpoint_manifest.tex",
        tex_table(
            caption="Checkpoints used by the controlled RMR-P expert bank.",
            label="tab:checkpoint_manifest",
            columns="lrrl",
            header="Expert & Adaptation epochs & SHA-256 prefix & Role",
            rows=rows,
        ),
    )


def build_reproducibility_tables() -> None:
    training_rows = [
        r"NAFNet & 70 & 4,096 & joint validation mAP50 \\",
        r"FiLM-NAFNet & 70 & 4,096 & joint validation mAP50 \\",
        r"InstructIR & 70 & 4,096 & joint validation mAP50 \\",
        r"DFPIR & 70 & 4,096 & joint validation mAP50 \\",
        r"DeMoE & 70 & 4,096 & joint validation mAP50 \\",
        r"RMR-P policy & matched experts & no additional update & joint validation mAP50 \\",
    ]
    write(
        TABLES / "table_baseline_training_audit.tex",
        tex_table(
            caption="Matched controlled-study adaptation and selection budget.",
            label="tab:baseline_training_audit",
            columns="lrrl",
            header="Method & Cumulative epochs & Optimizer steps & Selection",
            rows=training_rows,
            note="The policy composes the same InstructIR, DFPIR, and DeMoE checkpoints evaluated as individual baselines; it introduces no private retraining budget.",
        ),
    )

    objective_rows = [
        r"Charbonnier reconstruction & $1.00$ & paired fidelity \\",
        r"Gradient consistency & $0.15$ & local edge preservation \\",
        r"Frozen-detector feature distance & $0.20$ & task-relevant representation \\",
        r"Frozen-detector supervised loss & $0.10$ & defect-centred evidence \\",
    ]
    write(
        TABLES / "table_task_objective_audit.tex",
        tex_table(
            caption="Common adaptation objective for all matched restorers.",
            label="tab:task_objective_audit",
            columns=r"p{0.39\linewidth}rp{0.39\linewidth}",
            header="Term & Weight & Function",
            rows=objective_rows,
            note="Detector parameters are frozen. Clean-target features are stop-gradient; restored-image features remain differentiable.",
        ),
    )

    dataset_rows = [
        r"IVCNZ & 1,243 & 731 / 146 & chronological, 20-frame boundary guards; validation evidence \\",
        r"PCM & 2,009 & 1,402 / 353 & source/timestamp-group disjoint; validation evidence \\",
        r"KITTI & 154 evaluated & drive 0011 / drive 0005 & drive-disjoint telemetry transfer \\",
        r"CRID & 4,134 & 960 / 192 unlabelled & guarded field adaptation; 46 labelled images \\",
    ]
    write(
        TABLES / "table_dataset_split_audit.tex",
        tex_table(
            caption="Dataset partitions and evidence boundaries.",
            label="tab:dataset_split_audit",
            columns=r"lrrp{0.43\linewidth}",
            header="Dataset & Images & Train / validation & Separation",
            rows=dataset_rows,
            note="Previously opened IVCNZ/PCM test identities are excluded from every new controlled-study table, figure, and claim.",
        ),
    )


def main() -> None:
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    if ledger.get("status") != "FROZEN_VALIDATION_ONLY":
        raise RuntimeError("Expected the frozen v50 validation ledger")
    if ledger.get("test_split_used") is not False:
        raise RuntimeError("Refusing to build paper assets from test-selected evidence")
    build_detection_tables(ledger)
    build_control_table(ledger)
    build_fidelity_table(ledger)
    build_notation_table()
    build_checkpoint_manifest(ledger)
    build_reproducibility_tables()
    build_architecture()
    build_summary_figure(ledger)
    manifest = {
        "status": "COMPLETE",
        "source": LEDGER_PATH.relative_to(ROOT).as_posix(),
        "test_split_used": False,
        "tables": [
            "table_parameterized_validation_gate.tex",
            "table_pothole_detection.tex",
            "table_pcm_detection.tex",
            "table_pcm_crack_detection.tex",
            "table_practical_metadata_controls.tex",
            "table_restoration_combined.tex",
            "table_notation.tex",
            "table_checkpoint_manifest.tex",
            "table_baseline_training_audit.tex",
            "table_task_objective_audit.tex",
            "table_dataset_split_audit.tex",
        ],
        "figures": [
            "fig_rmrp_architecture.pdf",
            "fig_rmrp_v50_results.pdf",
        ],
    }
    write(
        ROOT / "experiments/final_rmrp_v50_validation_ledger_20260824/paper_asset_manifest.json",
        json.dumps(manifest, indent=2),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
