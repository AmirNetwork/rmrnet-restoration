from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_ieee_tits_rmrnet"
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"
EXP = ROOT / "experiments" / "loss_telemetry_assets"


RUNS = {
    "IVCNZ pothole": ROOT / "runs" / "rmrnet_v27_taskloss_pothole_yolo11s" / "history.json",
    "PCM multi-class": ROOT / "runs" / "rmrnet_v27_taskloss_pcm_yolo11s" / "history.json",
}


def read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def series(history: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row.get(key, 0.0)) for row in history]


def write_loss_component_table() -> None:
    rows = [
        (
            "Base restoration",
            r"$\|I_r-I_c\|_1$, gradients, Fourier magnitude, defect-weighted fidelity, and detector-visibility terms",
            "Keeps the image recognizable, prevents color/structure drift, and remains the dominant optimization signal.",
            "The detector-guided terms are intentionally low-weight regularizers rather than a replacement for image restoration.",
        ),
        (
            "Code supervision and sparse basis",
            "SmoothL1 supervision for the image-estimated corruption code plus an L1 penalty on monotone basis gates",
            "Forces the conditioning code to represent the known degradation state and discourages unnecessary basis activation.",
            "This prevents the FiLM vector from becoming an uninterpretable free latent token.",
        ),
        (
            "TDP with CQMix",
            "Frozen YOLO11s backbone-feature matching between restored/clean images with cross-quality patch mixing",
            "Preserves detector-relevant features while patch mixing reduces coherent adversarial shortcuts.",
            "Detector parameters are frozen; clean features use no-gradient targets.",
        ),
        (
            "Jacobian stability",
            "Hutchinson-estimated detector-feature Jacobian penalty",
            "Discourages high-frequency perturbations that make the detector feature field locally unstable.",
            "One random projection is used to keep training cost bounded.",
        ),
        (
            "Post-detection contour measurement",
            "Guarded MorphACWE/Snake refinement inside detector boxes",
            "Converts detections into area, perimeter, compactness, and overlap metrics after restoration.",
            "This is not part of the final training loss; failures are reported explicitly instead of hidden.",
        ),
        (
            "Anchor and evidence non-regression",
            "Weak detector-feature anchor to the degraded input plus road-evidence lower-crop constraints",
            "Keeps restoration from erasing useful crack/rim evidence or drifting away from the detector operating domain.",
            "This is a safeguard; it is grouped with train-time regularizers in ablations.",
        ),
        (
            "Validation output gate",
            "Validation-tuned pass-through/restored policy before YOLO",
            "Prevents unconditional restoration on native or weakly degraded images where the detector already works best.",
            "Test labels are never used for gate tuning.",
        ),
    ]

    body = "\n".join(
        f"{name} & {math} & {role} & {note} \\\\"
        for name, math, role, note in rows
    )
    tex = rf"""\begin{{table*}}[!t]
\centering
\caption{{Plain-language role of each optimization component. The deployed core is the metadata-conditioned restorer and validation gate; detector-feature, Jacobian, anchor, and evidence terms are train-time regularizers that shape detector-visible road evidence. Active contours are kept outside the training loss and used after detection for measurement.}}
\label{{tab:loss_component_intuition}}
\scriptsize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabularx}}{{\textwidth}}{{p{{0.16\textwidth}}p{{0.25\textwidth}}p{{0.29\textwidth}}p{{0.22\textwidth}}}}
\toprule
Component & Mathematical term & Intuition & Safeguard / claim boundary \\
\midrule
{body}
\bottomrule
\end{{tabularx}}
\end{{table*}}
"""
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "table_loss_component_intuition.tex").write_text(tex, encoding="utf-8")


def save_loss_trends() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), dpi=190)
    colors = {"IVCNZ pothole": "#e15759", "PCM multi-class": "#4e79a7"}

    for label, path in RUNS.items():
        hist = read_history(path)
        if not hist:
            continue
        epochs = series(hist, "epoch")
        axes[0, 0].plot(epochs, series(hist, "loss_restoration"), marker="o", color=colors[label], label=label)
        axes[0, 0].plot(epochs, series(hist, "loss"), linestyle="--", color=colors[label], alpha=0.75)
        axes[0, 1].plot(epochs, series(hist, "loss_tdp_weighted"), marker="o", color=colors[label], label=f"{label} TDP")
        axes[0, 1].plot(epochs, series(hist, "loss_jacobian_weighted"), marker="s", color=colors[label], linestyle="--", label=f"{label} Jacobian")
        axes[1, 0].plot(epochs, series(hist, "evidence_edge_degraded"), color=colors[label], alpha=0.35, linestyle=":")
        axes[1, 0].plot(epochs, series(hist, "evidence_edge_restored"), color=colors[label], marker="o", label=f"{label} restored")
        axes[1, 0].plot(epochs, series(hist, "evidence_edge_clean"), color=colors[label], alpha=0.55, linestyle="--")
        axes[1, 1].plot(epochs, series(hist, "val_psnr"), marker="o", color=colors[label], label=label)

    axes[0, 0].set_title("Base loss remains dominant")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].set_title("Low-weight detector regularizers")
    axes[0, 1].set_ylabel("weighted loss")
    axes[0, 1].legend(frameon=False, fontsize=7, ncol=1)
    axes[1, 0].set_title("Road-edge evidence stays between degraded and clean")
    axes[1, 0].set_ylabel("lower-crop edge evidence")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    style_handles = [
        Line2D([0], [0], color="#666666", linestyle=":", label="degraded reference"),
        Line2D([0], [0], color="#666666", linestyle="-", marker="o", label="restored"),
        Line2D([0], [0], color="#666666", linestyle="--", label="clean reference"),
    ]
    axes[1, 0].legend(
        handles=handles + style_handles,
        frameon=False,
        fontsize=7,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        borderaxespad=0.0,
    )
    axes[1, 1].set_title("Validation restoration quality during training")
    axes[1, 1].set_ylabel("PSNR (dB)")
    axes[1, 1].legend(frameon=False, fontsize=8)

    for ax in axes.ravel():
        ax.set_xlabel("epoch")
        ax.grid(axis="y", color="#d8dee6", linewidth=0.7, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_loss_trends.png", bbox_inches="tight")
    plt.close(fig)


def save_telemetry_flow() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.8, 4.8), dpi=190)
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    boxes = [
        (0.35, 4.55, 2.35, 0.9, "Road-damage\ncontrolled degradation", "#dbeafe"),
        (0.35, 2.85, 2.35, 0.9, "KITTI raw\nOXTS telemetry", "#dcfce7"),
        (0.35, 1.15, 2.35, 0.9, "Deployment\ncamera/vehicle logs", "#fef3c7"),
        (3.45, 4.55, 2.1, 0.9, "synthetic proxy\nscenario metadata", "#eff6ff"),
        (3.45, 2.85, 2.1, 0.9, "speed, yaw rate,\nacceleration, exposure", "#f0fdf4"),
        (3.45, 1.15, 2.1, 0.9, "IMU, exposure,\nfocus, compression", "#fffbeb"),
        (6.15, 3.55, 1.75, 0.9, "normalized\n8-D code", "#f3e8ff"),
        (8.15, 3.55, 1.45, 0.9, "RMR-Net\nfusion gate", "#fee2e2"),
        (8.15, 2.25, 1.45, 0.9, "image-only\nfallback", "#e5e7eb"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#374151", linewidth=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    arrows = [
        ((2.7, 5.0), (3.45, 5.0)),
        ((2.7, 3.3), (3.45, 3.3)),
        ((2.7, 1.6), (3.45, 1.6)),
        ((5.55, 5.0), (6.15, 4.15)),
        ((5.55, 3.3), (6.15, 4.0)),
        ((5.55, 1.6), (6.15, 3.85)),
        ((7.9, 4.0), (8.15, 4.0)),
        ((7.15, 3.55), (8.15, 2.7)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.2, color="#374151"))

    ax.text(5.1, 0.45, "The road-damage benchmark uses proxy metadata; KITTI uses real OXTS telemetry with controlled blur; deployment can use native camera/vehicle logs.", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_telemetry_processing.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    write_loss_component_table()
    save_loss_trends()
    save_telemetry_flow()
    manifest = {
        "tables": [str(TABLES / "table_loss_component_intuition.tex")],
        "figures": [
            str(FIGURES / "fig_loss_trends.png"),
            str(FIGURES / "fig_telemetry_processing.png"),
        ],
    }
    (EXP / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
