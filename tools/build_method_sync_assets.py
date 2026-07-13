from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "paper_ieee_tits_rmrnet" / "figures"


def box(ax, xy, wh, text, face, edge="#24303a", fontsize=10):
    rect = Rectangle(xy, wh[0], wh[1], facecolor=face, edgecolor=edge, linewidth=1.6)
    ax.add_patch(rect)
    ax.text(
        xy[0] + wh[0] / 2,
        xy[1] + wh[1] / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    return rect


def arrow(ax, start, end, text=None, rad=0.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="->",
        mutation_scale=14,
        linewidth=1.7,
        color="#24303a",
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(patch)
    if text:
        ax.text(
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2 + 0.18,
            text,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#374151",
        )


def build_architecture() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.2, 6.0))
    ax.set_axis_off()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.4)

    box(ax, (0.35, 2.55), (1.55, 0.9), "Degraded\nroad image\n$I_d$", "#e9eff5")
    box(ax, (2.55, 3.75), (1.65, 0.8), "Metadata code\n$z_m$", "#eef5df")
    box(ax, (2.55, 2.15), (1.65, 0.8), "Image code\n$z_b=g_\\phi(I_d)$", "#f5eddf")
    box(ax, (4.75, 2.85), (1.9, 1.0), "Sparse monotone\nbasis + reliability\nfusion", "#dff2f0", fontsize=9)
    box(ax, (7.15, 3.8), (1.75, 0.8), "Task-evidence\nattention", "#f4e6df")
    box(ax, (7.15, 2.65), (1.75, 0.8), "FiLM + code-gated\nrestoration blocks", "#e9e3f4", fontsize=9)
    box(ax, (7.15, 1.5), (1.75, 0.8), "Bounded detail skip\n$\\eta_dG_dD(I_d)$", "#edf7e8", fontsize=9)
    box(ax, (9.45, 2.55), (1.45, 0.9), "Restored\nimage\n$I_r$", "#e9eff5")
    box(ax, (10.95, 2.55), (0.85, 0.9), "Gate\n$I_o$", "#fff4d6")

    arrow(ax, (1.9, 3.0), (2.55, 2.55))
    arrow(ax, (4.2, 4.15), (4.75, 3.55))
    arrow(ax, (4.2, 2.55), (4.75, 3.15))
    arrow(ax, (6.65, 3.35), (7.15, 4.2), "$z$")
    arrow(ax, (6.65, 3.25), (7.15, 3.05), "$z$")
    arrow(ax, (1.9, 3.0), (7.15, 1.9), rad=-0.15)
    arrow(ax, (8.9, 4.2), (9.45, 3.1))
    arrow(ax, (8.9, 3.05), (9.45, 3.0))
    arrow(ax, (8.9, 1.9), (9.45, 2.75))
    arrow(ax, (10.9, 3.0), (10.95, 3.0))

    ax.text(
        3.15,
        0.65,
        "Training losses: code supervision, image fidelity, YOLO-feature TDP/Jacobian, and TDAC boundary regularization",
        ha="center",
        va="center",
        fontsize=9,
        color="#374151",
    )
    ax.text(
        8.7,
        0.35,
        "Inference path: restorer + validation-tuned output gate; frozen YOLO/TDAC losses are train-time only",
        ha="center",
        va="center",
        fontsize=9,
        color="#374151",
    )

    fig.tight_layout()
    fig.savefig(FIGURES / "fig_rmrnet_architecture.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    build_architecture()
