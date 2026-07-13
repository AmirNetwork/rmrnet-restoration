from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_trc_rmrnet" / "figures"


INK = "#1f2c34"
MUTED = "#5e6b74"
LINE = "#263642"
BLUE = "#e6f0f8"
TEAL = "#d8f0ed"
GREEN = "#edf6e8"
PURPLE = "#eee9f7"
PEACH = "#f7e7dd"
YELLOW = "#fff5d6"
GRAY = "#f6f8fa"


def label(ax, x: float, y: float, text: str, *, size: float = 9.0, weight: str = "normal", color: str = INK) -> None:
    ax.text(x, y, text, ha="center", va="center", fontsize=size, weight=weight, color=color, family="DejaVu Sans")


def box(ax, x: float, y: float, w: float, h: float, text: str, color: str, *, size: float = 8.6, weight: str = "normal") -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.12",
        facecolor=color,
        edgecolor=LINE,
        linewidth=1.35,
    )
    ax.add_patch(patch)
    label(ax, x + w / 2, y + h / 2, text, size=size, weight=weight)


def arrow(ax, src: tuple[float, float], dst: tuple[float, float], *, rad: float = 0.0, color: str = LINE, lw: float = 1.6) -> None:
    ax.add_patch(
        FancyArrowPatch(
            src,
            dst,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.6, 6.0), dpi=320)
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.92])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    label(ax, 50, 57, "RMR-Net: Metadata-Conditioned Road Restoration for Transportation Perception", size=15.5, weight="bold")
    label(ax, 50, 54.5, "Deployed path is lightweight; detector losses are train-time, active contours are post-detection measurement.", size=8.7, color=MUTED)

    box(ax, 3, 36, 12.5, 8.5, "Degraded road\nimage\n$I_d$", BLUE, size=9.5, weight="bold")
    box(ax, 21, 42, 13.5, 6.0, "Image-estimated\ncode\n$\\hat z=g_\\phi(I_d)$", TEAL, size=8.7)
    box(ax, 21, 31, 13.5, 6.0, "Metadata\ncode\n$z_m=q(m)$", GREEN, size=8.7)
    box(ax, 40, 35.5, 15.5, 9.0, "Sparse monotone\nbasis + reliability\nfusion\n$z=\\alpha g(z_m)+(1-\\alpha)g(\\hat z)$", TEAL, size=7.8, weight="bold")
    box(ax, 62, 42, 14.0, 6.0, "Task-evidence\nattention", PEACH, size=8.8)
    box(ax, 62, 33, 14.0, 6.0, "FiLM conditioned\nresidual blocks", PURPLE, size=8.8)
    box(ax, 62, 24, 14.0, 6.0, "Bounded detail\nskip\n$\\eta_dG_dD(I_d)$", GREEN, size=8.4)
    box(ax, 82, 34.5, 11.5, 8.5, "Restored\nimage\n$I_r$", BLUE, size=9.5, weight="bold")
    box(ax, 95, 34.5, 4.4, 8.5, "Gate\n$I_o$", YELLOW, size=8.2, weight="bold")

    arrow(ax, (15.5, 40.3), (21, 45.0))
    arrow(ax, (15.5, 40.3), (40, 37.4), rad=-0.12)
    arrow(ax, (34.5, 45.0), (40, 41.8))
    arrow(ax, (34.5, 34.0), (40, 38.2))
    arrow(ax, (55.5, 40.0), (62, 45.0))
    arrow(ax, (55.5, 40.0), (62, 36.0))
    arrow(ax, (55.5, 40.0), (62, 27.0))
    arrow(ax, (76, 45.0), (82, 39.8))
    arrow(ax, (76, 36.0), (82, 38.3))
    arrow(ax, (76, 27.0), (82, 36.6))
    arrow(ax, (93.5, 38.7), (95, 38.7))

    ax.plot([46, 70], [18, 18], color="#cfd6dc", linewidth=1.2)
    label(ax, 58, 20.0, "Training objective", size=8.6, weight="bold", color=MUTED)
    box(ax, 9, 10, 15.5, 6.3, "Fidelity\n$L_1$ + edge +\nFourier + evidence", GRAY, size=7.8)
    box(ax, 29, 10, 15.5, 6.3, "Frozen YOLO\nfeatures\n$L_{TDP}$ + anchor", PEACH, size=7.8)
    box(ax, 49, 10, 15.5, 6.3, "Cascaded\nstability\n$L_J$", PEACH, size=7.8)
    box(ax, 69, 10, 15.5, 6.3, "Evidence\nnon-regression\n+ detail-copy guard", GRAY, size=7.5)
    for x in [24.5, 44.5, 64.5]:
        arrow(ax, (x, 13.2), (x + 4.5, 13.2), lw=1.1, color="#74808a")

    box(ax, 69, 2.5, 19.5, 4.8, "Post-detection boundary measurement:\nYOLO box $\\rightarrow$ guarded active contour\n(area, perimeter, compactness)", YELLOW, size=7.3)
    arrow(ax, (88.5, 38.7), (78.5, 7.3), rad=-0.18, lw=1.1, color="#7a6a1b")
    label(ax, 28, 4.7, "Final training sets active-contour loss weight to zero; contours are evaluated after detection.", size=8.0, color="#7a5a00")

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_rmrnet_architecture.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
