from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_ieee_tits_rmrnet" / "figures"


INK = "#1f2d35"
MUTED = "#596873"
LINE = "#263642"
BLUE = "#e6f0f8"
TEAL = "#d8f0ed"
GREEN = "#eef5df"
PURPLE = "#eee9f7"
PEACH = "#f5e6df"
YELLOW = "#fff5d6"
GRAY = "#f7f9fa"


def text(ax, x, y, s, size=9, weight="normal", color=INK, ha="center", va="center"):
    ax.text(x, y, s, fontsize=size, weight=weight, color=color, ha=ha, va=va, family="DejaVu Sans")


def arrow(ax, a, b, rad=0.0, lw=1.7, color=LINE, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle=style,
            mutation_scale=12,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def round_box(ax, x, y, w, h, label, fc, size=8.5, weight="normal", ec=LINE, lw=1.5):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.08",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    text(ax, x + w / 2, y + h / 2, label, size=size, weight=weight)
    return patch


def feature_stack(ax, x, y, w, h, n, color, label, sublabel="", edge=LINE):
    for i in range(n):
        dx = i * 0.12
        dy = i * 0.10
        ax.add_patch(Rectangle((x + dx, y + dy), w, h, facecolor=color, edgecolor=edge, linewidth=1.3))
    text(ax, x + w / 2 + 0.12 * (n - 1), y + h / 2 + 0.10 * (n - 1), label, size=8.7, weight="bold")
    if sublabel:
        text(ax, x + w / 2 + 0.12 * (n - 1), y - 0.35, sublabel, size=7.2, color=MUTED)


def film_block(ax, x, y, w, h, label):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=PURPLE, edgecolor=LINE, linewidth=1.4))
    text(ax, x + w / 2, y + h * 0.62, label, size=8.2, weight="bold")
    text(ax, x + w / 2, y + h * 0.33, "FiLM", size=7.4, color=MUTED)


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11.0, 5.8), dpi=320)
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.92])
    ax.set_xlim(0, 106)
    ax.set_ylim(0, 56)
    ax.axis("off")

    text(ax, 50, 53.8, "RMR-Net Architecture", size=18, weight="bold")
    text(ax, 50, 51.6, "Metadata-conditioned road restoration with image-only fallback and frozen-detector evaluation", size=9.5, color=MUTED)

    # Main image restoration path.
    text(ax, 6.5, 45.7, "Input", size=8, weight="bold", color=MUTED)
    feature_stack(ax, 3.2, 36.5, 6.8, 7.2, 3, BLUE, "$I_d$", "degraded road image")
    round_box(ax, 13.2, 38.2, 7.0, 4.0, "Stem\nConv", BLUE, size=8)
    feature_stack(ax, 24.2, 37.0, 7.0, 6.6, 4, PURPLE, "E1", "FiLM blocks")
    feature_stack(ax, 36.5, 37.4, 6.4, 5.8, 4, PURPLE, "E2", "downsample")
    feature_stack(ax, 48.5, 37.7, 5.8, 5.2, 5, PURPLE, "B", "bottleneck")
    feature_stack(ax, 61.2, 37.4, 6.4, 5.8, 4, PURPLE, "D2", "upsample + skip")
    feature_stack(ax, 74.0, 37.0, 7.0, 6.6, 4, PURPLE, "D1", "upsample + skip")
    round_box(ax, 85.2, 38.2, 7.0, 4.0, "Head\nConv", BLUE, size=8)
    feature_stack(ax, 94.0, 36.5, 6.3, 7.2, 3, BLUE, "$I_r$", "restored image")

    for a, b in [
        ((10.2, 40.4), (13.2, 40.4)),
        ((20.2, 40.4), (24.2, 40.4)),
        ((31.7, 40.4), (36.5, 40.4)),
        ((43.2, 40.4), (48.5, 40.4)),
        ((54.9, 40.4), (61.2, 40.4)),
        ((68.2, 40.4), (74.0, 40.4)),
        ((81.4, 40.4), (85.2, 40.4)),
        ((92.2, 40.4), (94.0, 40.4)),
    ]:
        arrow(ax, a, b)

    # Skip connections.
    arrow(ax, (31.5, 37.0), (74.2, 36.0), rad=0.22, lw=1.2, color="#6b7580")
    arrow(ax, (42.7, 37.2), (61.4, 36.5), rad=0.18, lw=1.2, color="#6b7580")
    text(ax, 52.0, 32.9, "U-Net style skip connections preserve road texture", size=7.5, color=MUTED)

    # Defect attention branch.
    round_box(ax, 12.0, 27.0, 13.2, 5.2, "Defect-edge\nattention\n$E(I_d)=||\\nabla I_d||$", PEACH, size=7.7)
    arrow(ax, (7.0, 36.5), (15.5, 32.2), rad=-0.08)
    arrow(ax, (19.0, 32.2), (24.5, 37.0), rad=-0.06)
    text(ax, 18.6, 25.3, "emphasizes crack / pothole boundary cues", size=7.3, color=MUTED)

    # Conditioning branch.
    ax.add_patch(
        FancyBboxPatch(
            (3.0, 8.5),
            61.0,
            14.2,
            boxstyle="round,pad=0.025,rounding_size=0.14",
            facecolor="#ffffff",
            edgecolor="#c7d0d6",
            linewidth=1.2,
        )
    )
    text(ax, 6.2, 21.5, "Conditioning branch", size=8.5, weight="bold", color=MUTED, ha="left")
    round_box(ax, 5.0, 14.0, 13.5, 4.8, "Image code\n$z_b=g_\\phi(I_d)$", TEAL, size=8.0)
    round_box(ax, 5.0, 9.4, 13.5, 3.5, "Metadata\n$m$", GREEN, size=8.0)
    round_box(ax, 23.5, 9.7, 14.0, 4.3, "Mapper\n$z_m=q(m)$", TEAL, size=8.0)
    round_box(ax, 42.0, 11.5, 17.0, 5.0, "Fuse code\n$z=z_b$\n$z=0.5(z_b+z_m)$", TEAL, size=7.8, weight="bold")
    arrow(ax, (18.5, 16.4), (42.0, 14.5), rad=-0.08)
    arrow(ax, (18.5, 11.2), (23.5, 11.8))
    arrow(ax, (37.5, 11.8), (42.0, 13.0))
    text(ax, 33.5, 8.0, "Road-damage mAP uses proxy metadata; KITTI validates real OXTS telemetry under controlled blur.", size=7.1, color="#7a5a00")

    # z conditioning bus, kept separate from the main image path.
    arrow(ax, (50.5, 16.5), (50.5, 28.7), lw=1.1, color="#2c7777")
    ax.plot([27.8, 77.2], [28.7, 28.7], color="#2c7777", linewidth=1.1)
    for x in [27.8, 39.6, 51.4, 64.2, 77.0]:
        arrow(ax, (x, 28.7), (x, 37.0), lw=0.95, color="#2c7777", style="-|>")
    text(ax, 58.5, 30.4, "8-D code $z$ modulates every restoration block", size=7.4, color="#2c7777")

    # FiLM detail inset.
    ax.add_patch(
        FancyBboxPatch(
            (67.0, 10.0),
            28.0,
            13.0,
            boxstyle="round,pad=0.025,rounding_size=0.14",
            facecolor="#ffffff",
            edgecolor="#c7d0d6",
            linewidth=1.2,
        )
    )
    text(ax, 70.2, 21.7, "Block detail", size=8.5, weight="bold", color=MUTED, ha="left")
    film_block(ax, 69.4, 15.0, 7.5, 4.2, "GN")
    film_block(ax, 78.1, 15.0, 7.5, 4.2, "FiLM(z)")
    film_block(ax, 86.8, 15.0, 6.0, 4.2, "DWConv")
    arrow(ax, (76.9, 17.1), (78.1, 17.1), lw=1.2)
    arrow(ax, (85.6, 17.1), (86.8, 17.1), lw=1.2)
    text(ax, 81.0, 12.4, "$\\mathrm{FiLM}(x,z)=(1+\\gamma(z))\\odot x+\\beta(z)$", size=7.3, color=MUTED)

    # Evaluation strip.
    round_box(ax, 23.0, 2.8, 13.5, 4.0, "Training loss\n$L_1$ + edge + freq + code", YELLOW, size=7.4)
    round_box(ax, 44.0, 2.8, 13.0, 4.0, "Frozen YOLO\n$D_\\psi(I_r)$", PEACH, size=8.0, weight="bold")
    round_box(ax, 64.5, 2.8, 18.5, 4.0, "Report\nmAP50, mAP50-95, runtime", YELLOW, size=7.2, weight="bold")
    arrow(ax, (36.5, 4.8), (44.0, 4.8), lw=1.3)
    arrow(ax, (57.0, 4.8), (64.5, 4.8), lw=1.3)
    text(ax, 53.0, 0.8, "Claim boundary: metadata-conditioned road/ITS restoration, not a universal blind restoration model.", size=8.0, color="#735800")

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT_DIR / f"fig_rmrnet_clean_architecture.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    build()
