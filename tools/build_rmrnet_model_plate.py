from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_ieee_tits_rmrnet" / "figures"


COLORS = {
    "ink": "#22313a",
    "muted": "#53616b",
    "image": "#e8f0f7",
    "meta": "#edf5df",
    "code": "#dff2ef",
    "net": "#ebe7f5",
    "edge": "#f6e9e2",
    "loss": "#f8f1dc",
    "det": "#f2e1d7",
    "line": "#263642",
}


def box(ax, x, y, w, h, text, fc, ec=None, fs=10, weight="normal", radius=0.04, lw=1.7):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec or COLORS["line"],
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=COLORS["ink"], weight=weight)
    return patch


def label(ax, x, y, text, fs=9, color=None, ha="center", va="center", weight="normal"):
    ax.text(x, y, text, fontsize=fs, color=color or COLORS["ink"], ha=ha, va=va, weight=weight)


def arrow(ax, start, end, rad=0.0, lw=1.8, color=None):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=lw,
            color=color or COLORS["line"],
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def pill(ax, x, y, w, h, text, fc, fs=7.2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.08",
        linewidth=1.0,
        edgecolor="#6f7a82",
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=COLORS["ink"])


def grouped_panel(ax, x, y, w, h, title, fc="#ffffff"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.014,rounding_size=0.04",
            linewidth=1.2,
            edgecolor="#c0c8ce",
            facecolor=fc,
        )
    )
    label(ax, x + 0.25, y + h - 0.22, title, fs=9, color=COLORS["muted"], ha="left", weight="bold")


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11.0, 8.5), dpi=300)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 76)
    ax.axis("off")

    label(ax, 50, 73.7, "RMR-Net: Metadata-Conditioned Road Image Restoration for ITS Perception", fs=17, weight="bold")
    label(
        ax,
        50,
        71.5,
        r"Goal: restore degraded road frames so a frozen detector $D_\psi$ recovers pothole/crack/manhole evidence, not just higher PSNR.",
        fs=10,
        color=COLORS["muted"],
    )

    grouped_panel(ax, 1.2, 56.5, 97.6, 12.8, "Inputs and conditioning code")
    box(ax, 3, 61.0, 13.3, 5.4, "Degraded road\nimage\n" + r"$I_d$", COLORS["image"], fs=10.0, weight="bold")
    box(
        ax,
        3,
        56.8,
        30.0,
        3.5,
        r"Optional metadata record $m$" + "\n" + "blur/gyro/accel, speed, exposure" + "\n" + "focus, noise, low light, JPEG",
        COLORS["meta"],
        fs=7.8,
    )

    box(ax, 21.5, 62.2, 15.2, 3.7, r"Image code encoder" + "\n" + r"$z_b=g_\phi(I_d)$", COLORS["code"], fs=9.2)
    box(ax, 36.0, 57.0, 17.4, 3.9, r"Metadata-to-code mapper" + "\n" + r"$z_m=q(m)$", COLORS["code"], fs=9.2)
    box(
        ax,
        56.6,
        58.8,
        18.0,
        5.8,
        r"Code fusion" + "\n" + r"$z=z_b$  (image-only)" + "\n" + r"$z=\frac{1}{2}(z_b+z_m)$  (metadata)",
        "#d8f0ed",
        fs=9.4,
        weight="bold",
    )
    arrow(ax, (16.3, 63.6), (21.5, 64.0))
    arrow(ax, (33.0, 58.5), (36.0, 58.9))
    arrow(ax, (36.7, 64.0), (56.6, 62.2))
    arrow(ax, (53.4, 58.9), (56.6, 60.5))

    label(ax, 82.5, 66.2, r"8-D road degradation code $z$", fs=9, weight="bold")
    code_names = ["motion-x", "motion-y", "vibration", "defocus", "noise", "low-light", "JPEG", "severity"]
    for i, name in enumerate(code_names):
        pill(ax, 76.5 + (i % 4) * 5.2, 62.7 - (i // 4) * 2.4, 4.8, 1.5, name, "#f7fbfa", fs=6.6)
    label(
        ax,
        85.0,
        57.35,
        "Disclosure: road-damage mAP uses proxy metadata.\nKITTI validates real OXTS telemetry under controlled blur.",
        fs=7.0,
        color="#6a4b00",
    )

    grouped_panel(ax, 1.2, 25.6, 97.6, 28.5, "Restoration network")
    box(ax, 3, 45.7, 12.0, 4.0, r"Stem" + "\n" + r"$s=\mathrm{Conv}(I_d)$", COLORS["image"], fs=9.2)
    box(ax, 3, 33.2, 16.2, 5.0, r"Defect-edge attention" + "\n" + r"$E(I_d)=\|\nabla I_d\|$" + "\n" + r"$\tilde{s}=s\odot(1+A(E))$", COLORS["edge"], fs=8.7)
    arrow(ax, (9, 45.7), (10.3, 38.2))

    stages = [
        (25.0, 43.9, 12.5, 5.5, "Encoder 1", r"$B_1(\tilde{s},z)$"),
        (42.4, 43.9, 12.5, 5.5, "Encoder 2", r"$\downarrow\,B_2(\cdot,z)$"),
        (59.8, 43.9, 12.5, 5.5, "Bottleneck", r"$B_3(\cdot,z)$"),
        (42.4, 31.2, 12.5, 5.5, "Decoder 2", r"$\uparrow\,B_4(\cdot,z)$"),
        (25.0, 31.2, 12.5, 5.5, "Decoder 1", r"$\uparrow\,B_5(\cdot,z)$"),
    ]
    for x, y, w, h, t, eq in stages:
        box(ax, x, y, w, h, t + "\n" + eq, COLORS["net"], fs=9.0, weight="bold")

    arrow(ax, (15.0, 47.7), (25.0, 46.7))
    arrow(ax, (37.5, 46.7), (42.4, 46.7))
    arrow(ax, (54.9, 46.7), (59.8, 46.7))
    arrow(ax, (66.0, 43.9), (49.0, 36.7), rad=-0.16)
    arrow(ax, (42.4, 33.9), (37.5, 33.9))
    arrow(ax, (25.0, 33.9), (19.2, 35.8))

    box(ax, 76.5, 40.4, 18.2, 8.7, r"FiLM-conditioned block" + "\n" + r"$\mathrm{FiLM}_\ell(x,z)$" + "\n" + r"$=(1+\gamma_\ell(z))\odot x+\beta_\ell(z)$" + "\n" + r"$B_\ell:x+\mathrm{DWConv}(\mathrm{FiLM}_\ell(\mathrm{GN}(x),z))$", "#f3effb", fs=8.4)
    arrow(ax, (74.6, 61.4), (85.6, 49.1), rad=-0.05)
    label(ax, 82.8, 52.7, r"$z$ modulates every block", fs=8.2, color=COLORS["muted"])

    box(ax, 76.5, 30.2, 18.2, 5.5, r"Residual output" + "\n" + r"$r=h_\theta(I_d,z)$" + "\n" + r"$I_r=\mathrm{clip}(I_d+r,0,1)$", COLORS["image"], fs=9.0, weight="bold")
    arrow(ax, (37.5, 33.9), (76.5, 32.9), rad=0.08)

    grouped_panel(ax, 1.2, 7.8, 48.0, 15.3, "Training objective")
    box(
        ax,
        3.0,
        11.1,
        44.5,
        8.8,
        r"$\mathcal{L}=\|I_r-I_c\|_1+\lambda_e\|\nabla I_r-\nabla I_c\|_1$"
        + "\n"
        + r"$+\lambda_f\||\mathcal{F}(I_r)|-|\mathcal{F}(I_c)|\|_1$"
        + "\n"
        + r"$+\lambda_d\,\|(1+M_c)\odot(I_r-I_c)\|_1+\lambda_z\,\mathrm{SmoothL1}(z_b,z_m)$",
        COLORS["loss"],
        fs=10.1,
    )
    label(ax, 25.0, 9.2, r"$M_c$: clean-image gradient mask used to weight defect-like structures", fs=8.2, color=COLORS["muted"])

    grouped_panel(ax, 51.0, 7.8, 47.8, 15.3, "ITS evaluation path")
    box(ax, 53.0, 15.2, 9.8, 4.5, r"Restored" + "\n" + r"$I_r$", COLORS["image"], fs=10, weight="bold")
    box(ax, 68.0, 15.2, 12.8, 4.5, r"Frozen detector" + "\n" + r"$D_\psi(I_r)$", COLORS["det"], fs=10, weight="bold")
    box(ax, 85.2, 14.6, 11.0, 5.8, r"Report" + "\n" + r"$\Delta$ mAP50" + "\n" + "mAP50--95\nruntime", "#f8f1dc", fs=9.2, weight="bold")
    arrow(ax, (62.8, 17.45), (68.0, 17.45))
    arrow(ax, (80.8, 17.45), (85.2, 17.45))
    label(
        ax,
        75.0,
        10.35,
        "Detector weights are frozen: improvements measure restoration benefit,\nnot detector retraining.",
        fs=8.0,
        color=COLORS["muted"],
    )

    ax.add_patch(Rectangle((1.1, 1.0), 97.8, 4.2, linewidth=1.0, edgecolor="#d7b55a", facecolor="#fff9e8"))
    label(
        ax,
        50,
        3.12,
        "Claim boundary: RMR-Net is a road/ITS metadata-conditioned restoration pipeline.\n"
        "It is not claimed as the first sensor-guided or universal blind restoration model.",
        fs=8.7,
        color="#614600",
        weight="bold",
    )

    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"fig_rmrnet_model_article_plate.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    build()
