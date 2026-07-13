from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_ieee_tits_rmrnet"
FIGURES = PAPER / "figures"
TABLES = PAPER / "tables"
EXP = ROOT / "experiments" / "v31_demoe_integration"


SCENARIOS = ["motion", "defocus", "low light", "mixed motion+low light"]

METHOD_COLORS = {
    "degraded": "#747b83",
    "DFPIR": "#5778a4",
    "DeMoE-auto": "#7f6dba",
    "DeMoE-scenario": "#b279a2",
    "NAFNet-road": "#59a14f",
    "RMR-Net": "#e15759",
}


POTHOLE_DETECTION = {
    "clean": {"clean": 0.652},
    "motion": {
        "degraded": 0.140,
        "DFPIR": 0.374,
        "DeMoE-auto": 0.371,
        "DeMoE-scenario": 0.382,
        "NAFNet-road": 0.190,
        "RMR-Net": 0.423,
    },
    "defocus": {
        "degraded": 0.154,
        "DFPIR": 0.211,
        "DeMoE-auto": 0.248,
        "DeMoE-scenario": 0.254,
        "NAFNet-road": 0.357,
        "RMR-Net": 0.366,
    },
    "low light": {
        "degraded": 0.517,
        "DFPIR": 0.475,
        "DeMoE-auto": 0.510,
        "DeMoE-scenario": 0.510,
        "NAFNet-road": 0.505,
        "RMR-Net": 0.525,
    },
    "mixed motion+low light": {
        "degraded": 0.280,
        "DFPIR": 0.307,
        "DeMoE-auto": 0.343,
        "DeMoE-scenario": 0.351,
        "NAFNet-road": 0.275,
        "RMR-Net": 0.426,
    },
}

PCM_DETECTION = {
    "clean": {"clean": 0.541},
    "motion": {
        "degraded": 0.203,
        "DFPIR": 0.294,
        "DeMoE-auto": 0.319,
        "DeMoE-scenario": 0.320,
        "NAFNet-road": 0.194,
        "RMR-Net": 0.303,
    },
    "defocus": {
        "degraded": 0.060,
        "DFPIR": 0.091,
        "DeMoE-auto": 0.136,
        "DeMoE-scenario": 0.152,
        "NAFNet-road": 0.174,
        "RMR-Net": 0.227,
    },
    "low light": {
        "degraded": 0.326,
        "DFPIR": 0.352,
        "DeMoE-auto": 0.317,
        "DeMoE-scenario": 0.263,
        "NAFNet-road": 0.364,
        "RMR-Net": 0.432,
    },
    "mixed motion+low light": {
        "degraded": 0.061,
        "DFPIR": 0.059,
        "DeMoE-auto": 0.082,
        "DeMoE-scenario": 0.080,
        "NAFNet-road": 0.083,
        "RMR-Net": 0.145,
    },
}

RESTORATION = {
    ("Pothole", "motion"): {
        "RMR-Net": (25.40, 0.695, 142.6),
        "NAFNet-road": (24.76, 0.647, 302.0),
        "DFPIR": (26.44, 0.710, 1335.1),
        "DeMoE-auto": (24.436898206550538, 0.6205550768477394, 388.1292240641395),
        "DeMoE-scenario": (24.70168939711274, 0.6253916178157622, 394.23204117645065),
    },
    ("Pothole", "defocus"): {
        "RMR-Net": (24.72, 0.630, 147.9),
        "NAFNet-road": (24.58, 0.702, 230.9),
        "DFPIR": (25.98, 0.672, 1308.9),
        "DeMoE-auto": (23.643576146211245, 0.5368887487261053, 392.9323085561247),
        "DeMoE-scenario": (23.692846493868167, 0.5423701279622348, 394.6492427807593),
    },
    ("Pothole", "low light"): {
        "RMR-Net": (29.16, 0.855, 148.1),
        "NAFNet-road": (31.61, 0.908, 249.9),
        "DFPIR": (16.26, 0.831, 1456.7),
        "DeMoE-auto": (16.67174708998718, 0.6531404256820679, 392.92318823533805),
        "DeMoE-scenario": (20.360029929955672, 0.7182869087247288, 395.06985187164446),
    },
    ("PCM", "motion"): {
        "RMR-Net": (27.78, 0.834, 171.8),
        "NAFNet-road": (25.50, 0.796, 54.8),
        "DFPIR": (28.65, 0.872, 313.7),
        "DeMoE-auto": (27.062790949469065, 0.8120474591357818, 436.86192185428894),
        "DeMoE-scenario": (27.073222203508198, 0.8122865609972683, 438.57146953644207),
    },
    ("PCM", "defocus"): {
        "RMR-Net": (26.94, 0.795, 175.8),
        "NAFNet-road": (25.88, 0.834, 52.1),
        "DFPIR": (28.43, 0.866, 306.0),
        "DeMoE-auto": (25.318664061730352, 0.7347552506438154, 358.6074933774693),
        "DeMoE-scenario": (25.174183863213006, 0.7364476142537515, 363.54670827813607),
    },
    ("PCM", "low light"): {
        "RMR-Net": (32.12, 0.900, 177.5),
        "NAFNet-road": (34.03, 0.937, 50.8),
        "DFPIR": (17.41, 0.831, 297.1),
        "DeMoE-auto": (14.083531407609021, 0.6708101109163651, 183.22168377487486),
        "DeMoE-scenario": (20.779984829531763, 0.7896017329582316, 183.49669768211254),
    },
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    return float(np.corrcoef(xa, ya)[0, 1])


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(rankdata(x), rankdata(y))


def kendall_tau(x: list[float], y: list[float]) -> float:
    concordant = 0
    discordant = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            sx = np.sign(x[i] - x[j])
            sy = np.sign(y[i] - y[j])
            if sx == 0 or sy == 0:
                continue
            if sx == sy:
                concordant += 1
            else:
                discordant += 1
    denom = concordant + discordant
    return float((concordant - discordant) / denom) if denom else 0.0


def save_grouped_detection(data: dict[str, dict[str, float]], methods: list[str], title: str, path: Path, clean: float) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    x = np.arange(len(SCENARIOS))
    width = 0.11 if len(methods) >= 7 else 0.13
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width
    for offset, method in zip(offsets, methods):
        vals = [data[scenario][method] for scenario in SCENARIOS]
        bars = ax.bar(x + offset, vals, width, label=method, color=METHOD_COLORS[method], edgecolor="white", linewidth=0.6)
        if method == "RMR-Net":
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{bar.get_height():.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )
    ax.axhline(clean, color="#1f2933", linestyle="--", linewidth=1.1, label="clean detector")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("motion", "motion blur", 1) if s == "motion" else s for s in SCENARIOS])
    ax.set_ylim(0, max(clean + 0.08, 0.58))
    ax.set_ylabel("YOLO11s mAP50")
    ax.set_title(title, pad=10)
    ax.grid(axis="y", color="#d8dee6", linewidth=0.7, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_demoe_router_figure(path: Path) -> None:
    labels = ["Pothole\nmotion", "Pothole\ndefocus", "Pothole\nlow light", "PCM\nmotion", "PCM\ndefocus", "PCM\nlow light"]
    keys = [("Pothole", "motion"), ("Pothole", "defocus"), ("Pothole", "low light"), ("PCM", "motion"), ("PCM", "defocus"), ("PCM", "low light")]
    psnr_auto = [RESTORATION[k]["DeMoE-auto"][0] for k in keys]
    psnr_scen = [RESTORATION[k]["DeMoE-scenario"][0] for k in keys]
    map_auto = [
        POTHOLE_DETECTION["motion"]["DeMoE-auto"],
        POTHOLE_DETECTION["defocus"]["DeMoE-auto"],
        POTHOLE_DETECTION["low light"]["DeMoE-auto"],
        PCM_DETECTION["motion"]["DeMoE-auto"],
        PCM_DETECTION["defocus"]["DeMoE-auto"],
        PCM_DETECTION["low light"]["DeMoE-auto"],
    ]
    map_scen = [
        POTHOLE_DETECTION["motion"]["DeMoE-scenario"],
        POTHOLE_DETECTION["defocus"]["DeMoE-scenario"],
        POTHOLE_DETECTION["low light"]["DeMoE-scenario"],
        PCM_DETECTION["motion"]["DeMoE-scenario"],
        PCM_DETECTION["defocus"]["DeMoE-scenario"],
        PCM_DETECTION["low light"]["DeMoE-scenario"],
    ]

    fig, axes = plt.subplots(2, 1, figsize=(9.4, 6.2), sharex=True)
    x = np.arange(len(labels))
    w = 0.36
    axes[0].bar(x - w / 2, psnr_auto, w, label="DeMoE-auto", color=METHOD_COLORS["DeMoE-auto"])
    axes[0].bar(x + w / 2, psnr_scen, w, label="DeMoE-scenario", color=METHOD_COLORS["DeMoE-scenario"])
    axes[1].bar(x - w / 2, map_auto, w, label="DeMoE-auto", color=METHOD_COLORS["DeMoE-auto"])
    axes[1].bar(x + w / 2, map_scen, w, label="DeMoE-scenario", color=METHOD_COLORS["DeMoE-scenario"])
    axes[0].set_ylabel("PSNR (dB)")
    axes[1].set_ylabel("YOLO11s mAP50")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[0].set_title("DeMoE routing audit: known degradation helps fidelity more than detection")
    for ax in axes:
        ax.grid(axis="y", color="#d8dee6", linewidth=0.7, alpha=0.75)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(ncol=2, fontsize=9, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_fidelity_scatter(path: Path) -> tuple[float, float, int]:
    points: list[dict[str, object]] = []
    detection_lookup = {
        ("Pothole", "motion"): POTHOLE_DETECTION["motion"],
        ("Pothole", "defocus"): POTHOLE_DETECTION["defocus"],
        ("Pothole", "low light"): POTHOLE_DETECTION["low light"],
        ("PCM", "motion"): PCM_DETECTION["motion"],
        ("PCM", "defocus"): PCM_DETECTION["defocus"],
        ("PCM", "low light"): PCM_DETECTION["low light"],
    }
    methods = ["RMR-Net", "NAFNet-road", "DFPIR", "DeMoE-auto", "DeMoE-scenario"]
    for key, rest in RESTORATION.items():
        dataset, scenario = key
        for method in methods:
            if method not in rest or method not in detection_lookup[key]:
                continue
            points.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "method": method,
                    "psnr": rest[method][0],
                    "map50": detection_lookup[key][method],
                }
            )
    write_csv(EXP / "fidelity_detection_pairs_with_demoe.csv", points)
    xs = [float(p["psnr"]) for p in points]
    ys = [float(p["map50"]) for p in points]
    rho = spearman(xs, ys)
    tau = kendall_tau(xs, ys)

    markers = {"motion": "o", "defocus": "s", "low light": "^", "mixed motion+low light": "D"}
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for method in methods:
        subset = [p for p in points if p["method"] == method]
        for scenario in SCENARIOS:
            sp = [p for p in subset if p["scenario"] == scenario]
            if not sp:
                continue
            ax.scatter(
                [p["psnr"] for p in sp],
                [p["map50"] for p in sp],
                color=METHOD_COLORS[method],
                marker=markers[scenario],
                s=52,
                edgecolor="white",
                linewidth=0.7,
                alpha=0.92,
                label=method if scenario == "motion" else None,
            )
    ax.set_xlabel("PSNR (dB)")
    ax.set_ylabel("YOLO11s mAP50")
    ax.set_title(f"Fidelity and detection diverge (Spearman {rho:.3f}, Kendall {tau:.3f})")
    ax.grid(color="#d8dee6", linewidth=0.7, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    method_legend = ax.legend(title="method", fontsize=8, title_fontsize=8, frameon=False, loc="upper left")
    ax.add_artist(method_legend)
    scenario_handles = [
        plt.Line2D([0], [0], marker=markers[s], color="none", markerfacecolor="#6b7280", markeredgecolor="white", markersize=7, label=s)
        for s in SCENARIOS
    ]
    ax.legend(handles=scenario_handles, title="scenario", fontsize=8, title_fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return rho, tau, len(points)


def write_correlation_table(rho: float, tau: float, n: int) -> None:
    tex = rf"""\begin{{table}}[!t]
\centering
\caption{{Rank correlation between full-reference restoration quality and YOLO11s detection utility after adding DeMoE-auto and DeMoE-scenario. The weak correlation supports reporting fidelity and perception metrics separately.}}
\label{{tab:fidelity_detection_correlation}}
\small
\begin{{tabular}}{{lrr}}
\toprule
Metric pair & Correlation & Rows \\
\midrule
PSNR vs. \mapfifty, Spearman $\rho$ & {rho:.3f} & {n} \\
PSNR vs. \mapfifty, Kendall $\tau$ & {tau:.3f} & {n} \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    (TABLES / "table_fidelity_detection_correlation.tex").write_text(tex, encoding="utf-8")
    write_csv(EXP / "fidelity_detection_correlation_with_demoe.csv", [{"metric": "spearman", "value": rho, "rows": n}, {"metric": "kendall", "value": tau, "rows": n}])


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)
    save_grouped_detection(
        POTHOLE_DETECTION,
        ["degraded", "DFPIR", "DeMoE-auto", "DeMoE-scenario", "NAFNet-road", "RMR-Net"],
        "Pothole detection recovery with DeMoE benchmark",
        FIGURES / "fig_pothole_detection_recovery.png",
        clean=0.652,
    )
    save_grouped_detection(
        PCM_DETECTION,
        ["degraded", "DFPIR", "DeMoE-auto", "DeMoE-scenario", "NAFNet-road", "RMR-Net"],
        "PCM multi-class detection recovery with DeMoE benchmark",
        FIGURES / "fig_pcm_detection_recovery.png",
        clean=0.541,
    )
    save_demoe_router_figure(FIGURES / "fig_demoe_router_audit.png")
    rho, tau, n = save_fidelity_scatter(FIGURES / "fig_fidelity_detection_correlation.png")
    write_correlation_table(rho, tau, n)
    print({"spearman": rho, "kendall": tau, "rows": n})


if __name__ == "__main__":
    main()
