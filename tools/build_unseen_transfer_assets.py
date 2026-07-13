from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_ieee_tits_rmrnet"
TAB_DIR = PAPER / "tables"
FIG_DIR = PAPER / "figures"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def scenario_label(name: str) -> str:
    return name.replace("_medium", "").replace("motion_horizontal", "motion").replace("_", " ")


def main() -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    degraded = read_rows(ROOT / "runs" / "bench_pcm_unseen_degraded" / "metrics.csv")
    learned = read_rows(ROOT / "runs" / "bench_pcm_unseen_potholeonly_rmr_naf" / "metrics.csv")
    dfpir_all = read_rows(ROOT / "runs" / "bench_pcm_test_rmr_metadata_naf_dfpir" / "metrics.csv")
    dfpir = [row for row in dfpir_all if row["model"] == "DFPIR-CVPR2025"]

    rows: list[dict[str, str | float]] = []
    for row in degraded:
        rows.append({"scenario": row["scenario"], "method": "Degraded input", "psnr": float(row["psnr"]), "ssim": float(row["ssim"]), "runtime": 0.0})
    for row in learned:
        method = "RMR pothole-only" if row["model"] == "RMR-Net" else "NAFNet pothole-only"
        rows.append({"scenario": row["scenario"], "method": method, "psnr": float(row["psnr"]), "ssim": float(row["ssim"]), "runtime": float(row["mean_runtime_ms"])})
    for row in dfpir:
        rows.append({"scenario": row["scenario"], "method": "DFPIR", "psnr": float(row["psnr"]), "ssim": float(row["ssim"]), "runtime": float(row["mean_runtime_ms"])})

    method_order = ["Degraded input", "RMR pothole-only", "NAFNet pothole-only", "DFPIR"]
    scenarios = ["motion_horizontal_medium", "defocus_medium", "lowlight_medium"]
    by = {(row["scenario"], row["method"]): row for row in rows}
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Unseen-source PCM restoration transfer. Learned road restorers are trained only on the IVCNZ pothole restoration source and then evaluated on the unseen PCM pothole/crack/manhole test source. DFPIR is an external generic restoration baseline.}",
        r"\label{tab:unseen_pcm_transfer}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Scenario & Method & PSNR$\uparrow$ & SSIM$\uparrow$ & Runtime ms$\downarrow$ \\",
        r"\midrule",
    ]
    for scenario in scenarios:
        group = [by[(scenario, method)] for method in method_order]
        max_psnr = max(row["psnr"] for row in group)
        max_ssim = max(row["ssim"] for row in group)
        min_runtime = min(row["runtime"] for row in group if row["runtime"] > 0)
        for method in method_order:
            row = by[(scenario, method)]
            psnr = f"{row['psnr']:.2f}"
            ssim = f"{row['ssim']:.3f}"
            runtime = f"{row['runtime']:.1f}"
            if abs(row["psnr"] - max_psnr) < 1e-9:
                psnr = rf"\textbf{{{psnr}}}"
            if abs(row["ssim"] - max_ssim) < 1e-9:
                ssim = rf"\textbf{{{ssim}}}"
            if row["runtime"] > 0 and abs(row["runtime"] - min_runtime) < 1e-9:
                runtime = rf"\textbf{{{runtime}}}"
            lines.append(f"{scenario_label(scenario)} & {method} & {psnr} & {ssim} & {runtime} \\\\")
        lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (TAB_DIR / "table_unseen_pcm_transfer.tex").write_text("\n".join(lines), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.4, 3.2), dpi=220)
    colors = {"Degraded input": "#9da7ad", "RMR pothole-only": "#139b75", "NAFNet pothole-only": "#809bce", "DFPIR": "#f4a261"}
    width = 0.18
    xs = range(len(scenarios))
    for idx, method in enumerate(method_order):
        ax.bar([x + (idx - 1.5) * width for x in xs], [by[(s, method)]["psnr"] for s in scenarios], width, label=method, color=colors[method])
    ax.set_xticks(list(xs))
    ax.set_xticklabels(["Motion", "Defocus", "Low light"])
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PCM unseen-source restoration transfer")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_unseen_pcm_transfer.png", bbox_inches="tight")
    plt.close(fig)
    print(TAB_DIR / "table_unseen_pcm_transfer.tex")


if __name__ == "__main__":
    main()
