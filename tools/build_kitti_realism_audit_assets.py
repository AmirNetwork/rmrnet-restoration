from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_ieee_tits_rmrnet"
FIG_DIR = PAPER / "figures"
TAB_DIR = PAPER / "tables"
RAW_SUMMARY = ROOT / "runs" / "kitti_realmeta_robustness_rawtelemetry_trained_30ep" / "summary.json"
FULL_SUMMARY = ROOT / "runs" / "kitti_realmeta_robustness_raw_audit_existing" / "summary.json"


def bf(value: str, bold: bool) -> str:
    return rf"\textbf{{{value}}}" if bold else value


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_raw_telemetry_table(raw: dict) -> None:
    rows = [
        ("Degraded input", "degraded", "No restoration"),
        (r"\rmr{} image-only", "blind", "No metadata"),
        (r"\rmr{} + scalar telemetry", "raw_scalar", "Speed, exposure, vibration; no direction"),
        (r"\rmr{} + raw OXTS telemetry", "raw_telemetry", "Speed, exposure, yaw-rate, acceleration"),
    ]
    values = raw["mean"]
    max_psnr = max(values[key]["psnr"] for _, key, _ in rows)
    max_ssim = max(values[key]["ssim"] for _, key, _ in rows)
    degraded_psnr = values["degraded"]["psnr"]
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{Stricter KITTI raw-telemetry audit. This \rmr{} checkpoint is trained and tested without the derived blur length and blur angle fields. Raw OXTS telemetry uses speed, exposure, yaw-rate, and acceleration from the vehicle log; scalar telemetry removes the yaw/lateral direction cue.}",
        r"\label{tab:kitti_rawtelemetry_audit}",
        r"\setlength{\tabcolsep}{2.3pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Condition & PSNR$\uparrow$ & SSIM$\uparrow$ & $\Delta$PSNR \\",
        r"\midrule",
    ]
    for label, key, _note in rows:
        psnr = values[key]["psnr"]
        ssim = values[key]["ssim"]
        lines.append(
            f"{label} & {bf(f'{psnr:.2f}', abs(psnr - max_psnr) < 1e-9)} & "
            f"{bf(f'{ssim:.3f}', abs(ssim - max_ssim) < 1e-9)} & {psnr - degraded_psnr:+.2f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (TAB_DIR / "table_kitti_rawtelemetry_audit.tex").write_text("\n".join(lines), encoding="utf-8")


def plot_raw_telemetry(raw: dict, full: dict) -> None:
    labels = ["Degraded", "Image-only", "Raw scalar", "Raw telemetry", "Full-prior upper bound"]
    psnr = [
        raw["mean"]["degraded"]["psnr"],
        raw["mean"]["blind"]["psnr"],
        raw["mean"]["raw_scalar"]["psnr"],
        raw["mean"]["raw_telemetry"]["psnr"],
        full["mean"]["true"]["psnr"],
    ]
    ssim = [
        raw["mean"]["degraded"]["ssim"],
        raw["mean"]["blind"]["ssim"],
        raw["mean"]["raw_scalar"]["ssim"],
        raw["mean"]["raw_telemetry"]["ssim"],
        full["mean"]["true"]["ssim"],
    ]
    colors = ["#9da7ad", "#72bda3", "#b9c46a", "#139b75", "#355c7d"]

    fig, axes = plt.subplots(1, 2, figsize=(8.1, 3.0), dpi=220)
    axes[0].bar(labels, psnr, color=colors)
    axes[0].axhline(psnr[0], color="#4a545c", lw=1.0, ls="--")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_ylim(min(psnr) - 0.25, max(psnr) + 0.25)
    axes[0].set_title("Raw telemetry without blur fields")
    axes[0].tick_params(axis="x", rotation=25, labelsize=7.0)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(labels, ssim, color=colors)
    axes[1].axhline(ssim[0], color="#4a545c", lw=1.0, ls="--")
    axes[1].set_ylabel("SSIM")
    axes[1].set_ylim(min(ssim) - 0.015, max(ssim) + 0.015)
    axes[1].set_title("Structure recovery")
    axes[1].tick_params(axis="x", rotation=25, labelsize=7.0)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Held-out KITTI drive 0011: stricter real-telemetry audit", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_kitti_rawtelemetry_audit.png", bbox_inches="tight")
    plt.close(fig)


def update_summary(raw: dict, full: dict) -> None:
    path = PAPER / "KITTI_REALMETA_SUMMARY.json"
    summary = load(path)
    summary["raw_telemetry_no_blur_fields_audit"] = {
        "train_sequences": raw["no_leakage_check"]["train_sequences"],
        "test_sequences": raw["no_leakage_check"]["test_sequences"],
        "checkpoint": str(ROOT / "runs" / "rmrnet_kitti_rawtelemetry_splitB_30ep" / "rcadnet_last.pth"),
        "metadata_fields_removed": ["blur_length_px", "blur_angle_deg", "telemetry_strength", "blur_scale"],
        "degraded_psnr": raw["mean"]["degraded"]["psnr"],
        "image_only_psnr": raw["mean"]["blind"]["psnr"],
        "raw_scalar_psnr": raw["mean"]["raw_scalar"]["psnr"],
        "raw_telemetry_psnr": raw["mean"]["raw_telemetry"]["psnr"],
        "raw_telemetry_ssim": raw["mean"]["raw_telemetry"]["ssim"],
        "raw_telemetry_gain_vs_degraded_psnr": raw["mean"]["raw_telemetry"]["psnr"] - raw["mean"]["degraded"]["psnr"],
        "raw_telemetry_gain_vs_blind_psnr": raw["mean"]["raw_telemetry"]["psnr"] - raw["mean"]["blind"]["psnr"],
        "full_prior_upper_bound_psnr": full["mean"]["true"]["psnr"],
        "no_sequence_overlap": raw["no_leakage_check"]["sequence_overlap"] == [],
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    raw = load(RAW_SUMMARY)
    full = load(FULL_SUMMARY)
    write_raw_telemetry_table(raw)
    plot_raw_telemetry(raw, full)
    update_summary(raw, full)
    print(json.dumps({"table": str(TAB_DIR / "table_kitti_rawtelemetry_audit.tex"), "figure": str(FIG_DIR / "fig_kitti_rawtelemetry_audit.png")}, indent=2))


if __name__ == "__main__":
    main()
