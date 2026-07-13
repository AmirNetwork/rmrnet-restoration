from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_ieee_tits_rmrnet"
TAB = PAPER / "tables"
FIG = PAPER / "figures"

POTH_DET = ROOT / "runs" / "detection_eval_revised" / "pothole_test_revised.csv"
PCM_DET = ROOT / "runs" / "detection_eval_revised" / "pcm_test_revised_640.csv"
POTH_META = ROOT / "runs" / "detection_eval_revised" / "pothole_metadata_ablation_revised.csv"
PCM_PER = ROOT / "runs" / "detection_eval_revised" / "pcm_per_class_revised_640.csv"
POTH_RMR_REST = ROOT / "runs" / "bench_pothole_test_rmr_revised_evidence_30ep" / "metrics.csv"
PCM_RMR_REST = ROOT / "runs" / "bench_pcm_test_rmr_revised_evidence_30ep" / "metrics.csv"
POTH_OLD_REST = ROOT / "runs" / "bench_pothole_test_rmr_metadata_naf_dfpir" / "metrics.csv"
PCM_OLD_REST = ROOT / "runs" / "bench_pcm_test_rmr_metadata_naf_dfpir" / "metrics.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(value: str | float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def bf(text: str) -> str:
    return "\\textbf{" + text + "}"


def split_name(name: str) -> tuple[str, str]:
    if name == "clean":
        return "clean", "clean"
    scenario, method = name.split("_", 1)
    pretty = {
        "degraded": "degraded",
        "rmr_revised": "RMR-Net",
        "nafnet": "NAFNet-road",
        "dfpir": "DFPIR",
    }
    return scenario, pretty.get(method, method)


def write_table(path: Path, caption: str, label: str, headers: list[str], body: list[list[str]], *, wide: bool = True) -> None:
    env = "table*" if wide else "table"
    align = "ll" + "r" * max(0, len(headers) - 2)
    lines = [
        f"\\begin{{{env}}}[!t]",
        "\\centering",
        "\\caption{" + caption + "}",
        "\\label{" + label + "}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{" + align + "}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    lines.extend(" & ".join(row) + " \\\\" for row in body)
    lines.extend(["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def detection_table(src: Path, out_name: str, caption: str, label: str) -> None:
    data = rows(src)
    by = {r["name"]: r for r in data}
    body: list[list[str]] = []
    order = ["clean"]
    for scenario in ["motion", "defocus", "lowlight"]:
        order.extend([f"{scenario}_degraded", f"{scenario}_rmr_revised", f"{scenario}_nafnet", f"{scenario}_dfpir"])
    for key in order:
        r = by[key]
        scenario, method = split_name(key)
        map50 = f(r["map50"])
        map95 = f(r["map50_95"])
        if scenario != "clean":
            group = [by[f"{scenario}_{m}"] for m in ["rmr_revised", "nafnet", "dfpir"]]
            if float(r["map50"]) == max(float(g["map50"]) for g in group):
                map50 = bf(map50)
            if float(r["map50_95"]) == max(float(g["map50_95"]) for g in group):
                map95 = bf(map95)
        body.append([scenario, method, map50, map95, f(r["precision"]), f(r["recall"])])
    write_table(TAB / out_name, caption, label, ["Scenario", "Input", "mAP50", "mAP50--95", "Prec.", "Rec."], body)


def crack_table() -> None:
    data = rows(PCM_PER)
    by = {(r["eval_name"], r["class_name"]): r for r in data}
    selected = [
        "clean",
        "motion_degraded",
        "motion_rmr_revised",
        "motion_dfpir",
        "defocus_degraded",
        "defocus_rmr_revised",
        "defocus_nafnet",
        "defocus_dfpir",
        "lowlight_degraded",
        "lowlight_rmr_revised",
        "lowlight_dfpir",
    ]
    body: list[list[str]] = []
    for key in selected:
        r = by[(key, "crack")]
        scenario, method = split_name(key)
        map50 = f(r["map50"])
        map95 = f(r["map50_95"])
        if scenario != "clean":
            group_keys = [name for name in selected if name.startswith(scenario + "_") and not name.endswith("_degraded")]
            group = [by[(name, "crack")] for name in group_keys]
            if float(r["map50"]) == max(float(g["map50"]) for g in group):
                map50 = bf(map50)
            if float(r["map50_95"]) == max(float(g["map50_95"]) for g in group):
                map95 = bf(map95)
        body.append([scenario, method, map50, map95, f(r["precision"]), f(r["recall"])])
    write_table(TAB / "table_pcm_crack_detection.tex", "Crack-specific detection recovery on the PCM road-damage dataset.", "tab:pcm_crack", ["Scenario", "Input", "Crack mAP50", "Crack mAP50--95", "Prec.", "Rec."], body)


def metadata_table() -> None:
    by = {r["name"]: r for r in rows(POTH_META)}
    body = []
    for scenario in ["motion", "defocus", "lowlight"]:
        meta = by[f"{scenario}_rmr_revised"]
        blind = by[f"{scenario}_rmr_revised_blind"]
        gain = float(meta["map50"]) - float(blind["map50"])
        gain_text = f(gain)
        meta_text = f(meta["map50"])
        if gain > 0:
            gain_text = bf(gain_text)
            meta_text = bf(meta_text)
        body.append([scenario, f(blind["map50"]), meta_text, gain_text])
    write_table(
        TAB / "table_metadata_ablation.tex",
        "Metadata-conditioned versus image-only RMR-Net on held-out pothole detection.",
        "tab:metadata_ablation",
        ["Scenario", "Image-only mAP50", "Metadata mAP50", "Gain"],
        body,
        wide=False,
    )


def restoration_table() -> None:
    revised = {("Pothole", r["scenario"], "RMR-Net"): r for r in rows(POTH_RMR_REST)}
    revised.update({("PCM", r["scenario"], "RMR-Net"): r for r in rows(PCM_RMR_REST)})
    body_rows: list[tuple[str, str, str, dict[str, str]]] = []
    for dataset, old_path in [("Pothole", POTH_OLD_REST), ("PCM", PCM_OLD_REST)]:
        old = rows(old_path)
        for scenario in ["motion_horizontal_medium", "defocus_medium", "lowlight_medium"]:
            body_rows.append((dataset, scenario, "RMR-Net", revised[(dataset, scenario, "RMR-Net")]))
            for r in old:
                if r["scenario"] == scenario and r["model"] in {"NAFNet-road", "DFPIR-CVPR2025"}:
                    label = r["model"].replace("DFPIR-CVPR2025", "DFPIR")
                    body_rows.append((dataset, scenario, label, r))
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for dataset, scenario, _model, r in body_rows:
        grouped.setdefault((dataset, scenario), []).append(r)
    body: list[list[str]] = []
    for dataset, scenario, model, r in body_rows:
        group = grouped[(dataset, scenario)]
        psnr = f(r["psnr"], 2)
        ssim = f(r["ssim"])
        runtime = f(r["mean_runtime_ms"], 1)
        if float(r["psnr"]) == max(float(g["psnr"]) for g in group):
            psnr = bf(psnr)
        if float(r["ssim"]) == max(float(g["ssim"]) for g in group):
            ssim = bf(ssim)
        if float(r["mean_runtime_ms"]) == min(float(g["mean_runtime_ms"]) for g in group):
            runtime = bf(runtime)
        short_scenario = scenario.replace("_horizontal", "").replace("_medium", "")
        body.append([dataset, short_scenario, model, psnr, ssim, runtime])
    write_table(
        TAB / "table_restoration_combined.tex",
        "Full-reference restoration and GPU runtime on the held-out road test sets.",
        "tab:restoration_combined",
        ["Dataset", "Scenario", "Model", "PSNR", "SSIM", "ms/img"],
        body,
    )


def detection_bar(src: Path, out_name: str, title: str) -> None:
    by = {r["name"]: float(r["map50"]) for r in rows(src)}
    scenarios = ["motion", "defocus", "lowlight"]
    methods = ["degraded", "rmr_revised", "nafnet", "dfpir"]
    labels = ["Degraded", "RMR-Net", "NAFNet-road", "DFPIR"]
    colors = ["#8c939c", "#139b75", "#3b78a8", "#b95745"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=240)
    xs = range(len(scenarios))
    width = 0.18
    for i, method in enumerate(methods):
        vals = [by[f"{s}_{method}"] for s in scenarios]
        ax.bar([x + (i - 1.5) * width for x in xs], vals, width, color=colors[i], label=labels[i])
    ax.axhline(by["clean"], color="#222222", linestyle="--", linewidth=1)
    ax.text(2.23, by["clean"] + 0.01, "clean", fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(["Motion", "Defocus", "Low light"])
    ax.set_ylabel("Frozen YOLO mAP50")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, max(0.65, by["clean"] + 0.08))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.24))
    fig.tight_layout()
    fig.savefig(FIG / out_name, bbox_inches="tight")
    plt.close(fig)


def metadata_bar() -> None:
    by = {r["name"]: float(r["map50"]) for r in rows(POTH_META)}
    scenarios = ["motion", "defocus", "lowlight"]
    fig, ax = plt.subplots(figsize=(5.4, 3.0), dpi=240)
    xs = range(len(scenarios))
    ax.bar([x - 0.17 for x in xs], [by[f"{s}_rmr_revised_blind"] for s in scenarios], 0.34, label="image-only", color="#72bda3")
    ax.bar([x + 0.17 for x in xs], [by[f"{s}_rmr_revised"] for s in scenarios], 0.34, label="metadata", color="#139b75")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(["Motion", "Defocus", "Low light"])
    ax.set_ylabel("Pothole mAP50")
    ax.set_ylim(0, 0.62)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_metadata_ablation.png", bbox_inches="tight")
    plt.close(fig)


def summary() -> None:
    out = {
        "training": {
            "checkpoint": str(ROOT / "runs" / "rmrnet_revised_evidence_30ep" / "rcadnet_best.pth"),
            "epochs": 30,
            "best_val_psnr": 27.303161792003127,
        },
        "detection": {
            "pothole": str(POTH_DET),
            "pcm": str(PCM_DET),
            "pcm_per_class": str(PCM_PER),
        },
        "restoration": {
            "pothole_rmr": str(POTH_RMR_REST),
            "pcm_rmr": str(PCM_RMR_REST),
        },
    }
    (PAPER / "REVISED_RESULTS_SUMMARY.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    detection_table(POTH_DET, "table_pothole_detection.tex", "Held-out pothole detection before and after restoration.", "tab:pothole_detection")
    detection_table(PCM_DET, "table_pcm_detection.tex", "Held-out PCM pothole/crack/manhole detection before and after restoration.", "tab:pcm_detection")
    crack_table()
    metadata_table()
    restoration_table()
    detection_bar(POTH_DET, "fig_pothole_detection_recovery.png", "Pothole dataset downstream detection recovery")
    detection_bar(PCM_DET, "fig_pcm_detection_recovery.png", "PCM multi-class road-damage detection recovery")
    metadata_bar()
    summary()
    print(PAPER)


if __name__ == "__main__":
    main()
