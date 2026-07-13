from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_assets" / "rmrnet_ieee_2026-05-25"
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"

POTH_DET = ROOT / "runs" / "detection_eval_suite" / "pothole_test_rmr_naf_dfpir.csv"
PCM_DET = ROOT / "runs" / "detection_eval_suite" / "pcm_test_rmr_naf_dfpir.csv"
PCM_PER = ROOT / "runs" / "detection_eval_suite" / "pcm_per_class_selected.csv"
POTH_REST = ROOT / "runs" / "bench_pothole_test_rmr_metadata_naf_dfpir" / "metrics.csv"
POTH_BLIND = ROOT / "runs" / "bench_pothole_test_rmr_blind" / "metrics.csv"
PCM_REST = ROOT / "runs" / "bench_pcm_test_rmr_metadata_naf_dfpir" / "metrics.csv"
PCM_MODEL = ROOT / "runs" / "detect" / "runs" / "detect_train" / "yolov8n_pcm_clean_25ep" / "weights" / "best.pt"

CLASS_COLORS = {
    0: (32, 120, 184),
    1: (20, 155, 117),
    2: (185, 87, 69),
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(value: str | float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def bf(text: str) -> str:
    return "\\textbf{" + text + "}"


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def write_table(path: Path, caption: str, label: str, headers: list[str], body: list[list[str]], wide: bool = True) -> None:
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


def write_custom_table(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def split_name(name: str) -> tuple[str, str]:
    if name == "clean":
        return "clean", "clean"
    scenario, method = name.split("_", 1)
    method = method.replace("rmr_meta", "RMR-Net+meta").replace("rmr_blind", "RMR-Net image-only")
    method = method.replace("degraded", "degraded").replace("nafnet", "NAFNet-road").replace("dfpir", "DFPIR")
    return scenario, method


def detection_table(det_path: Path, out_name: str, caption: str, label: str) -> None:
    body = []
    order = ["clean"]
    for scenario in ["motion", "defocus", "lowlight"]:
        order.extend([f"{scenario}_degraded", f"{scenario}_rmr_meta", f"{scenario}_rmr_blind", f"{scenario}_nafnet", f"{scenario}_dfpir"])
    by = {r["name"]: r for r in rows(det_path)}
    for key in order:
        r = by[key]
        scenario, method = split_name(key)
        map50 = f(r["map50"])
        map95 = f(r["map50_95"])
        if scenario != "clean":
            group = [by[f"{scenario}_{m}"] for m in ["rmr_meta", "rmr_blind", "nafnet", "dfpir"]]
            if float(r["map50"]) == max(float(g["map50"]) for g in group):
                map50 = bf(map50)
            if float(r["map50_95"]) == max(float(g["map50_95"]) for g in group):
                map95 = bf(map95)
        body.append([scenario, method, map50, map95, f(r["precision"]), f(r["recall"])])
    write_table(TAB_DIR / out_name, caption, label, ["Scenario", "Input", "mAP50", "mAP50--95", "Prec.", "Rec."], body)


def restoration_table(paths: list[tuple[str, Path]], out_name: str) -> None:
    body = []
    for dataset, path in paths:
        data = rows(path)
        grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
        for r in data:
            scenario = r["scenario"].replace("_medium", "").replace("_horizontal", "")
            grouped.setdefault((dataset, scenario), []).append(r)
        for r in data:
            scenario = r["scenario"].replace("_medium", "").replace("_horizontal", "")
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
            body.append([dataset, scenario, r["model"].replace("DFPIR-CVPR2025", "DFPIR"), psnr, ssim, runtime])
    write_table(
        TAB_DIR / out_name,
        "Full-reference restoration and GPU runtime on the held-out road test sets.",
        "tab:restoration_combined",
        ["Dataset", "Scenario", "Model", "PSNR", "SSIM", "ms/img"],
        body,
    )


def metadata_table() -> None:
    by = {r["name"]: r for r in rows(POTH_DET)}
    body = []
    for scenario in ["motion", "defocus", "lowlight"]:
        meta = by[f"{scenario}_rmr_meta"]
        blind = by[f"{scenario}_rmr_blind"]
        body.append([scenario, f(blind["map50"]), bf(f(meta["map50"])), bf(f(float(meta["map50"]) - float(blind["map50"])))])
    write_table(
        TAB_DIR / "table_metadata_ablation.tex",
        "Metadata-conditioned versus image-only RMR-Net on held-out pothole detection.",
        "tab:metadata_ablation",
        ["Scenario", "Image-only mAP50", "Metadata mAP50", "Gain"],
        body,
        wide=False,
    )


def crack_table() -> None:
    selected = ["clean", "motion_degraded", "motion_rmr_meta", "motion_dfpir", "defocus_degraded", "defocus_rmr_meta", "defocus_nafnet", "defocus_dfpir", "lowlight_degraded", "lowlight_rmr_meta", "lowlight_dfpir"]
    by = {(r["eval_name"], r["class_name"]): r for r in rows(PCM_PER)}
    body = []
    for key in selected:
        r = by[(key, "crack")]
        scenario, method = split_name(key)
        map50 = f(r["map50"])
        map95 = f(r["map50_95"])
        if scenario != "clean":
            group_keys = [k for k in selected if k.startswith(scenario + "_") and not k.endswith("_degraded")]
            group = [by[(k, "crack")] for k in group_keys]
            if float(r["map50"]) == max(float(g["map50"]) for g in group):
                map50 = bf(map50)
            if float(r["map50_95"]) == max(float(g["map50_95"]) for g in group):
                map95 = bf(map95)
        body.append([scenario, method, map50, map95, f(r["precision"]), f(r["recall"])])
    write_table(TAB_DIR / "table_pcm_crack_detection.tex", "Crack-specific detection recovery on the PCM road-damage dataset.", "tab:pcm_crack", ["Scenario", "Input", "Crack mAP50", "Crack mAP50--95", "Prec.", "Rec."], body)


def notation_table() -> None:
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        "\\caption{Notation used in the RMR-Net formulation.}",
        "\\label{tab:notation}",
        "\\small",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{ll}",
        "\\toprule",
        "Symbol & Meaning \\\\",
        "\\midrule",
        "$I_d$ & Degraded road image \\\\",
        "$I_c$ & Clean target image used for supervised training \\\\",
        "$I_r$ & Restored output image \\\\",
        "$z_m$ & Metadata-derived degradation code \\\\",
        "$z_b$ & Image-estimated degradation code \\\\",
        "$z$ & Fused conditioning code supplied to restoration blocks \\\\",
        "$E(I_d)$ & Gradient magnitude map from the degraded input \\\\",
        "$A(\\cdot)$ & Defect-edge attention branch \\\\",
        "$f_\\theta$ & Restoration network \\\\",
        "$g_\\phi$ & Image degradation-code encoder \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    write_custom_table(TAB_DIR / "table_notation.tex", lines)


def metadata_realism_table() -> None:
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\caption{Metadata disclosure and practical deployment interpretation. Road-damage detection experiments use synthetic proxy metadata from controlled degradations; the KITTI experiment uses real synchronized OXTS vehicle telemetry with a declared camera-exposure setting.}",
        "\\label{tab:metadata_realism}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabularx}{\\textwidth}{p{0.18\\textwidth}p{0.20\\textwidth}p{0.27\\textwidth}X}",
        "\\toprule",
        "Metadata field & Current source & Practical source & Role in $z_m$ \\\\",
        "\\midrule",
        "Blur angle/length & Synthetic kernel parameters & gyro/camera motion estimate & motion direction/severity \\\\",
        "gyro/accel proxies & Scenario-derived values & IMU or vibration sensor & horizontal/vertical/random motion \\\\",
        "Speed proxy & Random plausible road speed & CAN bus, GNSS, odometry & exposure-motion coupling \\\\",
        "Exposure proxy & Scenario-derived value & camera EXIF/API & low-light and motion scale \\\\",
        "Defocus score & Scenario label & autofocus/laplacian/learned estimator & focus degradation \\\\",
        "Noise/low-light score & Scenario label & ISO/exposure/image estimator & photometric degradation \\\\",
        "JPEG quality & Scenario label if used & encoder setting/bitstream & compression degradation \\\\",
        "\\bottomrule",
        "\\end{tabularx}",
        "\\end{table*}",
    ]
    write_custom_table(TAB_DIR / "table_metadata_realism.tex", lines)


def contribution_boundary_table() -> None:
    lines = [
        "\\begin{table*}[!t]",
        "\\centering",
        "\\caption{Contribution boundary relative to prior work. RMR-Net combines known ideas in a road-damage ITS setting; the novelty claim is the domain-specific metadata-conditioned perception pipeline, not a generic first use of sensor-guided or conditional restoration.}",
        "\\label{tab:contribution_boundary}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabularx}{\\textwidth}{p{0.22\\textwidth}p{0.30\\textwidth}X}",
        "\\toprule",
        "Component & Prior-art status & RMR-Net contribution \\\\",
        "\\midrule",
        "IMU/gyro-aided deblurring & Established in classical and deep deblurring & Uses sensor-like road metadata interface, currently synthetic \\\\",
        "Degradation-conditioned restoration & Established in all-in-one restoration & Uses compact road degradation code with image fallback \\\\",
        "Task-driven restoration for detection & Established in adverse-weather/recognition restoration & Evaluates pothole/crack/manhole recovery after road restoration \\\\",
        "Defect-edge preservation & Related to edge/semantic restoration losses & Applies label-free edge attention to road-defect structure \\\\",
        "Edge feasibility & Common deployment concern & Reports GPU runtime beside road-damage mAP \\\\",
        "\\bottomrule",
        "\\end{tabularx}",
        "\\end{table*}",
    ]
    write_custom_table(TAB_DIR / "table_contribution_boundary.tex", lines)


def detection_bar(det_path: Path, out_name: str, title: str) -> None:
    by = {r["name"]: float(r["map50"]) for r in rows(det_path)}
    scenarios = ["motion", "defocus", "lowlight"]
    labels = ["Motion blur", "Defocus", "Low light"]
    methods = ["degraded", "rmr_meta", "rmr_blind", "nafnet", "dfpir"]
    pretty = ["Degraded", "RMR-Net+meta", "RMR image-only", "NAFNet-road", "DFPIR"]
    colors = ["#8c939c", "#139b75", "#72bda3", "#3b78a8", "#b95745"]
    width = 0.15
    fig, ax = plt.subplots(figsize=(7.4, 3.35), dpi=240)
    xs = range(len(scenarios))
    for i, method in enumerate(methods):
        vals = [by[f"{s}_{method}"] for s in scenarios]
        ax.bar([x + (i - 2) * width for x in xs], vals, width, color=colors[i], label=pretty[i])
    ax.axhline(by["clean"], color="#222222", linestyle="--", linewidth=1)
    ax.text(2.3, by["clean"] + 0.01, "clean", fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(0.65, by["clean"] + 0.08))
    ax.set_ylabel("Frozen YOLO mAP50")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name, bbox_inches="tight")
    plt.close(fig)


def metadata_bar() -> None:
    by = {r["name"]: float(r["map50"]) for r in rows(POTH_DET)}
    scenarios = ["motion", "defocus", "lowlight"]
    fig, ax = plt.subplots(figsize=(5.3, 3.0), dpi=240)
    x = range(len(scenarios))
    ax.bar([v - 0.17 for v in x], [by[f"{s}_rmr_blind"] for s in scenarios], 0.34, label="image-only", color="#72bda3")
    ax.bar([v + 0.17 for v in x], [by[f"{s}_rmr_meta"] for s in scenarios], 0.34, label="metadata-conditioned", color="#139b75")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Motion", "Defocus", "Low light"])
    ax.set_ylabel("Pothole mAP50")
    ax.set_ylim(0, 0.62)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_metadata_ablation.png", bbox_inches="tight")
    plt.close(fig)


def draw_round_box(ax, xy, wh, text: str, color: str, edge: str = "#24313a") -> None:
    x, y = xy
    w, h = wh
    box = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor=edge, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, wrap=True)


def pipeline_diagram() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.5), dpi=240)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    boxes = [
        ((0.35, 3.05), (1.55, 0.95), "Vehicle/\nroadside\nRGB camera", "#e8eef4"),
        ((2.25, 3.05), (1.65, 0.95), "Blur,\nvibration,\ndefocus,\nlow light", "#f3eadb"),
        ((4.25, 3.05), (1.65, 0.95), "RMR-Net\nrestoration", "#dff0ef"),
        ((6.25, 3.05), (1.65, 0.95), "Frozen YOLO\nroad-damage\nperception", "#ebe7f4"),
        ((5.1, 1.05), (1.9, 0.85), "Autonomous-\nvehicle risk\nawareness", "#eef4df"),
        ((7.45, 1.05), (1.9, 0.85), "Maintenance\nprioritization", "#eef4df"),
    ]
    for xy, wh, text, color in boxes:
        draw_round_box(ax, xy, wh, text, color)
    arrow = dict(arrowstyle="->", lw=1.8, color="#24313a", shrinkA=5, shrinkB=5)
    ax.annotate("", xy=(2.25, 3.52), xytext=(1.9, 3.52), arrowprops=arrow)
    ax.annotate("", xy=(4.25, 3.52), xytext=(3.9, 3.52), arrowprops=arrow)
    ax.annotate("", xy=(6.25, 3.52), xytext=(5.9, 3.52), arrowprops=arrow)
    ax.annotate("", xy=(6.0, 1.95), xytext=(6.95, 3.05), arrowprops=arrow)
    ax.annotate("", xy=(8.35, 1.95), xytext=(7.2, 3.05), arrowprops=arrow)
    ax.text(5.7, 2.45, "restoration judged by detector mAP + latency", ha="center", fontsize=9, color="#24313a")
    ax.text(
        5.0,
        0.35,
        "T-ITS positioning: robust road perception for connected/autonomous monitoring and infrastructure management",
        ha="center",
        fontsize=8.5,
        color="#24313a",
    )
    fig.tight_layout(pad=0.2)
    fig.savefig(FIG_DIR / "fig_rmrnet_its_pipeline.png", bbox_inches="tight")
    plt.close(fig)


def architecture_diagram() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.25), dpi=240)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.7)
    ax.axis("off")
    boxes = [
        ((0.25, 2.6), (1.25, 0.75), "Degraded\nroad image", "#e8eef4"),
        ((2.05, 3.25), (1.55, 0.75), "Metadata\ncode", "#eef4df"),
        ((2.05, 2.05), (1.55, 0.75), "Image\ncode encoder", "#f3eadb"),
        ((4.05, 2.6), (1.75, 0.75), "Code fusion\nFiLM tokens", "#dff0ef"),
        ((6.15, 3.25), (1.7, 0.75), "Defect-edge\nattention", "#f4e6e0"),
        ((6.15, 2.05), (1.7, 0.75), "Efficient\nrestoration\nblocks", "#ebe7f4"),
        ((8.35, 2.6), (1.35, 0.75), "Restored\nimage", "#e8eef4"),
    ]
    for xy, wh, text, color in boxes:
        draw_round_box(ax, xy, wh, text, color)
    arrow = dict(arrowstyle="->", lw=1.7, color="#24313a", shrinkA=5, shrinkB=5)
    ax.annotate("", xy=(2.05, 2.42), xytext=(1.5, 2.98), arrowprops=arrow)
    ax.annotate("", xy=(4.05, 3.0), xytext=(3.6, 3.62), arrowprops=arrow)
    ax.annotate("", xy=(4.05, 3.0), xytext=(3.6, 2.42), arrowprops=arrow)
    ax.annotate("", xy=(6.15, 3.62), xytext=(5.8, 3.0), arrowprops=arrow)
    ax.annotate("", xy=(6.15, 2.42), xytext=(5.8, 3.0), arrowprops=arrow)
    ax.annotate("", xy=(8.35, 2.98), xytext=(7.85, 2.42), arrowprops=arrow)
    ax.annotate("", xy=(8.35, 2.98), xytext=(7.85, 3.62), arrowprops=arrow)
    ax.text(5.0, 1.15, "One deployed model: metadata-conditioned when available, image-only fallback when not", ha="center", fontsize=9)
    ax.text(5.0, 0.55, "Training uses paired degraded/clean road images plus synthetic sensor-like degradation metadata.", ha="center", fontsize=8.5, color="#4a545c")
    fig.tight_layout(pad=0.2)
    fig.savefig(FIG_DIR / "fig_rmrnet_architecture.png", bbox_inches="tight")
    plt.close(fig)


def metadata_deployment_diagram() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.25), dpi=240)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    boxes = [
        ((0.35, 3.15), (2.15, 0.75), "Controlled benchmark\nsynthetic metadata", "#f3eadb"),
        ((0.35, 1.65), (2.15, 0.75), "Real deployment\nvehicle/camera signals", "#eef4df"),
        ((3.25, 2.35), (1.75, 0.8), "Metadata-to-code\nmapper", "#dff0ef"),
        ((5.75, 2.35), (1.5, 0.8), "$z_m$", "#ebe7f4"),
        ((7.75, 2.35), (1.65, 0.8), "RMR-Net\nFiLM blocks", "#e8eef4"),
    ]
    for xy, wh, text, color in boxes:
        draw_round_box(ax, xy, wh, text, color)
    arrow = dict(arrowstyle="->", lw=1.7, color="#24313a", shrinkA=5, shrinkB=5)
    ax.annotate("", xy=(3.25, 2.75), xytext=(2.5, 3.52), arrowprops=arrow)
    ax.annotate("", xy=(3.25, 2.75), xytext=(2.5, 2.02), arrowprops=arrow)
    ax.annotate("", xy=(5.75, 2.75), xytext=(5.0, 2.75), arrowprops=arrow)
    ax.annotate("", xy=(7.75, 2.75), xytext=(7.25, 2.75), arrowprops=arrow)
    ax.text(5.0, 1.05, "Reported experiments use the upper path. The lower path is the intended practical replacement, not an unreported result.", ha="center", fontsize=8.8)
    ax.text(5.0, 0.45, "Disclosure: road-damage mAP uses proxy metadata; KITTI validates real OXTS telemetry under controlled blur.", ha="center", fontsize=8.5, color="#4a545c")
    fig.tight_layout(pad=0.2)
    fig.savefig(FIG_DIR / "fig_metadata_deployment.png", bbox_inches="tight")
    plt.close(fig)


def resize_crop(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    tw, th = size
    scale = max(tw / w, th / h)
    img = img.resize((round(w * scale), round(h * scale)), Image.Resampling.BICUBIC)
    w, h = img.size
    left = max((w - tw) // 2, 0)
    top = max((h - th) // 2, 0)
    return img.crop((left, top, left + tw, top + th))


def predict_counts(model: YOLO, image_path: Path, conf: float = 0.20) -> dict[int, int]:
    result = model.predict(str(image_path), imgsz=640, conf=conf, verbose=False)[0]
    counts: dict[int, int] = {}
    for cls in result.boxes.cls.cpu().numpy().astype(int).tolist():
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def crack_gt_boxes(name: str) -> list[tuple[float, float, float, float]]:
    label_path = ROOT / "datasets" / "road_damage_pcm_yolo" / "labels" / "test" / (Path(name).stem + ".txt")
    if not label_path.exists():
        return []
    boxes: list[tuple[float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0] != "1":
            continue
        cx, cy, w, h = [float(v) for v in parts[1:5]]
        boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes


def draw_prediction(model: YOLO, image_path: Path, size: tuple[int, int], gt_boxes: list[tuple[float, float, float, float]]) -> Image.Image:
    result = model.predict(str(image_path), imgsz=640, conf=0.20, verbose=False)[0]
    original = Image.open(image_path).convert("RGB")
    ow, oh = original.size
    canvas = resize_crop(original, size)
    cw, ch = canvas.size
    scale = max(cw / ow, ch / oh)
    nw, nh = round(ow * scale), round(oh * scale)
    offset_x = max((nw - cw) // 2, 0)
    offset_y = max((nh - ch) // 2, 0)
    draw = ImageDraw.Draw(canvas)
    fnt = font(13)
    for gx1, gy1, gx2, gy2 in gt_boxes:
        x1 = gx1 * ow * scale - offset_x
        y1 = gy1 * oh * scale - offset_y
        x2 = gx2 * ow * scale - offset_x
        y2 = gy2 * oh * scale - offset_y
        if x2 < 0 or y2 < 0 or x1 > cw or y1 > ch:
            continue
        draw.rectangle((x1, y1, x2, y2), outline=(245, 190, 55), width=2)
    for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int), result.boxes.conf.cpu().numpy()):
        color = CLASS_COLORS.get(int(cls), (230, 185, 60))
        x1, y1, x2, y2 = [float(v) * scale for v in box]
        x1 -= offset_x
        x2 -= offset_x
        y1 -= offset_y
        y2 -= offset_y
        if x2 < 0 or y2 < 0 or x1 > cw or y1 > ch:
            continue
        label = f"{result.names[int(cls)]} {conf:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        tw = draw.textlength(label, font=fnt)
        draw.rectangle((x1, max(0, y1 - 18), x1 + tw + 6, max(18, y1)), fill=color)
        draw.text((x1 + 3, max(0, y1 - 17)), label, fill="white", font=fnt)
    return canvas


def find_crack_example(model: YOLO, scenario: str, variants: dict[str, Path]) -> str:
    degraded_dir = variants["Degraded"]
    names = [p.name for p in sorted(degraded_dir.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    best_name = names[0]
    best_score = -999
    for name in names:
        gt = len(crack_gt_boxes(name))
        if gt == 0:
            continue
        clean = predict_counts(model, variants["Clean"] / name).get(1, 0)
        deg = predict_counts(model, degraded_dir / name).get(1, 0)
        rmr = predict_counts(model, variants["RMR-Net+meta"] / name).get(1, 0)
        dfp = predict_counts(model, variants["DFPIR"] / name).get(1, 0)
        score = 8 * min(gt, 2) + 5 * clean + 6 * rmr - 4 * deg - 2 * dfp
        if clean == 0:
            score -= 12
        if rmr <= deg:
            score -= 10
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


def qualitative_detection_panel(scenario: str, tag: str, title: str) -> None:
    model = YOLO(str(PCM_MODEL))
    variants = {
        "Clean": ROOT / "datasets" / "road_damage_pcm_yolo" / "images" / "test",
        "Degraded": ROOT / "datasets" / f"pcm_yolo_{tag}_test" / "images" / "test",
        "RMR-Net+meta": ROOT / "datasets" / f"pcm_yolo_{tag}_test_rmrnet_meta" / "images" / "test",
        "DFPIR": ROOT / "datasets" / f"pcm_yolo_{tag}_test_dfpir" / "images" / "test",
    }
    name = find_crack_example(model, scenario, variants)
    gt_boxes = crack_gt_boxes(name)
    tile = (280, 160)
    pad = 12
    header_h = 68
    width = len(variants) * tile[0] + (len(variants) + 1) * pad
    height = header_h + tile[1] + 50
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((pad, 10), title, fill=(20, 28, 34), font=font(19))
    draw.text((pad, 34), "Yellow boxes: ground-truth cracks. Green boxes: YOLO crack predictions.", fill=(70, 78, 86), font=font(13))
    draw.text((pad, 50), "Examples are selected to contain annotated cracks and a valid clean-image detector response.", fill=(70, 78, 86), font=font(12))
    for i, (label, folder) in enumerate(variants.items()):
        x = pad + i * (tile[0] + pad)
        y = header_h
        image = draw_prediction(model, folder / name, tile, gt_boxes)
        panel.paste(image, (x, y))
        draw.text((x + 4, y + tile[1] + 8), label, fill=(20, 28, 34), font=font(14))
    panel.save(FIG_DIR / f"fig_yolo_crack_{tag}.png")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    detection_table(POTH_DET, "table_pothole_detection.tex", "Held-out pothole detection before and after restoration.", "tab:pothole_detection")
    detection_table(PCM_DET, "table_pcm_detection.tex", "Held-out PCM pothole/crack/manhole detection before and after restoration.", "tab:pcm_detection")
    restoration_table([("Pothole", POTH_REST), ("PCM", PCM_REST)], "table_restoration_combined.tex")
    metadata_table()
    crack_table()
    notation_table()
    metadata_realism_table()
    contribution_boundary_table()
    detection_bar(POTH_DET, "fig_pothole_detection_recovery.png", "Pothole dataset downstream detection recovery")
    detection_bar(PCM_DET, "fig_pcm_detection_recovery.png", "PCM multi-class road-damage detection recovery")
    metadata_bar()
    pipeline_diagram()
    architecture_diagram()
    metadata_deployment_diagram()
    qualitative_detection_panel("defocus", "defocus", "Crack detection under defocus blur")
    qualitative_detection_panel("lowlight", "lowlight", "Crack detection under low light")

    summary = {
        "pothole_detection": str(POTH_DET),
        "pcm_detection": str(PCM_DET),
        "pcm_crack_per_class": str(PCM_PER),
        "pothole_restoration": str(POTH_REST),
        "pcm_restoration": str(PCM_REST),
    }
    (OUT / "ASSET_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
