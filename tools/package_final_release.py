"""Package the final RMR-Net paper, evidence, and code artifacts.

The script deliberately creates clean release folders instead of zipping the
entire working paper directory, because the working directory contains old
scratch notes from earlier audits. The manuscript remains the source of truth:
only figures and tables referenced by ``manuscript.tex`` are copied into the
paper package.
"""

from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_ieee_tits_rmrnet"
EXPERIMENT = ROOT / "experiments" / "roboflow_geotagged_v5_native_real" / "final_all49"
RELEASE = ROOT / "release_final_20260622"


PAPER_DOCS = [
    "manuscript.tex",
    "references.bib",
    "IEEEtran.cls",
    "IEEEtran.bst",
    "README_SUBMISSION.md",
    "SOURCE_LINKS.md",
    "SUBMISSION_CHECKLIST.md",
    "GEOTAGGED_V5_NATIVE_REAL_NOTES.md",
    "RESULT_PROVENANCE_TABLE.csv",
    "SUPPLEMENTARY_README.md",
    "cover_letter.md",
]

CODE_PATHS = [
    "models",
    "rcadnet",
    "losses",
    "baselines",
    "configs",
    "tools",
    "train_rcadnet.py",
    "train_rmrnet.py",
    "train_road_baseline.py",
    "benchmark_unified_restoration.py",
    "benchmark_adapter_rcadnet.py",
    "infer_rcadnet.py",
    "requirements-windows-gpu.txt",
    "requirements-demoe-extra.txt",
    "requirements-detection-extra.txt",
    "requirements-dfpir-extra.txt",
    "README.md",
    "EXPERIMENT_PROTOCOL.md",
    "REPRODUCIBILITY_ARTIFACTS_README.md",
    "DEMOE_INTEGRATION.md",
]

EVIDENCE_FILES = [
    "final_native_field_manifest.json",
    "paper_metric_summary_all49.csv",
    "geotagged_eta_sweep.csv",
    "geotagged_tau_sweep.csv",
]

PROVENANCE_FILES = [
    "configs/rmrnet_headline.yaml",
    "experiments/revised_loss_ablation/ablation_protocol.json",
    "experiments/revised_loss_ablation/ablation_metrics.csv",
    "experiments/revised_loss_ablation/selection_summary.csv",
    "experiments/revised_loss_ablation/best_by_val_map.json",
    "experiments/v27_taskloss_yolo11s_eval/V27_TASKLOSS_SELECTION_MANIFEST.json",
    "experiments/v27_taskloss_yolo11s_eval/pothole_val_rmrnet_v27_epochs.csv",
    "experiments/v27_taskloss_yolo11s_eval/pcm_val_rmrnet_v27_epochs.csv",
    "experiments/v27_taskloss_yolo11s_eval/pothole_test_yolo11s_baselines_taskloss_v27.csv",
    "experiments/v27_taskloss_yolo11s_eval/pcm_test_yolo11s_baselines_taskloss_v27.csv",
]

RUN_PROVENANCE_DIRS = [
    "runs/revised_loss_ablation/full_model",
    "runs/revised_loss_ablation/jacobian",
    "runs/revised_loss_ablation/tdp_cqmix",
    "runs/revised_loss_ablation/tdp",
    "runs/revised_loss_ablation/detail_skip",
    "runs/revised_loss_ablation/attention",
    "runs/revised_loss_ablation/code_supervision",
    "runs/revised_loss_ablation/base_only",
]


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree_filtered(src: Path, dst: Path) -> None:
    ignore = shutil.ignore_patterns(
        "__pycache__",
        "*.pyc",
        ".pytest_cache",
        "*.zip",
        "*.pth",
        "*.pt",
        "*.pth.tar",
        "*.onnx",
        "*.engine",
        # Retired internal manuscript builders are intentionally kept in the
        # working tree for provenance but excluded from the reviewer-facing
        # code bundle so the release has one clean paper story.
        "build_paper_assets.py",
        "build_ieee_tits_assets.py",
        "build_native_blur_assets.py",
        "build_v*.py",
        "build_rmrnet_allinone_giant.py",
        "build_rmrnet_readable_monolith.py",
    )
    shutil.copytree(src, dst, ignore=ignore)


def referenced_assets() -> tuple[set[str], set[str]]:
    manuscript = (PAPER / "manuscript.tex").read_text(encoding="utf-8")
    tables = set(re.findall(r"\\input\{tables/([^}]+)\}", manuscript))
    figures = set(re.findall(r"\\includegraphics(?:\[[^\]]+\])?\{figures/([^}]+)\}", manuscript))
    return tables, figures


def write_release_manifest(path: Path, tables: set[str], figures: set[str]) -> None:
    metrics_csv = EXPERIMENT / "paper_metric_summary_all49.csv"
    eta_csv = EXPERIMENT / "geotagged_eta_sweep.csv"
    tau_csv = EXPERIMENT / "geotagged_tau_sweep.csv"
    rows = []
    if metrics_csv.exists():
        with metrics_csv.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    best = max(rows, key=lambda r: float(r.get("f1_iou10", 0.0)), default={})
    text = [
        "# Final RMR-Net Release Notes",
        "",
        "This release package is generated from the current manuscript source and the all-49 Sony native-image field-test run.",
        "",
        "## Key synced evidence",
        "",
        "- The manuscript frames the native-real experiment as a high-quality Sony native-image safety test, not as a severe natural-blur benchmark.",
        "- The geotagged field test uses all 49 matched annotated cam1 images at native 4752 x 3168 resolution.",
        "- Roboflow annotations are mapped back to the original images; Roboflow-resized pixels are not used for restoration.",
        "- Real EXIF and pose/geotag metadata are joined where available.",
        "- Residual-strength eta and gate-threshold tau sweeps are included for deployment policy analysis.",
        "- Active contours are a post-detection measurement stage; active-contour loss is not part of the final training objective.",
        "- The final configuration in `configs/rmrnet_headline.yaml` sets active-contour training loss to zero.",
        "- No-active-contour ablation provenance is included under `provenance/`.",
        "",
        "## All-49 field-test headline",
        "",
    ]
    if best:
        text.append(
            f"- Best relaxed grouped F1@IoU0.10 in the all-49 table: {best.get('method')} "
            f"with F1={float(best.get('f1_iou10', 0.0)):.3f}, "
            f"precision={float(best.get('precision_iou10', 0.0)):.3f}, "
            f"recall={float(best.get('recall_iou10', 0.0)):.3f}, "
            f"FP/image={float(best.get('false_pos_per_image_conf25', 0.0)):.2f}."
        )
    text.extend(
        [
            "",
            "## Files copied",
            "",
            f"- Referenced tables: {len(tables)}",
            f"- Referenced figures: {len(figures)}",
            f"- Evidence CSVs: {', '.join(p for p in EVIDENCE_FILES if (EXPERIMENT / p).exists())}",
            f"- Eta sweep CSV: {eta_csv.relative_to(ROOT) if eta_csv.exists() else 'missing'}",
            f"- Tau sweep CSV: {tau_csv.relative_to(ROOT) if tau_csv.exists() else 'missing'}",
            "",
            "## Build note",
            "",
            "A TeX installation was not available in this local environment. Compile `manuscript.tex` in Overleaf or with `pdflatex`/`bibtex` locally.",
            "",
        ]
    )
    path.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    paper_out = RELEASE / "paper_source"
    evidence_out = RELEASE / "evidence_summary"
    code_out = RELEASE / "code_release"
    reset_dir(RELEASE)
    paper_out.mkdir(parents=True)
    evidence_out.mkdir(parents=True)
    code_out.mkdir(parents=True)

    tables, figures = referenced_assets()

    for name in PAPER_DOCS:
        src = PAPER / name
        if src.exists():
            copy_file(src, paper_out / name)

    for table in sorted(tables):
        src = PAPER / "tables" / f"{table}.tex"
        if not src.exists():
            raise FileNotFoundError(src)
        copy_file(src, paper_out / "tables" / src.name)

    for fig in sorted(figures):
        src = PAPER / "figures" / fig
        if not src.exists():
            raise FileNotFoundError(src)
        copy_file(src, paper_out / "figures" / src.name)

    write_release_manifest(paper_out / "FINAL_RELEASE_NOTES.md", tables, figures)

    for name in EVIDENCE_FILES:
        src = EXPERIMENT / name
        if src.exists():
            copy_file(src, evidence_out / name)
    for subdir in ["figures", "sharpness_audit", "native_tiled_eval", "native_tiled_eval_overlay"]:
        src = EXPERIMENT / subdir
        if src.exists():
            shutil.copytree(src, evidence_out / subdir)
    if (PAPER / "tables").exists():
        shutil.copytree(PAPER / "tables", evidence_out / "paper_tables")
    if (PAPER / "figures").exists():
        selected = evidence_out / "paper_figures_selected"
        selected.mkdir(parents=True, exist_ok=True)
        for fig in sorted(figures):
            copy_file(PAPER / "figures" / fig, selected / fig)

    provenance = evidence_out / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    for rel in PROVENANCE_FILES:
        src = ROOT / rel
        if src.exists():
            copy_file(src, provenance / rel)
    for rel in RUN_PROVENANCE_DIRS:
        src = ROOT / rel
        if not src.exists():
            continue
        dst = provenance / rel
        dst.mkdir(parents=True, exist_ok=True)
        for name in ["audit_config.json", "history.json", "best_by_val_map.json", "selection_summary.csv"]:
            item = src / name
            if item.exists():
                copy_file(item, dst / name)

    for rel in CODE_PATHS:
        src = ROOT / rel
        if not src.exists():
            continue
        dst = code_out / rel
        if src.is_dir():
            copy_tree_filtered(src, dst)
        else:
            copy_file(src, dst)
    write_release_manifest(code_out / "FINAL_RELEASE_NOTES.md", tables, figures)

    print(f"paper_source={paper_out}")
    print(f"evidence_summary={evidence_out}")
    print(f"code_release={code_out}")


if __name__ == "__main__":
    main()
