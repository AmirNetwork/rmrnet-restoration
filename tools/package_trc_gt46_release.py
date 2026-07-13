from __future__ import annotations

"""Package the current TRC manuscript, code, and compact GT46 evidence.

This package is intentionally lean.  Native-resolution GT46 images and all
restored full-resolution method folders are several gigabytes, so the evidence
zip stores the reproducible metrics, detector CSV/GeoJSON outputs, paper
figures, and provenance instead of duplicating every image.
"""

import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAMP = "20260709_gt46_native_gate"

PAPER_DIR = ROOT / "paper_trc_rmrnet"
GT46_RUN = "yolo26rev_imgsz1280_conf010_clsD00-015_D10-025_D20-025_gamma085"
GT46_ROOT = ROOT / "experiments" / "gt46_yolo26_coordinate_revised"
GT46_EVAL = GT46_ROOT / f"evaluation_all_methods_{GT46_RUN}"
GT46_DET = GT46_ROOT / f"method_detections_{GT46_RUN}"
GT46_POLICY = GT46_ROOT / "native_evidence_policy_sweep_yolo26_coordinate"
GATE_OUTPUT = ROOT / "experiments" / "roboflow_geotagged_v5_native_real" / "v45_native_gate_gamma085_eval_sets"

PAPER_ZIP = ROOT / f"rmrnet_trc_paper_source_{STAMP}.zip"
CODE_ZIP = ROOT / f"rmrnet_trc_code_release_{STAMP}.zip"
EVIDENCE_ZIP = ROOT / f"rmrnet_trc_gt46_evidence_{STAMP}.zip"
STAGE = ROOT / "_trc_release_stage_gt46"


CODE_DIRS = [
    "models",
    "rcadnet",
    "losses",
    "baselines",
    "configs",
    "tools",
    "Yolo26_coordinate",
]

ROOT_FILES = [
    "README.md",
    "requirements-windows-gpu.txt",
    "requirements-detection-extra.txt",
    "requirements-dfpir-extra.txt",
    "requirements-demoe-extra.txt",
    "train_rcadnet.py",
    "train_rmrnet.py",
    "train_road_baseline.py",
    "benchmark_adapter_rcadnet.py",
    "benchmark_unified_restoration.py",
]


def should_skip(path: Path) -> bool:
    if "__pycache__" in path.parts:
        return True
    if path.suffix.lower() in {".pyc", ".pyo", ".zip", ".pdf"}:
        return True
    return False


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, *, suffixes: set[str] | None = None) -> None:
    if not src.exists():
        return
    for item in src.rglob("*"):
        if item.is_dir() or should_skip(item):
            continue
        if suffixes is not None and item.suffix.lower() not in suffixes:
            continue
        copy_file(item, dst / item.relative_to(src))


def zip_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for item in sorted(src.rglob("*")):
            if item.is_dir() or should_skip(item):
                continue
            archive.write(item, item.relative_to(src.parent))


def package_paper() -> None:
    zip_dir(PAPER_DIR, PAPER_ZIP)


def package_code() -> None:
    code_stage = STAGE / "rmrnet_trc_code_release"
    if code_stage.exists():
        shutil.rmtree(code_stage)
    code_stage.mkdir(parents=True)

    for rel in ROOT_FILES:
        copy_file(ROOT / rel, code_stage / rel)
    for rel in CODE_DIRS:
        copy_tree(ROOT / rel, code_stage / rel)

    readme = code_stage / "RUN_GT46_NATIVE_FIELD.md"
    readme.write_text(
        "# GT46 Native-Field Reproduction\n\n"
        "Activate the Windows GPU environment from the project root:\n\n"
        "```powershell\n"
        ".\\.venv\\Scripts\\activate\n"
        "```\n\n"
        "Regenerate the conservative native-gate images:\n\n"
        "```powershell\n"
        "python tools\\restore_native_yolo_split.py --data experiments\\roboflow_geotagged_v5_native_real\\native_real_yolo_newroad6\\data.yaml --split test --scenario native_real --out experiments\\roboflow_geotagged_v5_native_real\\v45_native_gate_gamma085_eval_sets --models rmr_metadata_gated --device cuda --tile 768 --overlap 96 --jpeg-quality 95 --rcadnet-weights runs\\gt46_detection_heavy_yolo26_olddata_3ep\\rcadnet_epoch_003.pth\n"
        "```\n\n"
        "Run the revised coordinate-aware YOLO26 detector/evaluator for all reported GT46 methods:\n\n"
        "```powershell\n"
        "python tools\\run_gt46_yolo26_coordinate_all_methods.py --detector-script Yolo26_coordinate\\Yolo26_coordinate_revised.py --device 0 --imgsz 1280 --conf 0.10 --class-conf \"D00=0.15,D10=0.25,D20=0.25\" --workers 1 --atlas-max 12 --tag yolo26rev_imgsz1280_conf010_clsD00-015_D10-025_D20-025_gamma085 --force-detect\n"
        "python tools\\make_gt46_coordinate_paper_assets.py\n"
        "```\n\n"
        "GT46 labels are used only by `tools/eval_yolo26_coordinate_gt46.py` for final scoring; they are not used to train RMR-Net or YOLO26.\n",
        encoding="utf-8",
    )
    zip_dir(code_stage, CODE_ZIP)


def package_evidence() -> None:
    evidence_stage = STAGE / "rmrnet_trc_gt46_evidence"
    if evidence_stage.exists():
        shutil.rmtree(evidence_stage)
    evidence_stage.mkdir(parents=True)

    copy_tree(PAPER_DIR / "tables", evidence_stage / "paper_trc_rmrnet" / "tables")
    copy_tree(PAPER_DIR / "figures", evidence_stage / "paper_trc_rmrnet" / "figures")
    copy_file(PAPER_DIR / "RESULT_PROVENANCE_TABLE.csv", evidence_stage / "paper_trc_rmrnet" / "RESULT_PROVENANCE_TABLE.csv")
    copy_file(PAPER_DIR / "RESPONSE_TO_REVIEWER.md", evidence_stage / "paper_trc_rmrnet" / "RESPONSE_TO_REVIEWER.md")

    for name in [
        "summary_metrics.csv",
        "class_metrics_iou10.csv",
        "README_RESULTS.md",
        "manifest.json",
        "table_geotagged_native_pilot.tex",
        "fig_gt46_coordinate_prf.png",
        "fig_gt46_coordinate_examples.jpg",
        "gt46_yolo26_coordinate_atlas.jpg",
    ]:
        copy_file(GT46_EVAL / name, evidence_stage / "experiments" / "gt46_yolo26_coordinate_revised" / GT46_EVAL.name / name)

    # Detector CSV/GeoJSON outputs are compact and sufficient to audit metrics.
    for method_dir in sorted(GT46_DET.iterdir() if GT46_DET.exists() else []):
        if not method_dir.is_dir():
            continue
        out_dir = evidence_stage / "experiments" / "gt46_yolo26_coordinate_revised" / GT46_DET.name / method_dir.name
        for suffix in {".csv", ".geojson", ".json", ".txt"}:
            copy_tree(method_dir, out_dir, suffixes={suffix})

    for name in ["policy_sweep_summary.csv", "fig_gt46_native_gate_sweep.png"]:
        copy_file(GT46_POLICY / name, evidence_stage / "experiments" / "gt46_yolo26_coordinate_revised" / GT46_POLICY.name / name)

    for name in ["restore_summary.csv"]:
        copy_file(GATE_OUTPUT / name, evidence_stage / "experiments" / "roboflow_geotagged_v5_native_real" / GATE_OUTPUT.name / name)

    (evidence_stage / "README_EVIDENCE.md").write_text(
        "# Compact GT46 Evidence Package\n\n"
        "This zip contains the current TRC paper tables/figures plus the exact GT46 coordinate-detector CSV/GeoJSON outputs used to regenerate Table `geotagged_native_pilot`. "
        "Full native-resolution images and restored image folders are intentionally not duplicated because they are multi-gigabyte local data. "
        "The run uses `Yolo26_coordinate_revised.py`, frozen `YOLO26s_RDD_FRDC_Distilled_v2.pt`, `imgsz=1280`, `conf=0.10`, and class thresholds `D00=0.15,D10=0.25,D20=0.25`.\n",
        encoding="utf-8",
    )
    zip_dir(evidence_stage, EVIDENCE_ZIP)


def main() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    package_paper()
    package_code()
    package_evidence()
    for path in [PAPER_ZIP, CODE_ZIP, EVIDENCE_ZIP]:
        print(f"{path}\t{path.stat().st_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
