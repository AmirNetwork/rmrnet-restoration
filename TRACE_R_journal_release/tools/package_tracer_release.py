#!/usr/bin/env python3
"""Create and verify the final TRACE-R paper and code archives.

Only the accepted single-output implementation and the final controlled,
metadata-intervention, and CRID provenance
are packaged. Legacy fusion, active-contour, KITTI, and abandoned development
artifacts are intentionally excluded.

Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_ieee_tits_trace_r"
RELEASE_DOCS = ROOT / "release" / "trace_r"
COVER_LETTER = ROOT / "output" / "docx" / "TRACE_R_cover_letter_IEEE_TITS_20260831.docx"
TRANSITION_EMAIL = ROOT / "output" / "TRACE_R_submission_transition_email_20260831.txt"
CONTROLLED = Path(r"E:\TRACE_R_experiments\trace_locked_confirmatory_v72_20260828")
CONTROLS = Path(r"E:\TRACE_R_experiments\trace_metadata_controls_v66_20260828")
CRID = Path(r"E:\TRACE_R_experiments\crid320_sealed_test_20260831")
CRID_ANNOTATIONS = ROOT / "datasets" / "crid320_annotation_20260829"
CRID_DETECTOR = Path(r"E:\TRACE_R_experiments\crid320_detector_v1_20260829")
CRID_ADAPTATION = Path(r"E:\TRACE_R_experiments\crid320_matched_restorers_20260829")
CRID_DETAIL = Path(r"E:\TRACE_R_experiments\crid320_native_detail_sweep_20260830")
CRID_RAW_CONTROL = Path(r"E:\TRACE_R_experiments\crid320_native_raw_filter_control_20260831")
CRID_TRACE_VALIDATION = Path(r"E:\TRACE_R_experiments\crid320_trace_field_policy_v8_validation_20260831")
CRID_POLICY_IDENTITY = ROOT / "experiments" / "crid320_field_policy_identity_20260831.json"
TRAIN = Path(r"E:\TRACE_R_experiments\matched_budget_trace_v53_20260827")
NAFNET_RUN = Path(r"E:\TRACE_R_experiments\official_nafnet_matched_v68_20260828")

MODEL_FILES = (
    "models/__init__.py",
    "models/rmrnet.py",
    "models/rmrp_metadata_demoe.py",
    "models/rmrp_prompted_dfpir.py",
    "models/tracer_sensor_adapter.py",
    "models/tracer_sparse_wavelet.py",
)
RCAD_FILES = (
    "rcadnet/__init__.py",
    "rcadnet/dataset.py",
    "rcadnet/losses.py",
    "rcadnet/model.py",
    "rcadnet/physics_prior.py",
    "rcadnet/practical_metadata.py",
    "rcadnet/scenario_codes.py",
    "rcadnet/sensor_geometry.py",
    "rcadnet/spatial_physics.py",
    "rcadnet/synthetic_metadata.py",
    "rcadnet/task_losses.py",
)
BASELINE_FILES = (
    "baselines/__init__.py",
    "baselines/demoe_adapter.py",
    "baselines/dfpir_adapter.py",
    "baselines/instructir_adapter.py",
    "baselines/nafnet_metadata.py",
    "baselines/nafnet_road.py",
)
TOOL_FILES = (
    "tools/audit_practical_metadata_benchmark.py",
    "tools/audit_nafnet_checkpoint_stability.py",
    "tools/build_practical_metadata_benchmark.py",
    "tools/build_sony_collection_figure.py",
    "tools/build_tracer_journal_assets.py",
    "tools/build_crid320_annotation_pool.py",
    "tools/build_crid320_paper_assets.py",
    "tools/build_crid320_staged_trace_init.py",
    "tools/create_trace_r_cover_letter.py",
    "tools/evaluate_crid320_validation.py",
    "tools/export_crid320_field_policy_identity.py",
    "tools/freeze_crid320_annotations.py",
    "tools/freeze_crid320_trace_field_policy.py",
    "tools/generate_crid320_yolo26_proposals.py",
    "tools/eval_yolo_suite.py",
    "tools/prepare_crid_ins_all_metadata.py",
    "tools/restore_native_yolo_split.py",
    "tools/restore_yolo_split.py",
    "tools/run_crid320_matched_adaptation.ps1",
    "tools/run_crid320_sealed_test.py",
    "tools/run_crid320_validation_sweep.ps1",
    "tools/run_official_nafnet_correction.py",
    "tools/run_tracer_locked_confirmatory.py",
    "tools/run_tracer_metadata_controls.py",
    "tools/sweep_crid320_native_detail_validation.py",
    "tools/train_crid320_detector.py",
    "tools/train_crid320_restorer.py",
    "tools/validate_matched_restorer_suite.py",
    "tools/package_tracer_release.py",
    "tools/refresh_trace_r_asset_manifest.py",
)
TEST_FILES = (
    "tests/test_dfpir_official.py",
    "tests/test_losses.py",
    "tests/test_matched_detector_objective.py",
    "tests/test_nafnet_official.py",
    "tests/test_native_tile_blending.py",
    "tests/test_practical_metadata.py",
    "tests/test_rmrp_metadata_demoe.py",
    "tests/test_trace_state_loss.py",
    "tests/test_tracer_sensor_adapter.py",
)
PAPER_FILES = (
    "manuscript.tex",
    "manuscript.pdf",
    "references.bib",
    "IEEEtran.cls",
    "IEEEtran.bst",
    "README.md",
    "asset_manifest.json",
    "tables/table_controlled_summary.tex",
    "tables/table_inference_inputs.tex",
    "tables/table_condition_results.tex",
    "tables/table_fidelity.tex",
    "tables/table_metadata_controls.tex",
    "tables/table_crid.tex",
    "tables/table_architecture_audit.tex",
    "tables/table_crid_adaptation_audit.tex",
    "figures/fig_trace_architecture.pdf",
    "figures/fig_trace_controlled_results.pdf",
    "figures/fig_trace_crid_ap.pdf",
    "figures/fig_trace_ivcnz_qualitative.pdf",
    "figures/fig_trace_pcm_qualitative.pdf",
    "figures/fig_trace_crid_qualitative.pdf",
    "figures/fig_trace_crid_collection.pdf",
    "figures/qualitative_selection_manifest.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "output" / "releases" / "trace_r_final_20260831",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reset_directory(path: Path) -> None:
    resolved = path.resolve()
    allowed = (ROOT / "output" / "releases").resolve()
    if allowed not in resolved.parents:
        raise RuntimeError(f"Refusing to replace release directory outside {allowed}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def copy_one(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_manifest(root: Path, name: str) -> Path:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    manifest = root / name
    lines = [f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in files]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def archive_tree(root: Path, destination: Path) -> None:
    if destination.exists():
        destination.unlink()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, Path(root.name) / path.relative_to(root))


def stamp_python_authors(root: Path) -> None:
    """Add the release author to copied Python sources without changing logic."""
    author = "# Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>\n"
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "Amir Ghorbani" in text:
            continue
        lines = text.splitlines(keepends=True)
        index = 1 if lines and lines[0].startswith("#!") else 0
        lines.insert(index, author)
        path.write_text("".join(lines), encoding="utf-8")


def extract_and_verify(archive_path: Path, expected_root: Path, output: Path) -> dict[str, object]:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        archive.testzip()
        archive.extractall(output)
    extracted_root = output / expected_root.name
    expected = {
        path.relative_to(expected_root).as_posix(): sha256(path)
        for path in expected_root.rglob("*")
        if path.is_file()
    }
    actual = {
        path.relative_to(extracted_root).as_posix(): sha256(path)
        for path in extracted_root.rglob("*")
        if path.is_file()
    }
    return {
        "archive": str(archive_path),
        "sha256": sha256(archive_path),
        "size_bytes": archive_path.stat().st_size,
        "file_count": len(actual),
        "exact_match": actual == expected,
        "missing": sorted(set(expected) - set(actual)),
        "unexpected": sorted(set(actual) - set(expected)),
        "hash_mismatches": sorted(
            key for key in set(expected) & set(actual) if expected[key] != actual[key]
        ),
    }


def build_paper(stage: Path) -> None:
    for relative in PAPER_FILES:
        copy_one(PAPER / relative, stage / relative)
    copy_one(COVER_LETTER, stage / COVER_LETTER.name)
    copy_one(TRANSITION_EMAIL, stage / TRANSITION_EMAIL.name)
    (stage / "BUILD.txt").write_text(
        "pdflatex -interaction=nonstopmode manuscript.tex\n"
        "bibtex manuscript\n"
        "pdflatex -interaction=nonstopmode manuscript.tex\n"
        "pdflatex -interaction=nonstopmode manuscript.tex\n",
        encoding="utf-8",
    )


def build_code(stage: Path) -> None:
    for source in RELEASE_DOCS.iterdir():
        if source.is_file():
            copy_one(source, stage / source.name)
    copy_one(ROOT / "train_matched_restorer.py", stage / "train_matched_restorer.py")
    for relative in (*MODEL_FILES, *RCAD_FILES, *BASELINE_FILES, *TOOL_FILES, *TEST_FILES):
        copy_one(ROOT / relative, stage / relative)

    # Configuration and source-of-truth ledgers are copied byte-for-byte.
    copy_one(TRAIN / "train" / "audit_config.json", stage / "configs" / "trace_r_training_audit_config.json")
    copy_one(TRAIN / "train" / "history.csv", stage / "configs" / "trace_r_training_history.csv")
    copy_one(TRAIN / "validation" / "best_by_val_map.json", stage / "configs" / "trace_r_best_by_val_map.json")
    copy_one(
        NAFNET_RUN / "train" / "audit_config.json",
        stage / "configs" / "official_nafnet_training_audit_config.json",
    )
    copy_one(
        NAFNET_RUN / "validation" / "best_by_val_map.json",
        stage / "configs" / "official_nafnet_best_by_val_map.json",
    )
    copy_one(
        NAFNET_RUN / "validation" / "stability_audit_v1" / "checkpoint_stability.json",
        stage / "provenance" / "nafnet" / "checkpoint_stability.json",
    )
    copy_one(
        NAFNET_RUN / "validation" / "stability_audit_v1" / "checkpoint_stability.csv",
        stage / "provenance" / "nafnet" / "checkpoint_stability.csv",
    )
    copy_one(
        CONTROLLED / "frozen_official_nafnet_before_test.json",
        stage / "provenance" / "controlled" / "frozen_official_nafnet_before_test.json",
    )
    copy_one(CONTROLLED / "final_provenance_ledger.json", stage / "provenance" / "controlled" / "final_provenance_ledger.json")
    copy_one(CONTROLLED / "detection" / "aggregate_metrics.csv", stage / "provenance" / "controlled" / "aggregate_metrics.csv")
    copy_one(CONTROLLED / "detection" / "all_condition_metrics.csv", stage / "provenance" / "controlled" / "all_condition_metrics.csv")
    copy_one(CONTROLLED / "detection" / "clean_ceiling.csv", stage / "provenance" / "controlled" / "clean_ceiling.csv")
    copy_one(CONTROLLED / "fidelity" / "all_summary.csv", stage / "provenance" / "controlled" / "fidelity_summary.csv")
    copy_one(CONTROLS / "metadata_control_summary.csv", stage / "provenance" / "metadata_controls" / "metadata_control_summary.csv")
    for name in ("SEALED_TEST_OPENED.json", "SEALED_TEST_COMPLETE.json"):
        copy_one(CRID / name, stage / "provenance" / "crid" / name)
    for name in (
        "annotation_freeze.json",
        "annotations_frozen.json",
        "locked_split.json",
        "split_manifest.csv",
        "split_manifest.json",
    ):
        copy_one(CRID_ANNOTATIONS / name, stage / "provenance" / "crid" / "annotations" / name)
    copy_one(
        CRID_ANNOTATIONS / "exports" / "latest" / "export_audit.json",
        stage / "provenance" / "crid" / "annotations" / "export_audit.json",
    )
    for name in (
        "args.yaml",
        "data_train_val.yaml",
        "detector_selection_freeze.json",
        "detector_training_audit.json",
        "results.csv",
    ):
        copy_one(CRID_DETECTOR / name, stage / "provenance" / "crid" / "detector" / name)
    # The final TRACE-R field checkpoint is a validation-only composition of
    # the selected DFPIR candidate and an identity-initialized sensor policy.
    # Copy only the four matched candidate-training audits here; the exact
    # TRACE-R composition is recorded separately by CRID_POLICY_IDENTITY.
    for method in ("nafnet", "instructir", "dfpir", "demoe"):
        for name in ("adaptation_audit.json", "adaptation_complete.json", "training_history.json"):
            copy_one(
                CRID_ADAPTATION / method / name,
                stage / "provenance" / "crid" / "adaptation" / method / name,
            )
    copy_one(
        CRID_DETAIL / "detail_sweep_selection.json",
        stage / "provenance" / "crid" / "validation" / "trace_detail_sweep_selection.json",
    )
    copy_one(
        CRID_RAW_CONTROL / "detail_sweep_selection.json",
        stage / "provenance" / "crid" / "validation" / "native_filter_control.json",
    )
    copy_one(
        CRID_TRACE_VALIDATION / "rmrp" / "trace_crid_field_policy_frozen" / "validation_selection.json",
        stage / "provenance" / "crid" / "validation" / "trace_frozen_policy_validation.json",
    )
    copy_one(
        CRID_POLICY_IDENTITY,
        stage / "provenance" / "crid" / "trace_field_policy_identity.json",
    )
    sealed = json.loads((CRID / "SEALED_TEST_COMPLETE.json").read_text(encoding="utf-8"))
    for result in sealed["results"]:
        report = result.get("validation_report")
        if report:
            copy_one(
                Path(report),
                stage / "provenance" / "crid" / "validation" / f"{result['method']}_selection.json",
            )
    copy_one(PAPER / "asset_manifest.json", stage / "provenance" / "paper" / "asset_manifest.json")
    copy_one(
        PAPER / "figures" / "qualitative_selection_manifest.json",
        stage / "provenance" / "paper" / "qualitative_selection_manifest.json",
    )

    # Include the exact submitted source snapshot but not LaTeX build products.
    copy_one(PAPER / "manuscript.tex", stage / "paper_source" / "manuscript.tex")
    copy_one(PAPER / "references.bib", stage / "paper_source" / "references.bib")
    for name in (
        "table_controlled_summary.tex",
        "table_inference_inputs.tex",
        "table_condition_results.tex",
        "table_fidelity.tex",
        "table_metadata_controls.tex",
        "table_crid.tex",
        "table_architecture_audit.tex",
        "table_crid_adaptation_audit.tex",
    ):
        copy_one(PAPER / "tables" / name, stage / "paper_source" / "tables" / name)
    stamp_python_authors(stage)


def main() -> None:
    args = parse_args()
    release_root = args.out.resolve()
    reset_directory(release_root)
    paper_stage = release_root / "TRACE_R_IEEE_TITS_paper_20260831"
    code_stage = release_root / "TRACE_R_code_and_provenance_20260831"
    paper_stage.mkdir()
    code_stage.mkdir()
    build_paper(paper_stage)
    build_code(code_stage)
    write_manifest(paper_stage, "MANIFEST.sha256")
    write_manifest(code_stage, "MANIFEST.sha256")

    paper_zip = release_root / f"{paper_stage.name}.zip"
    code_zip = release_root / f"{code_stage.name}.zip"
    archive_tree(paper_stage, paper_zip)
    archive_tree(code_stage, code_zip)

    verify_root = release_root / "verification_extract"
    paper_check = extract_and_verify(paper_zip, paper_stage, verify_root / "paper")
    code_check = extract_and_verify(code_zip, code_stage, verify_root / "code")
    verification = {
        "status": "verified" if paper_check["exact_match"] and code_check["exact_match"] else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper": paper_check,
        "code": code_check,
        "controlled_selected_checkpoint_sha256": "a79e2a775e576f17cfe78688484985830e89de7fbe582eca10d43cc4e0cf59db",
        "crid_selected_checkpoint_sha256": "f67019ef510b47f6a21f9036e6f5a3f74479154f54288e3937c76d174b3ff045",
        "detector_output_fusion": False,
        "single_restored_image_per_method": True,
    }
    (release_root / "release_verification.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )
    if verification["status"] != "verified":
        raise RuntimeError(json.dumps(verification, indent=2))
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
