#!/usr/bin/env python3
"""Build and verify the paper and code archives for the frozen RMR-P v50 release.

The script intentionally packages only the manuscript dependencies and the
implementation/evidence needed to reproduce the reported validation study.
Datasets, detector weights, and third-party restoration checkpoints are not
redistributed; their expected provenance and hashes are documented instead.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = ROOT / "paper_automation_in_construction_rmrnet"
RELEASE_ROOT = ROOT / "output" / "releases" / "20260824_rmrp_v50"
PDF_ROOT = ROOT / "output" / "pdf"

PAPER_ZIP = RELEASE_ROOT / "RMRP_AutomationInConstruction_final_20260824.zip"
CODE_ZIP = RELEASE_ROOT / "RMRP_v50_code_and_reproducibility_20260824.zip"
FINAL_PDF = PDF_ROOT / "RMRP_AutomationInConstruction_final_20260824.pdf"

LEDGER = ROOT / "experiments" / "final_rmrp_v50_validation_ledger_20260824"
QUALITATIVE_MANIFEST = (
    ROOT
    / "experiments"
    / "final_rmrp_v50_qualitative_20260824"
    / "qualitative_manifest.json"
)

ROOT_FILES = (
    "README.md",
    "CITATION.cff",
    "THIRD_PARTY_LICENSES.md",
    "requirements-windows-gpu.txt",
    "requirements-experiments.txt",
    "requirements-detection-extra.txt",
    "requirements-demoe-extra.txt",
    "requirements-dfpir-extra.txt",
    "train_matched_restorer.py",
    "train_rcadnet.py",
)

DOCUMENTATION = (
    "docs/RMRP_V50_REPRODUCIBILITY.md",
    "docs/practical_metadata_protocol.md",
    "docs/PRACTICAL_SENSOR_INTERFACE.md",
    "docs/MATCHED_TARGET_ADAPTATION.md",
    "configs/rmrp_v50_release.json",
)

TOOLS = (
    "tools/run_matched_training_suite.py",
    "tools/restore_yolo_split.py",
    "tools/eval_yolo_suite.py",
    "tools/validate_rmrp_expert_fusion.py",
    "tools/freeze_rmrp_v50_validation_ledger.py",
    "tools/evaluate_rmrp_v50_validation_fidelity.py",
    "tools/build_rmrp_v50_paper_assets.py",
    "tools/build_rmrp_v50_validation_qualitative.py",
    "tools/build_practical_metadata_benchmark.py",
    "tools/prepare_kitti_realmeta_restoration.py",
    "tools/eval_kitti_metadata_robustness.py",
    "tools/build_kitti_qualitative_atlas.py",
    "tools/prepare_crid_ins_all_metadata.py",
    "tools/prepare_crid_raw_sbg_metadata.py",
    "tools/build_crid_unlabelled_field_adaptation.py",
    "tools/run_crid46_sequence_disjoint_comparison.py",
    "tools/build_crid_direct_sbg_overlays.py",
    "tools/package_rmrp_v50_release.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def paper_dependencies() -> list[Path]:
    manuscript = ensure_file(PAPER_ROOT / "manuscript.tex")
    text = manuscript.read_text(encoding="utf-8")
    dependencies = {
        manuscript,
        ensure_file(PAPER_ROOT / "references.bib"),
        ensure_file(PAPER_ROOT / "manuscript.bbl"),
        ensure_file(PAPER_ROOT / "manuscript.pdf"),
    }

    for match in re.finditer(r"\\input\{([^}]+)\}", text):
        relative = Path(match.group(1))
        if not relative.suffix:
            relative = relative.with_suffix(".tex")
        dependencies.add(ensure_file(PAPER_ROOT / relative))

    image_pattern = r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}"
    for match in re.finditer(image_pattern, text):
        relative = Path(match.group(1))
        candidates = [relative] if relative.suffix else [
            relative.with_suffix(suffix)
            for suffix in (".pdf", ".png", ".jpg", ".jpeg")
        ]
        selected = next((PAPER_ROOT / item for item in candidates if (PAPER_ROOT / item).is_file()), None)
        if selected is None:
            raise FileNotFoundError(f"Unresolved manuscript image: {relative}")
        dependencies.add(selected)

    return sorted(dependencies)


def code_files() -> list[Path]:
    selected: set[Path] = set()
    for relative in (*ROOT_FILES, *DOCUMENTATION, *TOOLS):
        selected.add(ensure_file(ROOT / relative))

    for directory in ("models", "rcadnet", "baselines", "tests"):
        selected.update((ROOT / directory).rglob("*.py"))

    selected.update(path for path in LEDGER.rglob("*") if path.is_file())
    selected.add(ensure_file(QUALITATIVE_MANIFEST))
    return sorted(selected)


def write_archive(path: Path, files: list[Path], prefix: str, base: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source in files:
            relative = source.relative_to(base).as_posix()
            archive.write(source, f"{prefix}/{relative}")


def verify_archive(path: Path, required_suffixes: tuple[str, ...]) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt ZIP entry in {path.name}: {bad}")
        names = archive.namelist()
        for suffix in required_suffixes:
            if not any(name.endswith(suffix) for name in names):
                raise RuntimeError(f"Required entry missing from {path.name}: {suffix}")
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "entries": len(names),
    }


def verify_manuscript_pdf(path: Path) -> dict[str, object]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    if len(reader.pages) != 15:
        raise RuntimeError(f"Expected 15 manuscript pages, found {len(reader.pages)}")
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    required = (
        "RMR-P: Road Metadata-aware Restoration for Pavement Inspection",
        "Supplementary material",
        "FROZEN_VALIDATION_ONLY",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(f"Final PDF text markers missing: {missing}")
    if "??" in text:
        raise RuntimeError("Final PDF contains unresolved-reference marker '??'")
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "pages": len(reader.pages),
    }


def main() -> None:
    RELEASE_ROOT.mkdir(parents=True, exist_ok=True)
    PDF_ROOT.mkdir(parents=True, exist_ok=True)

    ledger = json.loads(ensure_file(LEDGER / "provenance_ledger.json").read_text(encoding="utf-8"))
    if ledger.get("status") != "FROZEN_VALIDATION_ONLY" or ledger.get("test_split_used") is not False:
        raise RuntimeError("Release requires the frozen validation-only ledger with test_split_used=false")

    manuscript_pdf = ensure_file(PAPER_ROOT / "manuscript.pdf")
    shutil.copy2(manuscript_pdf, FINAL_PDF)

    write_archive(PAPER_ZIP, paper_dependencies(), "RMRP_AutomationInConstruction", PAPER_ROOT)
    write_archive(CODE_ZIP, code_files(), "RMRP_v50_code", ROOT)

    verification = {
        "release": "RMR-P v50",
        "evidence_status": ledger["status"],
        "test_split_used": ledger["test_split_used"],
        "pdf": verify_manuscript_pdf(FINAL_PDF),
        "paper_zip": verify_archive(
            PAPER_ZIP,
            ("manuscript.tex", "references.bib", "manuscript.pdf"),
        ),
        "code_zip": verify_archive(
            CODE_ZIP,
            (
                "README.md",
                "models/rmrp_expert_fusion.py",
                "rcadnet/practical_metadata.py",
                "provenance_ledger.json",
            ),
        ),
    }

    if PAPER_ZIP.stat().st_size >= 50 * 1024 * 1024:
        raise RuntimeError("Paper ZIP exceeds the 50 MiB upload limit")

    manifest = RELEASE_ROOT / "release_manifest.json"
    manifest.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")

    # Reopen both archives from a clean temporary directory and verify that the
    # paper source retains every referenced dependency after extraction.
    tmp_root = ROOT / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="verify_rmrp_v50_", dir=tmp_root) as raw_tmp:
        verify_root = Path(raw_tmp)
        with zipfile.ZipFile(PAPER_ZIP) as archive:
            archive.extractall(verify_root / "paper")
        with zipfile.ZipFile(CODE_ZIP) as archive:
            archive.extractall(verify_root / "code")
        ensure_file(verify_root / "paper" / "RMRP_AutomationInConstruction" / "manuscript.tex")
        ensure_file(verify_root / "code" / "RMRP_v50_code" / "README.md")

    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
