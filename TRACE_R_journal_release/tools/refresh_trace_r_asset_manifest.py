#!/usr/bin/env python3
"""Refresh TRACE-R paper-asset hashes from the frozen result ledgers.

This utility performs no training or selection. It records the exact files used
by the manuscript so the release packager can reject superseded field assets.

Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROLLED = Path(r"E:\TRACE_R_experiments\trace_locked_confirmatory_v72_20260828")
CONTROLS = Path(r"E:\TRACE_R_experiments\trace_metadata_controls_v66_20260828")
CRID = Path(r"E:\TRACE_R_experiments\crid320_sealed_test_20260831")

ASSETS = (
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, default=ROOT / "paper_ieee_tits_trace_r")
    args = parser.parse_args()
    paper = args.paper.resolve()
    sources = {
        "controlled": CONTROLLED / "final_provenance_ledger.json",
        "metadata_controls": CONTROLS / "metadata_control_summary.csv",
        "crid": CRID / "SEALED_TEST_COMPLETE.json",
    }
    for path in (*sources.values(), *(paper / relative for relative in ASSETS)):
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = {
        "status": "paper_assets_complete",
        "training_or_selection_performed": False,
        "detector_output_fusion": False,
        "single_restored_image_per_method": True,
        "sources": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in sources.items()
        },
        "outputs": {
            relative.replace("/", "\\"): sha256(paper / relative)
            for relative in ASSETS
        },
        "manuscript_sha256": sha256(paper / "manuscript.tex"),
        "compiled_pdf_sha256": sha256(paper / "manuscript.pdf"),
    }
    destination = paper / "asset_manifest.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
