from __future__ import annotations

"""Audit the exact KITTI splitB protocol reported in the manuscript.

Author: Amir Ghorbani <amir.ghorbani@rmit.edu.au>

The reported table trains on drives 0001/0002/0005 and evaluates drive 0011.
This script intentionally does not audit older exploratory KITTI directories.
"""

import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = "kitti_realmeta_longexp_motion"
SPLITS = {
    "train": ROOT / "data" / "kitti_realmeta_longexp_train_splitB",
    "test": ROOT / "data" / "kitti_realmeta_longexp_test_splitB",
}
OUT = ROOT / "experiments" / "kitti_splitB_reported_protocol_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(values: list[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "mean": statistics.fmean(values),
        "maximum": max(values),
    }


def main() -> None:
    report: dict[str, object] = {
        "test_used_for_selection": False,
        "split_policy": {
            "train": [
                "2011_09_26_drive_0001_sync",
                "2011_09_26_drive_0002_sync",
                "2011_09_26_drive_0005_sync",
            ],
            "test": ["2011_09_26_drive_0011_sync"],
        },
        "splits": {},
    }
    names: dict[str, set[str]] = {}
    for split, root in SPLITS.items():
        scenario = root / "scenarios" / SCENARIO
        images = sorted((scenario / "gt").glob("*"))
        metadata_paths = sorted((scenario / "metadata").glob("*.json"))
        metadata = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_paths]
        protocol_path = root / "preparation_protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.exists() else {}
        names[split] = {path.name for path in images}
        report["splits"][split] = {  # type: ignore[index]
            "images": len(images),
            "metadata_records": len(metadata),
            "sequences": sorted({str(row["sequence"]) for row in metadata}),
            "blur_length_px": describe([float(row["blur_length_px"]) for row in metadata]),
            "blur_angle_deg": describe([float(row["blur_angle_deg"]) for row in metadata]),
            "speed_mps": describe([float(row["speed_mps"]) for row in metadata]),
            "yaw_rate_radps": describe([float(row["raw_oxts_yaw_rate_radps"]) for row in metadata]),
            "protocol_sha256": sha256(protocol_path) if protocol_path.exists() else None,
            "exposure_ms": metadata[0]["exposure_ms"] if metadata else None,
        }
    overlap = sorted(names["train"] & names["test"])
    train_sequences = set(report["splits"]["train"]["sequences"])  # type: ignore[index]
    test_sequences = set(report["splits"]["test"]["sequences"])  # type: ignore[index]
    sequence_overlap = sorted(train_sequences & test_sequences)
    report["filename_overlap"] = overlap
    report["sequence_overlap"] = sequence_overlap
    report["sequence_disjoint"] = not overlap and not sequence_overlap
    report["ok"] = (
        not overlap
        and not sequence_overlap
        and report["splits"]["train"]["images"] == 339  # type: ignore[index]
        and report["splits"]["test"]["images"] == 233  # type: ignore[index]
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
