# RMR-Net paper code package

Author: Amir Ghorbani (`amir.ghorbani@rmit.edu.au`)

This archive separates the implementation that produced the reported controlled IVCNZ/PCM checkpoints from later research variants.

## Start here

- `reported_controlled_main/` is the primary implementation for the headline IVCNZ and PCM results.
- `reported_checkpoints/` contains the validation-selected epoch-28 checkpoints and their recorded training configurations.
- `reported_results/` contains the validation-selection ledger and held-out result files used by the manuscript.
- `paper_provenance/` maps paper evidence to result artifacts.
- `FINAL_SUBMISSION_INTEGRITY_AUDIT.md` records the final manual-table,
  checkpoint-selection, split, and leakage audit.
- `CODE_PAPER_MAP.md` maps manuscript equations and policies to concrete classes and functions.
- `variants/current_research/` contains later source variants. These are included for traceability and are not presented as the source of the controlled epoch-28 results.

The primary implementation was selected by strict state-dictionary compatibility with both reported checkpoints: all 223 tensors match by name and shape, with no missing or unexpected keys. Run the included check before reproducing an experiment:

```powershell
python verify_reported_checkpoints.py
python -m compileall reported_controlled_main variants
```

For a Windows CUDA environment and the training/evaluation commands, see `reported_controlled_main/README.md`, `reported_controlled_main/EXPERIMENT_PROTOCOL.md`, and the two `*_audit_config.json` files. Dataset paths are machine-specific and must be changed locally. Checkpoint selection is validation-only; the held-out test metrics are not used to select an epoch. The manuscript tables were manually assembled before 15 July 2026 and were later verified against the frozen CSV artifacts included here. Known duplicate and temporal-adjacency findings in the legacy controlled splits are disclosed; this release does not claim that those source partitions are leakage-free.

Project files carry an Amir Ghorbani integration header. Where an adapter or utility incorporates a third-party method, the header identifies project integration only; original method authorship and licensing remain unchanged. Release-only comments and documentation do not alter executable behavior.

## Naming

The paper-facing name is **RMR-Net** (Road Metadata-aware Restoration Network). The primary source retains the historical `RCADNet` Python class name for checkpoint compatibility.

## Scope

The code archive does not redistribute third-party datasets or detector weights whose licenses require separate download. Their sources, versions, and local hashes are recorded in the manuscript and release documentation.
