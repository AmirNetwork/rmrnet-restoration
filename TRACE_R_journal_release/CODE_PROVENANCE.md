# Code Provenance

TRACE-R is authored and maintained by Amir Ghorbani
(`amir.ghorbani@rmit.edu.au`). This document separates original TRACE-R work,
project-written integration code, and third-party software used by the study.

## Original TRACE-R source

The following components implement the contribution described in the paper:

- `models/tracer_sensor_adapter.py`: reliability-aware state interface and
  hierarchy-wide low-rank adapters (paper Eqs. 6--9).
- `models/rmrp_metadata_demoe.py`: image/measurement state reconciliation and
  DeMoE route integration. The historical `rmrp` identifier is retained only
  for checkpoint compatibility.
- `models/rmrp_prompted_dfpir.py`: TRACE-R field interface for a DFPIR
  restoration substrate.
- `rcadnet/practical_metadata.py`, `rcadnet/sensor_geometry.py`, and
  `rcadnet/physics_prior.py`: exposure alignment, physical capture map,
  availability, and reliability.
- `rcadnet/losses.py` and `rcadnet/task_losses.py`: fidelity,
  detector-feature, detector-supervision, and state-calibration objectives.
- `train_matched_restorer.py`: staged optimization used for the controlled
  study.
- `tools/train_crid320_restorer.py` and the `tools/*crid320*` scripts: field
  calibration, validation-only selection, sealed evaluation, and asset build.
- `tools/run_tracer_locked_confirmatory.py` and
  `tools/run_tracer_metadata_controls.py`: controlled confirmatory evaluation
  and capture-record interventions.

The MIT license in this release applies to these project-authored files.

## Project-written compatibility code

Files under `baselines/` expose a common training and evaluation interface for
the published comparison methods. They do not claim authorship of the upstream
architectures or checkpoints.

- `baselines/demoe_adapter.py`: loader and interface for the official DeMoE
  repository.
- `baselines/dfpir_adapter.py`: loader and current-tensor compatibility checks
  for the official DFPIR repository.
- `baselines/instructir_adapter.py`: loader and instruction interface for the
  official InstructIR repository.
- `baselines/nafnet_road.py`: a project-local implementation of the published
  NAFNet GoPro configuration, with strict loading of official tensors and the
  upstream MIT notice retained.

## Third-party software and weights

DeMoE, DFPIR, InstructIR, NAFNet, Ultralytics detectors, public detector
checkpoints, and datasets retain their original authorship and licenses. Their
large source trees and weights are intentionally absent from this archive.
`DATA_AND_WEIGHTS.md` gives acquisition paths, while
`THIRD_PARTY_LICENSES.md` records the applicable upstream projects and terms.

## Verification boundary

The repository records hashes for the selected TRACE-R checkpoints, frozen
split manifests, result ledgers, paper tables, and figures. Rebuilding a table
or figure reads those frozen ledgers; it does not retrain a model or select a
new checkpoint. This boundary keeps manuscript generation separate from model
selection.
