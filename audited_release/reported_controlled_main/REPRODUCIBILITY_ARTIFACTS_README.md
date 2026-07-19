# RMR-Net Reproducibility Artifacts

This folder contains the small scripts and generated CSV/JSON files needed to
audit the reported manuscript tables. It intentionally does not include large
datasets, model checkpoints, or external benchmark archives.

## Included protocol scripts

- `tools/prepare_rdd2022_yolo.py`: converts the external RDD2022 VOC-style
  road-damage annotations to YOLO format. No RDD2022 result is claimed unless
  the dataset is downloaded and the full train/eval protocol is run.
- `tools/prepare_real_road_telemetry_pilot.py`: converts synchronized
  road-damage telemetry CSVs into the RMR-Net metadata format. It does not
  invent missing telemetry.
- `tools/train_fasterrcnn_yolo.py`: Faster R-CNN detector-family protocol for
  YOLO-format road splits. No non-YOLO detector number is claimed until this is
  run and added to provenance.
- `tools/snake_boundary_metrics.py`: detector-box or fixed-GT-box active-contour
  measurement and polygon metric evaluation.
- `tools/profile_rmrnet_complexity.py`: deployed model parameter/FLOP/runtime
  profiler.

## Included generated evidence

- `experiments/v29_review_completion`: split audits, Holm-corrected bootstrap
  tests, contour-yield bootstrap, fidelity/detection correlation, and claim
  boundary CSVs.
- `experiments/v30_submission_readiness`: PCM polygon contour accuracy,
  checkpoint-selection sensitivity, model complexity, and submission-readiness
  CSV/JSON files.
- `runs/snake_polygon_accuracy_v30`: only CSV/JSON summaries from the seeded PCM
  polygon audit, not large overlay images.

Every manuscript result should map to a row in
`paper_ieee_tits_rmrnet/RESULT_PROVENANCE_TABLE.csv`.
