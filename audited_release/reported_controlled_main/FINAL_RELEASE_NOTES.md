# Final RMR-Net Release Notes

This release package is generated from the current manuscript source and the all-49 Sony native-image field-test run.

## Key synced evidence

- The manuscript frames the native-real experiment as a high-quality Sony native-image safety test, not as a severe natural-blur benchmark.
- The geotagged field test uses all 49 matched annotated cam1 images at native 4752 x 3168 resolution.
- Roboflow annotations are mapped back to the original images; Roboflow-resized pixels are not used for restoration.
- Real EXIF and pose/geotag metadata are joined where available.
- Residual-strength eta and gate-threshold tau sweeps are included for deployment policy analysis.
- Active contours are a post-detection measurement stage; active-contour loss is not part of the final training objective.
- The final configuration in `configs/rmrnet_headline.yaml` sets active-contour training loss to zero.
- No-active-contour ablation provenance is included under `provenance/`.

## All-49 field-test headline

- Best relaxed grouped F1@IoU0.10 in the all-49 table: RMR-Net metadata with F1=0.167, precision=0.095, recall=0.705, FP/image=8.37.

## Files copied

- Referenced tables: 41
- Referenced figures: 26
- Evidence CSVs: final_native_field_manifest.json, paper_metric_summary_all49.csv, geotagged_eta_sweep.csv, geotagged_tau_sweep.csv
- Eta sweep CSV: experiments\roboflow_geotagged_v5_native_real\final_all49\geotagged_eta_sweep.csv
- Tau sweep CSV: experiments\roboflow_geotagged_v5_native_real\final_all49\geotagged_tau_sweep.csv

## Build note

A TeX installation was not available in this local environment. Compile `manuscript.tex` in Overleaf or with `pdflatex`/`bibtex` locally.
