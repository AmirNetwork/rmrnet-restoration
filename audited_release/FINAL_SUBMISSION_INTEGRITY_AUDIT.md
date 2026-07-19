# Final submission integrity audit

Audit date: 19 July 2026

The manuscript tables were manually assembled before 15 July 2026. They were therefore treated as claims to verify, not as source evidence. The final audit compared those claims against frozen machine-readable artifacts and checkpoint manifests.

## Passed checks

- The complete controlled IVCNZ and PCM table rows match `experiments/major_revision_evidence_20260715/master_controlled_results.csv` after manuscript rounding.
- The headline checkpoints load strictly with all 223 tensors matched. Their SHA-256 values are `3c6a1a8e...b538dd` (IVCNZ) and `e580d2bf...e7eec7` (PCM).
- Epoch 28 was selected independently for each controlled dataset by mean validation mAP50 over four degradation families. Test metrics were evaluated after selection and were not used to choose the epoch.
- The sequential PCM-defocus ablation table matches `experiments/major_revision_sequential_ablation_20260715/ablation_metrics.csv` after rounding.
- The reported KITTI intervention uses 339 training images from drives 0001, 0002, and 0005 and 233 test images from drive 0011. The sequence and filename overlap checks both return empty sets.
- The Sony direct-view table matches `experiments/major_revision_evidence_20260715/ilx_direct_single_view_metrics.csv`. The chronological 22/22 policy table and uncertainty values match their corresponding CSV artifacts.
- Active-contour loss is disabled in the reported controlled training objective. The contour procedure is post-detection analysis and does not generate the reported detector AP.
- The native-field checkpoint is not reused for the controlled IVCNZ/PCM results, and the controlled metadata path excludes native-image identity.

## Disclosed limitations

- A retrospective audit of the legacy source partitions found two exact IVCNZ train-test duplicate pairs, one PCM train-test duplicate pair, two PCM train-validation duplicate pairs, and temporal adjacency. The manuscript discloses this and limits these claims to within-source controlled degradation recovery.
- Controlled qualitative examples were chosen from held-out test images using matched-ground-truth recovery criteria. They are illustrative selections, are recorded in selection ledgers, and are not used for quantitative estimation.
- The Sony field corpus was inspected during earlier policy development. The final 22/22 chronological analysis is therefore described as a leakage-reduced reassessment, not a preregistered confirmatory experiment. Its confidence interval includes zero.
- Baselines use their documented native checkpoints or training recipes and do not all share identical information or optimization budgets. The manuscript states this limitation.
- KITTI uses real OXTS telemetry with a controlled blur intervention; it is not evidence of natural road-defect blur.

## Verdict

No evidence of fabricated metrics, hidden test-based checkpoint selection, cross-dataset checkpoint substitution, or detector-label leakage was found. The submission is internally auditable and suitable to submit with the stated limitations. It must not be described as leakage-free or as proving broad natural-blur field superiority.

