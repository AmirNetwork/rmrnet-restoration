# Paper-to-Code Map

This file links the notation in the manuscript to the implementation. It is meant to make the repository readable for reviewers and future users.

## Degradation Codes and Metadata Fusion

Paper symbols:

- `c_m`: metadata-derived degradation code.
- `c_b = g_phi(I_d)`: image-estimated degradation code.
- `b(u) = [u, u^2, sqrt(u + eps), u*s, u*(1-u)]`: sparse monotone basis, with severity `s = u_8`.
- `z = alpha e_m + (1-alpha) e_b`: fused conditioning code.

Code:

- `rcadnet/model.py::DegradationEncoder`
- `rcadnet/model.py::CodeBasisFusion._basis`
- `rcadnet/model.py::CodeBasisFusion.forward`
- `train_rcadnet.py::prepare_codes`

The image code is supervised during controlled-degradation training by `aux_code_weight` in `train_rcadnet.py`. Metadata dropout and jitter are also applied in `prepare_codes`.

## FiLM Conditioning

Paper equation:

```text
FiLM(x, z) = (1 + gamma(z)) * x + beta(z)
```

Code:

- `rcadnet/model.py::FiLM`
- `rcadnet/model.py::RCADBlock`
- `rcadnet/model.py::EvidenceConditionedBlock`

The implementation bounds `gamma` and `beta` with `tanh` and initializes the final projection to zero so the model begins close to identity modulation.

## Base Residual Restoration

Paper equation:

```text
I_b = clip(I_d + h_theta(I_d, z), 0, 1)
```

Code:

- `rcadnet/model.py::RCADNet._decode`

The residual head is zero-initialized for stable fine-tuning.

## Bounded Detail-Preserving Skip

Paper equation:

```text
I_r = clip(I_b + eta_d G_d * D(I_d), 0, 1)
```

Code:

- `rcadnet/model.py::EvidencePreservingDetailSkip`

`D(I_d)` is implemented as multi-scale high-pass detail. `G_d` is a learned evidence gate conditioned by decoder features, image evidence cues, and degradation code.

## Base Restoration Objective

Paper objective:

```text
L_base = L1 + lambda_e L_edge + lambda_f L_frequency
       + lambda_d L_defect_weighted + lambda_v L_visibility
```

Code:

- `rcadnet/losses.py::RCADLoss`
- `train_rcadnet.py`, where `base_weight` multiplies the base loss.

## Task-Driven Perceptual Loss and CQMix

Paper equations:

```text
L_TDP = sum_k alpha_k ||Phi_k(I_r or I_mix) - Phi_k(I_c)||^2
I_mix = M * I_r + (1 - M) * I_c
```

Code:

- `rcadnet/task_losses.py::FrozenDetectorFeatureExtractor`
- `rcadnet/task_losses.py::TaskDrivenPerceptualLoss`
- `rcadnet/task_losses.py::cross_quality_patch_mix`

Detector parameters are frozen. Clean features are computed under `torch.no_grad()`. The restored/mixed path stays differentiable.

## Cascaded Jacobian Regularization

Paper equation:

```text
L_J = E_v ||grad_{I_r} <Phi(I_r), v>||_2^2
```

Code:

- `rcadnet/task_losses.py::hutchinson_jacobian_penalty`

The implementation uses Rademacher Hutchinson probes to avoid materializing the full Jacobian.

## Evidence Non-Regression and Detector Anchor

Paper role:

- prevent restoration from suppressing detector-visible road evidence;
- keep restored images close to the detector operating distribution.

Code:

- `rcadnet/task_losses.py::road_evidence_nonregression_loss`
- `rcadnet/task_losses.py::DetectorInputAnchorLoss`
- `rcadnet/task_losses.py::CompositeTaskLoss`

## Active Contours

Final paper role:

- post-detection measurement only;
- not part of the final training objective.

Code:

- legacy disabled train-time ablation: `rcadnet/task_losses.py::ActiveContourGeometryLoss`;
- post-detection measurement: `tools/snake_boundary_metrics.py`.

The final config sets `active_contour_weight: 0.0`.

## G46 Field Detector

Paper role:

- native-resolution field safety audit with real EXIF/pose metadata;
- no detector fine-tuning on G46 labels.

Code:

- `Yolo26_coordinate/Yolo26_coordinate.py`
- `tools/eval_yolo26_coordinate_gt46.py`
- `tools/run_gt46_yolo26_coordinate_all_methods.py`
