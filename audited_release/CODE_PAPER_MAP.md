# Code-to-paper map

Author: Amir Ghorbani (`amir.ghorbani@rmit.edu.au`)

This map links the mathematical definitions in the Automation in Construction manuscript to the archived implementation that produced the reported controlled checkpoints. Line numbers may shift when comments are added; class and function names are stable.

| Paper definition | Implementation |
|---|---|
| Image and metadata degradation codes; reliability-gated fusion | `reported_controlled_main/rcadnet/model.py`: `DegradationCodeEncoder`, `CodeBasisFusion` |
| Non-periodic basis $b(u)=[u,u^2,\sqrt{u+\epsilon},su,u(1-u)]$ | `reported_controlled_main/rcadnet/model.py`: `CodeBasisFusion._basis` |
| FiLM conditioning $(1+\gamma(z))\odot x+\beta(z)$ | `reported_controlled_main/rcadnet/model.py`: `FiLM.forward` |
| Bounded detail update $I_r=\mathrm{clip}(I_b+\eta_dG_d\odot D_h,0,1)$ | `reported_controlled_main/rcadnet/model.py`: `EvidencePreservingDetailSkip.forward` |
| Base fidelity objective $\mathcal{L}_{\mathrm{base}}$ | `reported_controlled_main/rcadnet/losses.py`: `RoadRestorationLoss.forward` |
| Cross-quality patch mixing | `reported_controlled_main/rcadnet/task_losses.py`: `cross_quality_patch_mix` |
| Detector-feature loss $\mathcal{L}_{\mathrm{TDP}}$ | `reported_controlled_main/rcadnet/task_losses.py`: `TaskDrivenPerceptualLoss` |
| Hutchinson Jacobian estimator $\mathcal{L}_{\mathrm{J}}$ | `reported_controlled_main/rcadnet/task_losses.py`: `hutchinson_jacobian_penalty` |
| Detector anchor, evidence non-regression, and detail-copy terms | `reported_controlled_main/rcadnet/task_losses.py`: `DetectorInputAnchorLoss`, `road_evidence_nonregression_loss`, `detail_copy_regularization` |
| Composite task objective | `reported_controlled_main/rcadnet/task_losses.py`: `CompositeTaskLoss.forward` |
| Same-backward-pass optimization and task-loss warm-up | `reported_controlled_main/train_rcadnet.py`: training epoch loop and `task_warmup_scale` |
| Validation-only checkpoint selection | `reported_controlled_main/train_rcadnet.py`: validation mAP history and best-checkpoint block |
| Controlled qualitative atlas and deterministic selection ledger | `reported_controlled_main/tools/build_detection_qualitative_atlas.py` |
| Sony held-out detector-evidence atlas | `reported_controlled_main/tools/build_sony_field_qualitative_atlas.py` |
| Sony collection-platform figure | `reported_controlled_main/tools/build_sony_collection_figure.py` |

The active-contour routines are post-detection measurement tools. Their training weight is zero for every reported restoration checkpoint, as stated in the manuscript and recorded configurations.
