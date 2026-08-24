# Matched Target-Adaptation Protocol

This protocol compares RMR-P, NAFNet, sensor-conditioned FiLM-NAFNet, DeMoE,
DFPIR, and InstructIR after an equal target-domain adaptation budget on the
union of the IVCNZ and PCM training splits.

## Shared conditions

- balanced sampling over two datasets and five conditions: clean, motion,
  defocus, low light, and mixed motion-low-light;
- 30 epochs, 512 sampled patches per epoch, effective batch size 4;
- identical 128 x 128 paired crops and defect-focused crop probability;
- AdamW, learning rate `2e-5`, cosine decay, weight decay `1e-4`;
- the same frozen IVCNZ or PCM detector for each corresponding image;
- common objective

  `L_common = L_charbonnier + 0.15 L_gradient + 0.08 L_detector-feature`;

- checkpoints at epochs 0, 5, 10, 15, 20, 25, and 30;
- selection by mean validation mAP50 across both datasets and all four degraded
  conditions. Test images and labels are not read during training or selection.

Public pretraining histories are necessarily different and are recorded by
checkpoint SHA-256. The protocol equalizes target-domain data, crops,
optimization, detector guidance, and selection. DFPIR uses FP32 micro-batches
of one with four-step accumulation because FP16 is numerically unstable on the
6 GB GPU; the effective batch and optimizer-step count remain unchanged.

FiLM-NAFNet is the sensor-conditioned control. It receives the same public
82-value packet as RMR-P, but has no image-state estimator, cause-wise
reliability gate, physical proposal, or private state supervision. Its FiLM
layers are zero-initialized after loading the same road-trained NAFNet weights;
an automated test verifies that plain and FiLM-NAFNet produce exactly the same
output before matched adaptation.

## RMR-P-specific supervision

RMR-P alone receives the proposed sensor modality. On training data, private
generator labels supervise its eight-dimensional corruption state. These
labels are not model inputs and are absent from validation/test sidecars. The
deployed input is an 82-value public packet containing 11 three-axis gyroscope
samples, 11 three-axis accelerometer samples, camera settings, vehicle/timing
context, and reliability/availability values.

The deterministic sensor proposal follows

`Delta_theta = T_exp * integral(omega(t) dt)`.

A bounded learned calibration combines this proposal with image evidence and
per-cause reliability. Thus high ISO means elevated noise risk, not proof that
visible noise is present; the image branch can reject an inconsistent sensor
proposal. Balanced modality dropout trains all eight camera/IMU/vehicle
availability patterns. Measurement jitter covers noisy packets. A fraction of
training images receives a different sample's internally consistent sensor
record; the sensor-only target follows that donor while the joint image target
does not, teaching the compatibility gate to reject a misaligned packet.

## Commands

```powershell
python tools/run_matched_training_suite.py `
  --out-root experiments/matched_target_adaptation_v1_20260818

python tools/audit_matched_protocol.py `
  --training-root experiments/matched_target_adaptation_v1_20260818 `
  --out experiments/matched_target_adaptation_v1_20260818/protocol_audit.json

python tools/validate_matched_restorer_suite.py `
  --training-root experiments/matched_target_adaptation_v1_20260818 `
  --out experiments/matched_validation_v1_20260818
```

The validation selector first uses a deterministic 64-image subset to retain
two checkpoints per model, then selects between those checkpoints on the full
validation splits. Only the selected checkpoints may enter the final
evaluation stage.
