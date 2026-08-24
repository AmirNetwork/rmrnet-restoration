# Practical Metadata Protocol

RMR-Net treats metadata as a set of independently available and noisy
measurements. It does not use a single metadata-present switch.

## Public inference packet

The 82-value packet contains measurements that a synchronized road-monitoring
system can provide:

- 11 three-axis gyroscope samples spanning the exposure;
- 11 three-axis accelerometer samples spanning the exposure;
- exposure, ISO/gain, focal length, aperture, autofocus/focus proxy, and rolling
  readout;
- vehicle speed and yaw rate;
- timing offset and IMU noise estimates;
- continuous camera, IMU, and vehicle reliability values; and
- a global any-metadata-present indicator.

Private synthetic renderer parameters are never included in this public packet.
They are retained only as train/validation supervision for the corruption-state
heads.

## Partial availability

Camera, IMU, and vehicle fields may fail independently. The model therefore
supports all eight availability states:

| Index | Camera | IMU | Vehicle |
|---:|:---:|:---:|:---:|
| 0 | no | no | no |
| 1 | no | no | yes |
| 2 | no | yes | no |
| 3 | no | yes | yes |
| 4 | yes | no | no |
| 5 | yes | no | yes |
| 6 | yes | yes | no |
| 7 | yes | yes | yes |

Unavailable measurement fields and their reliability declarations are zeroed.
Measurement noise is then added only to fields that remain available.

## Joint corruption inference

The degraded image produces an image corruption estimate
`z_I = g_I(I_d)`. The available sensor packet produces a sensor estimate and
cause-wise support values. A joint posterior refiner combines the image
features, image estimate, sensor estimate, and support:

`z = g_J(I_d, z_I, z_M, q)`.

Thus metadata do not replace image evidence. Reliable synchronized IMU can
directly support motion-PSF coordinates; camera fields support exposure,
low-light, and focus-related coordinates; vehicle state is contextual evidence.
When a coordinate is unsupported or a modality is absent, its fusion weight
falls back toward the image estimate.

## Training curriculum

The detector-aware PCM run keeps a full packet for 50% of samples. For the
other 50%, it samples the seven non-full states uniformly. It also perturbs
available measurements with normalized Gaussian noise. Motion, defocus,
low-light, and mixed motion/low-light examples are all included.

The objective contains restoration fidelity, detector-feature preservation,
detector-Jacobian stability, evidence non-regression, metadata-control
advantage, and corruption-state supervision in one backward pass. Active-contour
loss is disabled. Validation mAP50, never test mAP, selects the checkpoint.

After selection is frozen, the evaluator reports full, camera-only, IMU-only,
vehicle-only, pairwise partial, unavailable, shuffled, and noisy controls.
