# Practical Sensor Interface for RMR-Net

RMR-Net does not treat IMU data as a generic degradation label. A gyroscope
observes camera angular velocity during exposure. It can constrain motion blur,
but it cannot directly observe defocus, photon noise, low illumination, or JPEG
compression. Those causes require camera settings and image evidence.

## Cause observability

The conditioning state is assembled cause by cause:

| Degradation state | Primary observable evidence | Important limitation |
| --- | --- | --- |
| Motion direction and magnitude | Exposure-synchronized gyro trajectory, exposure time, focal length, rolling-shutter timing | Translation and scene depth are only partly observable from a single vehicle IMU |
| Irregular vibration | High-frequency gyro and accelerometer variation during exposure | Vehicle acceleration is not identical to optical camera motion |
| Defocus | Focus position/error, autofocus confidence, aperture, image evidence | IMU does not observe optical focus |
| Low illumination and sensor noise | Exposure, ISO, analog gain, image brightness/noise evidence | Camera response and denoising differ by device |
| Compression | Codec quality or quantization telemetry when available; otherwise image evidence | The current practical packet has no codec field |

This division is enforced in code by a cause-wise reliability vector. For
example, absent or unreliable autofocus telemetry sets the metadata correction
for defocus to zero even when valid IMU samples are present.

## Inference packet

The released practical interface contains 82 normalized values:

- 11 synchronized gyroscope samples, each with x/y/z angular velocity (33);
- 11 synchronized accelerometer samples, each with x/y/z acceleration (33);
- exposure, ISO, analog gain, focal length, aperture, focus-error proxy,
  autofocus confidence, and rolling-shutter readout (8);
- vehicle speed, vehicle yaw rate, timestamp offset, and IMU noise (4);
- camera, IMU, and vehicle reliability plus metadata availability (4).

Signed inertial values and signed offsets are in `[-1, 1]`. Camera settings and
reliabilities are in `[0, 1]`. Values are normalized with constants fixed by the
training protocol, never with statistics from an evaluation split.

## Causal conversion

For an exposure of duration `T`, the gyro trace gives the angular trajectory

```text
Delta theta = integral_0^T omega(t) dt.
```

The packet contract expects IMU axes to be rotated into the camera frame by a
fixed mounting calibration. The deterministic encoder forms normalized
image-plane motion coordinates from the in-plane trajectory; focal length and
rolling-readout fields are available to the bounded calibration branch. A
bounded learned residual calibrates mounting bias, timestamp error, and the
synthetic-to-sensor scale. A second calibrated code drives the physical motion
candidate. Defocus and low-light coordinates use camera fields only when their
reliability is nonzero; compression remains image-estimated because the current
packet has no codec telemetry.

The image branch independently estimates the eight degradation coordinates.
The fused state is

```text
z = z_image + alpha .* (z_sensor - z_image),
```

where `alpha` is cause-wise, depends on sensor reliability and disagreement,
and is forced to zero when the corresponding metadata is unavailable.
The training target is also cause-wise:

```text
alpha_positive_target = 0.85 * sensor_cause_reliability
alpha_counterfactual_target = 0.02
```

This avoids asking the gate to trust an unobservable cause. Counterfactual
packets rotate the gyro trajectory, attenuate vibration evidence, and
contradict exposure/focus settings while preserving availability and
reliability declarations. They are training controls, not synthetic inputs
used at inference.

## Physical candidate

The sensor-conditioned inverse is deliberately not a hard replacement. The
deployed practical path computes

```text
I_out = I_neural
      + A(x) * g_motion * g_sensor * (I_physical - I_neural),
```

where `A(x)` is a bounded learned confidence map, `g_motion` suppresses the
motion inverse for non-motion corruption, and `g_sensor` is derived from IMU
quality and synchronization. The gate is initialized nearly closed. During
paired training only, its target is

```text
A_target(x) = g_motion * g_sensor
            * sigmoid((e_neural(x) - e_physical(x)) / 0.02),
```

where the two errors are measured against the clean training target. This
target is stop-gradient and is not available or required at inference. It
teaches the confidence map to use inertial inversion only where that candidate
actually improves the image, including under counterfactual sensor packets.

## Training-only information

PCM and IVCNZ have known synthetic renderer parameters. On training and
validation only, those parameters supervise the sensor-to-cause calibration.
They are never concatenated to the inference packet. Public test sidecars omit
both calibration targets and all renderer parameters.

The practical road experiment is therefore a noisy sensor-emulation study over
controlled image degradation, not a claim of naturally synchronized road IMU.
The external GyroBlur audit uses released raw gyro traces without test-time
calibration. KITTI provides real OXTS telemetry under separately controlled
blur. These evidence blocks must remain distinct in reporting.

## Code locations

- Packet schema and encoder: `rcadnet/practical_metadata.py`
- Reliability-gated image/sensor fusion: `rcadnet/model.py`
- Sensor-emulation builder: `tools/build_practical_metadata_benchmark.py`
- Calibration pretraining: `tools/pretrain_practical_sensor_encoder.py`
- Validation-only selection: `tools/evaluate_practical_sensor_sweep.py`
- Corrected promotion driver: `tools/continue_practical_sensor_gatefix.py`
- External raw-gyro audit: `tools/eval_gyroblur_subset.py`
