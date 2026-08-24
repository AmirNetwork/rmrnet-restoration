from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from rcadnet.practical_metadata import (
    CONTEXT_START,
    PRACTICAL_SENSOR_DIM,
    SENSOR_PACKET_NAMES,
)
from tools.prepare_crid_raw_sbg_metadata import build_packet


def base_metadata(timestamp: str) -> dict:
    packet = [0.0] * PRACTICAL_SENSOR_DIM
    packet[CONTEXT_START + 12] = 0.95
    packet[CONTEXT_START + 14] = 0.90
    packet[CONTEXT_START + 15] = 1.0
    return {
        "exif": {
            "DateTimeOriginal": timestamp,
            "SubsecTimeOriginal": "500",
            "OffsetTimeOriginal": "+00:00",
        },
        "practical_sensor_packet": {
            name: value for name, value in zip(SENSOR_PACKET_NAMES, packet)
        },
    }


def synthetic_sbg(center: float) -> dict[str, np.ndarray]:
    time = np.linspace(center - 0.1, center + 0.1, 41)
    phase = np.linspace(-1.0, 1.0, len(time))
    gyro = np.stack([phase, phase * 0.5, -phase], axis=1)
    accel = np.stack([phase * 2.0, 9.81 + phase, phase * 0.25], axis=1)
    return {"time": time, "gyro": gyro, "accel": accel}


def test_direct_packet_is_populated_only_inside_recording() -> None:
    capture = datetime(2026, 6, 9, 4, 56, 0, 500000, tzinfo=timezone.utc)
    sbg = synthetic_sbg(capture.timestamp())
    packet, provenance, covered = build_packet(
        base_metadata("2026:06:09 04:56:00"),
        sbg,
        half_window_seconds=0.025,
        gyro_full_scale=4.0,
        accel_full_scale=19.62,
    )
    assert covered
    assert len(packet) == PRACTICAL_SENSOR_DIM
    assert any(abs(value) > 0 for value in packet[:CONTEXT_START])
    assert provenance["angular_rate_source"] == "direct SBG measurement"
    assert provenance["annotation_blind"] is True


def test_out_of_interval_packet_marks_imu_unavailable() -> None:
    center = datetime(2026, 6, 9, 4, 56, 10, tzinfo=timezone.utc).timestamp()
    sbg = synthetic_sbg(center)
    packet, provenance, covered = build_packet(
        base_metadata("2026:06:09 04:55:00"),
        sbg,
        half_window_seconds=0.025,
        gyro_full_scale=4.0,
        accel_full_scale=19.62,
    )
    assert not covered
    assert all(value == 0.0 for value in packet[:CONTEXT_START])
    assert packet[CONTEXT_START + 13] == 0.0
    assert packet[CONTEXT_START + 12] == 0.95
    assert packet[CONTEXT_START + 14] == 0.90
    assert provenance["angular_rate_source"] == "unavailable"


def test_out_of_interval_packet_can_retain_audited_ins_trajectory() -> None:
    center = datetime(2026, 6, 9, 4, 56, 10, tzinfo=timezone.utc).timestamp()
    sbg = synthetic_sbg(center)
    metadata = base_metadata("2026:06:09 04:55:00")
    values = list(metadata["practical_sensor_packet"].values())
    values[0] = 0.10
    values[CONTEXT_START - 1] = -0.20
    values[CONTEXT_START + 13] = 0.72
    metadata["practical_sensor_packet"] = {
        name: value for name, value in zip(SENSOR_PACKET_NAMES, values)
    }
    metadata["practical_sensor_provenance"] = {
        "angular_rate_source": "derived from locally fitted roll/pitch/yaw",
        "acceleration_source": "derived from locally fitted NED velocity",
    }
    packet, provenance, covered = build_packet(
        metadata,
        sbg,
        half_window_seconds=0.025,
        gyro_full_scale=4.0,
        accel_full_scale=19.62,
        outside_sbg_policy="ins-derived",
    )
    assert not covered
    assert packet[0] == 0.10
    assert packet[CONTEXT_START - 1] == -0.20
    assert packet[CONTEXT_START + 13] == 0.65
    assert "derived" in provenance["angular_rate_source"]
    assert provenance["direct_sbg_available"] is False
