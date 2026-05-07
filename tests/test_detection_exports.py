from __future__ import annotations

# костыль, так как jetson не работает с python 3.11
try:
    from datetime import UTC, datetime
except ImportError:
    from datetime import timezone
    UTC = timezone.utc

from fire_uav.module_core.schema import TelemetrySample
from fire_uav.services.detection_exports import (
    make_confirmed_detection_log_entry,
    make_raw_detection_log_entry,
)


def _telemetry() -> TelemetrySample:
    return TelemetrySample(
        lat=55.0,
        lon=37.0,
        alt=120.0,
        alt_agl=30.0,
        yaw=90.0,
        pitch=-15.0,
        roll=1.5,
        battery=0.75,
        camera_mount_pitch_deg=32.0,
        timestamp=datetime(2026, 3, 11, 12, 0, 0, tzinfo=UTC),
        source="sim",
    )


def test_make_raw_detection_log_entry_adds_structured_summary_and_telemetry() -> None:
    entry = make_raw_detection_log_entry(
        timestamp="2026-03-11T12:00:01Z",
        count=1,
        best_confidence=0.91,
        best_bbox=[10, 20, 30, 40],
        classes=[2],
        detections=[
            {
                "class_id": 2,
                "confidence": 0.91,
                "bbox": [10, 20, 30, 40],
                "camera_id": "cam0",
                "timestamp": "2026-03-11T12:00:01Z",
            }
        ],
        telemetry=_telemetry(),
    )

    assert entry["count"] == 1
    assert entry["summary"]["best_detection"]["confidence"] == 0.91
    assert entry["telemetry"]["position"]["lat"] == 55.0
    assert entry["telemetry"]["orientation"]["yaw"] == 90.0


def test_make_confirmed_detection_log_entry_matches_normalized_contract() -> None:
    entry = make_confirmed_detection_log_entry(
        {
            "object_id": "track-7",
            "class_id": 2,
            "confidence": 0.88,
            "lat": 55.123,
            "lon": 37.456,
            "track_id": 7,
            "timestamp": datetime(2026, 3, 11, 12, 0, 5, tzinfo=UTC),
        },
        telemetry=_telemetry(),
    )

    assert entry["object"]["id"] == "track-7"
    assert entry["object"]["class_id"] == 2
    assert entry["geolocation"]["lat"] == 55.123
    assert entry["confidence"] == 0.88
    assert entry["time"]["detected_at"] == "2026-03-11T12:00:05+00:00"
    assert entry["telemetry"]["position"]["lon"] == 37.0
