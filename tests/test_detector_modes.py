"""Tests for body/head detector mode integration."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts import run_video_demo
from src.vision.detection_filter import DetectionRegionFilter
from src.vision.detector import DEFAULT_HEAD_MODEL_PATH, normalize_detector_mode
from src.vision.line_counter import LineConfig, LineManager
from src.vision.zone_manager import (
    ZoneConfig,
    ZoneManager,
    bottom_center_from_bbox,
    center_from_bbox,
    count_zone_occupancy,
)


def test_detector_mode_validation() -> None:
    """Only body/head modes should be accepted."""
    assert normalize_detector_mode("body") == "body"
    assert normalize_detector_mode(" HEAD ") == "head"
    with pytest.raises(ValueError, match="body.*head"):
        normalize_detector_mode("faces")


def test_run_video_demo_parses_detector_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """The video demo CLI should expose detector-mode parsing."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_video_demo.py",
            "--source",
            "data/input_videos/sample.mp4",
            "--output",
            "data/outputs/head_demo.mp4",
            "--zones-config",
            "configs/zones.example.json",
            "--db",
            "data/outputs/head_analytics.db",
            "--detector-mode",
            "head",
        ],
    )
    args = run_video_demo.parse_args()
    assert args.detector_mode == "head"


def test_missing_head_model_message_mentions_expected_path() -> None:
    """Head mode should fail with a clear expected-path message."""
    message = run_video_demo.validate_head_model("models/fine_tuned/head_detector/weights/missing.pt")
    assert message is not None
    assert str(DEFAULT_HEAD_MODEL_PATH) in message
    assert "Do not train locally" in message


def test_body_point_strategy_uses_bottom_center() -> None:
    """Full-body analytics should use the bottom-center of the box."""
    bbox = (2.0, 2.0, 6.0, 10.0)
    assert bottom_center_from_bbox(bbox) == (4.0, 10.0)

    zone_polygon = [[0, 8], [8, 8], [8, 12], [0, 12]]
    tracks = [{"track_id": 1, "bbox": bbox}]
    assert count_zone_occupancy(tracks, zone_polygon, point_strategy="bottom_center") == 1
    assert count_zone_occupancy(tracks, zone_polygon, point_strategy="center") == 0


def test_head_point_strategy_uses_center_for_zones_and_lines() -> None:
    """Head analytics should use bbox center for occupancy and crossings."""
    bbox = (2.0, -4.0, 6.0, 0.0)
    assert center_from_bbox(bbox) == (4.0, -2.0)

    manager = ZoneManager(
        zones=[
            ZoneConfig(
                id="head_zone",
                name="Head Zone",
                polygon=[(0.0, -3.0), (8.0, -3.0), (8.0, -1.0), (0.0, -1.0)],
                warning_threshold=1,
                critical_threshold=2,
            )
        ],
        point_strategy="center",
    )
    assert manager.update([{"track_id": 1, "bbox": bbox}])["head_zone"] == 1

    line = LineManager(
        config=LineConfig(id="gate", name="Gate", start=(0.0, 0.0), end=(10.0, 0.0)),
        point_strategy="center",
    )
    line.update([{"track_id": 1, "bbox": (2.0, -4.0, 6.0, 0.0)}])
    counts = line.update([{"track_id": 1, "bbox": (2.0, 2.0, 6.0, 6.0)}])
    assert counts["IN"] == 1


def test_comparison_script_help_command() -> None:
    """The comparison script should expose a working help command."""
    completed = subprocess.run(
        [sys.executable, "scripts/run_comparison_demo.py", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--head-model" in completed.stdout
    assert "--body-model" in completed.stdout


def test_cctv_head_filter_removes_train_false_positive() -> None:
    """The CCTV head filter should reject train/track-region false positives."""
    detection_filter = DetectionRegionFilter.from_zones_file(
        "configs/zones.cctv_platform.example.json",
        detector_mode="head",
    )
    assert detection_filter is not None

    train_front_false_positive = {
        "track_id": 9,
        "bbox": (946.0, 468.0, 976.0, 497.0),
        "confidence": 0.37,
        "class_name": "head/passenger",
        "frame_index": 267,
        "timestamp": 10.68,
        "detector_mode": "head",
    }
    valid_platform_head = {
        "track_id": 394,
        "bbox": (252.0, 559.0, 276.0, 585.0),
        "confidence": 0.64,
        "class_name": "head/passenger",
        "frame_index": 267,
        "timestamp": 10.68,
        "detector_mode": "head",
    }

    from src.vision.detector import NormalizedDetection

    false_detection = NormalizedDetection(**train_front_false_positive)
    valid_detection = NormalizedDetection(**valid_platform_head)
    assert not detection_filter.keep(false_detection)
    assert detection_filter.keep(valid_detection)
