"""Tests for source parsing and stream reader helpers."""

from __future__ import annotations

from src.config import infer_camera_id, infer_source_type, load_camera_settings, parse_source_value, redact_source_uri
from src.video.stream_reader import StreamReader


def test_parse_source_value_handles_webcam_index() -> None:
    assert parse_source_value("0") == 0
    assert parse_source_value(1) == 1
    assert parse_source_value("data/input_videos/sample.mp4") == "data/input_videos/sample.mp4"


def test_infer_source_type_handles_rtsp_and_http() -> None:
    assert infer_source_type("rtsp://example.com/live") == "rtsp"
    assert infer_source_type("https://example.com/live.m3u8") == "http"
    assert infer_source_type("0") == "webcam"


def test_source_redaction_and_camera_id_helpers() -> None:
    assert redact_source_uri("rtsp://user:secret@example.com:554/live") == "rtsp://***@example.com:554/live"
    assert redact_source_uri("data/input_videos/sample.mp4") == "data/input_videos/sample.mp4"
    assert infer_camera_id("data/input_videos/sample.mp4", "file") == "sample"
    assert infer_camera_id(0, "webcam") == "webcam_0"


def test_stream_reader_source_exists_for_live_and_file_sources() -> None:
    assert StreamReader(source="rtsp://example.com/live", source_type="rtsp").source_exists()
    assert StreamReader(source="data/input_videos/sample.mp4", source_type="file").source_exists()
    assert not StreamReader(source="data/input_videos/missing.mp4", source_type="file").source_exists()


def test_load_camera_settings_allows_disabled_rtsp_placeholder() -> None:
    cameras = load_camera_settings("configs/cameras.yaml")
    assert "demo_platform" in cameras
    assert "station_rtsp_example" in cameras
    assert cameras["station_rtsp_example"].enabled is False
