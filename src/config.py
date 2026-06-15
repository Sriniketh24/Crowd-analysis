"""Shared configuration helpers for app, cameras, and runtime settings."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml

ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _expand_env(value: object) -> object:
    """Expand ``${VAR}`` placeholders recursively within config data."""
    if isinstance(value, str):
        return ENV_PATTERN.sub(lambda match: os.getenv(match.group(1), ""), value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _expand_env(item) for key, item in value.items()}
    return value


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict, returning an empty dict when missing."""
    config_path = Path(path)
    if not config_path.exists():
        return {}
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in YAML config: {config_path}")
    expanded = _expand_env(loaded)
    if not isinstance(expanded, dict):
        raise ValueError(f"Expanded YAML config is not a mapping: {config_path}")
    return expanded


def parse_source_value(raw_source: str | int | None) -> str | int | None:
    """Convert numeric webcam strings into integers."""
    if raw_source is None:
        return None
    if isinstance(raw_source, int):
        return raw_source
    stripped = str(raw_source).strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped


def infer_source_type(source: str | int | None) -> str:
    """Infer source type from the source value."""
    if isinstance(source, int):
        return "webcam"
    source_str = str(source or "").strip().lower()
    if source_str.startswith("rtsp://"):
        return "rtsp"
    if source_str.startswith("http://") or source_str.startswith("https://"):
        return "http"
    if source_str.isdigit():
        return "webcam"
    return "file"


def infer_camera_id(source: str | int | None, source_type: str | None = None) -> str:
    """Build a stable non-secret camera ID for ad-hoc sources."""
    resolved_source_type = (source_type or infer_source_type(source)).lower()
    if resolved_source_type == "file":
        stem = Path(str(source or "")).stem
        return stem or "file_camera"
    if resolved_source_type == "webcam":
        return f"webcam_{source}"
    return "ad_hoc_camera"


def redact_source_uri(source: str | int | None) -> str:
    """Return a display/log-safe source string with URL credentials redacted."""
    if source is None:
        return ""
    if isinstance(source, int):
        return str(source)

    source_str = str(source)
    source_type = infer_source_type(source_str)
    if source_type not in {"rtsp", "http"}:
        return source_str

    try:
        parsed = urlsplit(source_str)
    except ValueError:
        return source_str
    if "@" not in parsed.netloc:
        return source_str

    host = parsed.hostname or ""
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    redacted_netloc = f"***@{host}" if host else "***"
    return urlunsplit((parsed.scheme, redacted_netloc, parsed.path, parsed.query, parsed.fragment))


@dataclass(slots=True)
class ModelSettings:
    """Model settings loaded from ``configs/app.yaml``."""

    weights: str = "yolo11n.pt"
    accuracy_weights: str = "yolo11s.pt"
    legacy_fallback_weights: str = "yolov8n.pt"
    device: str = "cpu"
    confidence: float = 0.35
    iou: float = 0.5
    person_class_id: int = 0
    imgsz: int = 640
    augment: bool = False
    max_det: int = 300
    head_confidence: float = 0.15
    head_imgsz: int = 1536
    head_max_det: int = 1000
    half: bool = False
    profile: str = "cpu_demo"
    use_fine_tuned_if_available: bool = True


@dataclass(slots=True)
class TrackerSettings:
    """Tracker settings loaded from ``configs/app.yaml``."""

    type: str = "bytetrack"
    config_overrides: dict[str, Any] = field(default_factory=dict)
    head_config_overrides: dict[str, Any] = field(default_factory=dict)
    head_stitching: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeSettings:
    """Runtime settings loaded from ``configs/app.yaml``."""

    frame_stride: int = 1
    max_fps: float = 0.0
    resize_width: int | None = None
    reconnect_backoff_seconds: float = 2.0
    max_reconnect_attempts: int = 3
    stream_read_timeout_seconds: float = 15.0


@dataclass(slots=True)
class PersistenceSettings:
    """Persistence settings loaded from ``configs/app.yaml``."""

    db_url: str = "sqlite:///data/outputs/analytics.db"
    snapshot_interval_seconds: float = 5.0


@dataclass(slots=True)
class OutputSettings:
    """Output settings loaded from ``configs/app.yaml``."""

    default_output_video: str = "data/outputs/demo.mp4"
    default_report_csv: str = "data/outputs/report.csv"


@dataclass(slots=True)
class AppSettings:
    """Structured application settings."""

    app_name: str = "railway-crowd-analytics"
    environment: str = "development"
    log_level: str = "INFO"
    model: ModelSettings = field(default_factory=ModelSettings)
    tracker: TrackerSettings = field(default_factory=TrackerSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    persistence: PersistenceSettings = field(default_factory=PersistenceSettings)
    outputs: OutputSettings = field(default_factory=OutputSettings)


@dataclass(slots=True)
class CameraSettings:
    """Structured camera/source configuration."""

    camera_id: str
    name: str
    source: str | int
    source_type: str
    enabled: bool = True
    description: str = ""
    zones_config: str = "configs/zones.example.json"
    threshold_profile: str | None = None
    frame_width: int | None = None
    frame_height: int | None = None
    fps_limit: float | None = None
    frame_stride: int | None = None
    store_output_video: bool = True


def load_app_settings(path: str | Path = "configs/app.yaml") -> AppSettings:
    """Load and normalize app settings."""
    raw = load_yaml_file(path)
    model_raw = raw.get("model", {})
    tracker_raw = raw.get("tracker", {})
    runtime_raw = raw.get("runtime", {})
    persistence_raw = raw.get("persistence", {})
    outputs_raw = raw.get("outputs", {})
    app_raw = raw.get("app", {})

    tracker_overrides = dict(tracker_raw.get("config_overrides", {}))
    if "track_activation_threshold" in tracker_raw:
        tracker_overrides.setdefault("track_high_thresh", tracker_raw["track_activation_threshold"])
        tracker_overrides.setdefault("new_track_thresh", tracker_raw["track_activation_threshold"])
    if "lost_track_buffer" in tracker_raw:
        tracker_overrides.setdefault("track_buffer", tracker_raw["lost_track_buffer"])
    if "minimum_matching_threshold" in tracker_raw:
        tracker_overrides.setdefault("match_thresh", tracker_raw["minimum_matching_threshold"])

    return AppSettings(
        app_name=str(app_raw.get("name", "railway-crowd-analytics")),
        environment=str(app_raw.get("environment", "development")),
        log_level=str(app_raw.get("log_level", "INFO")),
        model=ModelSettings(
            weights=str(model_raw.get("weights", "yolo11n.pt")),
            accuracy_weights=str(model_raw.get("accuracy_weights", "yolo11s.pt")),
            legacy_fallback_weights=str(model_raw.get("legacy_fallback_weights", "yolov8n.pt")),
            device=str(model_raw.get("device", "cpu")),
            confidence=float(model_raw.get("confidence", 0.35)),
            iou=float(model_raw.get("iou", 0.5)),
            person_class_id=int(model_raw.get("person_class_id", 0)),
            imgsz=int(model_raw.get("imgsz", 640)),
            augment=bool(model_raw.get("augment", False)),
            max_det=int(model_raw.get("max_det", 300)),
            head_confidence=float(model_raw.get("head_confidence", 0.15)),
            head_imgsz=int(model_raw.get("head_imgsz", 1536)),
            head_max_det=int(model_raw.get("head_max_det", 1000)),
            half=bool(model_raw.get("half", False)),
            profile=str(model_raw.get("profile", "cpu_demo")),
            use_fine_tuned_if_available=bool(model_raw.get("use_fine_tuned_if_available", True)),
        ),
        tracker=TrackerSettings(
            type=str(tracker_raw.get("type", "bytetrack")).strip().lower(),
            config_overrides=tracker_overrides,
            head_config_overrides=dict(tracker_raw.get("head_config_overrides", {})),
            head_stitching=dict(tracker_raw.get("head_stitching", {})),
        ),
        runtime=RuntimeSettings(
            frame_stride=max(1, int(runtime_raw.get("frame_stride", 1))),
            max_fps=float(runtime_raw.get("max_fps", 0.0)),
            resize_width=(
                int(runtime_raw["resize_width"])
                if runtime_raw.get("resize_width") not in (None, "", 0)
                else None
            ),
            reconnect_backoff_seconds=float(runtime_raw.get("reconnect_backoff_seconds", 2.0)),
            max_reconnect_attempts=max(0, int(runtime_raw.get("max_reconnect_attempts", 3))),
            stream_read_timeout_seconds=float(runtime_raw.get("stream_read_timeout_seconds", 15.0)),
        ),
        persistence=PersistenceSettings(
            db_url=str(persistence_raw.get("db_url", "sqlite:///data/outputs/analytics.db")),
            snapshot_interval_seconds=float(
                persistence_raw.get("snapshot_interval_seconds", 5.0)
            ),
        ),
        outputs=OutputSettings(
            default_output_video=str(outputs_raw.get("default_output_video", "data/outputs/demo.mp4")),
            default_report_csv=str(outputs_raw.get("default_report_csv", "data/outputs/report.csv")),
        ),
    )


def load_camera_settings(path: str | Path = "configs/cameras.yaml") -> dict[str, CameraSettings]:
    """Load configured cameras keyed by camera ID."""
    raw = load_yaml_file(path)
    cameras_raw = raw.get("cameras", [])
    if not isinstance(cameras_raw, list):
        raise ValueError("Config field 'cameras' must be a list.")
    cameras: dict[str, CameraSettings] = {}
    for idx, item in enumerate(cameras_raw):
        if not isinstance(item, dict):
            raise ValueError(f"Camera entry at index {idx} must be an object.")
        camera_id = str(item.get("id") or item.get("camera_id") or f"camera_{idx + 1}")
        source = parse_source_value(item.get("source"))
        enabled = bool(item.get("enabled", True))
        if source in (None, "") and enabled:
            raise ValueError(f"Camera '{camera_id}' is missing a source.")
        source_type = str(item.get("source_type") or infer_source_type(source)).lower()
        privacy = item.get("privacy", {})
        store_output_video = True
        if isinstance(privacy, dict):
            store_output_video = bool(privacy.get("store_output_video", True))
        cameras[camera_id] = CameraSettings(
            camera_id=camera_id,
            name=str(item.get("name", camera_id)),
            source=source if source not in (None, "") else "",
            source_type=source_type,
            enabled=enabled,
            description=str(item.get("description", "")),
            zones_config=str(item.get("zones_config", "configs/zones.example.json")),
            threshold_profile=(
                str(item["threshold_profile"]) if item.get("threshold_profile") not in (None, "") else None
            ),
            frame_width=int(item["frame_width"]) if item.get("frame_width") else None,
            frame_height=int(item["frame_height"]) if item.get("frame_height") else None,
            fps_limit=float(item["fps_limit"]) if item.get("fps_limit") not in (None, "") else None,
            frame_stride=(
                max(1, int(item["frame_stride"])) if item.get("frame_stride") not in (None, "") else None
            ),
            store_output_video=store_output_video,
        )
    return cameras


def choose_camera(
    cameras: dict[str, CameraSettings],
    camera_id: str | None = None,
) -> CameraSettings | None:
    """Select a configured camera by ID or return the first enabled camera."""
    if camera_id:
        return cameras.get(camera_id)
    for camera in cameras.values():
        if camera.enabled:
            return camera
    return None
