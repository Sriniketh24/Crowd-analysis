"""Metric calculation helpers for flow and occupancy summaries."""

from __future__ import annotations

from dataclasses import dataclass


def calculate_flow_rate(count: int, window_seconds: int) -> float:
    """Compute events per minute for a count over a time window."""
    if window_seconds <= 0:
        return 0.0
    return (count / window_seconds) * 60.0


@dataclass(slots=True)
class ZoneMetrics:
    """Aggregated metrics for one zone."""

    zone_id: str
    occupancy: int
    unique_tracks_seen: int
    alert_level: str
    recorded_at: str | None = None


@dataclass(slots=True)
class LineMetrics:
    """Aggregated directional line metrics."""

    line_name: str
    in_label: str
    out_label: str
    in_count: int
    out_count: int


@dataclass(slots=True)
class CameraMetrics:
    """Per-camera summary metrics for latest processed frame."""

    camera_id: str
    source_type: str
    total_detections: int
    unique_passengers_seen: int
    processing_fps: float
    reconnect_count: int
