"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health endpoint response model."""

    status: str
    service: str = "railway-crowd-analytics"


class ZoneMetricResponse(BaseModel):
    """Latest occupancy metrics for a configured zone."""

    zone_id: str
    occupancy: int
    alert_level: str
    frame_index: int
    recorded_at: datetime


class LineMetricResponse(BaseModel):
    """Directional line crossing metrics summary."""

    line_id: str
    direction: str
    count: int


class LatestMetricsResponse(BaseModel):
    """Latest frame-level analytics snapshot."""

    session_id: int
    camera_id: str
    source_type: str
    frame_index: int
    source_timestamp: float
    recorded_at: datetime
    total_detections: int
    unique_passengers_seen: int
    processing_fps: float
    reconnect_count: int
    zones: list[ZoneMetricResponse]
    line_counts: list[LineMetricResponse]


class CameraSummaryResponse(BaseModel):
    """Configured camera metadata."""

    camera_id: str
    name: str
    source_type: str
    description: str | None = None
    zones_config_path: str | None = None
    enabled: bool


class CameraHealthResponse(BaseModel):
    """Latest camera health/status event."""

    camera_id: str
    status: str
    reconnect_count: int
    dropped_frames: int
    message: str | None = None
    recorded_at: datetime


class CameraCountsResponse(BaseModel):
    """Per-camera aggregate counts."""

    camera_id: str
    session_id: int | None
    total_unique_passengers: int
    total_detections_latest: int
    line_counts: list[LineMetricResponse]


class AlertItemResponse(BaseModel):
    """One persisted crowd alert record."""

    id: int
    session_id: int
    camera_id: str
    zone_id: str
    alert_level: str
    occupancy: int
    frame_index: int
    recorded_at: datetime


class AlertsResponse(BaseModel):
    """List of crowd alerts."""

    alerts: list[AlertItemResponse]


class SessionItemResponse(BaseModel):
    """One processing session persisted in analytics DB."""

    id: int
    camera_id: str
    source: str
    source_type: str
    model_weights: str | None = None
    tracker_type: str | None = None
    started_at: datetime
    ended_at: datetime | None
    total_frames: int | None
    total_unique_passengers: int | None


class SessionsResponse(BaseModel):
    """List of processing sessions."""

    sessions: list[SessionItemResponse]


class ProcessVideoRequest(BaseModel):
    """Request payload for running the local video processing pipeline."""

    video_path: str = Field(..., description="Local path, webcam index, or RTSP/HTTP URL.")
    zones_config_path: str = Field(..., description="Path to zone/line configuration JSON.")
    output_path: str = Field(..., description="Path for annotated output video.")
    db_path: str | None = Field(
        default=None,
        description="Optional path to SQLite database for analytics logging.",
    )
    camera_id: str | None = None
    source_type: str | None = None
    model: str | None = None
    tracker: str | None = None


class ProcessVideoResponse(BaseModel):
    """Response payload after pipeline processing is triggered."""

    status: str
    message: str
    output_path: str
    db_path: str | None = None
