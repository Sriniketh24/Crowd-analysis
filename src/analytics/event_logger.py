"""API-independent event logger for writing analytics to SQLite."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from src.analytics.database import (
    CameraRegistry,
    CrowdAlert,
    FrameProcessed,
    LineCrossingEvent,
    RunSession,
    StreamHealthEvent,
    ZoneOccupancy,
    db_session,
    get_session_factory,
    init_db,
)
from src.vision.line_counter import LineCrossing
from src.vision.detector import normalize_detector_mode


class AnalyticsLogger:
    """Writes pipeline analytics to SQLite at configurable snapshot intervals."""

    def __init__(
        self,
        db_url: str,
        camera_id: str,
        snapshot_interval_secs: float = 5.0,
        detector_mode: str = "body",
    ) -> None:
        self._camera_id = camera_id
        self._snapshot_interval = snapshot_interval_secs
        self._detector_mode = normalize_detector_mode(detector_mode)

        engine = init_db(db_url)
        self._Session = get_session_factory(engine)

        self._run_session_id: int | None = None
        self._last_snapshot_time: float = 0.0
        self._prev_alert_levels: dict[str, str] = {}
        self._unique_track_ids_seen: set[int] = set()

    def register_camera(
        self,
        *,
        camera_name: str,
        source_type: str,
        description: str = "",
        zones_config_path: str | None = None,
        enabled: bool = True,
    ) -> None:
        """Insert or update camera registry metadata."""
        with db_session(self._Session) as session:
            row = session.get(CameraRegistry, self._camera_id)
            if row is None:
                row = CameraRegistry(
                    camera_id=self._camera_id,
                    name=camera_name,
                    source_type=source_type,
                    description=description or None,
                    zones_config_path=zones_config_path,
                    enabled=1 if enabled else 0,
                )
                session.add(row)
            else:
                row.name = camera_name
                row.source_type = source_type
                row.description = description or None
                row.zones_config_path = zones_config_path
                row.enabled = 1 if enabled else 0

    def start_session(
        self,
        source: str,
        *,
        source_type: str = "file",
        model_weights: str | None = None,
        detector_mode: str | None = None,
        tracker_type: str | None = None,
        zones_config_path: str | None = None,
    ) -> int:
        """Open a new run session and return its database ID."""
        if detector_mode is not None:
            self._detector_mode = normalize_detector_mode(detector_mode)
        with db_session(self._Session) as session:
            run = RunSession(
                camera_id=self._camera_id,
                source=source,
                source_type=source_type,
                model_weights=model_weights,
                detector_mode=self._detector_mode,
                tracker_type=tracker_type,
                zones_config_path=zones_config_path,
                started_at=datetime.now(timezone.utc),
            )
            session.add(run)
            session.flush()
            self._run_session_id = run.id

        self._last_snapshot_time = time.monotonic()
        self._prev_alert_levels = {}
        self._unique_track_ids_seen = set()
        self.log_stream_health(status="started", reconnect_count=0, dropped_frames=0, message=None)
        return self._run_session_id  # type: ignore[return-value]

    def end_session(self, total_frames: int, *, total_unique_passengers: int | None = None) -> None:
        """Close the active run session, recording end time and frame count."""
        if self._run_session_id is None:
            return
        with db_session(self._Session) as session:
            run = session.get(RunSession, self._run_session_id)
            if run is not None:
                run.ended_at = datetime.now(timezone.utc)
                run.total_frames = total_frames
                run.total_unique_passengers = (
                    total_unique_passengers
                    if total_unique_passengers is not None
                    else len(self._unique_track_ids_seen)
                )
        self.log_stream_health(
            status="completed",
            reconnect_count=0,
            dropped_frames=0,
            message=f"Processed {total_frames} frames.",
        )
        self._run_session_id = None

    def log_stream_health(
        self,
        *,
        status: str,
        reconnect_count: int,
        dropped_frames: int,
        message: str | None,
    ) -> None:
        """Persist a stream-health status row."""
        if self._run_session_id is None:
            return
        with db_session(self._Session) as session:
            session.add(
                StreamHealthEvent(
                    run_session_id=self._run_session_id,
                    camera_id=self._camera_id,
                    status=status,
                    reconnect_count=reconnect_count,
                    dropped_frames=dropped_frames,
                    message=message,
                )
            )

    def log_frame(
        self,
        frame_index: int,
        source_timestamp: float,
        total_detections: int,
        zone_occupancy: dict[str, int],
        zone_alerts: dict[str, str],
        line_counts: dict[str, dict[str, int]],
        *,
        line_events: list[LineCrossing] | None = None,
        track_ids: list[int] | None = None,
        processing_fps: float = 0.0,
        source_type: str = "file",
        reconnect_count: int = 0,
        dropped_frames: int = 0,
        detector_mode: str | None = None,
    ) -> None:
        """Log one processed frame and derived analytics events."""
        if self._run_session_id is None:
            raise RuntimeError("call start_session() before log_frame()")
        mode = normalize_detector_mode(detector_mode or self._detector_mode)

        now = datetime.now(timezone.utc)
        rows: list = []

        valid_track_ids = [track_id for track_id in (track_ids or []) if track_id >= 0]
        self._unique_track_ids_seen.update(valid_track_ids)
        rows.append(
            FrameProcessed(
                run_session_id=self._run_session_id,
                camera_id=self._camera_id,
                frame_index=frame_index,
                source_timestamp=source_timestamp,
                source_type=source_type,
                detector_mode=mode,
                recorded_at=now,
                total_detections=total_detections,
                tracked_objects=len(valid_track_ids),
                unique_passengers_seen=len(self._unique_track_ids_seen),
                processing_fps=processing_fps,
                reconnect_count=reconnect_count,
                dropped_frames=dropped_frames,
            )
        )

        elapsed = time.monotonic() - self._last_snapshot_time
        if elapsed >= self._snapshot_interval:
            for zone_id, count in zone_occupancy.items():
                rows.append(
                    ZoneOccupancy(
                        run_session_id=self._run_session_id,
                        camera_id=self._camera_id,
                        detector_mode=mode,
                        zone_id=zone_id,
                        frame_index=frame_index,
                        count=count,
                        alert_level=zone_alerts.get(zone_id, "NORMAL"),
                        recorded_at=now,
                    )
                )
            self._last_snapshot_time = time.monotonic()

        for event in line_events or []:
            rows.append(
                LineCrossingEvent(
                    run_session_id=self._run_session_id,
                    camera_id=self._camera_id,
                    detector_mode=mode,
                    line_id=event.line_id,
                    line_name=event.line_name,
                    track_id=event.track_id,
                    direction=event.direction,
                    frame_index=event.frame_index,
                    recorded_at=now,
                )
            )

        for zone_id, level in zone_alerts.items():
            prev_level = self._prev_alert_levels.get(zone_id)
            if level != prev_level:
                rows.append(
                    CrowdAlert(
                        run_session_id=self._run_session_id,
                        camera_id=self._camera_id,
                        detector_mode=mode,
                        zone_id=zone_id,
                        alert_level=level,
                        occupancy=zone_occupancy.get(zone_id, 0),
                        frame_index=frame_index,
                        recorded_at=now,
                    )
                )
                self._prev_alert_levels[zone_id] = level

        with db_session(self._Session) as session:
            session.add_all(rows)

        if reconnect_count > 0 or dropped_frames > 0:
            self.log_stream_health(
                status="streaming",
                reconnect_count=reconnect_count,
                dropped_frames=dropped_frames,
                message=None,
            )
