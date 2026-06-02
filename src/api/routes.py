"""Route registration helpers for the FastAPI application."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func

from src.analytics.database import (
    CameraRegistry,
    CrowdAlert,
    FrameProcessed,
    LineCrossingEvent,
    RunSession,
    StreamHealthEvent,
    ZoneOccupancy,
    get_session_factory,
    init_db,
)
from src.api.schemas import (
    AlertItemResponse,
    AlertsResponse,
    CameraCountsResponse,
    CameraHealthResponse,
    CameraSummaryResponse,
    HealthResponse,
    LatestMetricsResponse,
    LineMetricResponse,
    ProcessVideoRequest,
    ProcessVideoResponse,
    SessionItemResponse,
    SessionsResponse,
    ZoneMetricResponse,
)

router = APIRouter()
DEFAULT_DB_PATH = Path("data/outputs/analytics.db")


def _db_path_or_default(db_path: str | None) -> Path:
    return Path(db_path) if db_path else DEFAULT_DB_PATH


def _session_factory(db_path: str | None):
    resolved = _db_path_or_default(db_path)
    if resolved.suffix.lower() != ".db":
        raise HTTPException(status_code=400, detail=f"Database path must end with '.db': {resolved}")
    engine = init_db(f"sqlite:///{resolved.resolve()}")
    return get_session_factory(engine)


def _latest_session_for_camera(session, camera_id: str | None = None):
    query = session.query(RunSession)
    if camera_id:
        query = query.filter(RunSession.camera_id == camera_id)
    return query.order_by(RunSession.started_at.desc(), RunSession.id.desc()).first()


def _latest_frame_for_session(session, session_id: int):
    return (
        session.query(FrameProcessed)
        .filter(FrameProcessed.run_session_id == session_id)
        .order_by(FrameProcessed.frame_index.desc())
        .first()
    )


def _latest_zone_rows(session, session_id: int) -> list[ZoneOccupancy]:
    rows = (
        session.query(ZoneOccupancy)
        .filter(ZoneOccupancy.run_session_id == session_id)
        .order_by(ZoneOccupancy.frame_index.desc(), ZoneOccupancy.recorded_at.desc())
        .all()
    )
    latest_by_zone: dict[str, ZoneOccupancy] = {}
    for row in rows:
        if row.zone_id not in latest_by_zone:
            latest_by_zone[row.zone_id] = row
    return list(latest_by_zone.values())


def _line_counts_for_session(session, session_id: int) -> list[LineMetricResponse]:
    rows = (
        session.query(
            LineCrossingEvent.line_id,
            LineCrossingEvent.direction,
            func.count(LineCrossingEvent.id),
        )
        .filter(LineCrossingEvent.run_session_id == session_id)
        .group_by(LineCrossingEvent.line_id, LineCrossingEvent.direction)
        .all()
    )
    return [
        LineMetricResponse(line_id=line_id, direction=direction, count=count)
        for line_id, direction, count in rows
    ]


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/cameras", response_model=list[CameraSummaryResponse])
def get_cameras(db_path: str | None = Query(default=None)) -> list[CameraSummaryResponse]:
    Session = _session_factory(db_path)
    with Session() as session:
        rows = session.query(CameraRegistry).order_by(CameraRegistry.camera_id.asc()).all()
        return [
            CameraSummaryResponse(
                camera_id=row.camera_id,
                name=row.name,
                source_type=row.source_type,
                description=row.description,
                zones_config_path=row.zones_config_path,
                enabled=bool(row.enabled),
            )
            for row in rows
        ]


@router.get("/cameras/{camera_id}/health", response_model=CameraHealthResponse)
def get_camera_health(camera_id: str, db_path: str | None = Query(default=None)) -> CameraHealthResponse:
    Session = _session_factory(db_path)
    with Session() as session:
        row = (
            session.query(StreamHealthEvent)
            .filter(StreamHealthEvent.camera_id == camera_id)
            .order_by(StreamHealthEvent.recorded_at.desc(), StreamHealthEvent.id.desc())
            .first()
        )
        if row is None:
            raise HTTPException(status_code=404, detail=f"No health data found for camera '{camera_id}'.")
        return CameraHealthResponse(
            camera_id=row.camera_id,
            status=row.status,
            reconnect_count=row.reconnect_count,
            dropped_frames=row.dropped_frames,
            message=row.message,
            recorded_at=row.recorded_at,
        )


@router.get("/metrics/latest", response_model=LatestMetricsResponse)
def get_latest_metrics(db_path: str | None = Query(default=None)) -> LatestMetricsResponse:
    Session = _session_factory(db_path)
    with Session() as session:
        latest_session = _latest_session_for_camera(session)
        if latest_session is None:
            raise HTTPException(status_code=404, detail="No processing sessions found. Run /process-video first.")
        latest_frame = _latest_frame_for_session(session, latest_session.id)
        if latest_frame is None:
            raise HTTPException(status_code=404, detail=f"No frame metrics found for session {latest_session.id}.")
        zones = _latest_zone_rows(session, latest_session.id)
        return LatestMetricsResponse(
            session_id=latest_session.id,
            camera_id=latest_frame.camera_id,
            source_type=latest_frame.source_type,
            frame_index=latest_frame.frame_index,
            source_timestamp=latest_frame.source_timestamp,
            recorded_at=latest_frame.recorded_at,
            total_detections=latest_frame.total_detections,
            unique_passengers_seen=latest_frame.unique_passengers_seen,
            processing_fps=latest_frame.processing_fps,
            reconnect_count=latest_frame.reconnect_count,
            zones=[
                ZoneMetricResponse(
                    zone_id=row.zone_id,
                    occupancy=row.count,
                    alert_level=row.alert_level,
                    frame_index=row.frame_index,
                    recorded_at=row.recorded_at,
                )
                for row in zones
            ],
            line_counts=_line_counts_for_session(session, latest_session.id),
        )


@router.get("/metrics/zones", response_model=list[ZoneMetricResponse])
def get_zone_metrics(db_path: str | None = Query(default=None)) -> list[ZoneMetricResponse]:
    Session = _session_factory(db_path)
    with Session() as session:
        latest_session = _latest_session_for_camera(session)
        if latest_session is None:
            raise HTTPException(status_code=404, detail="No sessions found. Process a video first.")
        return [
            ZoneMetricResponse(
                zone_id=row.zone_id,
                occupancy=row.count,
                alert_level=row.alert_level,
                frame_index=row.frame_index,
                recorded_at=row.recorded_at,
            )
            for row in _latest_zone_rows(session, latest_session.id)
        ]


@router.get("/cameras/{camera_id}/latest", response_model=LatestMetricsResponse)
def get_camera_latest(camera_id: str, db_path: str | None = Query(default=None)) -> LatestMetricsResponse:
    Session = _session_factory(db_path)
    with Session() as session:
        latest_session = _latest_session_for_camera(session, camera_id=camera_id)
        if latest_session is None:
            raise HTTPException(status_code=404, detail=f"No sessions found for camera '{camera_id}'.")
        latest_frame = _latest_frame_for_session(session, latest_session.id)
        if latest_frame is None:
            raise HTTPException(status_code=404, detail=f"No frame metrics found for camera '{camera_id}'.")
        zones = _latest_zone_rows(session, latest_session.id)
        return LatestMetricsResponse(
            session_id=latest_session.id,
            camera_id=latest_frame.camera_id,
            source_type=latest_frame.source_type,
            frame_index=latest_frame.frame_index,
            source_timestamp=latest_frame.source_timestamp,
            recorded_at=latest_frame.recorded_at,
            total_detections=latest_frame.total_detections,
            unique_passengers_seen=latest_frame.unique_passengers_seen,
            processing_fps=latest_frame.processing_fps,
            reconnect_count=latest_frame.reconnect_count,
            zones=[
                ZoneMetricResponse(
                    zone_id=row.zone_id,
                    occupancy=row.count,
                    alert_level=row.alert_level,
                    frame_index=row.frame_index,
                    recorded_at=row.recorded_at,
                )
                for row in zones
            ],
            line_counts=_line_counts_for_session(session, latest_session.id),
        )


@router.get("/cameras/{camera_id}/counts", response_model=CameraCountsResponse)
def get_camera_counts(camera_id: str, db_path: str | None = Query(default=None)) -> CameraCountsResponse:
    Session = _session_factory(db_path)
    with Session() as session:
        latest_session = _latest_session_for_camera(session, camera_id=camera_id)
        if latest_session is None:
            raise HTTPException(status_code=404, detail=f"No sessions found for camera '{camera_id}'.")
        latest_frame = _latest_frame_for_session(session, latest_session.id)
        return CameraCountsResponse(
            camera_id=camera_id,
            session_id=latest_session.id,
            total_unique_passengers=latest_session.total_unique_passengers or 0,
            total_detections_latest=latest_frame.total_detections if latest_frame is not None else 0,
            line_counts=_line_counts_for_session(session, latest_session.id),
        )


@router.get("/cameras/{camera_id}/occupancy", response_model=list[ZoneMetricResponse])
def get_camera_occupancy(camera_id: str, db_path: str | None = Query(default=None)) -> list[ZoneMetricResponse]:
    Session = _session_factory(db_path)
    with Session() as session:
        latest_session = _latest_session_for_camera(session, camera_id=camera_id)
        if latest_session is None:
            raise HTTPException(status_code=404, detail=f"No sessions found for camera '{camera_id}'.")
        return [
            ZoneMetricResponse(
                zone_id=row.zone_id,
                occupancy=row.count,
                alert_level=row.alert_level,
                frame_index=row.frame_index,
                recorded_at=row.recorded_at,
            )
            for row in _latest_zone_rows(session, latest_session.id)
        ]


@router.get("/cameras/{camera_id}/alerts", response_model=AlertsResponse)
def get_camera_alerts(
    camera_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
    db_path: str | None = Query(default=None),
) -> AlertsResponse:
    Session = _session_factory(db_path)
    with Session() as session:
        rows = (
            session.query(CrowdAlert)
            .filter(CrowdAlert.camera_id == camera_id)
            .order_by(CrowdAlert.recorded_at.desc(), CrowdAlert.id.desc())
            .limit(limit)
            .all()
        )
        return AlertsResponse(
            alerts=[
                AlertItemResponse(
                    id=row.id,
                    session_id=row.run_session_id,
                    camera_id=row.camera_id,
                    zone_id=row.zone_id,
                    alert_level=row.alert_level,
                    occupancy=row.occupancy,
                    frame_index=row.frame_index,
                    recorded_at=row.recorded_at,
                )
                for row in rows
            ]
        )


@router.get("/alerts", response_model=AlertsResponse)
def get_alerts(limit: int = Query(default=100, ge=1, le=1000), db_path: str | None = Query(default=None)) -> AlertsResponse:
    Session = _session_factory(db_path)
    with Session() as session:
        rows = (
            session.query(CrowdAlert)
            .order_by(CrowdAlert.recorded_at.desc(), CrowdAlert.id.desc())
            .limit(limit)
            .all()
        )
        return AlertsResponse(
            alerts=[
                AlertItemResponse(
                    id=row.id,
                    session_id=row.run_session_id,
                    camera_id=row.camera_id,
                    zone_id=row.zone_id,
                    alert_level=row.alert_level,
                    occupancy=row.occupancy,
                    frame_index=row.frame_index,
                    recorded_at=row.recorded_at,
                )
                for row in rows
            ]
        )


@router.get("/sessions", response_model=SessionsResponse)
def get_sessions(limit: int = Query(default=50, ge=1, le=500), db_path: str | None = Query(default=None)) -> SessionsResponse:
    Session = _session_factory(db_path)
    with Session() as session:
        rows = (
            session.query(RunSession)
            .order_by(RunSession.started_at.desc(), RunSession.id.desc())
            .limit(limit)
            .all()
        )
        return SessionsResponse(
            sessions=[
                SessionItemResponse(
                    id=row.id,
                    camera_id=row.camera_id,
                    source=row.source,
                    source_type=row.source_type,
                    model_weights=row.model_weights,
                    tracker_type=row.tracker_type,
                    started_at=row.started_at,
                    ended_at=row.ended_at,
                    total_frames=row.total_frames,
                    total_unique_passengers=row.total_unique_passengers,
                )
                for row in rows
            ]
        )


@router.post("/process-video", response_model=ProcessVideoResponse)
@router.post("/analysis/run", response_model=ProcessVideoResponse)
def process_video(payload: ProcessVideoRequest) -> ProcessVideoResponse:
    from scripts.run_video_demo import run_pipeline

    zones_config_path = Path(payload.zones_config_path)
    output_path = Path(payload.output_path)
    db_path = Path(payload.db_path) if payload.db_path else None
    if not zones_config_path.exists():
        raise HTTPException(status_code=400, detail=f"Zones config not found: {zones_config_path}")

    args = Namespace(
        config=Path("configs/app.yaml"),
        camera_config=Path("configs/cameras.yaml"),
        camera_id=payload.camera_id,
        source=payload.video_path,
        source_type=payload.source_type,
        output=str(output_path),
        zones_config=zones_config_path,
        model=payload.model,
        device=None,
        tracker=payload.tracker,
        confidence=None,
        max_fps=None,
        frame_stride=None,
        resize_width=None,
        show=False,
        db=str(db_path) if db_path else None,
        snapshot_interval=None,
    )
    exit_code = run_pipeline(args)
    if exit_code != 0:
        raise HTTPException(status_code=400, detail="Video processing failed. Check source path/stream and zone config.")
    return ProcessVideoResponse(
        status="ok",
        message="Video processed successfully.",
        output_path=str(output_path),
        db_path=str(db_path) if db_path else None,
    )
