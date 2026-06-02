"""Tests for the FastAPI backend.

These use an isolated temp SQLite DB via the ``db_path`` query parameter so the
project database is never touched. They confirm the API stays healthy and never
crashes (HTTP 500) on an empty database.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.analytics.database import CameraRegistry, RunSession, StreamHealthEvent, db_session, get_session_factory, init_db
from src.api.app import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    """/health must return 200 with a service status payload."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "railway-crowd-analytics"


def test_alerts_empty_db_returns_empty_list(tmp_path) -> None:
    """/alerts on a fresh DB returns valid JSON with no alerts (no crash)."""
    db = tmp_path / "empty.db"
    response = client.get("/alerts", params={"db_path": str(db)})
    assert response.status_code == 200
    assert response.json() == {"alerts": []}


def test_sessions_empty_db_returns_empty_list(tmp_path) -> None:
    """/sessions on a fresh DB returns valid JSON with no sessions (no crash)."""
    db = tmp_path / "empty.db"
    response = client.get("/sessions", params={"db_path": str(db)})
    assert response.status_code == 200
    assert response.json() == {"sessions": []}


def test_metrics_latest_empty_db_no_server_error(tmp_path) -> None:
    """/metrics/latest on an empty DB returns a clean 404, never a 500 crash."""
    db = tmp_path / "empty.db"
    response = client.get("/metrics/latest", params={"db_path": str(db)})
    assert response.status_code == 404
    assert "detail" in response.json()


def test_metrics_zones_empty_db_no_server_error(tmp_path) -> None:
    """/metrics/zones on an empty DB returns a clean response, never a 500 crash."""
    db = tmp_path / "empty.db"
    response = client.get("/metrics/zones", params={"db_path": str(db)})
    assert response.status_code in (200, 404)
    # Valid JSON either way; no server error.
    assert response.json() is not None


def test_invalid_db_path_suffix_rejected(tmp_path) -> None:
    """A db_path that is not a .db file is rejected with a helpful 400."""
    response = client.get("/alerts", params={"db_path": str(tmp_path / "bad.txt")})
    assert response.status_code == 400
    assert ".db" in response.json()["detail"]


def test_cameras_endpoint_returns_registry_rows(tmp_path) -> None:
    db = tmp_path / "api.db"
    Session = get_session_factory(init_db(f"sqlite:///{db}"))
    with db_session(Session) as session:
        session.add(
            CameraRegistry(
                camera_id="cam_demo",
                name="Demo Camera",
                source_type="file",
                description="demo",
                zones_config_path="configs/zones.example.json",
                enabled=1,
            )
        )
    response = client.get("/cameras", params={"db_path": str(db)})
    assert response.status_code == 200
    assert response.json()[0]["camera_id"] == "cam_demo"


def test_camera_health_endpoint_returns_latest_status(tmp_path) -> None:
    db = tmp_path / "api.db"
    Session = get_session_factory(init_db(f"sqlite:///{db}"))
    with db_session(Session) as session:
        run = RunSession(camera_id="cam_demo", source="sample.mp4", source_type="file")
        session.add(run)
        session.flush()
        session.add(
            StreamHealthEvent(
                run_session_id=run.id,
                camera_id="cam_demo",
                status="streaming",
                reconnect_count=1,
                dropped_frames=3,
                message="ok",
            )
        )
    response = client.get("/cameras/cam_demo/health", params={"db_path": str(db)})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "streaming"
    assert body["reconnect_count"] == 1
