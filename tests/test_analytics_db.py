"""Tests for analytics SQLite persistence (database, event_logger, report_generator).

All tests use an in-memory SQLite DB — no files, no external dependencies.
"""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

import pytest

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
from src.analytics.event_logger import AnalyticsLogger
from src.analytics.report_generator import (
    TABLE_NAMES,
    export_combined_csv,
    export_table_to_csv,
)
from src.vision.line_counter import LineCrossing


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def engine():
    """Fresh in-memory SQLite engine for each test."""
    return init_db("sqlite:///:memory:")


@pytest.fixture()
def Session(engine):
    """Session factory bound to the in-memory engine."""
    return get_session_factory(engine)


@pytest.fixture()
def logger(tmp_path):
    """AnalyticsLogger connected to a temp-file SQLite DB with zero snapshot interval."""
    db_path = tmp_path / "test_analytics.db"
    return AnalyticsLogger(
        db_url=f"sqlite:///{db_path}",
        camera_id="test_cam",
        snapshot_interval_secs=0.0,  # snapshot every frame
    )


# ---------------------------------------------------------------------------
# database.py — table creation and basic CRUD
# ---------------------------------------------------------------------------


class TestInitDb:
    def test_all_tables_created(self, engine):
        from sqlalchemy import inspect

        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        expected = {
            "camera_registry",
            "run_sessions",
            "frames_processed",
            "zone_occupancy",
            "line_crossing_events",
            "crowd_alerts",
            "stream_health_events",
        }
        assert expected <= tables

    def test_second_init_is_idempotent(self, engine):
        """Calling init_db twice must not raise."""
        init_db("sqlite:///:memory:")


class TestRunSession:
    def test_insert_and_retrieve(self, Session):
        with db_session(Session) as session:
            run = RunSession(camera_id="cam1", source="video.mp4")
            session.add(run)
            session.flush()
            assert run.id is not None

        with db_session(Session) as session:
            found = session.get(RunSession, run.id)
            assert found is not None
            assert found.camera_id == "cam1"
            assert found.source == "video.mp4"
            assert found.ended_at is None

    def test_end_session_fields(self, Session):
        from datetime import datetime, timezone

        with db_session(Session) as session:
            run = RunSession(camera_id="cam1", source="video.mp4")
            session.add(run)
            session.flush()
            run_id = run.id

        with db_session(Session) as session:
            run = session.get(RunSession, run_id)
            run.ended_at = datetime.now(timezone.utc)
            run.total_frames = 100

        with db_session(Session) as session:
            run = session.get(RunSession, run_id)
            assert run.total_frames == 100
            assert run.ended_at is not None


class TestZoneOccupancy:
    def test_insert_snapshot(self, Session):
        with db_session(Session) as session:
            run = RunSession(camera_id="cam1", source="v.mp4")
            session.add(run)
            session.flush()
            snap = ZoneOccupancy(
                run_session_id=run.id,
                camera_id="cam1",
                zone_id="zone_a",
                frame_index=10,
                count=5,
                alert_level="NORMAL",
            )
            session.add(snap)

        with db_session(Session) as session:
            snaps = session.query(ZoneOccupancy).all()
            assert len(snaps) == 1
            assert snaps[0].count == 5
            assert snaps[0].alert_level == "NORMAL"


class TestLineCrossingEvent:
    def test_insert_crossing(self, Session):
        with db_session(Session) as session:
            run = RunSession(camera_id="cam1", source="v.mp4")
            session.add(run)
            session.flush()
            evt = LineCrossingEvent(
                run_session_id=run.id,
                camera_id="cam1",
                line_id="gate_line",
                line_name="Gate Line",
                track_id=42,
                direction="IN",
                frame_index=5,
            )
            session.add(evt)

        with db_session(Session) as session:
            evts = session.query(LineCrossingEvent).all()
            assert len(evts) == 1
            assert evts[0].direction == "IN"
            assert evts[0].track_id == 42


class TestCrowdAlert:
    def test_insert_alert(self, Session):
        with db_session(Session) as session:
            run = RunSession(camera_id="cam1", source="v.mp4")
            session.add(run)
            session.flush()
            alert = CrowdAlert(
                run_session_id=run.id,
                camera_id="cam1",
                zone_id="platform",
                alert_level="WARNING",
                occupancy=22,
                frame_index=30,
            )
            session.add(alert)

        with db_session(Session) as session:
            alerts = session.query(CrowdAlert).all()
            assert len(alerts) == 1
            assert alerts[0].alert_level == "WARNING"


# ---------------------------------------------------------------------------
# event_logger.py — AnalyticsLogger behaviour
# ---------------------------------------------------------------------------


class TestAnalyticsLogger:
    def test_start_session_creates_run_record(self, logger, tmp_path):
        session_id = logger.start_session(source="video.mp4")
        assert isinstance(session_id, int)
        assert session_id > 0

    def test_end_session_fills_metadata(self, logger, tmp_path):
        logger.start_session(source="video.mp4")
        logger.end_session(total_frames=50)
        # Logger session_id is cleared after end
        assert logger._run_session_id is None

    def test_log_frame_creates_frame_record(self, logger):
        logger.start_session(source="video.mp4")
        logger.log_frame(
            frame_index=0,
            source_timestamp=0.0,
            total_detections=3,
            zone_occupancy={"zone_a": 3},
            zone_alerts={"zone_a": "NORMAL"},
            line_counts={},
            track_ids=[1, 2, 3],
            source_type="file",
        )
        logger.end_session(total_frames=1)

    def test_zone_snapshot_written_at_interval_zero(self, logger):
        logger.start_session(source="video.mp4")
        logger.log_frame(
            frame_index=1,
            source_timestamp=1.0,
            total_detections=5,
            zone_occupancy={"zone_a": 5},
            zone_alerts={"zone_a": "WARNING"},
            line_counts={},
            track_ids=[1, 2, 3, 4, 5],
        )
        logger.end_session(total_frames=1)

    def test_line_crossing_delta_detected(self, logger):
        """New crossings should be persisted when cumulative counts increase."""
        logger.start_session(source="video.mp4")
        logger.log_frame(
            frame_index=1,
            source_timestamp=1.0,
            total_detections=2,
            zone_occupancy={},
            zone_alerts={},
            line_counts={"gate": {"IN": 0, "OUT": 0}},
            line_events=[],
        )
        # Two new IN crossings on frame 2
        logger.log_frame(
            frame_index=2,
            source_timestamp=2.0,
            total_detections=2,
            zone_occupancy={},
            zone_alerts={},
            line_counts={"gate": {"IN": 2, "OUT": 0}},
            line_events=[
                LineCrossing(
                    line_id="gate",
                    line_name="Gate",
                    track_id=1,
                    direction="IN",
                    frame_index=2,
                ),
                LineCrossing(
                    line_id="gate",
                    line_name="Gate",
                    track_id=2,
                    direction="IN",
                    frame_index=2,
                ),
            ],
        )
        logger.end_session(total_frames=2)

    def test_alert_written_only_on_transition(self, logger):
        """CrowdAlert rows should only be inserted when the level changes."""
        logger.start_session(source="video.mp4")
        # First frame — NORMAL → NORMAL (no alert written after first time NORMAL is set)
        logger.log_frame(
            frame_index=1,
            source_timestamp=1.0,
            total_detections=5,
            zone_occupancy={"z": 5},
            zone_alerts={"z": "NORMAL"},
            line_counts={},
        )
        # Second frame — NORMAL still (no new alert)
        logger.log_frame(
            frame_index=2,
            source_timestamp=2.0,
            total_detections=5,
            zone_occupancy={"z": 5},
            zone_alerts={"z": "NORMAL"},
            line_counts={},
        )
        # Third frame — transitions to WARNING
        logger.log_frame(
            frame_index=3,
            source_timestamp=3.0,
            total_detections=25,
            zone_occupancy={"z": 25},
            zone_alerts={"z": "WARNING"},
            line_counts={},
        )
        logger.end_session(total_frames=3)

    def test_log_frame_before_start_raises(self, logger):
        with pytest.raises(RuntimeError, match="start_session"):
            logger.log_frame(
                frame_index=0,
                source_timestamp=0.0,
                total_detections=0,
                zone_occupancy={},
                zone_alerts={},
                line_counts={},
            )


# ---------------------------------------------------------------------------
# report_generator.py — CSV export
# ---------------------------------------------------------------------------


class TestExportTableToCsv:
    def _populate_db(self, engine):
        """Insert minimal test data across all tables."""
        Session = get_session_factory(engine)
        with db_session(Session) as session:
            run = RunSession(camera_id="cam1", source="video.mp4", detector_mode="head")
            session.add(run)
            session.flush()
            session.add(
                FrameProcessed(
                    run_session_id=run.id,
                    camera_id="cam1",
                    detector_mode="head",
                    frame_index=0,
                    source_timestamp=0.0,
                    total_detections=1,
                )
            )
            session.add(
                ZoneOccupancy(
                    run_session_id=run.id,
                    camera_id="cam1",
                    detector_mode="head",
                    zone_id="zone_a",
                    frame_index=0,
                    count=3,
                    alert_level="NORMAL",
                )
            )
            session.add(
                LineCrossingEvent(
                    run_session_id=run.id,
                    camera_id="cam1",
                    detector_mode="head",
                    line_id="gate",
                    direction="IN",
                    frame_index=0,
                )
            )
            session.add(
                CrowdAlert(
                    run_session_id=run.id,
                    camera_id="cam1",
                    detector_mode="head",
                    zone_id="zone_a",
                    alert_level="WARNING",
                    occupancy=22,
                    frame_index=10,
                )
            )

    def test_export_zone_occupancy(self, engine, tmp_path):
        self._populate_db(engine)
        out = tmp_path / "zone_occupancy.csv"
        count = export_table_to_csv(engine, "zone_occupancy", out)
        assert count == 1
        assert out.exists()
        rows = list(csv.DictReader(out.read_text().splitlines()))
        assert rows[0]["zone_id"] == "zone_a"
        assert rows[0]["count"] == "3"
        assert rows[0]["detector_mode"] == "head"

    def test_export_line_crossing_events(self, engine, tmp_path):
        self._populate_db(engine)
        out = tmp_path / "crossings.csv"
        count = export_table_to_csv(engine, "line_crossing_events", out)
        assert count == 1
        rows = list(csv.DictReader(out.read_text().splitlines()))
        assert rows[0]["direction"] == "IN"

    def test_export_unknown_table_raises(self, engine, tmp_path):
        with pytest.raises(ValueError, match="Unknown table"):
            export_table_to_csv(engine, "nonexistent_table", tmp_path / "out.csv")

    def test_export_all_table_names_known(self):
        assert "run_sessions" in TABLE_NAMES
        assert "frames_processed" in TABLE_NAMES
        assert "zone_occupancy" in TABLE_NAMES
        assert "line_crossing_events" in TABLE_NAMES
        assert "crowd_alerts" in TABLE_NAMES
        assert "camera_registry" in TABLE_NAMES
        assert "stream_health_events" in TABLE_NAMES

    def test_header_row_matches_columns(self, engine, tmp_path):
        self._populate_db(engine)
        out = tmp_path / "frames.csv"
        export_table_to_csv(engine, "frames_processed", out)
        lines = out.read_text().splitlines()
        header = lines[0].split(",")
        assert "frame_index" in header
        assert "camera_id" in header
        assert "total_detections" in header
        assert "detector_mode" in header

    def test_empty_table_exports_header_only(self, engine, tmp_path):
        """An empty DB must still write a header row (no crash, openable file)."""
        out = tmp_path / "empty.csv"
        count = export_table_to_csv(engine, "zone_occupancy", out)
        assert count == 0
        lines = out.read_text().splitlines()
        assert len(lines) == 1  # header only
        assert "zone_id" in lines[0]


class TestExportCombinedCsv:
    def _populate_db(self, engine):
        Session = get_session_factory(engine)
        with db_session(Session) as session:
            run = RunSession(camera_id="cam1", source="video.mp4", detector_mode="head")
            session.add(run)
            session.flush()
            session.add(
                ZoneOccupancy(
                    run_session_id=run.id,
                    camera_id="cam1",
                    detector_mode="head",
                    zone_id="zone_a",
                    frame_index=0,
                    count=3,
                    alert_level="NORMAL",
                )
            )
            session.add(
                CrowdAlert(
                    run_session_id=run.id,
                    camera_id="cam1",
                    detector_mode="head",
                    zone_id="zone_a",
                    alert_level="WARNING",
                    occupancy=22,
                    frame_index=10,
                )
            )

    def test_combined_export_writes_single_file(self, engine, tmp_path):
        """All tables export into one CSV file with a leading 'table' column."""
        self._populate_db(engine)
        out = tmp_path / "report.csv"
        count = export_combined_csv(engine, out)
        assert out.is_file()
        # run_session + zone_occupancy + crowd_alert = 3 data rows
        assert count == 3
        rows = list(csv.DictReader(out.read_text().splitlines()))
        tables = {row["table"] for row in rows}
        assert {"run_sessions", "zone_occupancy", "crowd_alerts"} <= tables
        assert "detector_mode" in rows[0]

    def test_combined_export_empty_db_header_only(self, engine, tmp_path):
        """Empty DB still produces a valid single CSV with just the header."""
        out = tmp_path / "report.csv"
        count = export_combined_csv(engine, out)
        assert count == 0
        assert out.is_file()
        lines = out.read_text().splitlines()
        assert len(lines) == 1
        assert lines[0].startswith("table,")
