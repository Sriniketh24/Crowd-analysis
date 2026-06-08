"""SQLite persistence layer using SQLAlchemy ORM."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


class CameraRegistry(Base):
    """Configured camera/source registry."""

    __tablename__ = "camera_registry"

    camera_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    source_type: Mapped[str] = mapped_column(String, nullable=False, default="file")
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    zones_config_path: Mapped[str | None] = mapped_column(String, nullable=True)
    enabled: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class RunSession(Base):
    """One execution run of the video pipeline."""

    __tablename__ = "run_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    source: Mapped[str] = mapped_column(String, nullable=False)
    source_type: Mapped[str] = mapped_column(String, nullable=False, default="file")
    model_weights: Mapped[str | None] = mapped_column(String, nullable=True)
    detector_mode: Mapped[str] = mapped_column(String, nullable=False, default="body")
    tracker_type: Mapped[str | None] = mapped_column(String, nullable=True)
    zones_config_path: Mapped[str | None] = mapped_column(String, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    total_frames: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_unique_passengers: Mapped[int | None] = mapped_column(Integer, nullable=True)

    frames: Mapped[list[FrameProcessed]] = relationship(
        "FrameProcessed", back_populates="session", cascade="all, delete-orphan"
    )
    zone_snapshots: Mapped[list[ZoneOccupancy]] = relationship(
        "ZoneOccupancy", back_populates="session", cascade="all, delete-orphan"
    )
    crossings: Mapped[list[LineCrossingEvent]] = relationship(
        "LineCrossingEvent", back_populates="session", cascade="all, delete-orphan"
    )
    alerts: Mapped[list[CrowdAlert]] = relationship(
        "CrowdAlert", back_populates="session", cascade="all, delete-orphan"
    )
    health_events: Mapped[list[StreamHealthEvent]] = relationship(
        "StreamHealthEvent", back_populates="session", cascade="all, delete-orphan"
    )


class FrameProcessed(Base):
    """One processed frame record."""

    __tablename__ = "frames_processed"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("run_sessions.id", ondelete="CASCADE"), nullable=False
    )
    camera_id: Mapped[str] = mapped_column(String, nullable=False)
    frame_index: Mapped[int] = mapped_column(Integer, nullable=False)
    source_timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    source_type: Mapped[str] = mapped_column(String, nullable=False, default="file")
    detector_mode: Mapped[str] = mapped_column(String, nullable=False, default="body")
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    total_detections: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tracked_objects: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    unique_passengers_seen: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processing_fps: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    reconnect_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    dropped_frames: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    session: Mapped[RunSession] = relationship("RunSession", back_populates="frames")

    __table_args__ = (
        Index("ix_frames_camera_recorded", "camera_id", "recorded_at"),
        Index("ix_frames_session", "run_session_id"),
    )


class ZoneOccupancy(Base):
    """Periodic per-zone occupancy snapshot."""

    __tablename__ = "zone_occupancy"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("run_sessions.id", ondelete="CASCADE"), nullable=False
    )
    camera_id: Mapped[str] = mapped_column(String, nullable=False)
    detector_mode: Mapped[str] = mapped_column(String, nullable=False, default="body")
    zone_id: Mapped[str] = mapped_column(String, nullable=False)
    frame_index: Mapped[int] = mapped_column(Integer, nullable=False)
    count: Mapped[int] = mapped_column(Integer, nullable=False)
    alert_level: Mapped[str] = mapped_column(String, nullable=False, default="NORMAL")
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    session: Mapped[RunSession] = relationship("RunSession", back_populates="zone_snapshots")

    __table_args__ = (
        Index("ix_zone_camera_recorded", "camera_id", "recorded_at"),
        Index("ix_zone_session_zone", "run_session_id", "zone_id"),
    )


class LineCrossingEvent(Base):
    """One person crossing a virtual line."""

    __tablename__ = "line_crossing_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("run_sessions.id", ondelete="CASCADE"), nullable=False
    )
    camera_id: Mapped[str] = mapped_column(String, nullable=False)
    detector_mode: Mapped[str] = mapped_column(String, nullable=False, default="body")
    line_id: Mapped[str] = mapped_column(String, nullable=False)
    line_name: Mapped[str | None] = mapped_column(String, nullable=True)
    track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    direction: Mapped[str] = mapped_column(String, nullable=False)
    frame_index: Mapped[int] = mapped_column(Integer, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    session: Mapped[RunSession] = relationship("RunSession", back_populates="crossings")

    __table_args__ = (
        Index("ix_crossing_camera_recorded", "camera_id", "recorded_at"),
        Index("ix_crossing_session_line", "run_session_id", "line_id"),
    )


class CrowdAlert(Base):
    """Alert state change (level transition) for a zone."""

    __tablename__ = "crowd_alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("run_sessions.id", ondelete="CASCADE"), nullable=False
    )
    camera_id: Mapped[str] = mapped_column(String, nullable=False)
    detector_mode: Mapped[str] = mapped_column(String, nullable=False, default="body")
    zone_id: Mapped[str] = mapped_column(String, nullable=False)
    alert_level: Mapped[str] = mapped_column(String, nullable=False)
    occupancy: Mapped[int] = mapped_column(Integer, nullable=False)
    frame_index: Mapped[int] = mapped_column(Integer, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    session: Mapped[RunSession] = relationship("RunSession", back_populates="alerts")

    __table_args__ = (
        Index("ix_alert_camera_recorded", "camera_id", "recorded_at"),
        Index("ix_alert_session_zone", "run_session_id", "zone_id"),
    )


class StreamHealthEvent(Base):
    """Stream lifecycle and health status updates."""

    __tablename__ = "stream_health_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_session_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("run_sessions.id", ondelete="CASCADE"), nullable=False
    )
    camera_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    reconnect_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    dropped_frames: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    message: Mapped[str | None] = mapped_column(String, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    session: Mapped[RunSession] = relationship("RunSession", back_populates="health_events")


SQLITE_MIGRATIONS: dict[str, dict[str, str]] = {
    "run_sessions": {
        "source_type": "ALTER TABLE run_sessions ADD COLUMN source_type VARCHAR DEFAULT 'file'",
        "model_weights": "ALTER TABLE run_sessions ADD COLUMN model_weights VARCHAR",
        "detector_mode": "ALTER TABLE run_sessions ADD COLUMN detector_mode VARCHAR NOT NULL DEFAULT 'body'",
        "tracker_type": "ALTER TABLE run_sessions ADD COLUMN tracker_type VARCHAR",
        "zones_config_path": "ALTER TABLE run_sessions ADD COLUMN zones_config_path VARCHAR",
        "total_unique_passengers": "ALTER TABLE run_sessions ADD COLUMN total_unique_passengers INTEGER",
    },
    "frames_processed": {
        "source_type": "ALTER TABLE frames_processed ADD COLUMN source_type VARCHAR DEFAULT 'file'",
        "detector_mode": "ALTER TABLE frames_processed ADD COLUMN detector_mode VARCHAR NOT NULL DEFAULT 'body'",
        "tracked_objects": "ALTER TABLE frames_processed ADD COLUMN tracked_objects INTEGER DEFAULT 0",
        "unique_passengers_seen": "ALTER TABLE frames_processed ADD COLUMN unique_passengers_seen INTEGER DEFAULT 0",
        "processing_fps": "ALTER TABLE frames_processed ADD COLUMN processing_fps FLOAT DEFAULT 0.0",
        "reconnect_count": "ALTER TABLE frames_processed ADD COLUMN reconnect_count INTEGER DEFAULT 0",
        "dropped_frames": "ALTER TABLE frames_processed ADD COLUMN dropped_frames INTEGER DEFAULT 0",
    },
    "zone_occupancy": {
        "detector_mode": "ALTER TABLE zone_occupancy ADD COLUMN detector_mode VARCHAR NOT NULL DEFAULT 'body'",
    },
    "line_crossing_events": {
        "detector_mode": "ALTER TABLE line_crossing_events ADD COLUMN detector_mode VARCHAR NOT NULL DEFAULT 'body'",
        "line_name": "ALTER TABLE line_crossing_events ADD COLUMN line_name VARCHAR",
        "track_id": "ALTER TABLE line_crossing_events ADD COLUMN track_id INTEGER",
    },
    "crowd_alerts": {
        "detector_mode": "ALTER TABLE crowd_alerts ADD COLUMN detector_mode VARCHAR NOT NULL DEFAULT 'body'",
    },
}


def _ensure_sqlite_schema(engine) -> None:
    """Apply lightweight additive migrations for existing SQLite databases."""
    if engine.dialect.name != "sqlite":
        return
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as connection:
        for table_name, migrations in SQLITE_MIGRATIONS.items():
            if table_name not in existing_tables:
                continue
            existing_columns = {col["name"] for col in inspector.get_columns(table_name)}
            for column_name, sql in migrations.items():
                if column_name in existing_columns:
                    continue
                connection.execute(text(sql))


def get_engine(db_url: str):
    """Create a SQLAlchemy engine. Enables WAL mode for better concurrency on SQLite."""
    is_sqlite = db_url.startswith("sqlite")
    connect_args = {"check_same_thread": False} if is_sqlite else {}
    engine = create_engine(db_url, future=True, connect_args=connect_args)

    if is_sqlite and ":memory:" not in db_url:
        from sqlalchemy import event as sa_event

        @sa_event.listens_for(engine, "connect")
        def _set_wal_mode(dbapi_conn, _connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    return engine


def init_db(db_url: str):
    """Create engine, apply additive migrations, and create missing tables."""
    engine = get_engine(db_url)
    _ensure_sqlite_schema(engine)
    Base.metadata.create_all(engine)
    return engine


def get_session_factory(engine):
    """Return a configured sessionmaker bound to *engine*."""
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def db_session(session_factory) -> Generator[Session, None, None]:
    """Context manager for a single unit-of-work session with auto commit/rollback."""
    session: Session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
