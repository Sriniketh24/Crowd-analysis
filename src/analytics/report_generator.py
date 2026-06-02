"""Analytics report export utilities.

Supports exporting one or all analytics tables from a SQLite database to CSV.

Usage::

    from src.analytics.report_generator import export_table_to_csv, export_all_tables

    export_table_to_csv(engine, "zone_occupancy", Path("data/outputs/report.csv"))
    # or export every table into a directory:
    export_all_tables(engine, Path("data/outputs/reports/"))
"""

from __future__ import annotations

import csv
from pathlib import Path

from sqlalchemy import inspect, select

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

# Ordered list of (table_name, ORM model) for discovery and export.
_TABLES = [
    ("camera_registry", CameraRegistry),
    ("run_sessions", RunSession),
    ("frames_processed", FrameProcessed),
    ("zone_occupancy", ZoneOccupancy),
    ("line_crossing_events", LineCrossingEvent),
    ("crowd_alerts", CrowdAlert),
    ("stream_health_events", StreamHealthEvent),
]

TABLE_NAMES = [name for name, _ in _TABLES]


def _column_names(model) -> list[str]:
    """Return column names in declaration order for an ORM model."""
    mapper = inspect(model)
    return [col.key for col in mapper.columns]


def export_table_to_csv(
    engine,
    table_name: str,
    output_path: Path,
    *,
    camera_id: str | None = None,
    session_id: int | None = None,
) -> int:
    """Export one analytics table to CSV.

    Args:
        engine:      SQLAlchemy engine connected to the analytics database.
        table_name:  One of the known table names (see ``TABLE_NAMES``).
        output_path: Destination CSV file path; parent directories are created.
        camera_id:   Optional filter — only rows matching this camera_id.
        session_id:  Optional filter — only rows for this run_session_id.

    Returns:
        Number of rows written (excluding header).

    Raises:
        ValueError: If *table_name* is not recognised.
    """
    model_map = dict(_TABLES)
    if table_name not in model_map:
        raise ValueError(f"Unknown table '{table_name}'. Known: {TABLE_NAMES}")

    model = model_map[table_name]
    columns = _column_names(model)
    Session = get_session_factory(engine)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0

    with Session() as session, open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)

        stmt = select(model)
        if camera_id is not None and hasattr(model, "camera_id"):
            stmt = stmt.where(model.camera_id == camera_id)
        if session_id is not None and hasattr(model, "run_session_id"):
            stmt = stmt.where(model.run_session_id == session_id)

        for row in session.execute(stmt).scalars():
            writer.writerow([getattr(row, col) for col in columns])
            row_count += 1

    return row_count


def export_all_tables(
    engine,
    output_dir: Path,
    *,
    camera_id: str | None = None,
    session_id: int | None = None,
) -> dict[str, int]:
    """Export all analytics tables to separate CSV files in *output_dir*.

    Returns:
        Dict mapping table_name → row count for each exported file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, int] = {}
    for table_name, _ in _TABLES:
        dest = output_dir / f"{table_name}.csv"
        results[table_name] = export_table_to_csv(
            engine, table_name, dest, camera_id=camera_id, session_id=session_id
        )
    return results


def _union_columns() -> list[str]:
    """Return a ``table`` column followed by the union of all model columns."""
    seen: list[str] = []
    for _, model in _TABLES:
        for col in _column_names(model):
            if col not in seen:
                seen.append(col)
    return ["table", *seen]


def export_combined_csv(
    engine,
    output_path: Path,
    *,
    camera_id: str | None = None,
    session_id: int | None = None,
) -> int:
    """Export every analytics table into a single CSV file.

    Each row is tagged with a leading ``table`` column and uses the union of all
    table columns (unused columns are left blank). The header row is always
    written, so an empty database still produces a valid, openable CSV.

    Args:
        engine:      SQLAlchemy engine connected to the analytics database.
        output_path: Destination CSV file path; parent directories are created.
        camera_id:   Optional filter — only rows matching this camera_id.
        session_id:  Optional filter — only rows for this run_session_id.

    Returns:
        Total number of data rows written (excluding the header).
    """
    columns = _union_columns()
    Session = get_session_factory(engine)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with Session() as session, open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore", restval="")
        writer.writeheader()
        for table_name, model in _TABLES:
            model_cols = _column_names(model)
            stmt = select(model)
            if camera_id is not None and hasattr(model, "camera_id"):
                stmt = stmt.where(model.camera_id == camera_id)
            if session_id is not None and hasattr(model, "run_session_id"):
                stmt = stmt.where(model.run_session_id == session_id)
            for row in session.execute(stmt).scalars():
                record = {"table": table_name}
                for col in model_cols:
                    record[col] = getattr(row, col)
                writer.writerow(record)
                total += 1

    return total


def open_existing_db(db_path: str | Path) -> object:
    """Return an engine for an existing SQLite database (read-only safe).

    Creates tables (idempotent) so that an empty DB is still queryable.
    """
    url = f"sqlite:///{Path(db_path).resolve()}"
    return init_db(url)
