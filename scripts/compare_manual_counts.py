"""Compare detector counts with manually entered rough counts.

Manual counts are optional ground truth. Without them, the system can only be
estimated visually or by reviewing operational tuning metrics.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class ManualSegment:
    """A manually counted video segment loaded from the ground-truth CSV."""

    segment_id: str
    start_frame: int | None
    end_frame: int | None
    start_time_sec: float | None
    end_time_sec: float | None
    manual_count: float
    notes: str = ""


@dataclass(frozen=True)
class SegmentAverage:
    """Detector average count for one segment in one completed run session."""

    session_id: int
    avg_count: float | None


# Output columns are grouped to keep detection volume separate from measured
# accuracy: *_avg_count are operational detection volumes (always available
# from a run), while *_absolute_error / *_percentage_error are the only
# accuracy figures and exist solely because a manual_count was supplied.
OUTPUT_FIELDS = [
    "segment_id",
    "start_frame",
    "end_frame",
    "start_time_sec",
    "end_time_sec",
    "manual_count",
    # Provenance: which completed run session each volume came from.
    "body_session_id",
    "head_session_id",
    "hybrid_session_id",
    # Detection volume (operational observation, NOT accuracy).
    "body_avg_count",
    "head_avg_count",
    "hybrid_avg_count",
    # Measured accuracy vs the manual ground truth.
    "body_absolute_error",
    "head_absolute_error",
    "hybrid_absolute_error",
    "body_percentage_error",
    "head_percentage_error",
    "hybrid_percentage_error",
    "notes",
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare manual frame/time-range counts against the latest completed "
            "body, head, and hybrid detector sessions."
        )
    )
    parser.add_argument(
        "--manual-csv",
        type=Path,
        default=Path("data/manual_ground_truth/example_counts.csv"),
        help="CSV with segment_id, frame/time range, and manual_count columns.",
    )
    parser.add_argument(
        "--body-db",
        type=Path,
        default=Path("data/outputs/body_analytics.db"),
        help="SQLite DB containing completed body detector sessions.",
    )
    parser.add_argument(
        "--head-db",
        type=Path,
        default=Path("data/outputs/head_analytics.db"),
        help="SQLite DB containing completed head detector sessions.",
    )
    parser.add_argument(
        "--hybrid-db",
        type=Path,
        default=Path("data/outputs/hybrid_analytics.db"),
        help="SQLite DB containing completed hybrid detector sessions.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/outputs/manual_count_comparison.csv"),
        help="Output CSV path for per-segment body-vs-head-vs-hybrid accuracy metrics.",
    )
    return parser.parse_args()


def _print_instructions(manual_csv: Path) -> None:
    print("No usable manual ground truth counts were found.")
    print("No accuracy can be measured without manual labels; nothing was written.")
    print(f"Edit: {manual_csv}")
    print("For each row, fill manual_count after visually counting passengers in the selected range.")
    print("Use either start_frame/end_frame or start_time_sec/end_time_sec.")
    print("Then rerun this script after body/head/hybrid analytics databases have been generated.")


def _parse_optional_float(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(float(value))


def _load_manual_rows(path: Path) -> list[ManualSegment]:
    """Load usable manual-count rows, skipping blank template rows."""
    if not path.exists():
        return []
    rows: list[ManualSegment] = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_number, raw in enumerate(reader, start=2):
            try:
                manual_count = _parse_optional_float(raw.get("manual_count"))
            except ValueError as error:
                raise ValueError(f"Invalid manual_count at {path}:{row_number}") from error
            if manual_count is None:
                continue
            try:
                rows.append(
                    ManualSegment(
                        segment_id=raw.get("segment_id") or f"segment_{len(rows) + 1}",
                        start_frame=_parse_optional_int(raw.get("start_frame")),
                        end_frame=_parse_optional_int(raw.get("end_frame")),
                        start_time_sec=_parse_optional_float(raw.get("start_time_sec")),
                        end_time_sec=_parse_optional_float(raw.get("end_time_sec")),
                        manual_count=manual_count,
                        notes=raw.get("notes") or "",
                    )
                )
            except ValueError as error:
                raise ValueError(f"Invalid segment range at {path}:{row_number}") from error
    return rows


def _latest_completed_session_id(db_path: Path, detector_mode: str) -> int | None:
    """Return the newest completed session for the requested detector mode."""
    if not db_path.exists() or db_path.stat().st_size == 0:
        return None

    sql = """
        SELECT id
        FROM run_sessions
        WHERE detector_mode = ?
          AND ended_at IS NOT NULL
          AND total_frames IS NOT NULL
          AND total_frames > 0
        ORDER BY ended_at DESC, id DESC
        LIMIT 1
    """
    try:
        with sqlite3.connect(str(db_path), timeout=10) as conn:
            row = conn.execute(sql, (detector_mode,)).fetchone()
    except sqlite3.Error:
        return None
    return None if row is None else int(row[0])


def _segment_average_count(
    db_path: Path,
    segment: ManualSegment,
    *,
    detector_mode: str,
) -> SegmentAverage | None:
    """Average per-frame detections in the newest completed matching session."""
    session_id = _latest_completed_session_id(db_path, detector_mode)
    if session_id is None:
        return None

    filters: list[str] = []
    params: list[float | int | str] = [session_id, detector_mode]
    if segment.start_frame is not None:
        filters.append("frame_index >= ?")
        params.append(segment.start_frame)
    if segment.end_frame is not None:
        filters.append("frame_index <= ?")
        params.append(segment.end_frame)
    if segment.start_time_sec is not None:
        filters.append("source_timestamp >= ?")
        params.append(segment.start_time_sec)
    if segment.end_time_sec is not None:
        filters.append("source_timestamp <= ?")
        params.append(segment.end_time_sec)

    range_clause = f" AND {' AND '.join(filters)}" if filters else ""
    sql = (
        "SELECT AVG(total_detections) "
        "FROM frames_processed "
        "WHERE run_session_id = ? AND detector_mode = ?"
        f"{range_clause}"
    )
    try:
        with sqlite3.connect(str(db_path), timeout=10) as conn:
            value = conn.execute(sql, params).fetchone()[0]
    except sqlite3.Error:
        return None
    return SegmentAverage(session_id=session_id, avg_count=None if value is None else float(value))


def _error_stats(detected_count: float | None, manual_count: float) -> tuple[float | None, float | None]:
    """Calculate absolute and percentage error against manual ground truth."""
    if detected_count is None:
        return None, None
    absolute_error = abs(detected_count - manual_count)
    percentage_error = None if manual_count == 0 else (absolute_error / manual_count) * 100
    return absolute_error, percentage_error


def _format_number(value: float | int | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.2f}"


def _format_int(value: int | None) -> str:
    return "" if value is None else str(value)


def _comparison_row(
    segment: ManualSegment,
    body_average: SegmentAverage | None,
    head_average: SegmentAverage | None,
    hybrid_average: SegmentAverage | None,
) -> dict[str, str]:
    """Build one output CSV row for a manual segment."""
    body_count = None if body_average is None else body_average.avg_count
    head_count = None if head_average is None else head_average.avg_count
    hybrid_count = None if hybrid_average is None else hybrid_average.avg_count
    body_abs, body_pct = _error_stats(body_count, segment.manual_count)
    head_abs, head_pct = _error_stats(head_count, segment.manual_count)
    hybrid_abs, hybrid_pct = _error_stats(hybrid_count, segment.manual_count)
    return {
        "segment_id": segment.segment_id,
        "start_frame": _format_int(segment.start_frame),
        "end_frame": _format_int(segment.end_frame),
        "start_time_sec": _format_number(segment.start_time_sec),
        "end_time_sec": _format_number(segment.end_time_sec),
        "manual_count": _format_number(segment.manual_count),
        "body_session_id": "" if body_average is None else str(body_average.session_id),
        "head_session_id": "" if head_average is None else str(head_average.session_id),
        "hybrid_session_id": "" if hybrid_average is None else str(hybrid_average.session_id),
        "body_avg_count": _format_number(body_count),
        "head_avg_count": _format_number(head_count),
        "hybrid_avg_count": _format_number(hybrid_count),
        "body_absolute_error": _format_number(body_abs),
        "head_absolute_error": _format_number(head_abs),
        "hybrid_absolute_error": _format_number(hybrid_abs),
        "body_percentage_error": _format_number(body_pct),
        "head_percentage_error": _format_number(head_pct),
        "hybrid_percentage_error": _format_number(hybrid_pct),
        "notes": segment.notes,
    }


def main() -> None:
    """Run the manual-count comparison."""
    args = parse_args()
    manual_rows = _load_manual_rows(args.manual_csv)
    if not manual_rows:
        _print_instructions(args.manual_csv)
        return

    results = [
        _comparison_row(
            segment,
            _segment_average_count(args.body_db, segment, detector_mode="body"),
            _segment_average_count(args.head_db, segment, detector_mode="head"),
            _segment_average_count(args.hybrid_db, segment, detector_mode="hybrid"),
        )
        for segment in manual_rows
    ]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote manual count comparison: {args.output_csv}")
    missing_dbs = [
        db for db in (args.body_db, args.head_db, args.hybrid_db) if not db.exists()
    ]
    if missing_dbs:
        print(
            "Missing analytics DBs (their columns stay blank): "
            + ", ".join(str(db) for db in missing_dbs)
        )
    if any(
        not row[f"{mode}_session_id"]
        for row in results
        for mode in ("body", "head", "hybrid")
    ):
        print("At least one mode has no completed matching detector session for some rows.")
    print(
        "Columns *_avg_count are detection volume (operational observation), not accuracy."
    )
    print(
        "Accuracy is only the *_absolute_error / *_percentage_error columns, measured "
        "against your manual_count."
    )
    print("Percentage error is blank when manual_count is zero.")


if __name__ == "__main__":
    main()
