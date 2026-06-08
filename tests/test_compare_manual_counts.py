"""Tests for manual ground-truth comparison helpers."""

from __future__ import annotations

import sqlite3

import pytest

from scripts.compare_manual_counts import (
    ManualSegment,
    SegmentAverage,
    _comparison_row,
    _error_stats,
    _latest_completed_session_id,
    _load_manual_rows,
    _segment_average_count,
)


def _create_comparison_db(path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE run_sessions (
                id INTEGER PRIMARY KEY,
                detector_mode TEXT NOT NULL,
                started_at TEXT,
                ended_at TEXT,
                total_frames INTEGER
            );
            CREATE TABLE frames_processed (
                id INTEGER PRIMARY KEY,
                run_session_id INTEGER NOT NULL,
                detector_mode TEXT NOT NULL,
                frame_index INTEGER NOT NULL,
                source_timestamp REAL NOT NULL,
                total_detections INTEGER NOT NULL
            );
            """
        )
        conn.executemany(
            """
            INSERT INTO run_sessions (id, detector_mode, started_at, ended_at, total_frames)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (1, "body", "2026-06-08 09:00:00", "2026-06-08 09:01:00", 3),
                (2, "body", "2026-06-08 10:00:00", None, None),
                (3, "body", "2026-06-08 11:00:00", "2026-06-08 11:01:00", 3),
                (4, "head", "2026-06-08 12:00:00", "2026-06-08 12:01:00", 0),
                (5, "head", "2026-06-08 13:00:00", "2026-06-08 13:01:00", 3),
            ],
        )
        conn.executemany(
            """
            INSERT INTO frames_processed (
                run_session_id, detector_mode, frame_index, source_timestamp, total_detections
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (1, "body", 0, 0.0, 100),
                (1, "body", 1, 1.0, 100),
                (2, "body", 0, 0.0, 999),
                (3, "body", 0, 0.0, 8),
                (3, "body", 1, 1.0, 10),
                (3, "body", 2, 2.0, 12),
                (5, "head", 0, 0.0, 5),
                (5, "head", 1, 1.0, 7),
                (5, "head", 2, 2.0, 9),
            ],
        )


def test_load_manual_rows_skips_blank_template_rows_and_parses_ranges(tmp_path) -> None:
    manual_csv = tmp_path / "manual.csv"
    manual_csv.write_text(
        "\n".join(
            [
                "segment_id,start_frame,end_frame,start_time_sec,end_time_sec,manual_count,notes",
                "empty,0,10,,,,skip blank manual count",
                ",10.0,20,,5.5,12,usable row",
            ]
        ),
        encoding="utf-8",
    )

    rows = _load_manual_rows(manual_csv)

    assert rows == [
        ManualSegment(
            segment_id="segment_1",
            start_frame=10,
            end_frame=20,
            start_time_sec=None,
            end_time_sec=5.5,
            manual_count=12.0,
            notes="usable row",
        )
    ]


def test_load_manual_rows_reports_bad_manual_count_with_file_context(tmp_path) -> None:
    manual_csv = tmp_path / "manual.csv"
    manual_csv.write_text(
        "\n".join(
            [
                "segment_id,start_frame,end_frame,start_time_sec,end_time_sec,manual_count,notes",
                "bad,0,10,,,not-a-number,",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"manual\.csv:2"):
        _load_manual_rows(manual_csv)


def test_latest_completed_session_ignores_unfinished_and_zero_frame_sessions(tmp_path) -> None:
    db_path = tmp_path / "analytics.db"
    _create_comparison_db(db_path)

    assert _latest_completed_session_id(db_path, "body") == 3
    assert _latest_completed_session_id(db_path, "head") == 5


def test_segment_average_count_uses_latest_completed_session_only(tmp_path) -> None:
    db_path = tmp_path / "analytics.db"
    _create_comparison_db(db_path)
    segment = ManualSegment(
        segment_id="seg",
        start_frame=0,
        end_frame=2,
        start_time_sec=None,
        end_time_sec=None,
        manual_count=10,
    )

    average = _segment_average_count(db_path, segment, detector_mode="body")

    assert average == SegmentAverage(session_id=3, avg_count=10.0)


def test_segment_average_count_applies_time_range(tmp_path) -> None:
    db_path = tmp_path / "analytics.db"
    _create_comparison_db(db_path)
    segment = ManualSegment(
        segment_id="seg",
        start_frame=None,
        end_frame=None,
        start_time_sec=1.0,
        end_time_sec=2.0,
        manual_count=8,
    )

    average = _segment_average_count(db_path, segment, detector_mode="head")

    assert average == SegmentAverage(session_id=5, avg_count=8.0)


def test_error_stats_handles_normal_zero_and_missing_counts() -> None:
    assert _error_stats(10.0, 8.0) == (2.0, 25.0)
    assert _error_stats(2.0, 0.0) == (2.0, None)
    assert _error_stats(None, 8.0) == (None, None)


def test_comparison_row_writes_body_and_head_segment_metrics() -> None:
    segment = ManualSegment(
        segment_id="seg",
        start_frame=0,
        end_frame=2,
        start_time_sec=None,
        end_time_sec=None,
        manual_count=8.0,
        notes="checked by reviewer",
    )

    row = _comparison_row(
        segment,
        SegmentAverage(session_id=3, avg_count=10.0),
        SegmentAverage(session_id=5, avg_count=6.0),
    )

    assert row["manual_count"] == "8.00"
    assert row["body_avg_count"] == "10.00"
    assert row["head_avg_count"] == "6.00"
    assert row["body_absolute_error"] == "2.00"
    assert row["body_percentage_error"] == "25.00"
    assert row["head_absolute_error"] == "2.00"
    assert row["head_percentage_error"] == "25.00"
