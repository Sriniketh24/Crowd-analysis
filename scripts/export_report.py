"""Export analytics data from SQLite database to CSV.

Usage examples::

    # Export all tables to a single directory
    python scripts/export_report.py --db data/outputs/analytics.db --out data/outputs/reports/

    # Export a specific table to a single CSV file
    python scripts/export_report.py --db data/outputs/analytics.db --out data/outputs/report.csv --table zone_occupancy

    # Filter by camera
    python scripts/export_report.py --db data/outputs/analytics.db --out data/outputs/report.csv --table crowd_alerts --camera cam1

    # Filter by run session ID
    python scripts/export_report.py --db data/outputs/analytics.db --out data/outputs/report.csv --session 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.report_generator import (
    TABLE_NAMES,
    export_all_tables,
    export_combined_csv,
    export_table_to_csv,
    open_existing_db,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Export analytics data from SQLite to CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db",
        type=Path,
        required=True,
        help="Path to the analytics SQLite database file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help=(
            "Output path. If it ends in '.csv' a single CSV file is written "
            "(one table when --table is given, otherwise all tables combined). "
            "If it has no '.csv' suffix it is treated as a directory and one CSV "
            "per table is written."
        ),
    )
    parser.add_argument(
        "--table",
        type=str,
        choices=TABLE_NAMES,
        default=None,
        help="Export a single table. Omit to export all tables.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        metavar="CAMERA_ID",
        help="Filter rows to a specific camera_id.",
    )
    parser.add_argument(
        "--session",
        type=int,
        default=None,
        metavar="SESSION_ID",
        help="Filter rows to a specific run_session_id.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the export script."""
    args = parse_args()

    if not args.db.exists():
        print(f"Database not found: {args.db}", file=sys.stderr)
        print(
            "No analytics database exists yet. Generate one by running the video "
            "demo with --db, for example:",
            file=sys.stderr,
        )
        print(
            "  python scripts/run_video_demo.py "
            "--source data/input_videos/sample.mp4 "
            "--output data/outputs/demo.mp4 "
            "--zones-config configs/zones.example.json "
            f"--db {args.db}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    engine = open_existing_db(args.db)

    out_is_csv_file = args.out.suffix.lower() == ".csv"

    if args.table:
        # Single-table export → treat --out as a file path.
        count = export_table_to_csv(
            engine,
            args.table,
            args.out,
            camera_id=args.camera,
            session_id=args.session,
        )
        print(f"Exported {count} rows from '{args.table}' → {args.out}")
    elif out_is_csv_file:
        # Combined single-file export → all tables into one CSV at the exact path.
        count = export_combined_csv(
            engine,
            args.out,
            camera_id=args.camera,
            session_id=args.session,
        )
        print(f"Exported {count} rows from all tables → {args.out}")
    else:
        # Multi-table export → treat --out as a directory.
        results = export_all_tables(
            engine,
            args.out,
            camera_id=args.camera,
            session_id=args.session,
        )
        for table_name, count in results.items():
            dest = args.out / f"{table_name}.csv"
            print(f"  {table_name}: {count} rows → {dest}")
        total = sum(results.values())
        print(f"\nTotal: {total} rows exported to {args.out}/")


if __name__ == "__main__":
    main()
