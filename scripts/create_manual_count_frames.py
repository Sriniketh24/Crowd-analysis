"""Extract review frames and generate a blank ground-truth CSV template.

No model inference is run — this script only extracts frames and produces
a spreadsheet template that a human fills in by eye.

Usage
-----
    python3 scripts/create_manual_count_frames.py

    # Custom video / output locations:
    python3 scripts/create_manual_count_frames.py \\
        --source data/input_videos/sample.mp4 \\
        --out-dir data/manual_ground_truth/review_frames \\
        --output-csv data/manual_ground_truth/review_counts.csv \\
        --num-segments 8

How to fill in the CSV
----------------------
1. Open the generated CSV in a spreadsheet or text editor.
2. For each row, open the image listed in ``image_path`` (any image viewer).
3. Count every visible passenger by eye — count heads or standing figures,
   whichever is easier.  Do not guess; if the image is blurry or occluded
   and you cannot get a confident count, leave manual_count blank and write
   a note in the ``notes`` column.
4. Enter your count in the ``manual_count`` column.
5. Save the CSV and pass it to ``scripts/compare_manual_counts.py`` to
   compare your counts against the detector results.

Tips for counting
-----------------
* Zoom in — most image viewers let you press ``+`` or scroll.
* Count left-to-right row by row so you do not miss or double-count anyone.
* A person half-hidden behind another still counts as one passenger.
* Count only people visible in the frame; ignore anyone cut off at the edge
  if you cannot tell they are fully on the platform.
* The ``manual_count`` is the best count for the **middle frame** of the
  segment, not an average.  If the crowd changes noticeably over the segment,
  note that in ``notes``.

Do not modify segment_id, start_frame, end_frame, start_time_sec,
end_time_sec, or image_path columns — compare_manual_counts.py relies on them.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CSV_FIELDS = [
    "segment_id",
    "start_frame",
    "end_frame",
    "start_time_sec",
    "end_time_sec",
    "image_path",
    "manual_count",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract review frames and generate a blank ground-truth CSV template.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "After running, open each image in review_frames/ and fill in manual_count\n"
            "in the generated CSV.  Then run scripts/compare_manual_counts.py."
        ),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/input_videos/sample.mp4"),
        help="Input video file (default: data/input_videos/sample.mp4).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/manual_ground_truth/review_frames"),
        help="Directory for extracted JPEG frames.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/manual_ground_truth/review_counts.csv"),
        help="Output CSV template path.",
    )
    parser.add_argument(
        "--num-segments",
        type=int,
        default=6,
        metavar="N",
        help="Number of equal-length segments to divide the video into (default: 6).",
    )
    return parser.parse_args()


def _frame_to_time(frame_index: int, fps: float) -> float:
    """Convert a frame index to a timestamp in seconds."""
    return round(frame_index / fps, 3) if fps > 0 else 0.0


def extract_segment_frames(
    source: Path,
    out_dir: Path,
    num_segments: int,
) -> list[dict[str, object]]:
    """Open the video, split it into segments, extract the middle frame of each.

    Returns a list of dicts with segment metadata (no manual_count yet).
    """
    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0

    if total_frames <= 0:
        capture.release()
        raise RuntimeError(
            f"Could not read frame count from {source}.  "
            "The file may be corrupt or an unsupported codec."
        )

    print(f"Video : {source}")
    print(f"Frames: {total_frames}  FPS: {fps:.2f}  Duration: {total_frames / fps:.1f}s")

    out_dir.mkdir(parents=True, exist_ok=True)

    num_segments = max(1, min(num_segments, total_frames))
    segment_len = total_frames // num_segments
    segments: list[dict[str, object]] = []

    for seg_idx in range(num_segments):
        start_frame = seg_idx * segment_len
        # Last segment absorbs any remainder so no frames are skipped.
        end_frame = (
            (seg_idx + 1) * segment_len - 1
            if seg_idx < num_segments - 1
            else total_frames - 1
        )
        mid_frame = (start_frame + end_frame) // 2

        capture.set(cv2.CAP_PROP_POS_FRAMES, float(mid_frame))
        ok, frame = capture.read()
        if not ok or frame is None:
            print(f"  WARNING: could not read frame {mid_frame} for segment {seg_idx + 1}; skipping.")
            continue

        img_name = f"seg_{seg_idx + 1:03d}_frame_{mid_frame:06d}.jpg"
        img_path = out_dir / img_name
        cv2.imwrite(str(img_path), frame)

        # Store a path relative to the project root for portability.
        abs_img = img_path.resolve()
        try:
            rel_img = abs_img.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_img = abs_img

        segments.append(
            {
                "segment_id": f"seg_{seg_idx + 1:03d}",
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_sec": _frame_to_time(start_frame, fps),
                "end_time_sec": _frame_to_time(end_frame, fps),
                "image_path": str(rel_img),
            }
        )
        print(
            f"  Segment {seg_idx + 1:2d}: frames {start_frame}–{end_frame} "
            f"({_frame_to_time(start_frame, fps):.1f}s–{_frame_to_time(end_frame, fps):.1f}s) "
            f"→ {img_path.name}"
        )

    capture.release()
    return segments


def write_csv_template(segments: list[dict[str, object]], output_csv: Path) -> None:
    """Write blank-manual_count CSV template."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for seg in segments:
            writer.writerow(
                {
                    "segment_id": seg["segment_id"],
                    "start_frame": seg["start_frame"],
                    "end_frame": seg["end_frame"],
                    "start_time_sec": seg["start_time_sec"],
                    "end_time_sec": seg["end_time_sec"],
                    "image_path": seg["image_path"],
                    "manual_count": "",   # intentionally blank — human fills this in
                    "notes": "",
                }
            )


def _print_next_steps(output_csv: Path, out_dir: Path) -> None:
    print()
    print("=== Next steps ===")
    print(f"1. Open each image in:  {out_dir}/")
    print(f"2. Count passengers by eye and fill in manual_count in:")
    print(f"   {output_csv}")
    print("3. After running the body/head demo, compare counts with:")
    print("   python3 scripts/compare_manual_counts.py \\")
    print(f"       --manual-csv {output_csv}")
    print()
    print("The manual_count column is intentionally blank — no model counts are pre-filled.")


def main() -> None:
    args = parse_args()
    segments = extract_segment_frames(
        source=args.source,
        out_dir=args.out_dir,
        num_segments=args.num_segments,
    )
    if not segments:
        print("No segments extracted.  Check that the video file is readable.", file=sys.stderr)
        sys.exit(1)

    write_csv_template(segments, args.output_csv)
    print(f"\nWrote CSV template ({len(segments)} rows): {args.output_csv}")
    _print_next_steps(args.output_csv, args.out_dir)


if __name__ == "__main__":
    main()
