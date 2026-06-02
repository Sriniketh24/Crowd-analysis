"""Extract training frames from real public videos for manual annotation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_sample_frames import extract_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames from real Indian railway clips for manual labeling."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/indian_railway_videos/pexels_crowded_train_station_6023186.mp4",
        help="Input video file path.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/training_frames/raw",
        help="Directory where extracted frames are written.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=24,
        help="Number of evenly spaced frames to extract.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = extract_frames(source=args.source, out_dir=args.out, num=args.num)
    print(f"Extracted {len(written)} frame(s) for annotation into {args.out}")


if __name__ == "__main__":
    main()
