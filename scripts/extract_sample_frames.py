"""Extract representative frames from input videos for zone calibration.

Usage:
    python3 scripts/extract_sample_frames.py \
        --source data/input_videos/sample.mp4 \
        --out data/sample_frames \
        --num 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for frame extraction."""
    parser = argparse.ArgumentParser(description="Extract sample frames for zone calibration.")
    parser.add_argument(
        "--source",
        type=str,
        default="data/input_videos/sample.mp4",
        help="Input video file path.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/sample_frames",
        help="Directory where extracted frames are written.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=5,
        help="Number of evenly spaced frames to extract (default: 5).",
    )
    return parser.parse_args()


def extract_frames(source: str, out_dir: str, num: int) -> list[Path]:
    """Extract `num` evenly spaced frames from `source` into `out_dir`.

    Returns the list of written frame paths.
    """
    src_path = Path(source)
    if not src_path.exists():
        raise FileNotFoundError(f"Source video not found: {src_path}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(src_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {src_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    print(f"Video: {src_path}  {width}x{height}  fps={fps:.2f}  frames={total_frames}")

    num = max(1, num)
    if total_frames <= 0:
        # Fallback: read sequentially when frame count is unavailable.
        indices = list(range(num))
    else:
        step = max(1, total_frames // (num + 1))
        indices = [min(total_frames - 1, step * (i + 1)) for i in range(num)]

    written: list[Path] = []
    for i, frame_index in enumerate(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        dest = out_path / f"frame_{i:03d}_idx{frame_index:06d}.jpg"
        cv2.imwrite(str(dest), frame)
        written.append(dest)
        print(f"Wrote {dest}")

    capture.release()
    if not written:
        raise RuntimeError("No frames could be extracted from the source video.")
    return written


def main() -> None:
    """Run frame extraction entrypoint."""
    args = parse_args()
    written = extract_frames(source=args.source, out_dir=args.out, num=args.num)
    print(f"Extracted {len(written)} frame(s) to {args.out}")


if __name__ == "__main__":
    main()
