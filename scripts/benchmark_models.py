"""Benchmark YOLO model choices on a sample video source."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_app_settings
from src.video.stream_reader import StreamReader
from src.vision.detector import Detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark multiple YOLO model weights on a video source.")
    parser.add_argument("--source", type=str, default="data/input_videos/sample.mp4")
    parser.add_argument("--frames", type=int, default=120, help="Maximum frames to benchmark per model.")
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["yolo11n.pt", "yolo11s.pt", "yolov8n.pt"],
        help="Model names or local weights paths to benchmark.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/outputs/model_benchmark.csv"),
        help="Output CSV for benchmark summary.",
    )
    return parser.parse_args()


def benchmark_model(model_name: str, source: str, frames: int, confidence: float, device: str) -> dict[str, object]:
    reader = StreamReader(source=source)
    if not reader.source_exists():
        raise FileNotFoundError(f"Source not found: {source}")
    detector = Detector(weights_path=model_name, confidence=confidence, device=device)
    processed = 0
    detection_total = 0
    elapsed_total = 0.0
    for frame in reader.frames():
        started = time.perf_counter()
        result = detector.detect(frame.data, frame_index=frame.index, timestamp=frame.timestamp)
        elapsed_total += time.perf_counter() - started
        detection_total += len(result.detections)
        processed += 1
        if processed >= frames:
            break
    reader.release()
    fps = (processed / elapsed_total) if elapsed_total > 0 else 0.0
    avg_detections = (detection_total / processed) if processed > 0 else 0.0
    return {
        "model": model_name,
        "frames_processed": processed,
        "processing_time_seconds": round(elapsed_total, 4),
        "fps": round(fps, 3),
        "avg_detections_per_frame": round(avg_detections, 3),
        "status": "ok",
    }


def main() -> None:
    args = parse_args()
    settings = load_app_settings()
    confidence = float(args.confidence if args.confidence is not None else settings.model.confidence)
    device = args.device or settings.model.device
    results: list[dict[str, object]] = []

    for model_name in args.models:
        try:
            result = benchmark_model(model_name, args.source, args.frames, confidence, device)
            print(
                f"{model_name}: fps={result['fps']} avg_detections={result['avg_detections_per_frame']} "
                f"frames={result['frames_processed']}"
            )
            results.append(result)
        except Exception as exc:
            message = str(exc)
            print(f"{model_name}: skipped ({message})")
            results.append(
                {
                    "model": model_name,
                    "frames_processed": 0,
                    "processing_time_seconds": 0.0,
                    "fps": 0.0,
                    "avg_detections_per_frame": 0.0,
                    "status": f"skipped: {message}",
                }
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "frames_processed",
                "processing_time_seconds",
                "fps",
                "avg_detections_per_frame",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"Benchmark summary written to {args.output_csv}")


if __name__ == "__main__":
    main()
