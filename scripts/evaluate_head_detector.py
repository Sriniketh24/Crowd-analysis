"""Evaluate a fine-tuned YOLO head detector on a sample image and video."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import cv2

try:
    from ultralytics import YOLO
except ModuleNotFoundError:  # pragma: no cover - environment-specific dependency
    YOLO = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run sample inference with a fine-tuned YOLO head detector.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/fine_tuned/head_detector/weights/best.pt"),
        help="Fine-tuned head-detector weights.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/input_videos/sample.mp4"),
        help="Video source for annotated demo output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/outputs/head_demo.mp4"),
        help="Annotated output video path.",
    )
    parser.add_argument("--data", type=Path, default=Path("configs/head_dataset.yaml"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Resolve a CLI path relative to the project root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def find_sample_image(dataset_yaml: Path) -> Path | None:
    """Find one validation or training image for a quick still-image inference."""
    if not dataset_yaml.exists():
        return None

    import yaml

    config = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "path" not in config:
        return None

    dataset_path = Path(config["path"])
    base = dataset_path if dataset_path.is_absolute() else PROJECT_ROOT / dataset_path
    for split_key in ("val", "train"):
        rel = config.get(split_key)
        if not rel:
            continue
        images_dir = base / str(rel)
        if not images_dir.exists():
            continue
        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} and image_path.exists():
                return image_path
    return None


def run_sample_image(model: YOLO, image_path: Path, args: argparse.Namespace) -> None:
    """Run inference on one still image and print the detection count."""
    results = model.predict(
        source=str(image_path),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )
    boxes = results[0].boxes if results else None
    count = 0 if boxes is None else len(boxes)
    print(f"Sample image inference: {image_path} -> {count} heads detected")


def run_video(model: YOLO, source: Path, output: Path, args: argparse.Namespace) -> dict[str, float | int]:
    """Run video inference, save annotated output, and return summary stats."""
    if not source.exists():
        raise FileNotFoundError(f"Video source not found: {source}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid video dimensions for {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to create output video: {output}")

    frame_count = 0
    total_detections = 0
    max_detections = 0
    started_at = perf_counter()

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        results = model.predict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes
        detection_count = 0 if boxes is None else len(boxes)
        total_detections += detection_count
        max_detections = max(max_detections, detection_count)
        writer.write(result.plot())
        frame_count += 1

    elapsed = perf_counter() - started_at
    capture.release()
    writer.release()

    if frame_count == 0:
        raise RuntimeError(f"No frames were read from {source}")

    return {
        "frames": frame_count,
        "avg_heads": total_detections / frame_count,
        "max_heads": max_detections,
        "processing_fps": frame_count / elapsed if elapsed > 0 else 0.0,
    }


def main() -> None:
    """Run sample image and sample video inference."""
    args = parse_args()
    model_path = resolve_path(args.model)
    source = resolve_path(args.source)
    output = resolve_path(args.output)
    dataset_yaml = resolve_path(args.data)

    if YOLO is None:
        raise RuntimeError("Ultralytics is not installed. Install dependencies first: pip install -r requirements.txt")
    if not model_path.exists():
        raise FileNotFoundError(f"Fine-tuned head detector not found: {model_path}")

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    sample_image = find_sample_image(dataset_yaml)
    if sample_image is not None:
        run_sample_image(model, sample_image, args)
    else:
        print("Sample image inference skipped: no dataset image found.")

    stats = run_video(model, source, output, args)
    print("Video inference observations:")
    print(f"  source: {source}")
    print(f"  frames: {stats['frames']}")
    print(f"  average heads/frame: {stats['avg_heads']:.2f}")
    print(f"  max heads/frame: {stats['max_heads']}")
    print(f"  processing FPS: {stats['processing_fps']:.2f}")
    print(f"Output video: {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
