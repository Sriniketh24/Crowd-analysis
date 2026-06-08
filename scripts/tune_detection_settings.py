"""Sweep detection/tracking settings and write tuning observations to CSV.

These results are operational tuning signals, not true accuracy metrics. True
accuracy requires labelled ground truth or manual counts for the same frames.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path
from time import perf_counter

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_video_demo import load_thresholds, load_zone_and_line_managers
from src.config import load_app_settings, parse_source_value
from src.video.frame_processor import FrameProcessor
from src.video.video_writer import VideoWriter
from src.vision.annotator import annotate_frame
from src.vision.crowd_analyzer import CrowdAnalyzer
from src.vision.detection_filter import DetectionRegionFilter
from src.vision.detector import DEFAULT_HEAD_MODEL_PATH, normalize_detector_mode
from src.vision.tracker import Tracker


RESULT_COLUMNS = [
    "detector_mode",
    "model",
    "confidence",
    "imgsz",
    "iou",
    "tracker",
    "frames_processed",
    "total_detections",
    "unique_tracks",
    "avg_detections_per_frame",
    "processing_fps",
    "output_video_path",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for tuning sweeps."""
    parser = argparse.ArgumentParser(
        description="Run body/head detector setting sweeps and write tuning_results.csv."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/app.yaml"))
    parser.add_argument("--source", type=str, default="data/input_videos/sample.mp4")
    parser.add_argument("--zones-config", type=Path, default=Path("configs/zones.example.json"))
    parser.add_argument("--detector-mode", choices=["body", "head", "both"], default="both")
    parser.add_argument("--model", type=str, default=None, help="Model for the selected single detector mode.")
    parser.add_argument("--body-model", type=str, default=None)
    parser.add_argument("--head-model", type=str, default=str(DEFAULT_HEAD_MODEL_PATH))
    parser.add_argument("--confidence-values", type=str, default="0.15,0.20,0.25,0.30,0.40")
    parser.add_argument("--imgsz-values", type=str, default="640,960,1280")
    parser.add_argument("--iou-values", type=str, default="0.45,0.50,0.60")
    parser.add_argument("--trackers", type=str, default="bytetrack,botsort")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-det", type=int, default=None)
    parser.add_argument("--augment", action="store_true", help="Enable Ultralytics TTA for each run.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames per setting combination.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--quick", action="store_true", help="Use a short, low-cost sweep for smoke testing.")
    parser.add_argument("--save-videos", action="store_true", help="Save annotated videos for each setting.")
    parser.add_argument("--output-csv", type=Path, default=Path("data/outputs/tuning_results.csv"))
    parser.add_argument("--video-output-dir", type=Path, default=Path("data/outputs/tuning_videos"))
    return parser.parse_args()


def _parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values


def _parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def _parse_tracker_list(raw: str) -> list[str]:
    values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one tracker.")
    unsupported = [value for value in values if value not in {"bytetrack", "botsort"}]
    if unsupported:
        raise ValueError(f"Unsupported tracker(s): {', '.join(unsupported)}")
    return values


def _modes(detector_mode: str) -> list[str]:
    if detector_mode == "both":
        return ["body", "head"]
    return [normalize_detector_mode(detector_mode)]


def _model_for_mode(args: argparse.Namespace, settings, mode: str) -> str:
    if args.model and args.detector_mode != "both":
        return args.model
    if mode == "head":
        return args.head_model
    return args.body_model or settings.model.weights


def _video_path(args: argparse.Namespace, mode: str, conf: float, imgsz: int, iou: float, tracker: str) -> Path:
    stem = f"{mode}_conf{conf:.2f}_img{imgsz}_iou{iou:.2f}_{tracker}".replace(".", "p")
    return args.video_output_dir / f"{stem}.mp4"


def _open_source(source: str | int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source: {source}")
    return capture


def _run_one_setting(
    *,
    args: argparse.Namespace,
    settings,
    source: str | int,
    mode: str,
    model: str,
    confidence: float,
    imgsz: int,
    iou: float,
    tracker_type: str,
) -> dict[str, object]:
    capture = _open_source(source)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid video dimensions for source: {source}")

    point_strategy = "center" if mode == "head" else "bottom_center"
    zone_manager, line_managers = load_zone_and_line_managers(
        args.zones_config,
        target_width=width,
        target_height=height,
        point_strategy=point_strategy,
    )
    crowd_analyzer = CrowdAnalyzer(load_thresholds(Path("configs/thresholds.yaml"), zone_manager))
    detection_filter = DetectionRegionFilter.from_zones_file(args.zones_config, detector_mode=mode)
    tracker_overrides = {
        **settings.tracker.config_overrides,
        **(settings.tracker.head_config_overrides if mode == "head" else {}),
    }
    tracker = Tracker(
        tracker_type=tracker_type,
        weights_path=model,
        device=args.device or settings.model.device,
        confidence=confidence,
        iou=iou,
        person_class_id=settings.model.person_class_id,
        imgsz=imgsz,
        augment=args.augment,
        max_det=args.max_det if args.max_det is not None else (
            settings.model.head_max_det if mode == "head" else settings.model.max_det
        ),
        half=settings.model.half,
        tracker_config_overrides=tracker_overrides,
        accuracy_weights=settings.model.accuracy_weights,
        legacy_fallback_weights=settings.model.legacy_fallback_weights,
        use_fine_tuned_if_available=settings.model.use_fine_tuned_if_available if mode == "body" else False,
        detector_mode=mode,
    )
    processor = FrameProcessor(
        tracker=tracker,
        zone_manager=zone_manager,
        line_managers=line_managers,
        crowd_analyzer=crowd_analyzer,
        detection_filter=detection_filter,
        use_tracking=True,
    )

    writer: VideoWriter | None = None
    output_video_path = ""
    if args.save_videos:
        video_path = _video_path(args, mode, confidence, imgsz, iou, tracker_type)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = VideoWriter(output_path=str(video_path))
        writer.set_fps(fps)
        output_video_path = str(video_path)

    processed = 0
    total_detections = 0
    unique_tracks = 0
    started = perf_counter()
    source_frame_index = -1
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            source_frame_index += 1
            if args.frame_stride > 1 and source_frame_index % args.frame_stride != 0:
                continue
            timestamp = source_frame_index / fps if fps > 0 else float(source_frame_index)
            result = processor.process(frame=frame, frame_index=source_frame_index, timestamp=timestamp)
            total_detections += len(result.detections)
            unique_tracks = result.unique_passengers_seen
            processed += 1

            if writer is not None:
                annotated = annotate_frame(
                    frame=result.frame,
                    detections=result.detections,
                    zone_manager=zone_manager,
                    line_managers=line_managers,
                    zone_occupancy=result.zone_occupancy,
                    zone_alerts=result.zone_alerts,
                    line_counts=result.line_counts,
                    overlays={
                        "Mode": mode,
                        "Confidence": f"{confidence:.2f}",
                        "Img size": imgsz,
                        "IoU": f"{iou:.2f}",
                        "Tracker": tracker_type,
                        "Current detections": len(result.detections),
                        "Total detections": total_detections,
                        "Unique": result.unique_passengers_seen,
                    },
                )
                writer.write(annotated)

            if args.max_frames is not None and processed >= args.max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.close()

    elapsed = perf_counter() - started
    avg = total_detections / processed if processed else 0.0
    return {
        "detector_mode": mode,
        "model": model,
        "confidence": confidence,
        "imgsz": imgsz,
        "iou": iou,
        "tracker": tracker_type,
        "frames_processed": processed,
        "total_detections": total_detections,
        "unique_tracks": unique_tracks,
        "avg_detections_per_frame": f"{avg:.4f}",
        "processing_fps": f"{(processed / elapsed) if elapsed > 0 else 0.0:.4f}",
        "output_video_path": output_video_path,
    }


def main() -> None:
    """Run the tuning sweep and write the results CSV."""
    args = parse_args()
    settings = load_app_settings(args.config)
    source = parse_source_value(args.source)
    if source is None:
        raise SystemExit("No source was provided.")
    if isinstance(source, str) and not source.startswith(("rtsp://", "http://", "https://")):
        source_path = Path(source)
        if not source_path.exists():
            raise SystemExit(f"Source not found: {source}")
    if not args.zones_config.exists():
        raise SystemExit(f"Zones config not found: {args.zones_config}")

    confidences = _parse_float_list(args.confidence_values)
    imgsz_values = _parse_int_list(args.imgsz_values)
    iou_values = _parse_float_list(args.iou_values)
    trackers = _parse_tracker_list(args.trackers)

    if args.quick:
        args.max_frames = args.max_frames or 75
        confidences = confidences[:2]
        imgsz_values = [imgsz_values[min(1, len(imgsz_values) - 1)]]
        iou_values = [iou_values[min(1, len(iou_values) - 1)]]
        trackers = trackers[:1]

    rows: list[dict[str, object]] = []
    for mode, confidence, imgsz, iou, tracker_type in itertools.product(
        _modes(args.detector_mode),
        confidences,
        imgsz_values,
        iou_values,
        trackers,
    ):
        model = _model_for_mode(args, settings, mode)
        if mode == "head" and not Path(model).exists():
            print(f"Skipping head mode because model was not found: {model}")
            continue
        print(
            f"Running {mode}: model={model}, conf={confidence}, imgsz={imgsz}, "
            f"iou={iou}, tracker={tracker_type}",
            flush=True,
        )
        row = _run_one_setting(
            args=args,
            settings=settings,
            source=source,
            mode=mode,
            model=model,
            confidence=confidence,
            imgsz=imgsz,
            iou=iou,
            tracker_type=tracker_type,
        )
        rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote tuning results, not accuracy results: {args.output_csv}")
    print("True accuracy requires manual counts or labelled ground-truth boxes for these frames.")


if __name__ == "__main__":
    main()
