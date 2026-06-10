"""Run local video demo pipeline with configured camera sources."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    AppSettings,
    CameraSettings,
    choose_camera,
    infer_camera_id,
    infer_source_type,
    load_app_settings,
    load_camera_settings,
    parse_source_value,
    redact_source_uri,
)
from src.video.frame_processor import FrameProcessor
from src.video.stream_reader import StreamReader
from src.video.video_writer import VideoWriter
from src.vision.annotator import annotate_frame
from src.vision.crowd_analyzer import CrowdAnalyzer, ZoneAlertThresholds
from src.vision.detection_filter import DetectionRegionFilter
from src.vision.detector import DEFAULT_HEAD_MODEL_PATH, Detector, normalize_detector_mode
from src.vision.fusion_tracker import HybridTracker
from src.vision.hybrid_detector import HybridDetector, load_hybrid_roi_config
from src.vision.line_counter import LineConfig, LineManager
from src.vision.track_stitcher import TrackStitcher, TrackStitcherConfig
from src.vision.tracker import Tracker
from src.vision.zone_manager import PointStrategy, ZoneConfig, ZoneManager


DETECTOR_MODE_LABELS = {
    "body": "Full-Body Passenger Detection",
    "head": "Head-Based Passenger Detection",
    "hybrid": "Hybrid Body+Head Passenger Detection",
}


def parse_args() -> argparse.Namespace:
    """Parse command line args for video demo pipeline."""
    parser = argparse.ArgumentParser(description="Run YOLO person detection/tracking demo.")
    parser.add_argument("--config", type=Path, default=Path("configs/app.yaml"))
    parser.add_argument("--camera-config", type=Path, default=Path("configs/cameras.yaml"))
    parser.add_argument("--camera-id", type=str, default=None, metavar="CAMERA_ID")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Input source: file path, webcam index (e.g. 0), or RTSP/HTTP URL.",
    )
    parser.add_argument("--source-type", type=str, default=None, choices=["file", "webcam", "rtsp", "http"])
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output annotated video path.",
    )
    parser.add_argument(
        "--zones-config",
        type=Path,
        default=None,
        help="Path to zone/line JSON config file.",
    )
    parser.add_argument("--model", type=str, default=None, help="YOLO model weights path or model name.")
    parser.add_argument(
        "--detector-mode",
        type=str,
        choices=["body", "head", "hybrid"],
        default="body",
        help=(
            "Detection target: body uses pretrained person detection; head uses the fine-tuned "
            "head detector; hybrid runs body detection in near_body_zone and head detection in "
            "far_head_zone, then fuses both into one tracked stream."
        ),
    )
    parser.add_argument(
        "--body-model",
        type=str,
        default=None,
        help="Hybrid mode: body/person detection weights (defaults to the configured body model).",
    )
    parser.add_argument(
        "--head-model",
        type=str,
        default=None,
        help="Hybrid mode: head detection weights (defaults to the fine-tuned head detector).",
    )
    parser.add_argument(
        "--body-confidence",
        type=float,
        default=None,
        help="Hybrid mode: confidence threshold for the body detector.",
    )
    parser.add_argument(
        "--head-confidence",
        type=float,
        default=None,
        help="Hybrid mode: confidence threshold for the head detector.",
    )
    parser.add_argument(
        "--body-imgsz",
        type=int,
        default=None,
        help="Hybrid mode: inference image size for the body detector.",
    )
    parser.add_argument(
        "--head-imgsz",
        type=int,
        default=None,
        help="Hybrid mode: inference image size for the head detector.",
    )
    parser.add_argument(
        "--body-max-det",
        type=int,
        default=None,
        help="Hybrid mode: maximum detections per frame for the body detector.",
    )
    parser.add_argument(
        "--head-max-det",
        type=int,
        default=None,
        help="Hybrid mode: maximum detections per frame for the head detector.",
    )
    parser.add_argument("--device", type=str, default=None, help="Inference device: cpu, cuda, mps, etc.")
    parser.add_argument("--tracker", type=str, default=None, choices=["bytetrack", "botsort"])
    parser.add_argument("--confidence", type=float, default=None, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=None, help="YOLO non-max suppression IoU threshold.")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="YOLO inference image size. Head mode benefits from larger values on CCTV-angle video.",
    )
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable Ultralytics test-time augmentation when supported. Slower, sometimes improves recall.",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=None,
        help="Maximum detections per frame. Increase for dense crowds.",
    )
    parser.add_argument("--max-fps", type=float, default=None, help="Optional processing FPS cap.")
    parser.add_argument("--frame-stride", type=int, default=None, help="Process every Nth frame.")
    parser.add_argument("--resize-width", type=int, default=None, help="Resize frames before inference.")
    parser.add_argument("--show", action="store_true", help="Show annotated preview window while processing.")
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        metavar="DB_PATH",
        help="Optional path to SQLite analytics database.",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Seconds between zone-occupancy snapshots written to the DB.",
    )
    return parser.parse_args()


def load_thresholds(
    threshold_path: Path,
    zone_manager: ZoneManager,
    *,
    threshold_profile: str | None = None,
) -> dict[str, ZoneAlertThresholds]:
    """Load crowd alert thresholds by zone with fallback to zone config defaults."""
    parsed: dict[str, Any] = {}
    if threshold_path.exists():
        loaded = yaml.safe_load(threshold_path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            parsed = loaded

    threshold_root = parsed.get("thresholds", {}) if isinstance(parsed, dict) else {}
    default_thresholds = threshold_root.get("default", {}) if isinstance(threshold_root, dict) else {}
    zones_thresholds = threshold_root.get("zones", {}) if isinstance(threshold_root, dict) else {}
    profiles = parsed.get("profiles", {}) if isinstance(parsed, dict) else {}
    selected_profile = profiles.get(threshold_profile or "", {}) if isinstance(profiles, dict) else {}

    default_warning = int(default_thresholds.get("warning_count", default_thresholds.get("crowd_count", 20)))
    default_critical = int(default_thresholds.get("critical_count", default_thresholds.get("critical_count", 30)))
    default_dwell = float(default_thresholds.get("dwell_seconds", default_thresholds.get("crowd_dwell_seconds", 0.0)))
    default_clear = int(default_thresholds.get("clear_below_count", 0))

    by_zone: dict[str, ZoneAlertThresholds] = {}
    for zone in zone_manager.zones:
        zone_thresholds = zones_thresholds.get(zone.id, {}) if isinstance(zones_thresholds, dict) else {}
        profile_thresholds = selected_profile.get(zone.id, {}) if isinstance(selected_profile, dict) else {}
        combined = {**zone_thresholds, **profile_thresholds}
        warning = int(combined.get("warning_count", zone.warning_threshold or default_warning))
        critical = int(combined.get("critical_count", zone.critical_threshold or default_critical))
        warning = max(0, warning)
        critical = max(warning, critical)
        by_zone[zone.id] = ZoneAlertThresholds(
            warning=warning,
            critical=critical,
            dwell_seconds=float(combined.get("dwell_seconds", combined.get("crowd_dwell_seconds", default_dwell))),
            clear_below_count=int(combined.get("clear_below_count", default_clear)),
        )
    return by_zone


def _scale_point(point: tuple[float, float], scale_x: float, scale_y: float) -> tuple[float, float]:
    return (point[0] * scale_x, point[1] * scale_y)


def load_zone_and_line_managers(
    zones_config_path: Path,
    *,
    target_width: int | None = None,
    target_height: int | None = None,
    point_strategy: PointStrategy = "bottom_center",
    per_detection_anchor: bool = False,
) -> tuple[ZoneManager, list[LineManager]]:
    """Load zone and line configuration managers from one JSON file."""
    if not zones_config_path.exists():
        raise FileNotFoundError(f"Zones config not found: {zones_config_path}")

    raw = json.loads(zones_config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid zones config format: {zones_config_path}")

    ref_width = int(raw.get("frame_width", target_width or 0) or 0)
    ref_height = int(raw.get("frame_height", target_height or 0) or 0)
    scale_x = (target_width / ref_width) if target_width and ref_width else 1.0
    scale_y = (target_height / ref_height) if target_height and ref_height else 1.0

    zones_raw = raw.get("zones", [])
    if not isinstance(zones_raw, list):
        raise ValueError("Config field 'zones' must be a list.")
    zone_configs: list[ZoneConfig] = []
    for zone in zones_raw:
        if not isinstance(zone, dict):
            raise ValueError("Each zone entry must be an object.")
        zone_copy = copy.deepcopy(zone)
        if scale_x != 1.0 or scale_y != 1.0:
            zone_copy["polygon"] = [
                [point[0] * scale_x, point[1] * scale_y] for point in zone_copy.get("polygon", [])
            ]
        zone_configs.append(ZoneConfig.from_dict(zone_copy))
    zone_manager = ZoneManager(
        zones=zone_configs,
        point_strategy=point_strategy,
        per_detection_anchor=per_detection_anchor,
    )

    lines_raw = raw.get("lines", [])
    if not isinstance(lines_raw, list):
        raise ValueError("Config field 'lines' must be a list.")
    line_managers: list[LineManager] = []
    for idx, line in enumerate(lines_raw):
        if not isinstance(line, dict):
            raise ValueError(f"Line entry at index {idx} must be an object.")
        start_raw = line.get("start")
        end_raw = line.get("end")
        if not isinstance(start_raw, list | tuple) or len(start_raw) != 2:
            raise ValueError(f"Invalid line start for line index {idx}: {start_raw!r}")
        if not isinstance(end_raw, list | tuple) or len(end_raw) != 2:
            raise ValueError(f"Invalid line end for line index {idx}: {end_raw!r}")
        start = (float(start_raw[0]), float(start_raw[1]))
        end = (float(end_raw[0]), float(end_raw[1]))
        if scale_x != 1.0 or scale_y != 1.0:
            start = _scale_point(start, scale_x, scale_y)
            end = _scale_point(end, scale_x, scale_y)
        line_managers.append(
            LineManager(
                config=LineConfig(
                    id=str(line.get("id") or line.get("name") or f"line_{idx + 1}"),
                    name=str(line.get("name") or line.get("id") or f"line_{idx + 1}"),
                    start=start,
                    end=end,
                    in_label=str(line.get("in_label", "IN")),
                    out_label=str(line.get("out_label", "OUT")),
                ),
                point_strategy=point_strategy,
                per_detection_anchor=per_detection_anchor,
            )
        )
    return zone_manager, line_managers


def print_source_help(source: str | int) -> None:
    """Print actionable guidance when source is invalid or unavailable."""
    print(f"Unable to open source: {redact_source_uri(source)}")
    if str(source) == "data/input_videos/sample.mp4":
        print("No sample video found at data/input_videos/sample.mp4.")
        print("Please add the selected Pexels platform sample there, or pass --source with another path.")
    print("Provide one of the following:")
    print("  1) Existing video file path")
    print("  2) Webcam index (e.g. --source 0)")
    print('  3) RTSP/HTTP URL (prefer --source "$CAM1_RTSP_URL" or configs/cameras.yaml env placeholders)')


def resolve_runtime_args(args: argparse.Namespace, settings: AppSettings, camera: CameraSettings | None) -> dict[str, Any]:
    """Resolve effective runtime settings from config, camera, and CLI overrides."""
    detector_mode = normalize_detector_mode(args.detector_mode)
    source = parse_source_value(args.source) if args.source is not None else None
    source = source if source is not None else (camera.source if camera is not None else "data/input_videos/sample.mp4")
    source_type = args.source_type or (camera.source_type if camera is not None else None)
    source_type = source_type or infer_source_type(source)
    camera_id = args.camera_id or (camera.camera_id if camera is not None else infer_camera_id(source, source_type))
    store_output_video = camera.store_output_video if camera is not None else True
    if args.output is not None:
        store_output_video = True
    output = args.output or settings.outputs.default_output_video
    zones_config = args.zones_config or Path(camera.zones_config if camera is not None else "configs/zones.example.json")
    max_fps = args.max_fps if args.max_fps is not None else (
        camera.fps_limit if camera is not None and camera.fps_limit is not None else settings.runtime.max_fps
    )
    frame_stride = args.frame_stride if args.frame_stride is not None else (
        camera.frame_stride if camera is not None and camera.frame_stride is not None else settings.runtime.frame_stride
    )
    snapshot_interval = (
        args.snapshot_interval
        if args.snapshot_interval is not None
        else settings.persistence.snapshot_interval_seconds
    )
    model = args.model or (str(DEFAULT_HEAD_MODEL_PATH) if detector_mode == "head" else settings.model.weights)
    confidence = args.confidence if args.confidence is not None else settings.model.confidence
    if detector_mode == "head" and args.confidence is None:
        # Small heads in CCTV-angle video are often lower confidence than full
        # body boxes; keep explicit CLI/config values tunable.
        confidence = settings.model.head_confidence
    imgsz = args.imgsz if args.imgsz is not None else settings.model.imgsz
    if detector_mode == "head" and args.imgsz is None:
        # Avoid shrinking 1280x720 CCTV footage to 640px for head detection,
        # where many heads become too small for reliable inference.
        imgsz = settings.model.head_imgsz
    max_det = args.max_det if args.max_det is not None else settings.model.max_det
    if detector_mode == "head" and args.max_det is None:
        max_det = settings.model.head_max_det
    augment = settings.model.augment if args.augment is None else bool(args.augment)
    iou = args.iou if args.iou is not None else settings.model.iou

    # Hybrid mode runs two detectors with independent per-source overrides.
    body_model = args.body_model or args.model or settings.model.weights
    head_model = args.head_model or str(DEFAULT_HEAD_MODEL_PATH)
    body_confidence = (
        args.body_confidence if args.body_confidence is not None else settings.model.confidence
    )
    head_confidence = (
        args.head_confidence if args.head_confidence is not None else settings.model.head_confidence
    )
    body_imgsz = args.body_imgsz if args.body_imgsz is not None else settings.model.imgsz
    head_imgsz = args.head_imgsz if args.head_imgsz is not None else settings.model.head_imgsz
    body_max_det = args.body_max_det if args.body_max_det is not None else settings.model.max_det
    head_max_det = args.head_max_det if args.head_max_det is not None else settings.model.head_max_det

    per_detection_anchor = detector_mode == "hybrid"
    if detector_mode == "hybrid":
        point_strategy: PointStrategy = "bottom_center"
    elif detector_mode == "head":
        point_strategy = "center"
    else:
        point_strategy = "bottom_center"
    return {
        "source": source,
        "source_type": source_type,
        "output": output,
        "zones_config": Path(zones_config),
        "model": model,
        "detector_mode": detector_mode,
        "point_strategy": point_strategy,
        "per_detection_anchor": per_detection_anchor,
        "body_model": body_model,
        "head_model": head_model,
        "body_confidence": float(body_confidence),
        "head_confidence": float(head_confidence),
        "body_imgsz": int(body_imgsz),
        "head_imgsz": int(head_imgsz),
        "body_max_det": int(body_max_det),
        "head_max_det": int(head_max_det),
        "detector_mode_label": DETECTOR_MODE_LABELS[detector_mode],
        "confidence": float(confidence),
        "iou": float(iou),
        "device": args.device or settings.model.device,
        "tracker_type": args.tracker or settings.tracker.type,
        "db": args.db,
        "camera_id": camera_id,
        "camera_name": camera.name if camera is not None else camera_id,
        "camera_description": camera.description if camera is not None else "",
        "threshold_profile": camera.threshold_profile if camera is not None else None,
        "snapshot_interval": float(snapshot_interval),
        "max_fps": float(max_fps or 0.0),
        "frame_stride": int(frame_stride),
        "resize_width": args.resize_width if args.resize_width is not None else settings.runtime.resize_width,
        "imgsz": int(imgsz),
        "augment": bool(augment),
        "max_det": int(max_det),
        "store_output_video": store_output_video,
    }


def build_models(
    settings: AppSettings,
    *,
    model: str,
    confidence: float,
    device: str,
    tracker_type: str,
    detector_mode: str,
    imgsz: int,
    iou: float,
    augment: bool,
    max_det: int,
) -> tuple[Detector, Tracker]:
    """Build detector and tracker from settings and overrides."""
    mode = normalize_detector_mode(detector_mode)
    detector = Detector(
        weights_path=model,
        device=device,
        confidence=confidence,
        iou=iou,
        person_class_id=settings.model.person_class_id,
        imgsz=imgsz,
        augment=augment,
        max_det=max_det,
        half=settings.model.half,
        accuracy_weights=settings.model.accuracy_weights,
        legacy_fallback_weights=settings.model.legacy_fallback_weights,
        use_fine_tuned_if_available=settings.model.use_fine_tuned_if_available if mode == "body" else False,
        detector_mode=mode,
    )
    tracker = Tracker(
        tracker_type=tracker_type,
        weights_path=model,
        device=device,
        confidence=confidence,
        iou=iou,
        person_class_id=settings.model.person_class_id,
        imgsz=imgsz,
        augment=augment,
        max_det=max_det,
        half=settings.model.half,
        tracker_config_overrides={
            **settings.tracker.config_overrides,
            **(settings.tracker.head_config_overrides if mode == "head" else {}),
        },
        accuracy_weights=settings.model.accuracy_weights,
        legacy_fallback_weights=settings.model.legacy_fallback_weights,
        use_fine_tuned_if_available=settings.model.use_fine_tuned_if_available if mode == "body" else False,
        detector_mode=mode,
    )
    return detector, tracker


def build_hybrid_models(
    settings: AppSettings,
    effective: dict[str, Any],
    *,
    target_width: int | None,
    target_height: int | None,
) -> tuple[HybridDetector, HybridTracker]:
    """Build the hybrid body+head detector and the unified fusion tracker."""
    body_detector = Detector(
        weights_path=effective["body_model"],
        device=effective["device"],
        confidence=effective["body_confidence"],
        iou=effective["iou"],
        person_class_id=settings.model.person_class_id,
        imgsz=effective["body_imgsz"],
        augment=effective["augment"],
        max_det=effective["body_max_det"],
        half=settings.model.half,
        accuracy_weights=settings.model.accuracy_weights,
        legacy_fallback_weights=settings.model.legacy_fallback_weights,
        use_fine_tuned_if_available=settings.model.use_fine_tuned_if_available,
        detector_mode="body",
    )
    head_detector = Detector(
        weights_path=effective["head_model"],
        device=effective["device"],
        confidence=effective["head_confidence"],
        iou=effective["iou"],
        person_class_id=settings.model.person_class_id,
        imgsz=effective["head_imgsz"],
        augment=effective["augment"],
        max_det=effective["head_max_det"],
        half=settings.model.half,
        use_fine_tuned_if_available=False,
        detector_mode="head",
    )
    roi_config = load_hybrid_roi_config(
        effective["zones_config"],
        target_width=target_width,
        target_height=target_height,
    )
    # The body detector always runs full-frame in hybrid mode so passengers
    # outside any near zone are still detected; fusion de-dupes overlapping far
    # heads. Empty near_body_polygons disables both the ROI crop and the polygon
    # post-filter. The head detector stays restricted to far_head_polygons.
    roi_config.near_body_polygons = []
    hybrid_detector = HybridDetector(
        body_detector=body_detector,
        head_detector=head_detector,
        roi_config=roi_config,
    )
    hybrid_tracker = HybridTracker(prefer_bytetrack=True)
    return hybrid_detector, hybrid_tracker


def build_track_stitcher(settings: AppSettings, detector_mode: str) -> TrackStitcher | None:
    """Build the optional head-ID stitcher for head or hybrid modes."""
    mode = normalize_detector_mode(detector_mode)
    if mode not in {"head", "hybrid"}:
        return None
    raw = dict(settings.tracker.head_stitching)
    if not bool(raw.get("enabled", False)):
        return None
    config = TrackStitcherConfig(
        enabled=True,
        gap_frames=int(raw.get("gap_frames", 220)),
        dist_heads=float(raw.get("dist_heads", 5.0)),
        mode=str(raw.get("mode", "observation")),
        ambiguity_ratio=float(raw.get("ambiguity_ratio", 0.95)),
        max_speed_heads=float(raw.get("max_speed_heads", 0.85)),
        direction_weight=float(raw.get("direction_weight", 0.12)),
        max_direction_cost=float(raw.get("max_direction_cost", 0.85)),
        max_jump_heads=float(raw.get("max_jump_heads", 7.5)),
    )
    return TrackStitcher(config)


def validate_head_model(model: str) -> str | None:
    """Return an actionable error message when a head-model path is missing."""
    model_path = Path(model)
    if model_path.exists():
        return None
    expected = DEFAULT_HEAD_MODEL_PATH
    return (
        "Head detector model not found.\n"
        f"Expected Colab-trained weights at: {expected}\n"
        f"Requested model path: {model}\n"
        "Place the fine-tuned best.pt at the expected path or pass --model with an existing file. "
        "Do not train locally for this workflow."
    )


def run_pipeline(args: argparse.Namespace) -> int:
    """Run end-to-end stream read -> process -> annotate -> write pipeline."""
    settings = load_app_settings(args.config)
    cameras = load_camera_settings(args.camera_config) if args.camera_config.exists() else {}
    if args.camera_id:
        camera = choose_camera(cameras, args.camera_id)
        if camera is None:
            known = ", ".join(sorted(cameras)) or "(none configured)"
            print(f"Camera '{args.camera_id}' not found in {args.camera_config}. Known cameras: {known}")
            return 1
    elif args.source is None:
        camera = choose_camera(cameras)
    else:
        # Explicit sources are ad-hoc unless the caller names a configured camera.
        camera = None
    effective = resolve_runtime_args(args, settings, camera)

    if effective["detector_mode"] == "head":
        missing_model_message = validate_head_model(effective["model"])
        if missing_model_message:
            print(missing_model_message)
            return 1
    elif effective["detector_mode"] == "hybrid":
        missing_model_message = validate_head_model(effective["head_model"])
        if missing_model_message:
            print(missing_model_message)
            return 1

    reader = StreamReader(
        source=effective["source"],
        source_type=effective["source_type"],
        frame_stride=effective["frame_stride"],
        max_fps=effective["max_fps"],
        reconnect_backoff_seconds=settings.runtime.reconnect_backoff_seconds,
        max_reconnect_attempts=settings.runtime.max_reconnect_attempts,
        read_timeout_seconds=settings.runtime.stream_read_timeout_seconds,
        camera_id=effective["camera_id"],
    )

    if not reader.source_exists():
        print_source_help(effective["source"])
        return 1

    reader.open()
    if not reader.is_opened():
        print_source_help(effective["source"])
        return 1

    source_width = int(reader.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) if reader.capture else 0
    source_height = int(reader.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) if reader.capture else 0
    target_width = effective["resize_width"] if effective["resize_width"] else source_width or None
    target_height = None
    if target_width and source_width and source_height:
        target_height = int(source_height * (target_width / float(source_width)))

    try:
        zone_manager, line_managers = load_zone_and_line_managers(
            effective["zones_config"],
            target_width=target_width,
            target_height=target_height,
            point_strategy=effective["point_strategy"],
            per_detection_anchor=effective["per_detection_anchor"],
        )
    except (FileNotFoundError, ValueError, KeyError, TypeError) as error:
        print(f"Zone/line config load failed: {error}")
        reader.release()
        return 1

    thresholds_by_zone = load_thresholds(
        Path("configs/thresholds.yaml"),
        zone_manager,
        threshold_profile=effective["threshold_profile"],
    )
    crowd_analyzer = CrowdAnalyzer(thresholds_by_zone=thresholds_by_zone)
    detection_filter = DetectionRegionFilter.from_zones_file(
        effective["zones_config"],
        detector_mode=effective["detector_mode"],
    )

    hybrid_detector = None
    hybrid_tracker = None
    detector = None
    tracker = None
    track_stitcher = build_track_stitcher(settings, effective["detector_mode"])
    try:
        if effective["detector_mode"] == "hybrid":
            hybrid_detector, hybrid_tracker = build_hybrid_models(
                settings,
                effective,
                target_width=target_width,
                target_height=target_height,
            )
        else:
            detector, tracker = build_models(
                settings,
                model=effective["model"],
                confidence=effective["confidence"],
                device=effective["device"],
                tracker_type=effective["tracker_type"],
                detector_mode=effective["detector_mode"],
                imgsz=effective["imgsz"],
                iou=effective["iou"],
                augment=effective["augment"],
                max_det=effective["max_det"],
            )
    except RuntimeError as error:
        print(f"Model initialization failed: {error}")
        reader.release()
        return 1

    processor = FrameProcessor(
        detector=detector,
        tracker=tracker,
        zone_manager=zone_manager,
        line_managers=line_managers,
        crowd_analyzer=crowd_analyzer,
        detection_filter=detection_filter,
        use_tracking=True,
        resize_width=effective["resize_width"],
        hybrid_detector=hybrid_detector,
        hybrid_tracker=hybrid_tracker,
        track_stitcher=track_stitcher,
    )

    analytics_logger = None
    if effective["db"]:
        from src.analytics.event_logger import AnalyticsLogger

        db_url = f"sqlite:///{Path(effective['db']).resolve()}"
        Path(effective["db"]).parent.mkdir(parents=True, exist_ok=True)
        analytics_logger = AnalyticsLogger(
            db_url=db_url,
            camera_id=effective["camera_id"] or "camera",
            snapshot_interval_secs=effective["snapshot_interval"],
            detector_mode=effective["detector_mode"],
        )
        analytics_logger.register_camera(
            camera_name=effective["camera_name"],
            source_type=reader.source_type,
            description=effective["camera_description"],
            zones_config_path=str(effective["zones_config"]),
            enabled=True,
        )
        session_id = analytics_logger.start_session(
            source=redact_source_uri(effective["source"]),
            source_type=reader.source_type,
            model_weights=effective["model"],
            detector_mode=effective["detector_mode"],
            tracker_type=effective["tracker_type"],
            zones_config_path=str(effective["zones_config"]),
        )
        print(f"Analytics logging to: {effective['db']}  (session_id={session_id})")

    writer = VideoWriter(output_path=effective["output"]) if effective["store_output_video"] else None
    source_fps = reader.estimated_fps()
    if writer is not None:
        writer.set_fps(source_fps if source_fps > 0 else 20.0)

    processed = 0
    total_detections = 0
    last_status: tuple[int, int] = (-1, -1)
    total_unique_passengers = 0
    try:
        for frame in reader.frames():
            result = processor.process(
                frame=frame.data,
                frame_index=frame.index,
                timestamp=frame.timestamp,
            )
            total_unique_passengers = result.unique_passengers_seen
            total_detections += len(result.detections)
            if analytics_logger is not None:
                track_ids = [
                    detection.track_id
                    for detection in result.detections
                    if detection.track_id is not None
                ]
                analytics_logger.log_frame(
                    frame_index=frame.index,
                    source_timestamp=frame.source_timestamp,
                    total_detections=len(result.detections),
                    zone_occupancy=result.zone_occupancy,
                    zone_alerts=result.zone_alerts,
                    line_counts=result.line_counts,
                    line_events=result.line_events,
                    track_ids=[track_id for track_id in track_ids if track_id is not None],
                    processing_fps=result.processing_fps,
                    source_type=frame.source_type,
                    reconnect_count=frame.reconnect_count,
                    dropped_frames=frame.dropped_frames,
                    detector_mode=effective["detector_mode"],
                )
                status_key = (frame.reconnect_count, frame.dropped_frames)
                if status_key != last_status and (frame.reconnect_count > 0 or frame.dropped_frames > 0):
                    analytics_logger.log_stream_health(
                        status="streaming",
                        reconnect_count=frame.reconnect_count,
                        dropped_frames=frame.dropped_frames,
                        message="live stream counters updated",
                    )
                    last_status = status_key

            annotated = annotate_frame(
                frame=result.frame,
                detections=result.detections,
                zone_manager=zone_manager,
                line_managers=line_managers,
                zone_occupancy=result.zone_occupancy,
                zone_alerts=result.zone_alerts,
                line_counts=result.line_counts,
                overlays={
                    "Mode": effective["detector_mode_label"],
                    "Camera": effective["camera_id"] or effective["camera_name"],
                    "Source": reader.source_type,
                    "Frame": frame.index,
                    "Confidence": f"{effective['confidence']:.2f}",
                    "Img size": effective["imgsz"],
                    "IoU": f"{effective['iou']:.2f}",
                    "Tracker": effective["tracker_type"],
                    "Current detections": len(result.detections),
                    "Total detections": total_detections,
                    "Unique": result.unique_passengers_seen,
                    "Max det": effective["max_det"],
                    "Augment": "on" if effective["augment"] else "off",
                    "FPS": f"{result.processing_fps:.2f}",
                    "Reconnects": frame.reconnect_count,
                },
            )
            if writer is not None:
                writer.write(annotated)
            if args.show:
                cv2.imshow("Crowd Analytics Demo", annotated)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            processed += 1
    finally:
        reader.release()
        if writer is not None:
            writer.close()
        if args.show:
            cv2.destroyAllWindows()
        if analytics_logger is not None:
            analytics_logger.end_session(
                total_frames=processed,
                total_unique_passengers=total_unique_passengers,
            )

    if processed == 0:
        print_source_help(effective["source"])
        return 1

    print(f"Processed {processed} frames.")
    if writer is not None:
        print(f"Annotated output saved to: {effective['output']}")
    else:
        print("Annotated output video disabled by camera privacy config.")
    return 0


def main() -> None:
    """Run script entrypoint and return helpful errors on failure."""
    args = parse_args()
    exit_code = run_pipeline(args)
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
