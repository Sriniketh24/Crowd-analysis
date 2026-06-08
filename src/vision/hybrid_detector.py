"""Hybrid body+head detector with ROI-cropped inference and fusion.

For speed and accuracy each model only runs on the rectangular crop bounding
its configured ROI group:

  * ``near_body_zone`` -> full-body/person detection for near/visible people.
  * ``far_head_zone``  -> head detection for far/occluded/dense people.

Detections from each crop are translated back to full-frame coordinates,
post-filtered by the actual ROI polygons (not just the crop rectangle), and
then merged by :func:`src.vision.detection_fusion.fuse_body_head`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.vision.detection_fusion import (
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_UPPER_BODY_FRACTION,
    fuse_body_head,
)
from src.vision.detector import (
    DetectionResult,
    NormalizedDetection,
    hybrid_class_name_for_source,
)
from src.vision.zone_manager import Point, point_from_bbox, point_in_polygon

BBox = tuple[float, float, float, float]

# Anchor used to post-filter each source against its ROI polygons.
BODY_ANCHOR = "bottom_center"
HEAD_ANCHOR = "center"


@dataclass(slots=True)
class HybridRoiConfig:
    """ROI polygon groups for the hybrid detector, in full-frame coordinates."""

    near_body_polygons: list[list[Point]] = field(default_factory=list)
    far_head_polygons: list[list[Point]] = field(default_factory=list)


def translate_bbox(bbox: BBox, offset_x: float, offset_y: float) -> BBox:
    """Shift a crop-relative bbox back into full-frame coordinates."""
    x1, y1, x2, y2 = bbox
    return (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)


def crop_bounds_from_polygons(
    polygons: list[list[Point]],
    frame_width: int,
    frame_height: int,
    *,
    padding: int = 0,
) -> tuple[int, int, int, int] | None:
    """Return the integer crop rectangle bounding all ROI polygons.

    Returns ``None`` when no polygons are supplied (caller runs full frame).
    The rectangle is clamped to the frame and padded for boxes that straddle
    the ROI edge.
    """
    points = [point for polygon in polygons for point in polygon]
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1 = max(0, int(math.floor(min(xs))) - padding)
    y1 = max(0, int(math.floor(min(ys))) - padding)
    x2 = min(frame_width, int(math.ceil(max(xs))) + padding)
    y2 = min(frame_height, int(math.ceil(max(ys))) + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def detection_anchor_in_polygons(
    bbox: BBox,
    polygons: list[list[Point]],
    anchor: str,
) -> bool:
    """Return True when the detection anchor falls inside any ROI polygon."""
    if not polygons:
        return True
    anchor_point = point_from_bbox(bbox, anchor)  # type: ignore[arg-type]
    return any(point_in_polygon(anchor_point, polygon) for polygon in polygons)


def run_detector_in_rois(
    detector: object,
    frame: np.ndarray,
    polygons: list[list[Point]],
    *,
    source_type: str,
    anchor: str,
    frame_index: int,
    timestamp: float | None,
    padding: int = 0,
) -> list[NormalizedDetection]:
    """Run a detector on the ROI crop and return full-frame detections.

    ``detector`` must expose ``detect(frame, frame_index, timestamp)`` returning
    a :class:`DetectionResult`. When no polygons are configured the detector
    runs on the full frame with no polygon post-filter.
    """
    frame_height, frame_width = frame.shape[:2]
    bounds = crop_bounds_from_polygons(polygons, frame_width, frame_height, padding=padding)
    if bounds is None:
        crop = frame
        offset_x, offset_y = 0, 0
    else:
        x1, y1, x2, y2 = bounds
        crop = frame[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1
        if crop.size == 0:
            return []

    result = detector.detect(crop, frame_index=frame_index, timestamp=timestamp)  # type: ignore[attr-defined]
    detections: list[NormalizedDetection] = []
    for detection in result.detections:
        translated = translate_bbox(detection.bbox, offset_x, offset_y)
        if not detection_anchor_in_polygons(translated, polygons, anchor):
            continue
        detections.append(
            NormalizedDetection(
                track_id=detection.track_id,
                bbox=translated,
                confidence=detection.confidence,
                class_name=hybrid_class_name_for_source(source_type),
                frame_index=detection.frame_index,
                timestamp=detection.timestamp,
                detector_mode=source_type,  # type: ignore[arg-type]
                source_type=source_type,
            )
        )
    return detections


class HybridDetector:
    """Run body and head detectors on their ROI crops and fuse the results."""

    def __init__(
        self,
        body_detector: object,
        head_detector: object,
        roi_config: HybridRoiConfig,
        *,
        crop_padding: int = 8,
        upper_body_fraction: float = DEFAULT_UPPER_BODY_FRACTION,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ) -> None:
        """Store the two detectors plus ROI and fusion parameters."""
        self.body_detector = body_detector
        self.head_detector = head_detector
        self.roi_config = roi_config
        self.crop_padding = crop_padding
        self.upper_body_fraction = upper_body_fraction
        self.iou_threshold = iou_threshold

    def detect(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        timestamp: float | None = None,
    ) -> list[NormalizedDetection]:
        """Detect bodies and heads on their ROI crops and return fused output."""
        body_detections = run_detector_in_rois(
            self.body_detector,
            frame,
            self.roi_config.near_body_polygons,
            source_type="body",
            anchor=BODY_ANCHOR,
            frame_index=frame_index,
            timestamp=timestamp,
            padding=self.crop_padding,
        )
        head_detections = run_detector_in_rois(
            self.head_detector,
            frame,
            self.roi_config.far_head_polygons,
            source_type="head",
            anchor=HEAD_ANCHOR,
            frame_index=frame_index,
            timestamp=timestamp,
            padding=self.crop_padding,
        )
        return fuse_body_head(
            body_detections,
            head_detections,
            upper_body_fraction=self.upper_body_fraction,
            iou_threshold=self.iou_threshold,
        )

    def detect_result(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        timestamp: float | None = None,
    ) -> DetectionResult:
        """Detect and return a :class:`DetectionResult` for parity with Detector."""
        return DetectionResult(detections=self.detect(frame, frame_index, timestamp))


def _parse_polygons(value: object) -> list[list[Point]]:
    """Parse a ROI value into a list of polygons (each a list of points)."""
    if value is None:
        return []
    if isinstance(value, dict):
        if "polygons" in value:
            return _parse_polygons(value["polygons"])
        if "polygon" in value:
            return _parse_polygons(value["polygon"])
        return []
    if not isinstance(value, list) or not value:
        return []

    first = value[0]
    # A list of polygons: first element is itself a sequence of points.
    if isinstance(first, list | tuple) and first and isinstance(first[0], list | tuple):
        polygons: list[list[Point]] = []
        for polygon_raw in value:
            polygon = _parse_single_polygon(polygon_raw)
            if len(polygon) >= 3:
                polygons.append(polygon)
        return polygons

    # A single polygon: list of [x, y] points.
    polygon = _parse_single_polygon(value)
    return [polygon] if len(polygon) >= 3 else []


def _parse_single_polygon(raw: object) -> list[Point]:
    """Parse a single polygon (list of [x, y] points)."""
    if not isinstance(raw, list | tuple):
        return []
    polygon: list[Point] = []
    for point_raw in raw:
        if not isinstance(point_raw, list | tuple) or len(point_raw) != 2:
            continue
        polygon.append((float(point_raw[0]), float(point_raw[1])))
    return polygon


def _scale_polygons(
    polygons: list[list[Point]],
    scale_x: float,
    scale_y: float,
) -> list[list[Point]]:
    """Scale ROI polygons to target frame dimensions."""
    if scale_x == 1.0 and scale_y == 1.0:
        return polygons
    return [
        [(x * scale_x, y * scale_y) for x, y in polygon]
        for polygon in polygons
    ]


def load_hybrid_roi_config(
    config_path: str | Path,
    *,
    target_width: int | None = None,
    target_height: int | None = None,
) -> HybridRoiConfig:
    """Load near_body_zone / far_head_zone ROI groups from a zones JSON file."""
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid zones config format: {config_path}")

    ref_width = int(raw.get("frame_width", target_width or 0) or 0)
    ref_height = int(raw.get("frame_height", target_height or 0) or 0)
    scale_x = (target_width / ref_width) if target_width and ref_width else 1.0
    scale_y = (target_height / ref_height) if target_height and ref_height else 1.0

    near_body = _scale_polygons(_parse_polygons(raw.get("near_body_zone")), scale_x, scale_y)
    far_head = _scale_polygons(_parse_polygons(raw.get("far_head_zone")), scale_x, scale_y)
    return HybridRoiConfig(near_body_polygons=near_body, far_head_polygons=far_head)
