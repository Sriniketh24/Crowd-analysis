"""Fuse body and head detections into one non-double-counted stream.

The hybrid detector runs full-body detection for near/visible passengers and
head detection for far/occluded/dense passengers. Because a single person can
be picked up by both models, this module merges the two lists while suppressing
heads that clearly belong to an already-detected body.

Suppression rules (see project requirements):
  * A head whose center lies inside the upper part of a body box is the same
    person -> keep the body, drop the head.
  * A head that overlaps a body box strongly (IoU threshold) is the same
    visible person -> keep the body, drop the head.
  * A head with no matching body (e.g. far/dense crowd) is kept.
All body detections are always kept.
"""

from __future__ import annotations

from src.vision.detector import (
    BODY_PASSENGER_LABEL,
    HEAD_PASSENGER_LABEL,
    NormalizedDetection,
)

BBox = tuple[float, float, float, float]
Point = tuple[float, float]

# Default share of a body box (measured from the top) treated as the "head"
# region for the inside-upper-body suppression test.
DEFAULT_UPPER_BODY_FRACTION = 0.45
# Default IoU above which a head and body box are treated as the same person.
DEFAULT_IOU_THRESHOLD = 0.45


def center_point(bbox: BBox) -> Point:
    """Return the geometric center of a bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def upper_body_region(body_bbox: BBox, fraction: float = DEFAULT_UPPER_BODY_FRACTION) -> BBox:
    """Return the top ``fraction`` slice of a body box (the head region)."""
    x1, y1, x2, y2 = body_bbox
    clamped = min(max(fraction, 0.0), 1.0)
    upper_y2 = y1 + (y2 - y1) * clamped
    return (x1, y1, x2, upper_y2)


def point_in_bbox(point: Point, bbox: BBox) -> bool:
    """Return True when a point lies within a bounding box (inclusive)."""
    px, py = point
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2


def head_inside_upper_body(
    head_bbox: BBox,
    body_bbox: BBox,
    fraction: float = DEFAULT_UPPER_BODY_FRACTION,
) -> bool:
    """Return True when a head center sits in the upper part of a body box."""
    return point_in_bbox(center_point(head_bbox), upper_body_region(body_bbox, fraction))


def bbox_iou(box_a: BBox, box_b: BBox) -> float:
    """Return intersection-over-union for two bounding boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def _label_detection(detection: NormalizedDetection, source_type: str) -> NormalizedDetection:
    """Return a copy of ``detection`` carrying explicit hybrid source metadata."""
    class_name = HEAD_PASSENGER_LABEL if source_type == "head" else BODY_PASSENGER_LABEL
    return NormalizedDetection(
        track_id=detection.track_id,
        bbox=detection.bbox,
        confidence=detection.confidence,
        class_name=class_name,
        frame_index=detection.frame_index,
        timestamp=detection.timestamp,
        detector_mode=source_type,  # type: ignore[arg-type]
        source_type=source_type,
    )


def fuse_body_head(
    body_detections: list[NormalizedDetection],
    head_detections: list[NormalizedDetection],
    *,
    upper_body_fraction: float = DEFAULT_UPPER_BODY_FRACTION,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> list[NormalizedDetection]:
    """Merge body and head detections, suppressing double-counted heads."""
    fused: list[NormalizedDetection] = [
        _label_detection(body, "body") for body in body_detections
    ]
    body_boxes = [body.bbox for body in body_detections]
    for head in head_detections:
        if _head_matches_any_body(
            head.bbox,
            body_boxes,
            upper_body_fraction=upper_body_fraction,
            iou_threshold=iou_threshold,
        ):
            continue
        fused.append(_label_detection(head, "head"))
    return fused


def _head_matches_any_body(
    head_bbox: BBox,
    body_boxes: list[BBox],
    *,
    upper_body_fraction: float,
    iou_threshold: float,
) -> bool:
    """Return True when a head should be suppressed in favor of a body box."""
    for body_bbox in body_boxes:
        if head_inside_upper_body(head_bbox, body_bbox, upper_body_fraction):
            return True
        if bbox_iou(head_bbox, body_bbox) >= iou_threshold:
            return True
    return False
