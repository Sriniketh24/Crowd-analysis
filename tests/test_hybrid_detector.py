"""Tests for hybrid body+head detection, fusion, tracking, and anchors."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from src.vision.detection_fusion import (
    bbox_iou,
    fuse_body_head,
    head_inside_upper_body,
    upper_body_region,
)
from src.vision.detector import (
    DetectionResult,
    NormalizedDetection,
    normalize_detector_mode,
)
from src.vision.fusion_tracker import HybridTracker, bytetrack_available
from src.vision.hybrid_detector import (
    HybridDetector,
    HybridRoiConfig,
    crop_bounds_from_polygons,
    load_hybrid_roi_config,
    run_detector_in_rois,
    translate_bbox,
)
from src.vision.line_counter import LineConfig, LineManager
from src.vision.zone_manager import ZoneConfig, ZoneManager


def _det(
    bbox: tuple[float, float, float, float],
    *,
    track_id: int | None = None,
    confidence: float = 0.9,
    source: str = "body",
    frame_index: int = 0,
    timestamp: float = 0.0,
) -> NormalizedDetection:
    return NormalizedDetection(
        track_id=track_id,
        bbox=bbox,
        confidence=confidence,
        class_name=f"{source}/passenger",
        frame_index=frame_index,
        timestamp=timestamp,
        detector_mode=source,  # type: ignore[arg-type]
        source_type=source,
    )


class _StubDetector:
    """Detector stub returning fixed crop-relative detections."""

    def __init__(self, detections: list[NormalizedDetection]) -> None:
        self._detections = detections
        self.last_frame_shape: tuple[int, ...] | None = None

    def detect(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        timestamp: float | None = None,
    ) -> DetectionResult:
        self.last_frame_shape = frame.shape
        return DetectionResult(detections=list(self._detections))


# --- detector mode validation -------------------------------------------------


def test_detector_mode_validation_accepts_hybrid() -> None:
    assert normalize_detector_mode("hybrid") == "hybrid"
    assert normalize_detector_mode(" HYBRID ") == "hybrid"
    with pytest.raises(ValueError, match="body.*head.*hybrid"):
        normalize_detector_mode("torso")


# --- crop bbox coordinate translation ----------------------------------------


def test_translate_bbox_offsets_back_to_full_frame() -> None:
    assert translate_bbox((10.0, 20.0, 30.0, 40.0), 100.0, 200.0) == (
        110.0,
        220.0,
        130.0,
        240.0,
    )


def test_crop_bounds_from_polygons_clamps_to_frame() -> None:
    polygons = [[(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)]]
    assert crop_bounds_from_polygons(polygons, 640, 480) == (100, 100, 200, 200)
    # Padding extends bounds but stays clamped to the frame edges.
    assert crop_bounds_from_polygons(polygons, 150, 150, padding=50) == (50, 50, 150, 150)
    assert crop_bounds_from_polygons([], 640, 480) is None


def test_run_detector_in_rois_translates_and_polygon_filters() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    polygons = [[(100.0, 100.0), (200.0, 100.0), (200.0, 200.0), (100.0, 200.0)]]
    # Crop-relative detection: bottom-center (25, 40) -> full frame (125, 140).
    inside = _det((10.0, 10.0, 40.0, 40.0), source="body")
    detections = run_detector_in_rois(
        _StubDetector([inside]),
        frame,
        polygons,
        source_type="body",
        anchor="bottom_center",
        frame_index=7,
        timestamp=1.5,
    )
    assert len(detections) == 1
    assert detections[0].bbox == (110.0, 110.0, 140.0, 140.0)
    assert detections[0].source_type == "body"
    assert detections[0].class_name == "body/passenger"


def test_run_detector_in_rois_drops_detections_outside_polygon() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    polygons = [[(100.0, 100.0), (120.0, 100.0), (120.0, 120.0), (100.0, 120.0)]]
    # Bottom-center translates to (200, 200), far outside the small ROI polygon.
    outside = _det((90.0, 90.0, 110.0, 100.0), source="body")
    detections = run_detector_in_rois(
        _StubDetector([outside]),
        frame,
        polygons,
        source_type="body",
        anchor="bottom_center",
        frame_index=0,
        timestamp=0.0,
    )
    assert detections == []


# --- fusion: head suppression vs. retention ----------------------------------


def test_head_inside_upper_body_region() -> None:
    body = (100.0, 100.0, 140.0, 300.0)
    assert upper_body_region(body, 0.45) == (100.0, 100.0, 140.0, 190.0)
    assert head_inside_upper_body((110.0, 110.0, 130.0, 150.0), body, 0.45)
    # A head near the feet is not in the upper body region.
    assert not head_inside_upper_body((110.0, 270.0, 130.0, 295.0), body, 0.45)


def test_fuse_suppresses_head_inside_upper_body() -> None:
    body = _det((100.0, 100.0, 140.0, 300.0), source="body")
    head = _det((110.0, 110.0, 130.0, 150.0), source="head")
    fused = fuse_body_head([body], [head])
    assert len(fused) == 1
    assert fused[0].source_type == "body"
    assert fused[0].class_name == "body/passenger"


def test_fuse_suppresses_head_overlapping_body_strongly() -> None:
    body = (100.0, 100.0, 200.0, 200.0)
    head = (110.0, 110.0, 195.0, 195.0)
    assert bbox_iou(body, head) >= 0.45
    fused = fuse_body_head(
        [_det(body, source="body")],
        [_det(head, source="head")],
    )
    assert len(fused) == 1
    assert fused[0].source_type == "body"


def test_fuse_keeps_head_when_body_absent() -> None:
    head = _det((110.0, 110.0, 130.0, 150.0), source="head")
    fused = fuse_body_head([], [head])
    assert len(fused) == 1
    assert fused[0].source_type == "head"
    assert fused[0].class_name == "head/passenger"


def test_fuse_keeps_distant_head_with_unrelated_body() -> None:
    body = _det((100.0, 100.0, 140.0, 300.0), source="body")
    far_head = _det((500.0, 90.0, 520.0, 110.0), source="head")
    fused = fuse_body_head([body], [far_head])
    assert {d.source_type for d in fused} == {"body", "head"}
    assert len(fused) == 2


# --- hybrid detector end-to-end (stubbed models) -----------------------------


def test_hybrid_detector_fuses_roi_crops() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    near_body = [[(100.0, 400.0), (400.0, 400.0), (400.0, 700.0), (100.0, 700.0)]]
    far_head = [[(600.0, 100.0), (900.0, 100.0), (900.0, 300.0), (600.0, 300.0)]]
    roi = HybridRoiConfig(near_body_polygons=near_body, far_head_polygons=far_head)

    # Body crop origin (100, 400); detection bottom-center -> (250, 650) in ROI.
    body_stub = _StubDetector([_det((100.0, 50.0, 200.0, 250.0), source="body")])
    # Head crop origin (600, 100); detection center -> (750, 200) in ROI.
    head_stub = _StubDetector([_det((130.0, 80.0, 170.0, 120.0), source="head")])

    detector = HybridDetector(body_stub, head_stub, roi, crop_padding=0)
    fused = detector.detect(frame, frame_index=3, timestamp=2.0)

    sources = sorted(d.source_type for d in fused)
    assert sources == ["body", "head"]
    body_det = next(d for d in fused if d.source_type == "body")
    assert body_det.bbox == (200.0, 450.0, 300.0, 650.0)


# --- unified tracking after fusion -------------------------------------------


def test_hybrid_tracker_assigns_unified_ids_across_sources() -> None:
    tracker = HybridTracker(prefer_bytetrack=False, iou_match_threshold=0.3)
    fused = [
        _det((100.0, 100.0, 140.0, 300.0), source="body"),
        _det((500.0, 90.0, 520.0, 110.0), source="head"),
    ]
    first = tracker.update(fused, frame_index=0, timestamp=0.0)
    ids = sorted(d.track_id for d in first)
    assert ids == [1, 2]
    assert {d.source_type for d in first} == {"body", "head"}

    # Same boxes next frame -> same IDs from the single unified tracker.
    second = tracker.update(
        [
            _det((101.0, 101.0, 141.0, 301.0), source="body"),
            _det((501.0, 91.0, 521.0, 111.0), source="head"),
        ],
        frame_index=1,
        timestamp=0.1,
    )
    assert sorted(d.track_id for d in second) == [1, 2]
    # Source metadata survives tracking.
    body_track = next(d for d in second if d.source_type == "body")
    assert body_track.class_name == "body/passenger"


def test_hybrid_tracker_prefers_bytetrack_when_available() -> None:
    tracker = HybridTracker(prefer_bytetrack=True)
    expected = "bytetrack" if bytetrack_available() else "greedy_iou"
    assert tracker.backend == expected


# --- per-detection anchor: zone counting -------------------------------------


def test_hybrid_zone_counting_uses_per_detection_anchor() -> None:
    zone = ZoneConfig(
        id="z",
        name="Z",
        polygon=[(0.0, 0.0), (100.0, 0.0), (100.0, 25.0), (0.0, 25.0)],
        warning_threshold=1,
        critical_threshold=2,
    )
    # Head: center (50, 25) inside, bottom-center (50, 40) outside.
    head = {"track_id": 1, "bbox": (40.0, 10.0, 60.0, 40.0), "source_type": "head"}
    # Body: bottom-center (50, 20) inside, center (50, -5) outside.
    body = {"track_id": 2, "bbox": (40.0, -30.0, 60.0, 20.0), "source_type": "body"}

    hybrid = ZoneManager(zones=[zone], point_strategy="bottom_center", per_detection_anchor=True)
    assert hybrid.update([head, body])["z"] == 2

    # Without per-detection anchors a fixed bottom-center misses the head.
    fixed = ZoneManager(zones=[zone], point_strategy="bottom_center")
    assert fixed.update([head, body])["z"] == 1


# --- per-detection anchor: hybrid line crossing uses head center -------------


def test_hybrid_line_crossing_uses_head_center() -> None:
    line = LineManager(
        config=LineConfig(id="gate", name="Gate", start=(0.0, 0.0), end=(100.0, 0.0)),
        point_strategy="bottom_center",
        per_detection_anchor=True,
    )
    # Head center crosses the y=0 line between the two updates.
    line.update([{"track_id": 1, "bbox": (40.0, -30.0, 60.0, -10.0), "source_type": "head"}])
    counts = line.update([{"track_id": 1, "bbox": (40.0, 10.0, 60.0, 30.0), "source_type": "head"}])
    assert counts["IN"] == 1


# --- ROI config loading ------------------------------------------------------


def test_load_hybrid_roi_config_parses_zone_groups() -> None:
    roi = load_hybrid_roi_config("configs/zones.hybrid_platform.example.json")
    assert len(roi.near_body_polygons) == 1
    assert len(roi.far_head_polygons) == 1
    assert len(roi.near_body_polygons[0]) >= 3
    assert len(roi.far_head_polygons[0]) >= 3


def test_load_hybrid_roi_config_scales_to_target() -> None:
    roi = load_hybrid_roi_config(
        "configs/zones.hybrid_platform.example.json",
        target_width=640,
        target_height=360,
    )
    # Source frame is 1280x720, so a 640x360 target halves every coordinate.
    x, y = roi.near_body_polygons[0][0]
    assert x == pytest.approx(135 / 2)
    assert y == pytest.approx(720 / 2)


def test_run_video_demo_accepts_hybrid_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import run_video_demo

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_video_demo.py",
            "--source",
            "data/input_videos/sample.mp4",
            "--zones-config",
            "configs/zones.hybrid_platform.example.json",
            "--detector-mode",
            "hybrid",
            "--body-model",
            "yolo11n.pt",
            "--head-model",
            "models/fine_tuned/head_detector/weights/best.pt",
        ],
    )
    args = run_video_demo.parse_args()
    assert args.detector_mode == "hybrid"
    assert args.body_model == "yolo11n.pt"
    assert args.head_model.endswith("best.pt")
