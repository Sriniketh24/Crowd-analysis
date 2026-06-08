"""Frame pre-processing helpers used before detection and tracking."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.vision.detector import Detector, NormalizedDetection
from src.vision.crowd_analyzer import CrowdAnalyzer
from src.vision.detection_filter import DetectionRegionFilter
from src.vision.line_counter import LineCrossing, LineManager
from src.vision.zone_manager import ZoneManager
from src.vision.tracker import Tracker, normalize_tracks


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Return a preprocessed frame for model inference."""
    if frame is None:
        raise ValueError("Frame cannot be None.")
    return frame


@dataclass(slots=True)
class FrameProcessResult:
    """Output of one frame processing step."""

    frame: np.ndarray
    detections: list[NormalizedDetection]
    zone_occupancy: dict[str, int]
    zone_alerts: dict[str, str]
    line_counts: dict[str, dict[str, int]]
    line_events: list[LineCrossing]
    unique_passengers_seen: int
    processing_fps: float


class FrameProcessor:
    """Coordinate preprocessing with detection or tracking inference."""

    def __init__(
        self,
        detector: Detector | None = None,
        tracker: Tracker | None = None,
        zone_manager: ZoneManager | None = None,
        line_managers: list[LineManager] | None = None,
        crowd_analyzer: CrowdAnalyzer | None = None,
        detection_filter: DetectionRegionFilter | None = None,
        use_tracking: bool = True,
        resize_width: int | None = None,
    ) -> None:
        """Build a modular frame processor with optional resize step."""
        self.detector = detector
        self.tracker = tracker
        self.zone_manager = zone_manager
        self.line_managers = line_managers or []
        self.crowd_analyzer = crowd_analyzer
        self.detection_filter = detection_filter
        self.use_tracking = use_tracking and tracker is not None
        self.resize_width = resize_width
        self._unique_track_ids_seen: set[int] = set()
        self._last_processed_timestamp: float | None = None

    def _maybe_resize(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame while preserving aspect ratio when configured."""
        if self.resize_width is None:
            return frame
        height, width = frame.shape[:2]
        if width <= 0:
            return frame
        ratio = self.resize_width / float(width)
        target_height = int(height * ratio)
        return cv2.resize(frame, (self.resize_width, target_height))

    def process(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp: float,
    ) -> FrameProcessResult:
        """Run one frame through preprocessing and model pipeline."""
        prepared = preprocess_frame(frame)
        prepared = self._maybe_resize(prepared)

        if self.use_tracking and self.tracker is not None:
            tracked = self.tracker.track(
                frame=prepared,
                frame_index=frame_index,
                timestamp=timestamp,
            )
            normalized = normalize_tracks(tracked)
            if self.detection_filter is not None:
                normalized = self.detection_filter.filter(normalized)
            return self._build_result(prepared, normalized, frame_index=frame_index, timestamp=timestamp)

        if self.detector is None:
            return self._build_result(prepared, [])

        result = self.detector.detect(
            frame=prepared,
            frame_index=frame_index,
            timestamp=timestamp,
        )
        detections = result.detections
        if self.detection_filter is not None:
            detections = self.detection_filter.filter(detections)
        return self._build_result(prepared, detections, frame_index=frame_index, timestamp=timestamp)

    def _build_result(
        self,
        frame: np.ndarray,
        detections: list[NormalizedDetection],
        *,
        frame_index: int,
        timestamp: float,
    ) -> FrameProcessResult:
        """Compute analytics from tracked detections and package frame output."""
        zone_occupancy: dict[str, int] = {}
        zone_alerts: dict[str, str] = {}
        line_counts: dict[str, dict[str, int]] = {}
        line_events: list[LineCrossing] = []

        for detection in detections:
            if detection.track_id is not None and detection.track_id >= 0:
                self._unique_track_ids_seen.add(detection.track_id)

        if self.zone_manager is not None:
            zone_occupancy = self.zone_manager.update(detections)
            if self.crowd_analyzer is not None:
                zone_alerts = self.crowd_analyzer.evaluate(zone_occupancy, timestamp=timestamp)

        for line_manager in self.line_managers:
            line_counts[line_manager.config.id] = line_manager.update(
                detections,
                frame_index=frame_index,
                timestamp=timestamp,
            )
            line_events.extend(line_manager.recent_events)

        processing_fps = 0.0
        if self._last_processed_timestamp is not None and timestamp > self._last_processed_timestamp:
            processing_fps = 1.0 / (timestamp - self._last_processed_timestamp)
        self._last_processed_timestamp = timestamp

        return FrameProcessResult(
            frame=frame,
            detections=detections,
            zone_occupancy=zone_occupancy,
            zone_alerts=zone_alerts,
            line_counts=line_counts,
            line_events=line_events,
            unique_passengers_seen=len(self._unique_track_ids_seen),
            processing_fps=processing_fps,
        )
