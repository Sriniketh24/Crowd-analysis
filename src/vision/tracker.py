"""Tracking wrapper to assign stable IDs to detections across frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from time import time
from typing import Any

import yaml

try:
    from ultralytics import YOLO
except ModuleNotFoundError:  # pragma: no cover - environment-specific dependency
    YOLO = None  # type: ignore[assignment]

from src.vision.detector import (
    DEFAULT_PERSON_CLASS_ID,
    NormalizedDetection,
    build_model,
)

SUPPORTED_TRACKERS = {"bytetrack", "botsort"}


@dataclass(slots=True)
class TrackedObject:
    """Single tracked object with stable tracker identifier."""

    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_name: str
    frame_index: int
    timestamp: float


@dataclass(slots=True)
class LineageDiagnostics:
    """Lightweight tracker diagnostics for future evaluation/reporting."""

    tracker_type: str
    configured_path: str


class Tracker:
    """Ultralytics tracking wrapper with ByteTrack or BoT-SORT."""

    def __init__(
        self,
        tracker_type: str = "bytetrack",
        weights_path: str | None = None,
        device: str = "cpu",
        confidence: float = 0.25,
        iou: float = 0.5,
        person_class_id: int = DEFAULT_PERSON_CLASS_ID,
        imgsz: int = 640,
        half: bool = False,
        tracker_config_overrides: dict[str, Any] | None = None,
        accuracy_weights: str | None = None,
        legacy_fallback_weights: str | None = None,
        use_fine_tuned_if_available: bool = True,
    ) -> None:
        """Initialize tracking model and tracker mode."""
        if tracker_type not in SUPPORTED_TRACKERS:
            raise ValueError(
                f"Unsupported tracker type '{tracker_type}'. "
                f"Expected one of: {', '.join(sorted(SUPPORTED_TRACKERS))}"
            )
        self.tracker_type = tracker_type
        self.model = build_model(
            weights_path,
            accuracy_weights=accuracy_weights,
            legacy_fallback_weights=legacy_fallback_weights,
            use_fine_tuned_if_available=use_fine_tuned_if_available,
        )
        self.device = device
        self.confidence = confidence
        self.iou = iou
        self.person_class_id = person_class_id
        self.imgsz = imgsz
        self.half = half
        self._tracker_config_overrides = dict(tracker_config_overrides or {})
        self._tracker_config_path = self._build_tracker_config()
        self.diagnostics = LineageDiagnostics(
            tracker_type=self.tracker_type,
            configured_path=self._tracker_config_path,
        )

    def _build_tracker_config(self) -> str:
        """Create a tracker config file when overrides are supplied."""
        if not self._tracker_config_overrides:
            return f"{self.tracker_type}.yaml"
        tracker_config = {"tracker_type": self.tracker_type, **self._tracker_config_overrides}
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=f"_{self.tracker_type}.yaml",
            prefix="crowd_tracker_",
            delete=False,
            encoding="utf-8",
        ) as handle:
            yaml.safe_dump(tracker_config, handle, sort_keys=True)
            return handle.name

    def track(
        self,
        frame: object,
        frame_index: int = 0,
        timestamp: float | None = None,
    ) -> list[TrackedObject]:
        """Run frame tracking and return normalized tracked objects."""
        frame_ts = time() if timestamp is None else timestamp
        results = self.model.track(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            classes=[self.person_class_id],
            tracker=self._tracker_config_path,
            persist=True,
            device=self.device,
            imgsz=self.imgsz,
            half=self.half,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return []
        names = result.names or getattr(self.model, "names", {})

        tracks: list[TrackedObject] = []
        has_track_ids = boxes.id is not None
        for idx, box in enumerate(boxes):
            class_id = int(box.cls.item())
            if class_id != self.person_class_id:
                continue
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            track_id = (
                int(boxes.id[idx].item()) if has_track_ids and boxes.id is not None else -1
            )
            tracks.append(
                TrackedObject(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    confidence=float(box.conf.item()),
                    class_name=str(names.get(class_id, "person")),
                    frame_index=frame_index,
                    timestamp=frame_ts,
                )
            )
        return tracks

    def update(
        self,
        frame: object,
        frame_index: int = 0,
        timestamp: float | None = None,
    ) -> list[TrackedObject]:
        """Backward-compatible alias for track()."""
        return self.track(frame=frame, frame_index=frame_index, timestamp=timestamp)


def normalize_tracks(tracks: list[TrackedObject]) -> list[NormalizedDetection]:
    """Convert tracker outputs into normalized detection schema."""
    return [
        NormalizedDetection(
            track_id=track.track_id,
            bbox=track.bbox,
            confidence=track.confidence,
            class_name=track.class_name,
            frame_index=track.frame_index,
            timestamp=track.timestamp,
        )
        for track in tracks
    ]
