"""Single unified tracker for fused hybrid detections.

After body and head detections are merged into one list, the hybrid pipeline
must assign one set of stable track IDs across both sources (no independent
body/head IDs for the final count). This wraps ``supervision.ByteTrack`` when
available, falling back to a small built-in greedy-IoU tracker so the module
stays usable and testable without the optional dependency.
"""

from __future__ import annotations

from src.vision.detection_fusion import bbox_iou
from src.vision.detector import NormalizedDetection

try:  # pragma: no cover - import guarded for optional dependency
    import numpy as np
    import supervision as sv
except ModuleNotFoundError:  # pragma: no cover - environment-specific dependency
    np = None  # type: ignore[assignment]
    sv = None  # type: ignore[assignment]


def bytetrack_available() -> bool:
    """Return True when supervision.ByteTrack can be used."""
    return sv is not None and np is not None and hasattr(sv, "ByteTrack")


def _source_of(detection: NormalizedDetection) -> str:
    """Return the detection source, falling back to detector_mode."""
    return detection.source_type or detection.detector_mode or "body"


class HybridTracker:
    """Assign one set of track IDs to a fused body+head detection list."""

    def __init__(
        self,
        *,
        prefer_bytetrack: bool = True,
        iou_match_threshold: float = 0.3,
        max_age: int = 30,
    ) -> None:
        """Create a tracker, preferring ByteTrack when available."""
        self.iou_match_threshold = iou_match_threshold
        self.max_age = max_age
        self._use_bytetrack = prefer_bytetrack and bytetrack_available()
        self._bytetrack = sv.ByteTrack() if self._use_bytetrack else None
        # Greedy fallback state: track_id -> (bbox, frames_since_seen)
        self._next_id = 1
        self._active: dict[int, tuple[tuple[float, float, float, float], int]] = {}

    @property
    def backend(self) -> str:
        """Return the active tracking backend name."""
        return "bytetrack" if self._use_bytetrack else "greedy_iou"

    def update(
        self,
        detections: list[NormalizedDetection],
        frame_index: int = 0,
        timestamp: float | None = None,
    ) -> list[NormalizedDetection]:
        """Return the fused detections with unified track IDs assigned."""
        if self._use_bytetrack:
            return self._update_bytetrack(detections, frame_index, timestamp)
        return self._update_greedy(detections, frame_index, timestamp)

    def _update_bytetrack(
        self,
        detections: list[NormalizedDetection],
        frame_index: int,
        timestamp: float | None,
    ) -> list[NormalizedDetection]:
        """Track using supervision.ByteTrack, preserving source metadata."""
        assert sv is not None and np is not None  # narrowing for type checkers
        if not detections:
            self._bytetrack.update_with_detections(sv.Detections.empty())
            return []

        xyxy = np.array([d.bbox for d in detections], dtype=float)
        confidence = np.array([d.confidence for d in detections], dtype=float)
        class_id = np.array(
            [0 if _source_of(d) == "body" else 1 for d in detections], dtype=int
        )
        data = {
            "source_type": np.array([_source_of(d) for d in detections], dtype=object),
            "class_name": np.array([d.class_name for d in detections], dtype=object),
        }
        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            data=data,
        )
        tracked = self._bytetrack.update_with_detections(sv_detections)

        results: list[NormalizedDetection] = []
        tracker_ids = tracked.tracker_id
        for idx in range(len(tracked)):
            x1, y1, x2, y2 = (float(value) for value in tracked.xyxy[idx])
            track_id = int(tracker_ids[idx]) if tracker_ids is not None else -1
            source_type = str(tracked.data["source_type"][idx])
            results.append(
                NormalizedDetection(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    confidence=float(tracked.confidence[idx]),
                    class_name=str(tracked.data["class_name"][idx]),
                    frame_index=frame_index,
                    timestamp=timestamp if timestamp is not None else 0.0,
                    detector_mode=source_type,  # type: ignore[arg-type]
                    source_type=source_type,
                )
            )
        return results

    def _update_greedy(
        self,
        detections: list[NormalizedDetection],
        frame_index: int,
        timestamp: float | None,
    ) -> list[NormalizedDetection]:
        """Greedy IoU fallback tracker preserving source metadata."""
        for track_id in list(self._active):
            bbox, age = self._active[track_id]
            self._active[track_id] = (bbox, age + 1)

        used_ids: set[int] = set()
        results: list[NormalizedDetection] = []
        for detection in detections:
            best_id: int | None = None
            best_iou = self.iou_match_threshold
            for track_id, (bbox, _age) in self._active.items():
                if track_id in used_ids:
                    continue
                iou = bbox_iou(detection.bbox, bbox)
                if iou >= best_iou:
                    best_iou = iou
                    best_id = track_id
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
            used_ids.add(best_id)
            self._active[best_id] = (detection.bbox, 0)
            source_type = _source_of(detection)
            results.append(
                NormalizedDetection(
                    track_id=best_id,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    class_name=detection.class_name,
                    frame_index=frame_index,
                    timestamp=timestamp if timestamp is not None else detection.timestamp,
                    detector_mode=source_type,  # type: ignore[arg-type]
                    source_type=source_type,
                )
            )

        # Drop stale tracks that have not been matched recently.
        for track_id in list(self._active):
            if self._active[track_id][1] > self.max_age:
                del self._active[track_id]
        return results
