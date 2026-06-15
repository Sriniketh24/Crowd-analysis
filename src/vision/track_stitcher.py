"""Online track-ID stitching for short occlusion gaps.

The detector/tracker can legitimately create a new raw ID after a passenger is
hidden by a pole, train edge, or another person. This module remaps those short
fragments back to the previous ID when the motion is plausible and no two active
tracks would share the same final ID in the same frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.vision.detector import NormalizedDetection

BBox = tuple[float, float, float, float]


def _centroid(box: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _box_width(box: BBox) -> float:
    return max(1.0, box[2] - box[0])


def _direction_cost(a: tuple[float, float], b: tuple[float, float]) -> float:
    a_norm = math.hypot(a[0], a[1])
    b_norm = math.hypot(b[0], b[1])
    if a_norm < 1.0 or b_norm < 1.0:
        return 0.0
    cosine = (a[0] * b[0] + a[1] * b[1]) / max(1e-6, a_norm * b_norm)
    return (1.0 - max(-1.0, min(1.0, cosine))) / 2.0


def _appearance_crop_box(box: BBox, frame_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = box
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    center_x = (x1 + x2) / 2.0
    crop_width = box_width * 2.2
    crop_height = box_height * 3.0
    left = max(0, int(round(center_x - crop_width / 2.0)))
    right = min(width, int(round(center_x + crop_width / 2.0)))
    top = max(0, int(round(y1 - box_height * 0.25)))
    bottom = min(height, int(round(y1 + crop_height)))
    return (left, top, right, bottom)


def _appearance_histogram(frame: np.ndarray | None, box: BBox) -> np.ndarray | None:
    if frame is None:
        return None
    left, top, right, bottom = _appearance_crop_box(box, frame.shape)
    if right <= left or bottom <= top:
        return None
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
    if not np.isfinite(hist).all() or float(hist.sum()) <= 0:
        return None
    return hist


def _appearance_cost(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 1.0
    similarity = float(cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_CORREL))
    if not math.isfinite(similarity):
        return 1.0
    return max(0.0, min(1.0, (1.0 - similarity) / 2.0))


@dataclass(frozen=True, slots=True)
class TrackStitcherConfig:
    """Controls for online track-fragment stitching."""

    enabled: bool = False
    gap_frames: int = 220
    dist_heads: float = 5.0
    mode: str = "observation"
    ambiguity_ratio: float = 0.95
    max_speed_heads: float = 0.85
    appearance_weight: float = 0.0
    max_appearance_cost: float = 1.0
    direction_weight: float = 0.12
    max_direction_cost: float = 0.85
    max_jump_heads: float = 7.5
    history_points: int = 6
    max_archive_age_frames: int = 600


@dataclass(slots=True)
class _TrackState:
    root_id: int
    first_frame: int
    last_frame: int
    first_box: BBox
    last_box: BBox
    centers: list[tuple[int, tuple[float, float]]] = field(default_factory=list)
    widths: list[float] = field(default_factory=list)
    last_appearance: np.ndarray | None = None


class TrackStitcher:
    """Remap raw tracker IDs to more stable final IDs."""

    def __init__(self, config: TrackStitcherConfig | None = None) -> None:
        self.config = config or TrackStitcherConfig()
        self._states: dict[int, _TrackState] = {}
        self._raw_to_root: dict[int, int] = {}

    def update(
        self,
        detections: list[NormalizedDetection],
        *,
        frame_index: int,
        frame: np.ndarray | None = None,
    ) -> list[NormalizedDetection]:
        """Return detections with stitched track IDs when enabled."""
        if not self.config.enabled:
            return detections

        self._prune(frame_index)
        used_roots: set[int] = set()
        remapped: list[NormalizedDetection] = []
        for detection in sorted(detections, key=lambda item: item.track_id if item.track_id is not None else -1):
            raw_id = detection.track_id
            if raw_id is None or raw_id < 0:
                remapped.append(detection)
                continue
            appearance = _appearance_histogram(frame, detection.bbox) if self.config.appearance_weight > 0 else None
            root_id = self._raw_to_root.get(raw_id)
            if root_id is None:
                root_id = self._choose_root(raw_id, detection.bbox, frame_index, used_roots, appearance)
                self._raw_to_root[raw_id] = root_id
            if root_id in used_roots and root_id != raw_id:
                root_id = raw_id
                self._raw_to_root[raw_id] = raw_id
            used_roots.add(root_id)
            self._update_state(raw_id, root_id, detection.bbox, frame_index, appearance)
            remapped.append(
                NormalizedDetection(
                    track_id=root_id,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    class_name=detection.class_name,
                    frame_index=detection.frame_index,
                    timestamp=detection.timestamp,
                    detector_mode=detection.detector_mode,
                    source_type=detection.source_type,
                )
            )
        return remapped

    def _choose_root(
        self,
        raw_id: int,
        box: BBox,
        frame_index: int,
        used_roots: set[int],
        appearance: np.ndarray | None,
    ) -> int:
        candidates: list[tuple[float, int]] = []
        center = _centroid(box)
        width = _box_width(box)
        for old_raw_id, state in self._states.items():
            if old_raw_id == raw_id or state.root_id in used_roots:
                continue
            gap = frame_index - state.last_frame
            if gap <= 0 or gap > self.config.gap_frames:
                continue
            cost = self._candidate_cost(state, center, width, gap, appearance)
            if cost is not None:
                candidates.append((cost, state.root_id))

        if not candidates:
            return raw_id
        candidates.sort(key=lambda item: item[0])
        best_score, best_root = candidates[0]
        if len(candidates) > 1:
            second_score = candidates[1][0]
            if second_score > 0 and best_score / second_score > self.config.ambiguity_ratio:
                return raw_id
        return best_root

    def _candidate_cost(
        self,
        state: _TrackState,
        center: tuple[float, float],
        width: float,
        gap: int,
        appearance: np.ndarray | None,
    ) -> float | None:
        old_center = _centroid(state.last_box)
        raw_distance = math.hypot(center[0] - old_center[0], center[1] - old_center[1])
        max_width = max(width, _median(state.widths))
        raw_jump_heads = raw_distance / max(1.0, max_width)
        if self.config.max_jump_heads > 0 and raw_jump_heads > self.config.max_jump_heads:
            return None

        distance = raw_distance
        direction_mismatch = 0.0
        if self.config.mode in {"velocity", "observation"}:
            vx, vy = self._velocity(state, tail=True)
            predicted = (old_center[0] + vx * gap, old_center[1] + vy * gap)
            predicted_distance = math.hypot(center[0] - predicted[0], center[1] - predicted[1])
            distance = min(distance, predicted_distance)
            if self.config.mode == "observation":
                direction_mismatch = _direction_cost((vx, vy), (center[0] - old_center[0], center[1] - old_center[1]))
                if direction_mismatch > self.config.max_direction_cost:
                    return None

        threshold = self.config.dist_heads * max_width
        implied_speed_heads = distance / max(1, gap) / max(1.0, width)
        if distance > threshold or implied_speed_heads > self.config.max_speed_heads:
            return None
        appearance_mismatch = 0.0
        if self.config.appearance_weight > 0:
            appearance_mismatch = _appearance_cost(state.last_appearance, appearance)
            if appearance_mismatch > self.config.max_appearance_cost:
                return None
        spatial_cost = distance / max(1.0, threshold)
        motion_weight = max(0.0, 1.0 - self.config.direction_weight - self.config.appearance_weight)
        return (
            motion_weight * spatial_cost
            + self.config.direction_weight * direction_mismatch
            + self.config.appearance_weight * appearance_mismatch
        )

    def _update_state(
        self,
        raw_id: int,
        root_id: int,
        box: BBox,
        frame_index: int,
        appearance: np.ndarray | None,
    ) -> None:
        state = self._states.get(raw_id)
        center = _centroid(box)
        width = _box_width(box)
        if state is None:
            self._states[raw_id] = _TrackState(
                root_id=root_id,
                first_frame=frame_index,
                last_frame=frame_index,
                first_box=box,
                last_box=box,
                centers=[(frame_index, center)],
                widths=[width],
                last_appearance=appearance,
            )
            return
        state.root_id = root_id
        state.last_frame = frame_index
        state.last_box = box
        state.centers.append((frame_index, center))
        state.widths.append(width)
        if appearance is not None:
            state.last_appearance = appearance
        if len(state.centers) > self.config.history_points:
            state.centers = state.centers[-self.config.history_points :]
        if len(state.widths) > self.config.history_points:
            state.widths = state.widths[-self.config.history_points :]

    def _velocity(self, state: _TrackState, *, tail: bool) -> tuple[float, float]:
        centers = state.centers[-self.config.history_points :] if tail else state.centers[: self.config.history_points]
        if len(centers) < 2:
            return (0.0, 0.0)
        first_frame, first_center = centers[0]
        last_frame, last_center = centers[-1]
        dt = max(1, last_frame - first_frame)
        return ((last_center[0] - first_center[0]) / dt, (last_center[1] - first_center[1]) / dt)

    def _prune(self, frame_index: int) -> None:
        cutoff = frame_index - self.config.max_archive_age_frames
        stale = [raw_id for raw_id, state in self._states.items() if state.last_frame < cutoff]
        for raw_id in stale:
            del self._states[raw_id]
            self._raw_to_root.pop(raw_id, None)


def _median(values: list[float]) -> float:
    if not values:
        return 1.0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
