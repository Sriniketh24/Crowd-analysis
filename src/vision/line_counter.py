"""Line crossing counter logic for directional passenger flow metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.vision.zone_manager import PointStrategy, point_from_bbox, resolve_anchor_strategy

Point = tuple[float, float]


def _cross_product(a: Point, b: Point, c: Point) -> float:
    """Return cross product of vectors AB and AC."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _line_side(point: Point, start: Point, end: Point) -> int:
    """Return side of point relative to directed line."""
    cross = _cross_product(start, end, point)
    if cross > 0:
        return 1
    if cross < 0:
        return -1
    return 0


@dataclass(slots=True)
class LineCount:
    """Directional line crossing count state."""

    in_count: int = 0
    out_count: int = 0
    track_last_side: dict[int, int] = field(default_factory=dict)
    counted_directions: dict[int, set[str]] = field(default_factory=dict)


@dataclass(slots=True)
class LineConfig:
    """Line metadata and geometry used for directional counting."""

    id: str
    name: str
    start: Point
    end: Point
    in_label: str = "IN"
    out_label: str = "OUT"


@dataclass(slots=True)
class LineCrossing:
    """One detected directional line crossing event."""

    line_id: str
    line_name: str
    track_id: int
    direction: str
    frame_index: int
    timestamp: float | None = None


@dataclass(slots=True)
class LineManager:
    """Stateful manager to count directional line crossings."""

    config: LineConfig
    point_strategy: PointStrategy = "bottom_center"
    per_detection_anchor: bool = False
    in_count: int = 0
    out_count: int = 0
    track_last_side: dict[int, int] = field(default_factory=dict)
    counted_directions: dict[int, set[str]] = field(default_factory=dict)
    recent_events: list[LineCrossing] = field(default_factory=list)

    def _extract_track_id(self, track: object) -> int | None:
        if isinstance(track, dict):
            raw_id = track.get("track_id", track.get("id"))
        else:
            raw_id = getattr(track, "track_id", getattr(track, "id", None))
        if raw_id is None:
            return None
        return int(raw_id)

    def _extract_bbox(self, track: object) -> tuple[float, float, float, float] | None:
        if isinstance(track, dict):
            bbox_raw = track.get("bbox")
        else:
            bbox_raw = getattr(track, "bbox", None)
        if bbox_raw is None or len(bbox_raw) != 4:
            return None
        x1, y1, x2, y2 = bbox_raw
        return (float(x1), float(y1), float(x2), float(y2))

    def update(
        self,
        tracks: list[object],
        *,
        frame_index: int = 0,
        timestamp: float | None = None,
    ) -> dict[str, int]:
        """Update line crossing counts and return summary labels."""
        self.recent_events = []
        for track in tracks:
            track_id = self._extract_track_id(track)
            bbox = self._extract_bbox(track)
            if track_id is None or bbox is None:
                continue

            strategy = resolve_anchor_strategy(
                track, self.point_strategy, per_detection=self.per_detection_anchor
            )
            side = _line_side(point_from_bbox(bbox, strategy), self.config.start, self.config.end)
            previous = self.track_last_side.get(track_id)
            self.track_last_side[track_id] = side

            if previous is None or previous == 0 or side == 0 or previous == side:
                continue

            direction = self.config.in_label if previous < side else self.config.out_label
            counted = self.counted_directions.setdefault(track_id, set())
            if direction in counted:
                continue

            counted.add(direction)
            if direction == self.config.in_label:
                self.in_count += 1
            else:
                self.out_count += 1
            self.recent_events.append(
                LineCrossing(
                    line_id=self.config.id,
                    line_name=self.config.name,
                    track_id=track_id,
                    direction=direction,
                    frame_index=frame_index,
                    timestamp=timestamp,
                )
            )

        return {
            self.config.in_label: self.in_count,
            self.config.out_label: self.out_count,
        }


def update_line_count(tracks: list[object], line_start: tuple[int, int], line_end: tuple[int, int], current: LineCount) -> LineCount:
    """Backward-compatible helper for single-line directional counting."""
    manager = LineManager(
        config=LineConfig(
            id="line",
            name="line",
            start=(float(line_start[0]), float(line_start[1])),
            end=(float(line_end[0]), float(line_end[1])),
        ),
        point_strategy="bottom_center",
        in_count=current.in_count,
        out_count=current.out_count,
        track_last_side=dict(current.track_last_side),
        counted_directions={k: set(v) for k, v in current.counted_directions.items()},
    )
    manager.update(tracks)
    current.in_count = manager.in_count
    current.out_count = manager.out_count
    current.track_last_side = manager.track_last_side
    current.counted_directions = manager.counted_directions
    return current
