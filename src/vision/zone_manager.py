"""Zone occupancy utilities for polygon-based counting logic."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

BBox = tuple[float, float, float, float]
Point = tuple[float, float]
PointStrategy = Literal["bottom_center", "center"]


@dataclass(slots=True)
class ZoneConfig:
    """Static polygon zone configuration loaded from JSON."""

    id: str
    name: str
    polygon: list[Point]
    warning_threshold: int
    critical_threshold: int

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ZoneConfig":
        """Build a validated zone config from raw dictionary data."""
        zone_id = str(data["id"])
        name = str(data["name"])
        polygon_raw = data["polygon"]
        warning_threshold = int(data["warning_threshold"])
        critical_threshold = int(data["critical_threshold"])

        if not isinstance(polygon_raw, list) or len(polygon_raw) < 3:
            raise ValueError(f"Zone '{zone_id}' polygon must contain at least 3 points.")
        polygon: list[Point] = []
        for point in polygon_raw:
            if not isinstance(point, list | tuple) or len(point) != 2:
                raise ValueError(f"Zone '{zone_id}' has an invalid polygon point: {point!r}")
            polygon.append((float(point[0]), float(point[1])))

        if warning_threshold < 0 or critical_threshold < 0:
            raise ValueError(f"Zone '{zone_id}' thresholds must be non-negative.")
        if warning_threshold > critical_threshold:
            raise ValueError(
                f"Zone '{zone_id}' warning_threshold must be <= critical_threshold."
            )
        return cls(
            id=zone_id,
            name=name,
            polygon=polygon,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
        )


@dataclass(slots=True)
class ZoneState:
    """Runtime occupancy and unique-id state for one zone."""

    config: ZoneConfig
    current_occupancy: int = 0
    current_track_ids: set[int] = field(default_factory=set)
    unique_track_ids: set[int] = field(default_factory=set)


def extract_track_id(track: object) -> int | None:
    """Extract track ID from dict-like or object-like tracks."""
    if isinstance(track, dict):
        raw_id = track.get("track_id", track.get("id"))
    else:
        raw_id = getattr(track, "track_id", getattr(track, "id", None))
    if raw_id is None:
        return None
    return int(raw_id)


def extract_bbox(track: object) -> BBox | None:
    """Extract bounding box from dict-like or object-like tracks."""
    if isinstance(track, dict):
        bbox_raw = track.get("bbox")
    else:
        bbox_raw = getattr(track, "bbox", None)
    if bbox_raw is None:
        return None
    if len(bbox_raw) != 4:
        raise ValueError(f"Invalid bbox format: {bbox_raw!r}")
    x1, y1, x2, y2 = bbox_raw
    return (float(x1), float(y1), float(x2), float(y2))


def bottom_center_from_bbox(bbox: BBox) -> Point:
    """Return the bottom-center point of a bounding box."""
    x1, _y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def center_from_bbox(bbox: BBox) -> Point:
    """Return the center point of a bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_from_bbox(bbox: BBox, strategy: PointStrategy = "bottom_center") -> Point:
    """Return the configured reference point for analytics counting."""
    if strategy == "bottom_center":
        return bottom_center_from_bbox(bbox)
    if strategy == "center":
        return center_from_bbox(bbox)
    raise ValueError("point strategy must be either 'bottom_center' or 'center'")


def extract_source_type(track: object) -> str | None:
    """Extract a detection source ('body'/'head') from dict/object tracks."""
    if isinstance(track, dict):
        raw = track.get("source_type") or track.get("detector_mode")
    else:
        raw = getattr(track, "source_type", None) or getattr(track, "detector_mode", None)
    return str(raw) if raw is not None else None


def resolve_anchor_strategy(
    track: object,
    default_strategy: PointStrategy,
    *,
    per_detection: bool = False,
) -> PointStrategy:
    """Choose the anchor strategy for one track.

    With ``per_detection`` enabled (hybrid mode) head detections use the bbox
    center and body detections use the bottom-center, regardless of the manager
    default. Otherwise the manager default is used for every track.
    """
    if not per_detection:
        return default_strategy
    source = extract_source_type(track)
    if source == "head":
        return "center"
    if source == "body":
        return "bottom_center"
    return default_strategy


def _point_on_segment(point: Point, seg_start: Point, seg_end: Point) -> bool:
    """Return True if point lies exactly on a segment."""
    px, py = point
    x1, y1 = seg_start
    x2, y2 = seg_end
    cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
    if abs(cross) > 1e-9:
        return False
    within_x = min(x1, x2) - 1e-9 <= px <= max(x1, x2) + 1e-9
    within_y = min(y1, y2) - 1e-9 <= py <= max(y1, y2) + 1e-9
    return within_x and within_y


def point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    """Return True when a point is inside polygon (boundary included)."""
    if len(polygon) < 3:
        return False

    inside = False
    px, py = point
    prev_x, prev_y = polygon[-1]
    for curr_x, curr_y in polygon:
        if _point_on_segment(point, (prev_x, prev_y), (curr_x, curr_y)):
            return True

        intersects = (curr_y > py) != (prev_y > py)
        if intersects:
            x_at_py = (prev_x - curr_x) * (py - curr_y) / (prev_y - curr_y) + curr_x
            if px <= x_at_py:
                inside = not inside
        prev_x, prev_y = curr_x, curr_y
    return inside


def count_zone_occupancy(
    tracks: list[object],
    zone_polygon: list[list[int]],
    *,
    point_strategy: PointStrategy = "bottom_center",
    per_detection_anchor: bool = False,
) -> int:
    """Return people count inside a configured polygon zone."""
    polygon = [(float(x), float(y)) for x, y in zone_polygon]
    count = 0
    for track in tracks:
        bbox = extract_bbox(track)
        if bbox is None:
            continue
        strategy = resolve_anchor_strategy(
            track, point_strategy, per_detection=per_detection_anchor
        )
        if point_in_polygon(point_from_bbox(bbox, strategy), polygon):
            count += 1
    return count


class ZoneManager:
    """Manage occupancy and unique IDs for configured polygon zones."""

    def __init__(
        self,
        zones: list[ZoneConfig],
        *,
        point_strategy: PointStrategy = "bottom_center",
        per_detection_anchor: bool = False,
    ) -> None:
        point_from_bbox((0.0, 0.0, 1.0, 1.0), point_strategy)
        self.zones = zones
        self.point_strategy = point_strategy
        self.per_detection_anchor = per_detection_anchor
        self.zone_states: dict[str, ZoneState] = {
            zone.id: ZoneState(config=zone) for zone in zones
        }

    @classmethod
    def load_from_file(
        cls,
        config_path: str | Path,
        *,
        point_strategy: PointStrategy = "bottom_center",
        per_detection_anchor: bool = False,
    ) -> "ZoneManager":
        """Load zones from a JSON configuration file."""
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        zone_configs = [ZoneConfig.from_dict(zone) for zone in config.get("zones", [])]
        return cls(
            zones=zone_configs,
            point_strategy=point_strategy,
            per_detection_anchor=per_detection_anchor,
        )

    def update(self, tracks: list[object]) -> dict[str, int]:
        """Update occupancy and unique-track counters for all zones."""
        for state in self.zone_states.values():
            state.current_track_ids = set()

        for track in tracks:
            bbox = extract_bbox(track)
            track_id = extract_track_id(track)
            if bbox is None or track_id is None:
                continue
            strategy = resolve_anchor_strategy(
                track, self.point_strategy, per_detection=self.per_detection_anchor
            )
            track_point = point_from_bbox(bbox, strategy)
            for state in self.zone_states.values():
                if point_in_polygon(track_point, state.config.polygon):
                    state.current_track_ids.add(track_id)
                    state.unique_track_ids.add(track_id)

        for state in self.zone_states.values():
            state.current_occupancy = len(state.current_track_ids)

        return {zone_id: state.current_occupancy for zone_id, state in self.zone_states.items()}

    def get_zone_state(self, zone_id: str) -> ZoneState:
        """Return runtime state for one zone ID."""
        return self.zone_states[zone_id]
