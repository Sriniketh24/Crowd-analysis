"""Configurable post-detection filters for camera-specific cleanup."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.vision.detector import NormalizedDetection, normalize_detector_mode
from src.vision.zone_manager import Point, point_in_polygon


@dataclass(slots=True)
class DetectionRegionFilter:
    """Filter detections by include/ignore polygons and simple box constraints."""

    detector_mode: str
    include_polygons: list[list[Point]]
    ignore_polygons: list[list[Point]]
    point_strategy: str = "center"
    min_width: float = 0.0
    min_height: float = 0.0
    max_width: float | None = None
    max_height: float | None = None
    min_aspect_ratio: float | None = None
    max_aspect_ratio: float | None = None

    @classmethod
    def from_zones_file(
        cls,
        config_path: str | Path,
        *,
        detector_mode: str,
    ) -> "DetectionRegionFilter | None":
        """Build a mode-specific filter from a zones JSON file when configured."""
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        filters = config.get("detection_filters", {})
        if not isinstance(filters, dict):
            return None
        mode = normalize_detector_mode(detector_mode)
        raw = filters.get(mode)
        if not isinstance(raw, dict):
            return None

        return cls(
            detector_mode=mode,
            include_polygons=_parse_polygons(raw.get("include_polygons", [])),
            ignore_polygons=_parse_polygons(raw.get("ignore_polygons", [])),
            point_strategy=str(raw.get("point_strategy", "center")),
            min_width=float(raw.get("min_width", 0.0) or 0.0),
            min_height=float(raw.get("min_height", 0.0) or 0.0),
            max_width=_optional_float(raw.get("max_width")),
            max_height=_optional_float(raw.get("max_height")),
            min_aspect_ratio=_optional_float(raw.get("min_aspect_ratio")),
            max_aspect_ratio=_optional_float(raw.get("max_aspect_ratio")),
        )

    def keep(self, detection: NormalizedDetection) -> bool:
        """Return whether a detection passes the configured filters."""
        if normalize_detector_mode(detection.detector_mode) != self.detector_mode:
            return True

        x1, y1, x2, y2 = detection.bbox
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        if width < self.min_width or height < self.min_height:
            return False
        if self.max_width is not None and width > self.max_width:
            return False
        if self.max_height is not None and height > self.max_height:
            return False
        if height > 0:
            aspect_ratio = width / height
            if self.min_aspect_ratio is not None and aspect_ratio < self.min_aspect_ratio:
                return False
            if self.max_aspect_ratio is not None and aspect_ratio > self.max_aspect_ratio:
                return False

        point = self._reference_point(detection.bbox)
        if self.include_polygons and not any(point_in_polygon(point, polygon) for polygon in self.include_polygons):
            return False
        if any(point_in_polygon(point, polygon) for polygon in self.ignore_polygons):
            return False
        return True

    def filter(self, detections: list[NormalizedDetection]) -> list[NormalizedDetection]:
        """Return detections that pass this filter."""
        return [detection for detection in detections if self.keep(detection)]

    def _reference_point(self, bbox: tuple[float, float, float, float]) -> Point:
        x1, y1, x2, y2 = bbox
        if self.point_strategy == "top_center":
            return ((x1 + x2) / 2.0, y1)
        if self.point_strategy == "bottom_center":
            return ((x1 + x2) / 2.0, y2)
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _parse_polygons(raw: object) -> list[list[Point]]:
    if not isinstance(raw, list):
        return []
    polygons: list[list[Point]] = []
    for polygon_raw in raw:
        if not isinstance(polygon_raw, list):
            continue
        polygon: list[Point] = []
        for point_raw in polygon_raw:
            if not isinstance(point_raw, list | tuple) or len(point_raw) != 2:
                continue
            polygon.append((float(point_raw[0]), float(point_raw[1])))
        if len(polygon) >= 3:
            polygons.append(polygon)
    return polygons
