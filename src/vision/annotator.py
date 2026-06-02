"""Annotation helpers to draw overlays on processed frames."""

from __future__ import annotations

import cv2
import numpy as np

from src.vision.detector import NormalizedDetection
from src.vision.line_counter import LineManager
from src.vision.zone_manager import ZoneManager

ALERT_COLORS: dict[str, tuple[int, int, int]] = {
    "NORMAL": (0, 200, 0),
    "WARNING": (0, 200, 255),
    "CRITICAL": (0, 0, 255),
}


def annotate_frame(
    frame: np.ndarray,
    detections: list[NormalizedDetection],
    zone_manager: ZoneManager | None = None,
    line_managers: list[LineManager] | None = None,
    zone_occupancy: dict[str, int] | None = None,
    zone_alerts: dict[str, str] | None = None,
    line_counts: dict[str, dict[str, int]] | None = None,
    overlays: dict[str, object] | None = None,
) -> np.ndarray:
    """Draw detection/tracking overlays and lightweight metadata text."""
    annotated = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(value) for value in detection.bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
        track_prefix = (
            f"id={detection.track_id} " if detection.track_id is not None and detection.track_id >= 0 else ""
        )
        label = f"{track_prefix}{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

    if zone_manager is not None:
        _draw_zones(
            annotated=annotated,
            zone_manager=zone_manager,
            zone_occupancy=zone_occupancy or {},
            zone_alerts=zone_alerts or {},
        )

    if line_managers:
        _draw_lines(annotated=annotated, line_managers=line_managers, line_counts=line_counts or {})

    _draw_alert_banner(annotated, zone_alerts or {})

    if overlays:
        y_offset = 24
        for key, value in overlays.items():
            text = f"{key}: {value}"
            cv2.putText(
                annotated,
                text,
                (12, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y_offset += 26

    return annotated


def _draw_zones(
    annotated: np.ndarray,
    zone_manager: ZoneManager,
    zone_occupancy: dict[str, int],
    zone_alerts: dict[str, str],
) -> None:
    """Render zone polygons with names, occupancy, and alert levels."""
    for zone in zone_manager.zones:
        points = np.array(zone.polygon, dtype=np.int32)
        alert = zone_alerts.get(zone.id, "NORMAL")
        color = ALERT_COLORS.get(alert, ALERT_COLORS["NORMAL"])
        # Translucent fill so the underlying video stays visible: blend a filled
        # copy back onto the frame instead of painting an opaque polygon.
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [points], color=color)
        cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, dst=annotated)
        cv2.polylines(annotated, [points], isClosed=True, color=color, thickness=2)

        occupancy = zone_occupancy.get(zone.id, 0)
        text = f"{zone.name}: {occupancy} [{alert}]"
        anchor_x, anchor_y = points[0]
        cv2.putText(
            annotated,
            text,
            (int(anchor_x), max(20, int(anchor_y) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def _draw_lines(
    annotated: np.ndarray,
    line_managers: list[LineManager],
    line_counts: dict[str, dict[str, int]],
) -> None:
    """Render counting lines and directional IN/OUT counters."""
    y_offset = 70
    for line_manager in line_managers:
        start = (int(line_manager.config.start[0]), int(line_manager.config.start[1]))
        end = (int(line_manager.config.end[0]), int(line_manager.config.end[1]))
        cv2.line(annotated, start, end, (255, 0, 255), 2)
        mid_x = int((start[0] + end[0]) / 2)
        mid_y = int((start[1] + end[1]) / 2)
        cv2.putText(
            annotated,
            line_manager.config.name,
            (mid_x, max(20, mid_y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

        counts = line_counts.get(line_manager.config.id, {})
        in_label = line_manager.config.in_label
        out_label = line_manager.config.out_label
        in_count = counts.get(in_label, line_manager.in_count)
        out_count = counts.get(out_label, line_manager.out_count)

        cv2.putText(
            annotated,
            f"{line_manager.config.name} {in_label}:{in_count} {out_label}:{out_count}",
            (12, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
        y_offset += 24


def _draw_alert_banner(annotated: np.ndarray, zone_alerts: dict[str, str]) -> None:
    """Draw top-right summary of current crowd alert status."""
    if not zone_alerts:
        return
    priority = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}
    top_alert = max(zone_alerts.values(), key=lambda level: priority.get(level, 0))
    color = ALERT_COLORS.get(top_alert, ALERT_COLORS["NORMAL"])
    text = f"Alert: {top_alert}"
    text_size, _baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x = max(12, annotated.shape[1] - text_size[0] - 20)
    y = 28
    cv2.putText(
        annotated,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
