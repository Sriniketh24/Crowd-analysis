"""Crowd state analysis based on occupancy thresholds and dwell times."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ZoneAlertThresholds:
    """Alert thresholds for one zone."""

    warning: int
    critical: int
    dwell_seconds: float = 0.0
    clear_below_count: int = 0


@dataclass(slots=True)
class ZoneAlertState:
    """Stateful alert tracker for one zone."""

    current_level: str = "NORMAL"
    warning_since: float | None = None
    critical_since: float | None = None


def is_crowded(occupancy: int, threshold: int) -> bool:
    """Return whether occupancy exceeds crowd threshold."""
    return occupancy >= threshold


def crowd_alert_level(occupancy: int, warning_threshold: int, critical_threshold: int) -> str:
    """Return alert level for occupancy and configured thresholds."""
    if occupancy >= critical_threshold:
        return "CRITICAL"
    if occupancy >= warning_threshold:
        return "WARNING"
    return "NORMAL"


class CrowdAnalyzer:
    """Compute crowd alert levels for all zones."""

    def __init__(self, thresholds_by_zone: dict[str, ZoneAlertThresholds]) -> None:
        self.thresholds_by_zone = thresholds_by_zone
        self.state_by_zone: dict[str, ZoneAlertState] = {
            zone_id: ZoneAlertState() for zone_id in thresholds_by_zone
        }

    def evaluate(
        self,
        occupancy_by_zone: dict[str, int],
        timestamp: float | None = None,
    ) -> dict[str, str]:
        """Return alert level by zone ID.

        Zones without configured thresholds are reported as ``NORMAL`` rather than
        raising, so a config mismatch never crashes the pipeline.
        """
        alerts: dict[str, str] = {}
        now = timestamp if timestamp is not None else 0.0
        for zone_id, occupancy in occupancy_by_zone.items():
            thresholds = self.thresholds_by_zone.get(zone_id)
            if thresholds is None:
                alerts[zone_id] = "NORMAL"
                continue
            alerts[zone_id] = self._evaluate_zone(zone_id, occupancy, thresholds, now)
        return alerts

    def _evaluate_zone(
        self,
        zone_id: str,
        occupancy: int,
        thresholds: ZoneAlertThresholds,
        timestamp: float,
    ) -> str:
        """Apply dwell and hysteresis rules for one zone."""
        state = self.state_by_zone.setdefault(zone_id, ZoneAlertState())
        clear_below = thresholds.clear_below_count
        if clear_below and occupancy <= clear_below:
            state.current_level = "NORMAL"
            state.warning_since = None
            state.critical_since = None
            return state.current_level

        if occupancy >= thresholds.critical:
            if state.critical_since is None:
                state.critical_since = timestamp
            state.warning_since = state.warning_since or timestamp
            dwell_elapsed = timestamp - state.critical_since
            if dwell_elapsed >= thresholds.dwell_seconds:
                state.current_level = "CRITICAL"
            elif state.current_level == "NORMAL":
                state.current_level = "WARNING"
            return state.current_level

        state.critical_since = None
        if occupancy >= thresholds.warning:
            if state.warning_since is None:
                state.warning_since = timestamp
            dwell_elapsed = timestamp - state.warning_since
            if dwell_elapsed >= thresholds.dwell_seconds:
                state.current_level = "WARNING"
            return state.current_level

        state.warning_since = None
        state.current_level = "NORMAL"
        return state.current_level
