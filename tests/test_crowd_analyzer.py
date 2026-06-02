"""Tests for crowd analyzer threshold helpers."""

from src.vision.crowd_analyzer import CrowdAnalyzer, ZoneAlertThresholds, crowd_alert_level, is_crowded


def test_is_crowded_threshold_logic() -> None:
    """Crowd is true when occupancy reaches threshold."""
    assert is_crowded(10, 10)
    assert not is_crowded(9, 10)


def test_crowd_alert_warning_and_critical_levels() -> None:
    """Alert level should transition through NORMAL/WARNING/CRITICAL."""
    assert crowd_alert_level(occupancy=5, warning_threshold=10, critical_threshold=15) == "NORMAL"
    assert crowd_alert_level(occupancy=10, warning_threshold=10, critical_threshold=15) == "WARNING"
    assert crowd_alert_level(occupancy=20, warning_threshold=10, critical_threshold=15) == "CRITICAL"


def test_crowd_analyzer_evaluates_all_zones() -> None:
    """Analyzer should return per-zone alert levels from configured thresholds."""
    analyzer = CrowdAnalyzer(
        thresholds_by_zone={
            "platform_zone": ZoneAlertThresholds(warning=20, critical=30),
            "concourse_zone": ZoneAlertThresholds(warning=14, critical=20),
        }
    )
    alerts = analyzer.evaluate({"platform_zone": 22, "concourse_zone": 12}, timestamp=1.0)
    assert alerts == {"platform_zone": "WARNING", "concourse_zone": "NORMAL"}


def test_zone_without_thresholds_defaults_to_normal() -> None:
    """A zone missing from the threshold config must not raise; defaults to NORMAL."""
    analyzer = CrowdAnalyzer(thresholds_by_zone={})
    alerts = analyzer.evaluate({"unconfigured_zone": 999}, timestamp=1.0)
    assert alerts == {"unconfigured_zone": "NORMAL"}


def test_crowd_analyzer_honors_dwell_and_clear_thresholds() -> None:
    analyzer = CrowdAnalyzer(
        thresholds_by_zone={
            "platform_zone": ZoneAlertThresholds(
                warning=10,
                critical=15,
                dwell_seconds=2.0,
                clear_below_count=7,
            )
        }
    )
    assert analyzer.evaluate({"platform_zone": 12}, timestamp=0.0)["platform_zone"] == "NORMAL"
    assert analyzer.evaluate({"platform_zone": 12}, timestamp=3.0)["platform_zone"] == "WARNING"
    assert analyzer.evaluate({"platform_zone": 16}, timestamp=3.5)["platform_zone"] == "WARNING"
    assert analyzer.evaluate({"platform_zone": 16}, timestamp=6.0)["platform_zone"] == "CRITICAL"
    assert analyzer.evaluate({"platform_zone": 6}, timestamp=7.0)["platform_zone"] == "NORMAL"
