"""Tests for analytics metric helper placeholders."""

from src.analytics.metrics import calculate_flow_rate


def test_calculate_flow_rate_events_per_minute() -> None:
    """Flow helper converts count over window into per-minute rate."""
    assert calculate_flow_rate(30, 60) == 30.0


def test_calculate_flow_rate_handles_zero_window() -> None:
    """Zero or invalid windows should produce safe default rate."""
    assert calculate_flow_rate(30, 0) == 0.0
