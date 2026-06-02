"""Tests for directional line crossing counters."""

from src.vision.line_counter import LineConfig, LineCount, LineManager, update_line_count


def test_line_crossing_counts_direction() -> None:
    """Track moving from one side to the other should increment direction count."""
    manager = LineManager(
        config=LineConfig(id="gate", name="gate", start=(0.0, 0.0), end=(10.0, 0.0), in_label="IN", out_label="OUT")
    )

    manager.update([{"track_id": 1, "bbox": (2, -6, 4, -2)}])  # bottom y=-2 (one side)
    counts = manager.update([{"track_id": 1, "bbox": (2, 1, 4, 3)}])  # bottom y=3 (other side)
    assert counts["IN"] == 1
    assert counts["OUT"] == 0
    assert manager.recent_events[0].track_id == 1
    assert manager.recent_events[0].line_id == "gate"


def test_no_duplicate_count_for_same_track_direction() -> None:
    """Same track should not be counted twice in the same direction."""
    manager = LineManager(
        config=LineConfig(id="gate", name="gate", start=(0.0, 0.0), end=(10.0, 0.0), in_label="IN", out_label="OUT")
    )
    manager.update([{"track_id": 7, "bbox": (2, -6, 4, -2)}])
    manager.update([{"track_id": 7, "bbox": (2, 1, 4, 3)}])  # first IN
    manager.update([{"track_id": 7, "bbox": (2, -6, 4, -2)}])  # OUT
    counts = manager.update([{"track_id": 7, "bbox": (2, 1, 4, 3)}])  # IN again
    assert counts["IN"] == 1
    assert counts["OUT"] == 1


def test_update_line_count_wrapper_updates_state() -> None:
    """Legacy helper should retain internal state and apply counting."""
    state = LineCount(in_count=0, out_count=0)
    update_line_count([{"track_id": 3, "bbox": (0, -4, 2, -1)}], (0, 0), (10, 0), state)
    updated = update_line_count([{"track_id": 3, "bbox": (0, 1, 2, 4)}], (0, 0), (10, 0), state)
    assert updated.in_count == 1
    assert updated.out_count == 0
