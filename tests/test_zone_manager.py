"""Tests for zone occupancy and polygon helpers."""

from src.vision.zone_manager import ZoneManager, count_zone_occupancy, point_in_polygon


def test_point_inside_polygon() -> None:
    """Point-in-polygon should classify inside/outside points."""
    polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    assert point_in_polygon((5.0, 5.0), polygon)
    assert not point_in_polygon((20.0, 5.0), polygon)


def test_zone_occupancy_count_uses_bottom_center() -> None:
    """Occupancy should count only tracks whose bottom-center is inside polygon."""
    zone_polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
    tracks = [
        {"track_id": 1, "bbox": (1, 1, 3, 3)},  # bottom center (2,3) inside
        {"track_id": 2, "bbox": (9, 9, 12, 12)},  # bottom center (10.5,12) outside
        {"track_id": 3, "bbox": (4, 5, 6, 10)},  # bottom center (5,10) boundary
    ]
    assert count_zone_occupancy(tracks, zone_polygon) == 2


def test_zone_manager_tracks_current_and_unique_ids() -> None:
    """Zone manager should keep current occupancy and unique track history."""
    manager = ZoneManager.load_from_file("configs/zones.example.json")

    frame1 = [
        {"track_id": 10, "bbox": (150, 300, 200, 400)},
        {"track_id": 11, "bbox": (300, 320, 350, 420)},
    ]
    occupancy1 = manager.update(frame1)
    assert occupancy1["platform_zone"] == 2
    assert manager.get_zone_state("platform_zone").unique_track_ids == {10, 11}

    frame2 = [
        {"track_id": 11, "bbox": (320, 340, 360, 430)},
        {"track_id": 12, "bbox": (500, 350, 560, 440)},
    ]
    occupancy2 = manager.update(frame2)
    assert occupancy2["platform_zone"] == 2
    assert manager.get_zone_state("platform_zone").unique_track_ids == {10, 11, 12}
