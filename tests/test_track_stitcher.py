"""Tests for online track-ID stitching."""

from __future__ import annotations

import numpy as np

from src.vision.detector import NormalizedDetection
from src.vision.track_stitcher import TrackStitcher, TrackStitcherConfig


def _det(track_id: int, bbox: tuple[float, float, float, float], frame_index: int) -> NormalizedDetection:
    return NormalizedDetection(
        track_id=track_id,
        bbox=bbox,
        confidence=0.9,
        class_name="head/passenger",
        frame_index=frame_index,
        timestamp=frame_index / 25.0,
        detector_mode="head",
        source_type="head",
    )


def test_track_stitcher_remaps_short_occlusion_gap() -> None:
    stitcher = TrackStitcher(
        TrackStitcherConfig(
            enabled=True,
            gap_frames=20,
            dist_heads=4.0,
            mode="velocity",
            ambiguity_ratio=0.9,
            max_speed_heads=1.0,
            max_jump_heads=6.0,
        )
    )

    assert stitcher.update([_det(10, (100, 100, 120, 120), 0)], frame_index=0)[0].track_id == 10
    assert stitcher.update([_det(10, (104, 100, 124, 120), 1)], frame_index=1)[0].track_id == 10

    remapped = stitcher.update([_det(20, (128, 100, 148, 120), 8)], frame_index=8)

    assert remapped[0].track_id == 10


def test_track_stitcher_rejects_large_visible_jump() -> None:
    stitcher = TrackStitcher(
        TrackStitcherConfig(
            enabled=True,
            gap_frames=30,
            dist_heads=10.0,
            mode="velocity",
            ambiguity_ratio=1.0,
            max_speed_heads=5.0,
            max_jump_heads=3.0,
        )
    )

    stitcher.update([_det(1, (100, 100, 120, 120), 0)], frame_index=0)
    stitcher.update([_det(1, (102, 100, 122, 120), 1)], frame_index=1)
    result = stitcher.update([_det(2, (250, 100, 270, 120), 5)], frame_index=5)

    assert result[0].track_id == 2


def test_track_stitcher_prevents_same_root_in_one_frame() -> None:
    stitcher = TrackStitcher(
        TrackStitcherConfig(
            enabled=True,
            gap_frames=20,
            dist_heads=5.0,
            mode="velocity",
            ambiguity_ratio=1.0,
            max_speed_heads=2.0,
            max_jump_heads=6.0,
        )
    )

    stitcher.update([_det(1, (100, 100, 120, 120), 0)], frame_index=0)
    stitcher.update([_det(1, (104, 100, 124, 120), 1)], frame_index=1)
    result = stitcher.update(
        [
            _det(2, (128, 100, 148, 120), 8),
            _det(3, (130, 100, 150, 120), 8),
        ],
        frame_index=8,
    )

    assert [det.track_id for det in result] == [1, 3]


def test_track_stitcher_rejects_appearance_mismatch() -> None:
    stitcher = TrackStitcher(
        TrackStitcherConfig(
            enabled=True,
            gap_frames=20,
            dist_heads=5.0,
            mode="velocity",
            ambiguity_ratio=1.0,
            max_speed_heads=2.0,
            max_jump_heads=6.0,
            appearance_weight=0.2,
            max_appearance_cost=0.1,
        )
    )
    red_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    red_frame[:, :] = (0, 0, 255)
    blue_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    blue_frame[:, :] = (255, 0, 0)

    stitcher.update([_det(1, (100, 100, 120, 120), 0)], frame_index=0, frame=red_frame)
    stitcher.update([_det(1, (104, 100, 124, 120), 1)], frame_index=1, frame=red_frame)
    result = stitcher.update([_det(2, (128, 100, 148, 120), 8)], frame_index=8, frame=blue_frame)

    assert result[0].track_id == 2
