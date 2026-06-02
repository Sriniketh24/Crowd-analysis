"""Utilities to write annotated output video frames to disk."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VideoWriter:
    """OpenCV-based frame writer for annotated output videos."""

    def __init__(
        self,
        output_path: str,
        fps: float = 20.0,
        frame_size: tuple[int, int] | None = None,
        fourcc: str = "avc1",
    ) -> None:
        """Initialize writer metadata; actual writer opens on first write if needed."""
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.fourcc = fourcc
        self._writer: cv2.VideoWriter | None = None
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    def set_fps(self, fps: float) -> None:
        """Update output FPS before the writer is opened."""
        if self._writer is not None:
            return
        if fps > 0:
            self.fps = fps

    def _open(self, frame_size: tuple[int, int]) -> None:
        """Open writer handle using known frame size."""
        codec = cv2.VideoWriter_fourcc(*self.fourcc)
        self._writer = cv2.VideoWriter(self.output_path, codec, self.fps, frame_size)
        if not self._writer.isOpened():
            raise RuntimeError(f"Unable to open output video writer at: {self.output_path}")

    def write(self, frame: np.ndarray) -> None:
        """Write one frame to output video."""
        if frame is None:
            return
        if self._writer is None:
            height, width = frame.shape[:2]
            size = self.frame_size if self.frame_size is not None else (width, height)
            self._open(size)
        assert self._writer is not None
        self._writer.write(frame)

    def close(self) -> None:
        """Close writer resources if allocated by implementation."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
