"""Frame stream reader abstraction for files, webcams, RTSP, and HTTP streams."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterator

import cv2
import numpy as np

from src.config import infer_source_type, parse_source_value


@dataclass(slots=True)
class VideoFrame:
    """Container for one decoded frame and associated metadata."""

    index: int
    timestamp: float
    timestamp_ms: int
    data: np.ndarray
    source_timestamp: float
    source_type: str
    reconnect_count: int = 0
    dropped_frames: int = 0


@dataclass(slots=True)
class StreamStats:
    """Stream health and throughput counters."""

    source_type: str
    reconnect_count: int = 0
    dropped_frames: int = 0
    last_frame_time: float | None = None
    last_error: str | None = None
    status: str = "idle"


class StreamReader:
    """OpenCV-backed reader for local files and live video sources."""

    def __init__(
        self,
        source: str | int,
        *,
        source_type: str | None = None,
        frame_stride: int = 1,
        max_fps: float = 0.0,
        reconnect_backoff_seconds: float = 2.0,
        max_reconnect_attempts: int = 3,
        read_timeout_seconds: float = 15.0,
        camera_id: str | None = None,
    ) -> None:
        """Initialize a reader with source and live-stream policy."""
        self.source = parse_source_value(source)
        self.source_type = (source_type or infer_source_type(self.source)).lower()
        self.frame_stride = max(1, frame_stride)
        self.max_fps = max(0.0, max_fps)
        self.reconnect_backoff_seconds = max(0.0, reconnect_backoff_seconds)
        self.max_reconnect_attempts = max(0, max_reconnect_attempts)
        self.read_timeout_seconds = max(0.0, read_timeout_seconds)
        self.camera_id = camera_id or (
            Path(str(self.source)).stem if self.source_type == "file" else "camera"
        )
        self.capture: cv2.VideoCapture | None = None
        self.stats = StreamStats(source_type=self.source_type)
        self._last_yield_time: float | None = None

    @property
    def is_live_source(self) -> bool:
        """Return whether the source behaves like a live stream."""
        return self.source_type in {"webcam", "rtsp", "http"}

    def _resolve_capture_source(self) -> str | int:
        """Convert source to OpenCV-compatible value."""
        return self.source if self.source is not None else ""

    def open(self) -> None:
        """Open an OpenCV capture handle."""
        capture_source = self._resolve_capture_source()
        self.capture = cv2.VideoCapture(capture_source)
        self.stats.status = "opened" if self.is_opened() else "error"

    def is_opened(self) -> bool:
        """Return whether capture handle is currently open."""
        return self.capture is not None and self.capture.isOpened()

    def release(self) -> None:
        """Release capture resources."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.stats.status = "released"

    def source_exists(self) -> bool:
        """Validate source existence for local file paths."""
        if self.source_type in {"webcam", "rtsp", "http"}:
            return True
        return Path(str(self.source)).exists()

    def estimated_fps(self) -> float:
        """Return source FPS when OpenCV reports one, else 0."""
        if self.capture is None:
            return 0.0
        fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        return fps if fps > 0 else 0.0

    def _source_timestamp(self) -> float:
        """Return source timestamp in seconds when available."""
        if self.capture is None:
            return 0.0
        ms = float(self.capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        return ms / 1000.0 if ms > 0 else time.time()

    def _reconnect(self, attempt: int) -> bool:
        """Reconnect to live streams with exponential backoff."""
        self.release()
        backoff = self.reconnect_backoff_seconds * (2 ** max(0, attempt - 1))
        if backoff > 0:
            time.sleep(backoff)
        self.open()
        self.stats.reconnect_count += 1
        self.stats.status = "reconnected" if self.is_opened() else "error"
        return self.is_opened()

    def _enforce_max_fps(self) -> None:
        """Sleep between yielded frames to honor the configured cap."""
        if self.max_fps <= 0:
            return
        now = time.monotonic()
        if self._last_yield_time is not None:
            minimum_interval = 1.0 / self.max_fps
            elapsed = now - self._last_yield_time
            if elapsed < minimum_interval:
                time.sleep(minimum_interval - elapsed)
        self._last_yield_time = time.monotonic()

    def frames(self) -> Iterator[VideoFrame]:
        """Yield decoded frames from the source until stream completion."""
        if not self.is_opened():
            self.open()
        if not self.is_opened():
            self.stats.last_error = f"Unable to open source: {self.source}"
            return

        assert self.capture is not None
        frame_index = 0
        skipped_frames = 0
        reconnect_attempt = 0
        while True:
            started = time.monotonic()
            ok, frame = self.capture.read()
            if not ok or frame is None:
                self.stats.last_error = "read_failed"
                if not self.is_live_source or reconnect_attempt >= self.max_reconnect_attempts:
                    break
                reconnect_attempt += 1
                if self._reconnect(reconnect_attempt):
                    continue
                break

            reconnect_attempt = 0
            if self.read_timeout_seconds > 0 and time.monotonic() - started > self.read_timeout_seconds:
                self.stats.last_error = "read_timeout"
                if self.is_live_source and reconnect_attempt < self.max_reconnect_attempts:
                    reconnect_attempt += 1
                    if self._reconnect(reconnect_attempt):
                        continue
                break

            if frame_index % self.frame_stride != 0:
                skipped_frames += 1
                frame_index += 1
                continue

            self._enforce_max_fps()
            now = time.time()
            self.stats.last_frame_time = now
            self.stats.status = "streaming"
            self.stats.dropped_frames = skipped_frames
            yield VideoFrame(
                index=frame_index,
                timestamp=now,
                timestamp_ms=int(now * 1000),
                data=frame,
                source_timestamp=self._source_timestamp(),
                source_type=self.source_type,
                reconnect_count=self.stats.reconnect_count,
                dropped_frames=skipped_frames,
            )
            frame_index += 1
