"""Run head-only ID-stability experiments on a video.

This is the script version of ``railway_head_id_stability_colab.ipynb``. It is
intended for Colab/A100 and local smoke tests: detect heads, track them with a
tuned ByteTrack wrapper, optionally stitch short track fragments, and write
honest ID-stability metrics plus annotated videos.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import shutil
import sys
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.detector import Detector, NormalizedDetection

try:
    import supervision as sv
except Exception:  # pragma: no cover - optional runtime fallback
    sv = None  # type: ignore[assignment]


SAMPLE_VIDEO_URL = "https://videos.pexels.com/video-files/12049569/12049569-hd_1280_720_25fps.mp4"
DEFAULT_RPEE_WEIGHTS = Path("models/fine_tuned/head_detector_s/weights/best.pt")
DEFAULT_CROWDHUMAN_WEIGHTS = Path("crowdhuman_head_s_best.pt")


@dataclass(frozen=True)
class Arm:
    """One experiment arm."""

    name: str
    conf: float
    activation: float
    consec: int
    expand: bool
    stitch: bool
    stitch_gap_frames: int | None = None
    stitch_dist_heads: float | None = None
    stitch_mode: str | None = None
    stitch_ambiguity_ratio: float | None = None
    stitch_max_speed_heads: float | None = None


@dataclass(frozen=True)
class StabilityConfig:
    """ID-stability knobs shared across arms."""

    imgsz: int = 1536
    detector_iou: float = 0.55
    head_nms_iou: float = 0.60
    max_det: int = 1000
    min_confirmed_age: int = 3
    lost_track_buffer: int = 90
    match_thresh: float = 0.85
    box_expand_factor: float = 1.6
    stitch_gap_frames: int = 45
    stitch_dist_heads: float = 1.5
    stitch_mode: str = "spatial"
    stitch_ambiguity_ratio: float = 0.80
    stitch_max_speed_heads: float = 0.45
    switch_lookback: int = 45


DEFAULT_ARMS = (
    Arm("baseline", conf=0.16, activation=0.45, consec=1, expand=False, stitch=False),
    Arm("fix_0p16", conf=0.16, activation=0.20, consec=3, expand=True, stitch=True),
    Arm("fix_0p30", conf=0.30, activation=0.30, consec=3, expand=True, stitch=True),
    Arm(
        "fix_0p16_safe",
        conf=0.16,
        activation=0.20,
        consec=3,
        expand=True,
        stitch=True,
        stitch_gap_frames=120,
        stitch_dist_heads=2.5,
        stitch_mode="velocity",
        stitch_ambiguity_ratio=0.65,
        stitch_max_speed_heads=0.35,
    ),
    Arm(
        "fix_0p16_loose",
        conf=0.16,
        activation=0.20,
        consec=3,
        expand=True,
        stitch=True,
        stitch_gap_frames=90,
        stitch_dist_heads=3.0,
        stitch_mode="velocity",
        stitch_ambiguity_ratio=0.80,
        stitch_max_speed_heads=0.45,
    ),
    Arm(
        "fix_0p16_wide",
        conf=0.16,
        activation=0.20,
        consec=3,
        expand=True,
        stitch=True,
        stitch_gap_frames=150,
        stitch_dist_heads=4.0,
        stitch_mode="velocity",
        stitch_ambiguity_ratio=1.0,
        stitch_max_speed_heads=0.70,
    ),
)

MANUAL_GT = {80: 25, 240: 27, 400: 21, 560: 28, 720: 24}


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=Path("data/input_videos/sample.mp4"))
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_RPEE_WEIGHTS,
        help="Head detector weights. Default is the RPEE-trained head_detector_s model.",
    )
    parser.add_argument(
        "--model-label",
        default=None,
        help="Short label used in output folder names. Defaults to model stem/parent.",
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["baseline", "fix_0p16", "fix_0p30"],
        choices=[arm.name for arm in DEFAULT_ARMS],
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/outputs/head_id_stability"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    parser.add_argument(
        "--stitch-mode",
        choices=["spatial", "velocity"],
        default="spatial",
        help="Default stitching mode for arms without their own override.",
    )
    parser.add_argument("--download-sample", action="store_true")
    parser.add_argument("--make-side-by-side", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def ensure_video(video_path: Path, *, download_sample: bool) -> Path:
    """Ensure the input video exists, with Colab-friendly download/upload fallback."""
    if video_path.exists():
        return video_path
    if download_sample or video_path == Path("data/input_videos/sample.mp4"):
        video_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Video not found at {video_path}; downloading Pexels sample with browser headers...")
        req = urllib.request.Request(
            SAMPLE_VIDEO_URL,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as response, video_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            return video_path
        except Exception as exc:
            print(f"Download failed: {exc}")

    if "google.colab" in sys.modules:
        from google.colab import files  # type: ignore

        print("Upload the input .mp4 now.")
        uploaded = files.upload()
        if uploaded:
            name = next(iter(uploaded))
            video_path.parent.mkdir(parents=True, exist_ok=True)
            Path(name).replace(video_path)
            return video_path
    raise FileNotFoundError(
        f"No video at {video_path}. In Colab, run with --download-sample or upload sample.mp4."
    )


def resolve_model(model_path: Path) -> Path:
    """Resolve a model path, with Colab upload fallback."""
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path
    candidates = [DEFAULT_RPEE_WEIGHTS, DEFAULT_CROWDHUMAN_WEIGHTS, Path("best.pt")]
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            print(f"Requested model missing; using available model: {candidate}")
            return candidate
    if "google.colab" in sys.modules:
        from google.colab import files  # type: ignore

        print("Upload head detector .pt weights now.")
        uploaded = files.upload()
        if uploaded:
            name = next(iter(uploaded))
            target = Path("head_detector_best.pt")
            Path(name).replace(target)
            return target
    raise FileNotFoundError(f"Head detector weights not found: {model_path}")


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Compute IoU for xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def centroid(box: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return box center."""
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def box_width(box: tuple[float, float, float, float]) -> float:
    """Return positive box width."""
    return max(1.0, box[2] - box[0])


def expand_box(
    box: tuple[float, float, float, float],
    factor: float,
) -> tuple[float, float, float, float]:
    """Expand a box around its center."""
    cx, cy = centroid(box)
    width, height = box[2] - box[0], box[3] - box[1]
    return (
        cx - width * factor / 2,
        cy - height * factor / 2,
        cx + width * factor / 2,
        cy + height * factor / 2,
    )


def nms(detections: Iterable[NormalizedDetection], iou_threshold: float) -> list[NormalizedDetection]:
    """Simple NMS over normalized detections."""
    kept: list[NormalizedDetection] = []
    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        if all(bbox_iou(detection.bbox, old.bbox) < iou_threshold for old in kept):
            kept.append(detection)
    return kept


class TunedHeadTracker:
    """ByteTrack wrapper with head-specific association boxes and fallback tracker."""

    def __init__(self, frame_rate: float, arm: Arm, cfg: StabilityConfig) -> None:
        self.arm = arm
        self.cfg = cfg
        self.backend = "greedy_iou"
        self.tracker = None
        if sv is not None and hasattr(sv, "ByteTrack"):
            kwargs = self._bytetrack_kwargs(frame_rate)
            try:
                self.tracker = sv.ByteTrack(**kwargs)
                self.backend = "bytetrack:" + json.dumps(kwargs, sort_keys=True)
            except TypeError:
                self.tracker = sv.ByteTrack()
                self.backend = "bytetrack:defaults"
        self._next_id = 1
        self._active: dict[int, tuple[tuple[float, float, float, float], int]] = {}

    def _bytetrack_kwargs(self, frame_rate: float) -> dict[str, object]:
        params = inspect.signature(sv.ByteTrack).parameters  # type: ignore[union-attr]
        candidates = {
            "track_activation_threshold": self.arm.activation,
            "lost_track_buffer": self.cfg.lost_track_buffer,
            "minimum_matching_threshold": self.cfg.match_thresh,
            "minimum_consecutive_frames": self.arm.consec,
            "frame_rate": max(1, int(round(frame_rate or 25))),
            # Older supervision names:
            "track_thresh": self.arm.activation,
            "track_buffer": self.cfg.lost_track_buffer,
            "match_thresh": self.cfg.match_thresh,
        }
        return {key: value for key, value in candidates.items() if key in params}

    def update(
        self,
        detections: list[NormalizedDetection],
    ) -> list[tuple[int, tuple[float, float, float, float], float]]:
        """Assign track IDs to detections."""
        if self.tracker is not None and sv is not None:
            return self._update_bytetrack(detections)
        return self._update_greedy(detections)

    def _tracking_box(
        self, detection: NormalizedDetection
    ) -> tuple[float, float, float, float]:
        return expand_box(detection.bbox, self.cfg.box_expand_factor) if self.arm.expand else detection.bbox

    def _update_bytetrack(
        self,
        detections: list[NormalizedDetection],
    ) -> list[tuple[int, tuple[float, float, float, float], float]]:
        if not detections:
            try:
                self.tracker.update_with_detections(sv.Detections.empty())  # type: ignore[union-attr]
            except Exception:
                pass
            return []

        track_boxes = np.array([self._tracking_box(detection) for detection in detections], dtype=float)
        confidence = np.array([detection.confidence for detection in detections], dtype=float)
        class_id = np.zeros(len(detections), dtype=int)
        original = np.array([detection.bbox for detection in detections], dtype=float)
        data = {
            "ox1": original[:, 0],
            "oy1": original[:, 1],
            "ox2": original[:, 2],
            "oy2": original[:, 3],
        }
        sv_detections = sv.Detections(  # type: ignore[union-attr]
            xyxy=track_boxes,
            confidence=confidence,
            class_id=class_id,
            data=data,
        )
        tracked = self.tracker.update_with_detections(sv_detections)
        if tracked.tracker_id is None:
            return []

        results: list[tuple[int, tuple[float, float, float, float], float]] = []
        for idx in range(len(tracked)):
            original_box = (
                float(tracked.data["ox1"][idx]),
                float(tracked.data["oy1"][idx]),
                float(tracked.data["ox2"][idx]),
                float(tracked.data["oy2"][idx]),
            )
            conf = float(tracked.confidence[idx]) if tracked.confidence is not None else 0.0
            results.append((int(tracked.tracker_id[idx]), original_box, conf))
        return results

    def _update_greedy(
        self,
        detections: list[NormalizedDetection],
    ) -> list[tuple[int, tuple[float, float, float, float], float]]:
        for track_id in list(self._active):
            box, lost = self._active[track_id]
            self._active[track_id] = (box, lost + 1)

        used: set[int] = set()
        results: list[tuple[int, tuple[float, float, float, float], float]] = []
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            track_box = self._tracking_box(detection)
            best_id: int | None = None
            best_iou = 0.0
            for track_id, (box, lost) in self._active.items():
                if track_id in used or lost > self.cfg.lost_track_buffer:
                    continue
                score = bbox_iou(track_box, box)
                if score > best_iou:
                    best_id, best_iou = track_id, score
            if best_id is None or best_iou < max(0.10, 1.0 - self.cfg.match_thresh):
                best_id = self._next_id
                self._next_id += 1
            used.add(best_id)
            self._active[best_id] = (track_box, 0)
            results.append((best_id, detection.bbox, detection.confidence))

        for track_id in list(self._active):
            if self._active[track_id][1] > self.cfg.lost_track_buffer:
                del self._active[track_id]
        return results


def stitch_tracks(
    per_frame: list[list[tuple[int, tuple[float, float, float, float], float]]],
    *,
    gap_frames: int,
    dist_heads: float,
    mode: str = "spatial",
    ambiguity_ratio: float = 0.80,
    max_speed_heads: float = 0.45,
) -> dict[int, int]:
    """Merge short-gap track fragments by spatial continuity."""
    info: dict[int, dict[str, object]] = {}
    for frame_index, detections in enumerate(per_frame):
        for track_id, box, _conf in detections:
            if track_id not in info:
                info[track_id] = {
                    "first": frame_index,
                    "last": frame_index,
                    "first_c": centroid(box),
                    "last_c": centroid(box),
                    "centers": [],
                    "widths": [box_width(box)],
                }
            row = info[track_id]
            row["last"] = frame_index
            row["last_c"] = centroid(box)
            row["centers"].append((frame_index, centroid(box)))  # type: ignore[union-attr]
            row["widths"].append(box_width(box))  # type: ignore[union-attr]

    parent = {track_id: track_id for track_id in info}

    def find(track_id: int) -> int:
        while parent[track_id] != track_id:
            parent[track_id] = parent[parent[track_id]]
            track_id = parent[track_id]
        return track_id

    def union(earlier: int, later: int) -> None:
        root_a, root_b = find(earlier), find(later)
        if root_a != root_b:
            parent[root_b] = root_a

    def velocity(track_id: int, *, tail: bool) -> tuple[float, float]:
        centers = info[track_id]["centers"]  # type: ignore[assignment]
        if len(centers) < 2:
            return (0.0, 0.0)
        segment = centers[-6:] if tail else centers[:6]
        if len(segment) < 2:
            return (0.0, 0.0)
        f0, c0 = segment[0]
        f1, c1 = segment[-1]
        dt = max(1, int(f1) - int(f0))
        return ((c1[0] - c0[0]) / dt, (c1[1] - c0[1]) / dt)

    births = sorted(info, key=lambda track_id: int(info[track_id]["first"]))
    for born_id in births:
        born = info[born_id]
        born_frame = int(born["first"])
        born_center = born["first_c"]  # type: ignore[assignment]
        born_width = float(np.median(born["widths"]))  # type: ignore[arg-type]
        candidates: list[tuple[float, int]] = []
        for old_id, old in info.items():
            if old_id == born_id:
                continue
            gap = born_frame - int(old["last"])
            if gap <= 0 or gap > gap_frames:
                continue
            old_center = old["last_c"]  # type: ignore[assignment]
            distance = math.hypot(born_center[0] - old_center[0], born_center[1] - old_center[1])
            if mode == "velocity":
                vx, vy = velocity(old_id, tail=True)
                predicted = (old_center[0] + vx * gap, old_center[1] + vy * gap)
                predicted_distance = math.hypot(
                    born_center[0] - predicted[0],
                    born_center[1] - predicted[1],
                )
                distance = min(distance, predicted_distance)
            threshold = dist_heads * max(born_width, float(np.median(old["widths"])))  # type: ignore[arg-type]
            implied_speed_heads = distance / max(1, gap) / max(1.0, born_width)
            if distance <= threshold and implied_speed_heads <= max_speed_heads:
                candidates.append((distance / max(1.0, threshold), old_id))
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0])
        best_score, best_id = candidates[0]
        if len(candidates) > 1:
            second_score = candidates[1][0]
            # If the best candidate is not clearly better, do not stitch. Dense
            # crowds often have several plausible nearby heads; a skipped stitch
            # is safer than assigning one ID to two different people.
            if second_score > 0 and best_score / second_score > ambiguity_ratio:
                continue
        if best_id is not None:
            union(best_id, born_id)
    return {track_id: find(track_id) for track_id in info}


def video_meta(path: Path) -> tuple[float, int, int, int]:
    """Return fps, width, height, frame count."""
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return fps, width, height, frames


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """Open a Colab-safe video writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for codec in ("mp4v", "MJPG", "XVID"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
    raise RuntimeError(f"Cannot open VideoWriter for {path}")


def color_for(track_id: int) -> tuple[int, int, int]:
    """Deterministic BGR color by track ID."""
    raw = (int(track_id) * 2654435761) % (256**3)
    return (raw % 256, (raw // 256) % 256, (raw // 65536) % 256)


def model_label_for(path: Path, override: str | None) -> str:
    """Build a compact model label for output paths."""
    if override:
        return override
    if path == DEFAULT_RPEE_WEIGHTS:
        return "rpee_head_s"
    if path == DEFAULT_CROWDHUMAN_WEIGHTS:
        return "crowdhuman_head_s"
    return path.stem.replace(".", "_")


def run_arm(
    arm: Arm,
    detector: Detector,
    video_path: Path,
    out_dir: Path,
    cfg: StabilityConfig,
    *,
    max_frames: int,
) -> dict[str, object]:
    """Run one arm and return summary metrics."""
    fps, width, height, total_frames = video_meta(video_path)
    if max_frames > 0:
        total_frames = min(total_frames, max_frames)
    detector.confidence = arm.conf
    tracker = TunedHeadTracker(fps, arm, cfg)
    stitch_gap_frames = arm.stitch_gap_frames or cfg.stitch_gap_frames
    stitch_dist_heads = arm.stitch_dist_heads or cfg.stitch_dist_heads
    stitch_mode = arm.stitch_mode or cfg.stitch_mode
    stitch_ambiguity_ratio = arm.stitch_ambiguity_ratio or cfg.stitch_ambiguity_ratio
    stitch_max_speed_heads = arm.stitch_max_speed_heads or cfg.stitch_max_speed_heads

    per_frame: list[list[tuple[int, tuple[float, float, float, float], float]]] = []
    capture = cv2.VideoCapture(str(video_path))
    frame_index = 0
    with tqdm(total=total_frames if total_frames > 0 else None, desc=f"{arm.name} track") as progress:
        while True:
            if max_frames > 0 and frame_index >= max_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
            raw = detector.detect(frame, frame_index=frame_index, timestamp=frame_index / fps).detections
            raw = nms(raw, cfg.head_nms_iou)
            per_frame.append(tracker.update(raw))
            frame_index += 1
            progress.update(1)
    capture.release()

    remap = (
        stitch_tracks(
            per_frame,
            gap_frames=stitch_gap_frames,
            dist_heads=stitch_dist_heads,
            mode=stitch_mode,
            ambiguity_ratio=stitch_ambiguity_ratio,
            max_speed_heads=stitch_max_speed_heads,
        )
        if arm.stitch
        else {}
    )

    def root_id(track_id: int) -> int:
        return remap.get(track_id, track_id)

    stats: dict[int, dict[str, object]] = defaultdict(lambda: {"frames": set(), "boxes": []})
    for frame_number, detections in enumerate(per_frame):
        for track_id, box, _conf in detections:
            rid = root_id(track_id)
            stats[rid]["frames"].add(frame_number)  # type: ignore[union-attr]
            stats[rid]["boxes"].append(box)  # type: ignore[union-attr]

    visible = {rid: len(row["frames"]) for rid, row in stats.items()}
    confirmed = {rid for rid, count in visible.items() if count >= cfg.min_confirmed_age}
    first = {rid: min(row["frames"]) for rid, row in stats.items() if row["frames"]}  # type: ignore[arg-type]
    last = {rid: max(row["frames"]) for rid, row in stats.items() if row["frames"]}  # type: ignore[arg-type]

    per_frame_counts = []
    for detections in per_frame:
        ids = {root_id(track_id) for track_id, _box, _conf in detections if root_id(track_id) in confirmed}
        per_frame_counts.append(len(ids))

    raw_unique = len({track_id for detections in per_frame for track_id, _box, _conf in detections})
    unique_poststitch = len(stats)
    confirmed_unique = len(confirmed)
    peak_concurrent = max(per_frame_counts) if per_frame_counts else 0
    median_concurrent = float(np.median(per_frame_counts)) if per_frame_counts else 0.0
    short_lived = sum(1 for count in visible.values() if count < cfg.min_confirmed_age)
    median_lifespan = float(np.median([visible[rid] for rid in confirmed])) if confirmed else 0.0
    inflation = confirmed_unique / peak_concurrent if peak_concurrent else float("nan")

    first_center = {rid: centroid(stats[rid]["boxes"][0]) for rid in confirmed}  # type: ignore[index]
    last_center = {rid: centroid(stats[rid]["boxes"][-1]) for rid in confirmed}  # type: ignore[index]
    switch_events = 0
    for rid in confirmed:
        for other in confirmed:
            if other == rid:
                continue
            gap = first[rid] - last[other]
            if 0 < gap <= cfg.switch_lookback:
                distance = math.hypot(
                    first_center[rid][0] - last_center[other][0],
                    first_center[rid][1] - last_center[other][1],
                )
                first_box = stats[rid]["boxes"][0]  # type: ignore[index]
                if distance <= stitch_dist_heads * box_width(first_box):
                    switch_events += 1
                    break

    gt_errors = []
    for frame_id, gt_count in MANUAL_GT.items():
        if frame_id < len(per_frame_counts):
            gt_errors.append(abs(per_frame_counts[frame_id] - gt_count))
    gt_mae = float(np.mean(gt_errors)) if gt_errors else float("nan")

    per_frame_csv = out_dir / f"{arm.name}_per_frame.csv"
    pd.DataFrame(
        {
            "frame": list(range(len(per_frame_counts))),
            "current_visible_confirmed": per_frame_counts,
        }
    ).to_csv(per_frame_csv, index=False)

    lifespans_csv = out_dir / f"{arm.name}_lifespans.csv"
    pd.DataFrame(
        [
            {
                "track_id": rid,
                "visible_frames": visible[rid],
                "first_frame": first.get(rid),
                "last_frame": last.get(rid),
                "confirmed": rid in confirmed,
            }
            for rid in sorted(stats)
        ]
    ).to_csv(lifespans_csv, index=False)

    annotated_video = out_dir / f"{arm.name}_annotated.mp4"
    capture = cv2.VideoCapture(str(video_path))
    writer = open_writer(annotated_video, fps, (width, height))
    frame_index = 0
    with tqdm(total=len(per_frame), desc=f"{arm.name} draw") as progress:
        while frame_index < len(per_frame):
            ok, frame = capture.read()
            if not ok:
                break
            current_visible = 0
            for track_id, box, _conf in per_frame[frame_index]:
                rid = root_id(track_id)
                if rid not in confirmed:
                    continue
                current_visible += 1
                x1, y1, x2, y2 = (int(value) for value in box)
                color = color_for(rid)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    str(rid),
                    (x1, max(10, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            labels = [
                arm.name,
                f"visible(confirmed): {current_visible}",
                f"unique(confirmed): {confirmed_unique}",
                f"raw IDs: {raw_unique}",
            ]
            for idx, label in enumerate(labels):
                y = 24 + idx * 24
                cv2.putText(frame, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
            frame_index += 1
            progress.update(1)
    capture.release()
    writer.release()

    return {
        "arm": arm.name,
        "conf": arm.conf,
        "backend": tracker.backend,
        "stitch_gap_frames": stitch_gap_frames if arm.stitch else 0,
        "stitch_dist_heads": stitch_dist_heads if arm.stitch else 0,
        "stitch_mode": stitch_mode if arm.stitch else "none",
        "stitch_ambiguity_ratio": stitch_ambiguity_ratio if arm.stitch else 0,
        "stitch_max_speed_heads": stitch_max_speed_heads if arm.stitch else 0,
        "peak_concurrent_confirmed": peak_concurrent,
        "median_concurrent_confirmed": round(median_concurrent, 2),
        "raw_unique_ids_prestitch": raw_unique,
        "unique_ids_poststitch": unique_poststitch,
        "confirmed_unique": confirmed_unique,
        "inflation_factor": round(inflation, 2) if inflation == inflation else None,
        "residual_switch_events": switch_events,
        "short_lived_tracks": short_lived,
        "median_confirmed_lifespan": round(median_lifespan, 2),
        "gt_mae": round(gt_mae, 2) if gt_mae == gt_mae else None,
        "per_frame_csv": str(per_frame_csv),
        "lifespans_csv": str(lifespans_csv),
        "annotated_video": str(annotated_video),
    }


def make_side_by_side(summaries: list[dict[str, object]], out_path: Path, target_height: int = 480) -> None:
    """Render annotated videos side-by-side for visual review."""
    caps = [cv2.VideoCapture(str(summary["annotated_video"])) for summary in summaries]
    fps = caps[0].get(cv2.CAP_PROP_FPS) or 25.0
    sizes = []
    for cap in caps:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        sizes.append((max(1, int(width * target_height / max(1, height))), target_height))
    writer = open_writer(out_path, fps, (sum(width for width, _height in sizes), target_height))
    while True:
        frames = []
        for cap, (width, height) in zip(caps, sizes):
            ok, frame = cap.read()
            if not ok:
                frames = None
                break
            frames.append(cv2.resize(frame, (width, height)))
        if frames is None:
            break
        writer.write(cv2.hconcat(frames))
    for cap in caps:
        cap.release()
    writer.release()


def main() -> int:
    """Run selected arms and save outputs."""
    args = parse_args()
    video_path = ensure_video(args.video, download_sample=args.download_sample)
    model_path = resolve_model(args.model)
    label = model_label_for(model_path, args.model_label)
    cfg = StabilityConfig(imgsz=args.imgsz, stitch_mode=args.stitch_mode)
    selected = {arm.name: arm for arm in DEFAULT_ARMS}
    arms = [selected[name] for name in args.arms]

    out_dir = args.output_root / video_path.stem / label
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Video: {video_path}")
    print(f"Model: {model_path}")
    print(f"Output: {out_dir}")
    print(f"Arms: {[arm.name for arm in arms]}")

    detector = Detector(
        weights_path=str(model_path),
        device=args.device,
        confidence=0.16,
        iou=cfg.detector_iou,
        imgsz=cfg.imgsz,
        max_det=cfg.max_det,
        half=args.half,
        use_fine_tuned_if_available=False,
        detector_mode="head",
    )

    summaries = [
        run_arm(arm, detector, video_path, out_dir, cfg, max_frames=args.max_frames)
        for arm in arms
    ]
    table = pd.DataFrame(summaries)
    metrics_path = out_dir / "comparison_metrics.csv"
    table.to_csv(metrics_path, index=False)
    print("\n=== comparison_metrics ===")
    print(table[
        [
            "arm",
            "stitch_gap_frames",
            "stitch_dist_heads",
            "stitch_mode",
            "stitch_ambiguity_ratio",
            "stitch_max_speed_heads",
            "peak_concurrent_confirmed",
            "median_concurrent_confirmed",
            "raw_unique_ids_prestitch",
            "unique_ids_poststitch",
            "confirmed_unique",
            "inflation_factor",
            "residual_switch_events",
            "gt_mae",
        ]
    ].to_string(index=False))
    print(f"\nSaved: {metrics_path}")

    if args.make_side_by_side and len(summaries) > 1:
        side_path = out_dir / "side_by_side.mp4"
        make_side_by_side(summaries, side_path)
        print(f"Side-by-side: {side_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
