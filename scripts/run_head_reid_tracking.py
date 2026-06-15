"""Head tracking with appearance ReID on head+shoulder crops.

The head detector finds heads, but head-only crops are too weak for appearance
re-identification (most distinguishing signal is clothing/torso). So for ReID we
expand each head box DOWN into a head+shoulder crop, run a boxmot ReID tracker
(BoT-SORT / DeepOCSORT with an OSNet model) on those crops, and map the resulting
stable track IDs back onto the original head boxes for counting/display.

This directly targets the "person goes behind a pole and comes back as a new ID"
failure: the tracker re-acquires the same person by appearance after an occlusion
gap, instead of relying on motion/IoU alone.

Outputs honest ID-stability metrics directly comparable to
``scripts/run_head_id_stability.py`` (confirmed_unique, inflation_factor, ...).
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.detector import Detector  # noqa: E402

SAMPLE_VIDEO_URL = "https://videos.pexels.com/video-files/12049569/12049569-hd_1280_720_25fps.mp4"
DEFAULT_RPEE_WEIGHTS = Path("models/fine_tuned/head_detector_s/weights/best.pt")
DEFAULT_REID_WEIGHTS = "osnet_x0_25_msmt17.pt"

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class ReidArm:
    """One ReID tracker experiment arm."""

    name: str
    tracker_type: str  # botsort | deepocsort | bytetrack
    with_reid: bool = True
    track_buffer: int = 90
    track_high_thresh: float = 0.35
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.35
    match_thresh: float = 0.8
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.3


DEFAULT_ARMS: tuple[ReidArm, ...] = (
    ReidArm("botsort_reid", "botsort", with_reid=True, track_buffer=90),
    ReidArm("botsort_noreid", "botsort", with_reid=False, track_buffer=90),
    ReidArm("deepocsort_reid", "deepocsort", with_reid=True, track_buffer=90),
)


@dataclass
class TrackRecord:
    frames: list[int] = field(default_factory=list)
    centers: list[tuple[float, float]] = field(default_factory=list)
    widths: list[float] = field(default_factory=list)


def ensure_video(video_path: Path, *, download_sample: bool) -> Path:
    if video_path.exists():
        return video_path
    if not download_sample:
        raise FileNotFoundError(f"Video not found: {video_path}")
    print(f"Downloading sample video to {video_path} ...")
    video_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(SAMPLE_VIDEO_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(video_path, "wb") as fh:
        fh.write(resp.read())
    return video_path


def head_to_shoulder(box: BBox, frame_w: int, frame_h: int, *, width_scale: float, down_scale: float, up_pad: float) -> BBox:
    """Expand a head box into a head+shoulder crop for ReID appearance."""
    x1, y1, x2, y2 = box
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    new_w = w * width_scale
    sx1 = max(0.0, cx - new_w / 2.0)
    sx2 = min(float(frame_w), cx + new_w / 2.0)
    sy1 = max(0.0, y1 - h * up_pad)
    sy2 = min(float(frame_h), y2 + h * down_scale)
    return (sx1, sy1, sx2, sy2)


def build_tracker(arm: ReidArm, reid_weights: Path, device: str, half: bool):
    """Construct a boxmot tracker with overridden params for head tracking."""
    from boxmot.trackers.tracker_zoo import create_tracker, get_tracker_config

    cfg_path = get_tracker_config(arm.tracker_type)
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    params = {k: v["default"] for k, v in raw.items() if isinstance(v, dict) and "default" in v}

    # Apply head-tracking overrides where the param exists for this tracker.
    overrides = {
        "track_buffer": arm.track_buffer,
        "track_high_thresh": arm.track_high_thresh,
        "track_low_thresh": arm.track_low_thresh,
        "new_track_thresh": arm.new_track_thresh,
        "match_thresh": arm.match_thresh,
        "proximity_thresh": arm.proximity_thresh,
        "appearance_thresh": arm.appearance_thresh,
    }
    for key, value in overrides.items():
        if key in params:
            params[key] = value

    reid_capable = arm.tracker_type in {"botsort", "deepocsort", "strongsort", "hybridsort"}
    # ``with_reid`` is not a yaml key but the ReID trackers accept it as a kwarg;
    # set it explicitly so the no-ReID control does not try to embed with a None model.
    if reid_capable:
        params["with_reid"] = arm.with_reid
    weights = reid_weights if (reid_capable and arm.with_reid) else None
    return create_tracker(
        arm.tracker_type,
        tracker_config=cfg_path,
        reid_weights=weights,
        device=device,
        half=half,
        evolve_param_dict=params,
    )


def color_for(track_id: int) -> tuple[int, int, int]:
    rng = (track_id * 1234567) % 0xFFFFFF
    return (rng & 0xFF, (rng >> 8) & 0xFF, (rng >> 16) & 0xFF)


def run_arm(
    arm: ReidArm,
    *,
    detector: Detector,
    video_path: Path,
    out_dir: Path,
    reid_weights: Path,
    device: str,
    half: bool,
    width_scale: float,
    down_scale: float,
    up_pad: float,
    min_confirmed_age: int,
    max_frames: int,
    write_video: bool,
) -> dict[str, object]:
    tracker = build_tracker(arm, reid_weights, device, half)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0:
        total = min(total, max_frames)

    writer = None
    video_out = out_dir / f"{arm.name}_annotated.mp4"
    if write_video:
        for fourcc_name in ("mp4v", "MJPG", "XVID"):
            writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*fourcc_name), fps, (frame_w, frame_h))
            if writer.isOpened():
                break
            writer = None

    records: dict[int, TrackRecord] = defaultdict(TrackRecord)
    per_frame_present: list[set[int]] = []

    frame_index = 0
    pbar = tqdm(total=total, desc=f"{arm.name}")
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames > 0 and frame_index >= max_frames):
            break
        heads = detector.detect(frame, frame_index=frame_index, timestamp=frame_index / fps).detections
        head_boxes: list[BBox] = [tuple(map(float, d.bbox)) for d in heads]
        dets = np.zeros((len(heads), 6), dtype=np.float32)
        for i, (d, hb) in enumerate(zip(heads, head_boxes)):
            sx1, sy1, sx2, sy2 = head_to_shoulder(hb, frame_w, frame_h, width_scale=width_scale, down_scale=down_scale, up_pad=up_pad)
            dets[i] = [sx1, sy1, sx2, sy2, float(d.confidence), 0.0]

        if len(dets) == 0:
            dets = np.empty((0, 6), dtype=np.float32)
        out = np.asarray(tracker.update(dets, frame))

        present: set[int] = set()
        for row in out:
            track_id = int(row[4])
            det_ind = int(row[7]) if row.shape[0] > 7 and 0 <= int(row[7]) < len(head_boxes) else -1
            head_box = head_boxes[det_ind] if det_ind >= 0 else (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
            hx1, hy1, hx2, hy2 = head_box
            cx, cy = (hx1 + hx2) / 2.0, (hy1 + hy2) / 2.0
            w = max(1.0, hx2 - hx1)
            rec = records[track_id]
            rec.frames.append(frame_index)
            rec.centers.append((cx, cy))
            rec.widths.append(w)
            present.add(track_id)
            if writer is not None:
                col = color_for(track_id)
                cv2.rectangle(frame, (int(hx1), int(hy1)), (int(hx2), int(hy2)), col, 2)
                cv2.putText(frame, str(track_id), (int(hx1), int(hy1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        per_frame_present.append(present)
        if writer is not None:
            writer.write(frame)
        frame_index += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    if writer is not None:
        writer.release()

    return compute_metrics(arm, records, per_frame_present, min_confirmed_age, video_out if write_video else None)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(values))


def compute_metrics(
    arm: ReidArm,
    records: dict[int, TrackRecord],
    per_frame_present: list[set[int]],
    min_confirmed_age: int,
    video_out: Path | None,
) -> dict[str, object]:
    raw_unique = len(records)
    confirmed_ids = {tid for tid, rec in records.items() if len(rec.frames) >= min_confirmed_age}
    confirmed_unique = len(confirmed_ids)

    concurrent_confirmed = [len(present & confirmed_ids) for present in per_frame_present]
    peak_concurrent = max(concurrent_confirmed) if concurrent_confirmed else 0
    median_concurrent = _median([float(c) for c in concurrent_confirmed if c > 0]) or 0.0

    lifespans = [rec.frames[-1] - rec.frames[0] + 1 for tid, rec in records.items() if tid in confirmed_ids]
    median_lifespan = _median([float(x) for x in lifespans])

    # Gap-jump diagnostics: where a confirmed track has internal frame gaps
    # (occlusion bridged by tracker/ReID), measure the spatial jump in head widths.
    gap_jump_events = 0
    max_gap_jump_heads = 0.0
    for tid in confirmed_ids:
        rec = records[tid]
        med_w = _median(rec.widths) or 1.0
        for k in range(1, len(rec.frames)):
            gap = rec.frames[k] - rec.frames[k - 1]
            if gap <= 2:
                continue
            (px, py), (cx, cy) = rec.centers[k - 1], rec.centers[k]
            jump_heads = (np.hypot(cx - px, cy - py)) / med_w
            gap_jump_events += 1
            max_gap_jump_heads = max(max_gap_jump_heads, float(jump_heads))

    inflation = confirmed_unique / median_concurrent if median_concurrent > 0 else float("nan")

    return {
        "arm": arm.name,
        "tracker": arm.tracker_type,
        "with_reid": arm.with_reid,
        "track_buffer": arm.track_buffer,
        "appearance_thresh": arm.appearance_thresh,
        "peak_concurrent_confirmed": peak_concurrent,
        "median_concurrent_confirmed": round(median_concurrent, 1),
        "raw_unique_ids": raw_unique,
        "confirmed_unique": confirmed_unique,
        "inflation_factor": round(inflation, 2) if median_concurrent > 0 else None,
        "gap_jump_events": gap_jump_events,
        "max_gap_jump_heads": round(max_gap_jump_heads, 2),
        "median_confirmed_lifespan": round(median_lifespan, 1),
        "annotated_video": str(video_out) if video_out else "",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, default=Path("data/input_videos/sample.mp4"))
    p.add_argument("--download-sample", action="store_true")
    p.add_argument("--model", type=Path, default=DEFAULT_RPEE_WEIGHTS)
    p.add_argument("--reid-weights", type=Path, default=Path(DEFAULT_REID_WEIGHTS))
    p.add_argument("--arms", nargs="+", default=["botsort_reid", "botsort_noreid"])
    p.add_argument("--conf", type=float, default=0.16)
    p.add_argument("--imgsz", type=int, default=1536)
    p.add_argument("--device", default="cpu")
    p.add_argument("--half", action="store_true")
    p.add_argument("--max-det", type=int, default=1000)
    p.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    p.add_argument("--min-confirmed-age", type=int, default=3)
    p.add_argument("--shoulder-width-scale", type=float, default=1.8)
    p.add_argument("--shoulder-down-scale", type=float, default=2.5)
    p.add_argument("--shoulder-up-pad", type=float, default=0.1)
    p.add_argument("--output-root", type=Path, default=Path("data/outputs/head_reid"))
    p.add_argument("--no-video", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    video_path = ensure_video(args.video, download_sample=args.download_sample)
    model_path = args.model
    if not model_path.exists():
        raise FileNotFoundError(f"Head detector weights not found: {model_path}")

    out_dir = args.output_root / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = Detector(
        weights_path=str(model_path),
        device=args.device,
        confidence=args.conf,
        imgsz=args.imgsz,
        max_det=args.max_det,
        half=args.half,
        detector_mode="head",
        use_fine_tuned_if_available=False,
        class_name_override="head",
    )

    available = {a.name: a for a in DEFAULT_ARMS}
    rows: list[dict[str, object]] = []
    for name in args.arms:
        if name not in available:
            print(f"Skipping unknown arm '{name}'. Available: {list(available)}")
            continue
        arm = available[name]
        print(f"\n=== Arm: {arm.name} (tracker={arm.tracker_type}, with_reid={arm.with_reid}, buffer={arm.track_buffer}) ===")
        metrics = run_arm(
            arm,
            detector=detector,
            video_path=video_path,
            out_dir=out_dir,
            reid_weights=args.reid_weights,
            device=args.device,
            half=args.half,
            width_scale=args.shoulder_width_scale,
            down_scale=args.shoulder_down_scale,
            up_pad=args.shoulder_up_pad,
            min_confirmed_age=args.min_confirmed_age,
            max_frames=args.max_frames,
            write_video=not args.no_video,
        )
        rows.append(metrics)
        print("  ", {k: metrics[k] for k in ("confirmed_unique", "inflation_factor", "gap_jump_events", "max_gap_jump_heads")})

    table = pd.DataFrame(rows)
    csv_path = out_dir / "reid_comparison_metrics.csv"
    table.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
