"""Compare multi-object trackers on MOTChallenge sequences (e.g. MOT20).

Stage 0 of the ID-stability work: make ID-switching *measurable* before trying
to fix it. We run one fixed detector over each sequence and feed the identical
detections to several trackers, so the only variable is the tracker / ReID
component. Results are scored against the dataset ground truth with
``py-motmetrics`` (IDF1, ID-switches, MOTA, fragmentations).

This isolates the question we actually care about: does appearance-based ReID
(BoT-SORT / DeepOCSORT) reduce ID-switches versus motion-only tracking
(ByteTrack / OC-SORT)?

Designed to run in Colab/GPU (MOT20 is large and ReID wants a GPU). Example:

    pip install boxmot motmetrics ultralytics
    python scripts/eval_mot_tracker.py \
        --mot-root /content/MOT20/train \
        --seqs MOT20-01 MOT20-02 \
        --trackers bytetrack ocsort botsort deepocsort \
        --detector yolo11s.pt --conf 0.3 --imgsz 1280 \
        --max-frames 500 --device cuda:0 --half \
        --out-dir data/outputs/mot20_eval

Numbers are a *relative* same-detector comparison, not official MOT20
leaderboard scores (no distractor preprocessing / HOTA). That is enough to
prove which tracker keeps IDs most stable; leaderboard-grade eval (TrackEval)
can be layered on later.
"""

from __future__ import annotations

import argparse
import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

# py-motmetrics (1.4.0, the latest release) still calls np.asfarray, which NumPy
# removed in 2.0. Colab ships NumPy 2.x and boxmot/ultralytics require it, so we
# restore the helper here instead of downgrading NumPy. asfarray == asarray cast
# to a float dtype (defaulting to float64 when a non-float dtype is requested).
if not hasattr(np, "asfarray"):  # pragma: no cover - environment shim
    def _asfarray(a, dtype=np.float64):
        dtype = np.dtype(dtype)
        if not np.issubdtype(dtype, np.inexact):
            dtype = np.float64
        return np.asarray(a, dtype=dtype)

    np.asfarray = _asfarray  # type: ignore[attr-defined]

# Trackers that consume ReID appearance weights; the rest are motion-only.
REID_TRACKERS = {"botsort", "deepocsort", "strongsort", "hybridsort", "boosttrack"}
MOTION_TRACKERS = {"bytetrack", "ocsort", "sfsort"}
DEFAULT_TRACKERS = ("bytetrack", "ocsort", "botsort", "deepocsort")
DEFAULT_REID_WEIGHTS = "osnet_x0_25_msmt17.pt"
PERSON_CLASS_ID = 0


@dataclass(slots=True)
class Sequence:
    """A single MOTChallenge sequence on disk."""

    name: str
    img_dir: Path
    gt_path: Path | None
    frame_count: int


def _import_create_tracker():
    """Import boxmot's tracker factory across known module layouts."""
    try:
        from boxmot.trackers.tracker_zoo import create_tracker  # boxmot >= 11
    except ModuleNotFoundError:
        try:
            from boxmot.tracker_zoo import create_tracker  # older layout
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError(
                "boxmot is not installed. Run: pip install boxmot"
            ) from exc
    return create_tracker


def discover_sequences(mot_root: Path, wanted: Iterable[str] | None) -> list[Sequence]:
    """Find MOTChallenge sequences under ``mot_root`` (any dir containing img1/).

    ``mot_root`` is forgiving: point it at the split dir (``.../MOT20/train``),
    the dataset root (``.../MOT20``), or even ``/content`` — sequences are found
    recursively by locating ``img1`` folders, so a slightly-off path still works.
    """
    if not mot_root.exists():
        raise FileNotFoundError(
            f"--mot-root does not exist: {mot_root}\n"
            "The MOT20 download/unzip step likely did not complete. Expected a "
            "path like /content/MOT20/train containing MOT20-01/, MOT20-02/, ..."
        )
    wanted_set = {w.strip() for w in wanted} if wanted else None
    seq_dirs = sorted({p.parent for p in mot_root.rglob("img1") if p.is_dir()})
    sequences: list[Sequence] = []
    for seq_dir in seq_dirs:
        if wanted_set is not None and seq_dir.name not in wanted_set:
            continue
        img_dir = seq_dir / "img1"
        seqinfo = seq_dir / "seqinfo.ini"
        frame_count = 0
        if seqinfo.exists():
            parser = configparser.ConfigParser()
            parser.read(seqinfo)
            frame_count = parser.getint("Sequence", "seqLength", fallback=0)
        if frame_count <= 0:
            frame_count = sum(1 for _ in img_dir.glob("*.jpg"))
        gt_path = seq_dir / "gt" / "gt.txt"
        sequences.append(
            Sequence(
                name=seq_dir.name,
                img_dir=img_dir,
                gt_path=gt_path if gt_path.exists() else None,
                frame_count=frame_count,
            )
        )
    if not sequences:
        raise FileNotFoundError(
            f"No sequences found under {mot_root} (no img1/ folder anywhere below "
            f"it). Check the unzip step; expected e.g. {mot_root}/MOT20-01/img1/000001.jpg"
        )
    return sequences


def build_tracker(
    tracker_type: str,
    *,
    reid_weights: str,
    device: str,
    half: bool,
):
    """Create a boxmot tracker, attaching ReID weights only when relevant."""
    create_tracker = _import_create_tracker()
    kwargs = {"per_class": False}
    if tracker_type in REID_TRACKERS:
        kwargs.update(reid_weights=Path(reid_weights), device=device, half=half)
    return create_tracker(tracker_type, **kwargs)


def run_tracker_on_sequence(
    sequence: Sequence,
    tracker_type: str,
    detector,
    *,
    reid_weights: str,
    device: str,
    half: bool,
    conf: float,
    iou: float,
    imgsz: int,
    max_frames: int,
) -> list[tuple[int, int, float, float, float, float, float]]:
    """Detect + track one sequence; return MOT rows (frame, id, x, y, w, h, conf)."""
    import cv2

    tracker = build_tracker(
        tracker_type, reid_weights=reid_weights, device=device, half=half
    )
    frame_paths = sorted(sequence.img_dir.glob("*.jpg"))
    if max_frames > 0:
        frame_paths = frame_paths[:max_frames]

    rows: list[tuple[int, int, float, float, float, float, float]] = []
    for frame_index, frame_path in enumerate(frame_paths, start=1):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        prediction = detector.predict(
            source=frame,
            conf=conf,
            iou=iou,
            classes=[PERSON_CLASS_ID],
            imgsz=imgsz,
            device=device,
            half=half,
            verbose=False,
        )[0]
        boxes = prediction.boxes
        if boxes is None or len(boxes) == 0:
            dets = np.empty((0, 6), dtype=np.float32)
        else:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy().reshape(-1, 1)
            cls = np.zeros((xyxy.shape[0], 1), dtype=np.float32)
            dets = np.hstack([xyxy, confs, cls]).astype(np.float32)

        tracks = tracker.update(dets, frame)  # (M, 8): xyxy, id, conf, cls, det_idx
        for track in tracks:
            x1, y1, x2, y2, track_id = track[0], track[1], track[2], track[3], track[4]
            track_conf = track[5] if len(track) > 5 else 1.0
            rows.append(
                (
                    frame_index,
                    int(track_id),
                    float(x1),
                    float(y1),
                    float(x2 - x1),
                    float(y2 - y1),
                    float(track_conf),
                )
            )
    return rows


def write_mot_results(
    rows: list[tuple[int, int, float, float, float, float, float]], out_path: Path
) -> None:
    """Write tracker output in MOTChallenge format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for frame_index, track_id, x, y, w, h, conf in rows:
            handle.write(
                f"{frame_index},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
                f"{conf:.4f},-1,-1,-1\n"
            )


def score_sequences(
    gt_paths: dict[str, Path], result_paths: dict[str, Path]
):
    """Score tracker results against ground truth with py-motmetrics."""
    import motmetrics as mm

    accumulators = []
    names = []
    for name, gt_path in gt_paths.items():
        if name not in result_paths:
            continue
        gt = mm.io.loadtxt(str(gt_path), fmt="mot16", min_confidence=1)
        ts = mm.io.loadtxt(str(result_paths[name]), fmt="mot16")
        # When --max-frames caps the run, only score the frames we evaluated;
        # otherwise the untracked tail counts as misses and distorts MOTA/recall.
        if len(ts):
            last_frame = ts.index.get_level_values("FrameId").max()
            gt = gt[gt.index.get_level_values("FrameId") <= last_frame]
        accumulators.append(
            mm.utils.compare_to_groundtruth(gt, ts, "iou", distth=0.5)
        )
        names.append(name)
    if not accumulators:
        raise RuntimeError("No sequences had both ground truth and results to score.")

    metrics = [
        "idf1",
        "idp",
        "idr",
        "recall",
        "precision",
        "num_switches",
        "num_fragmentations",
        "mota",
        "motp",
        "mostly_tracked",
        "mostly_lost",
    ]
    handler = mm.metrics.create()
    summary = handler.compute_many(
        accumulators, metrics=metrics, names=names, generate_overall=True
    )
    return summary, handler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mot-root", required=True, type=Path,
                        help="MOTChallenge split dir, e.g. /path/to/MOT20/train")
    parser.add_argument("--seqs", nargs="*", default=None,
                        help="Sequence names to run (default: all under --mot-root)")
    parser.add_argument("--trackers", nargs="+", default=list(DEFAULT_TRACKERS),
                        help="Tracker types to compare")
    parser.add_argument("--detector", default="yolo11s.pt",
                        help="Ultralytics detector weights for person detection")
    parser.add_argument("--reid-weights", default=DEFAULT_REID_WEIGHTS,
                        help="ReID weights for appearance trackers (auto-downloaded)")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--max-frames", type=int, default=500,
                        help="Cap frames per sequence for a fast first pass (0 = all)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument("--half", action="store_true", help="Half precision (GPU)")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("data/outputs/mot20_eval"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the tracker comparison and print/save the metrics table."""
    args = parse_args(argv)
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError:
        raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")

    sequences = discover_sequences(args.mot_root, args.seqs)
    print(f"Found {len(sequences)} sequence(s): {', '.join(s.name for s in sequences)}")
    detector = YOLO(args.detector)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_tracker_summaries = {}
    for tracker_type in args.trackers:
        reid_tag = "ReID" if tracker_type in REID_TRACKERS else "motion-only"
        print(f"\n=== Tracker: {tracker_type} ({reid_tag}) ===")
        gt_paths: dict[str, Path] = {}
        result_paths: dict[str, Path] = {}
        for sequence in sequences:
            print(f"  - {sequence.name} ...", flush=True)
            rows = run_tracker_on_sequence(
                sequence, tracker_type, detector,
                reid_weights=args.reid_weights, device=args.device, half=args.half,
                conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                max_frames=args.max_frames,
            )
            res_path = args.out_dir / tracker_type / f"{sequence.name}.txt"
            write_mot_results(rows, res_path)
            result_paths[sequence.name] = res_path
            if sequence.gt_path is not None:
                gt_paths[sequence.name] = sequence.gt_path

        if not gt_paths:
            print("  (no ground truth found; wrote results only, skipping scoring)")
            continue
        import motmetrics as mm
        summary, handler = score_sequences(gt_paths, result_paths)
        per_tracker_summaries[tracker_type] = summary.loc[["OVERALL"]]
        print(mm.io.render_summary(
            summary, formatters=handler.formatters,
            namemap=mm.io.motchallenge_metric_names,
        ))

    if per_tracker_summaries:
        import pandas as pd
        combined = pd.concat(
            {t: s for t, s in per_tracker_summaries.items()}, names=["tracker"]
        ).reset_index(level=1, drop=True)
        out_csv = args.out_dir / "tracker_comparison.csv"
        combined.to_csv(out_csv)
        print("\n================ OVERALL COMPARISON (lower IDSW + higher IDF1 = better) ===")
        cols = ["idf1", "mota", "num_switches", "num_fragmentations"]
        print(combined[cols].sort_values("idf1", ascending=False).to_string())
        print(f"\nSaved: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
