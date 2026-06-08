"""Held-out FAR-region recall eval for the fine-tuned YOLO head detector.

This is a *read-only* evaluation helper. It NEVER trains and NEVER writes into
``models/fine_tuned/...``. It is purpose-built for Fix 5 ("improve recall on
tiny / occluded FAR heads"): it slices the held-out RPEE-Heads ``test`` split
into size buckets and reports recall per bucket, so small/far heads are scored
separately from large/near heads.

Why a new script (and not just ``scripts/evaluate_head_detector.py``)
--------------------------------------------------------------------
``scripts/evaluate_head_detector.py`` runs a single-image sanity check plus an
annotated video pass; it does not compute recall against ground truth, and it
does not break results out by head size. This helper adds the missing
quantitative, size-bucketed recall that the FAR-head work needs. It is additive
and leaves the existing evaluator untouched.

What "FAR" means here
---------------------
RPEE-Heads is overhead/elevated platform footage where head size is the most
reliable proxy for distance: distant passengers project to tiny boxes. We bucket
ground-truth boxes by pixel area (measured at the eval ``--imgsz``):

    tiny   : area  < 16x16   (the hardest / farthest heads)
    small  : 16x16 <= area < 32x32
    medium : 32x32 <= area < 96x96
    large  : area >= 96x96

Recall is computed by greedy IoU matching of predictions to ground truth at a
fixed IoU threshold (default 0.5), per bucket and overall.

Usage
-----
    python3 scripts/eval_far_heads.py \
        --model models/fine_tuned/head_detector/weights/best.pt \
        --images data/head_datasets/raw/rpee_heads/testing/test/images \
        --labels data/head_datasets/raw/rpee_heads/testing/test/labels \
        --imgsz 1280 --conf 0.10 --iou 0.5

Compare two checkpoints (baseline vs candidate) on the SAME held-out split:

    python3 scripts/eval_far_heads.py --model <baseline.pt> --imgsz 640  ...
    python3 scripts/eval_far_heads.py --model <candidate.pt> --imgsz 1280 ...

Only the candidate checkpoint should live OUTSIDE this repo (e.g. a Colab run
dir); do not overwrite the committed best.pt.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
import sys

import cv2

try:
    from ultralytics import YOLO
except ModuleNotFoundError:  # pragma: no cover - environment-specific dependency
    YOLO = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# (name, lower_px_side_inclusive, upper_px_side_exclusive). area is side*side.
BUCKETS = (
    ("tiny", 0, 16),
    ("small", 16, 32),
    ("medium", 32, 96),
    ("large", 96, 10 ** 9),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model",
        type=Path,
        default=Path("models/fine_tuned/head_detector/weights/best.pt"),
        help="Head-detector weights to evaluate (read-only).",
    )
    p.add_argument(
        "--images",
        type=Path,
        default=Path("data/head_datasets/raw/rpee_heads/testing/test/images"),
        help="Held-out test images dir (RPEE-Heads official test split).",
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=Path("data/head_datasets/raw/rpee_heads/testing/test/labels"),
        help="Matching YOLO label dir for --images.",
    )
    p.add_argument("--imgsz", type=int, default=1280, help="Inference image size (use 1280 for tiny heads).")
    p.add_argument("--conf", type=float, default=0.10, help="Confidence threshold (low favors recall).")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for a TP match.")
    p.add_argument("--max-det", type=int, default=1000, help="Max detections/image (crowds need >300).")
    p.add_argument("--device", type=str, default=None, help="cpu, mps, or 0.")
    p.add_argument("--limit", type=int, default=0, help="Optional cap on #images (0 = all). For quick smoke tests.")
    p.add_argument("--json-out", type=Path, default=None, help="Optional path to write a JSON summary.")
    return p.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass
class Bucket:
    name: str
    gt: int = 0
    tp: int = 0

    @property
    def recall(self) -> float:
        return self.tp / self.gt if self.gt else 0.0


@dataclass
class Eval:
    buckets: dict[str, Bucket] = field(default_factory=lambda: {n: Bucket(n) for n, _, _ in BUCKETS})
    total_pred: int = 0
    total_tp: int = 0
    images: int = 0

    @property
    def gt(self) -> int:
        return sum(b.gt for b in self.buckets.values())

    @property
    def tp(self) -> int:
        return sum(b.tp for b in self.buckets.values())


def bucket_for(side_px: float) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= side_px < hi:
            return name
    return BUCKETS[-1][0]


def load_gt(label_path: Path, img_w: int, img_h: int, imgsz: int) -> list[tuple[float, float, float, float, str]]:
    """Return GT boxes as (x1, y1, x2, y2) in *scaled-to-imgsz* pixels + size bucket.

    YOLO predicts at letterboxed imgsz; we score in original pixels but bucket by
    the box's side length scaled to imgsz so buckets line up with the model's
    effective resolution.
    """
    scale = imgsz / max(img_w, img_h) if max(img_w, img_h) else 1.0
    out: list[tuple[float, float, float, float, str]] = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _cls, xc, yc, w, h = (float(v) for v in parts)
        bx1 = (xc - w / 2) * img_w
        by1 = (yc - h / 2) * img_h
        bx2 = (xc + w / 2) * img_w
        by2 = (yc + h / 2) * img_h
        side = ((w * img_w) * (h * img_h)) ** 0.5 * scale
        out.append((bx1, by1, bx2, by2, bucket_for(side)))
    return out


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def match(preds: list[tuple[float, float, float, float]], gts: list, iou_thr: float, ev: Eval) -> None:
    """Greedy highest-confidence-first matching. ``preds`` must be conf-sorted desc."""
    used = [False] * len(gts)
    for pb in preds:
        best_j, best_iou = -1, iou_thr
        for j, g in enumerate(gts):
            if used[j]:
                continue
            v = iou(pb, (g[0], g[1], g[2], g[3]))
            if v >= best_iou:
                best_iou, best_j = v, j
        if best_j >= 0:
            used[best_j] = True
            ev.buckets[gts[best_j][4]].tp += 1


def main() -> int:
    args = parse_args()
    model_path = resolve(args.model)
    images_dir = resolve(args.images)
    labels_dir = resolve(args.labels)

    if YOLO is None:
        print("ERROR: ultralytics not installed. pip install -r requirements.txt", file=sys.stderr)
        return 2
    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 2
    if not images_dir.exists():
        print(f"ERROR: images dir not found: {images_dir}", file=sys.stderr)
        return 2

    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        print(f"ERROR: no images under {images_dir}", file=sys.stderr)
        return 2

    print(f"Model:  {model_path}")
    print(f"Images: {images_dir} ({len(image_paths)} images)")
    print(f"imgsz={args.imgsz} conf={args.conf} iou={args.iou} max_det={args.max_det}")
    model = YOLO(str(model_path))
    ev = Eval()

    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        img_h, img_w = frame.shape[:2]
        gts = load_gt(labels_dir / f"{img_path.stem}.txt", img_w, img_h, args.imgsz)
        for g in gts:
            ev.buckets[g[4]].gt += 1
        ev.images += 1

        res = model.predict(
            source=frame,
            conf=args.conf,
            iou=0.5,
            imgsz=args.imgsz,
            max_det=args.max_det,
            device=args.device,
            verbose=False,
        )[0]
        preds = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            order = confs.argsort()[::-1]
            preds = [tuple(xyxy[i]) for i in order]
        ev.total_pred += len(preds)
        match(preds, gts, args.iou, ev)

    print("\n" + "=" * 64)
    print("FAR-region recall by head-size bucket (held-out test split)")
    print("-" * 64)
    print(f"  {'bucket':<8} {'gt':>7} {'tp':>7} {'recall':>9}")
    for name, _, _ in BUCKETS:
        b = ev.buckets[name]
        print(f"  {name:<8} {b.gt:>7} {b.tp:>7} {b.recall:>9.3f}")
    print("-" * 64)
    overall_recall = ev.tp / ev.gt if ev.gt else 0.0
    print(f"  {'OVERALL':<8} {ev.gt:>7} {ev.tp:>7} {overall_recall:>9.3f}")
    print(f"  images={ev.images}  total_predictions={ev.total_pred}")
    far_b = ev.buckets["tiny"]
    print(f"\n  KEY METRIC (FAR/tiny heads <16px) recall = {far_b.recall:.3f}  "
          f"({far_b.tp}/{far_b.gt})")
    print("=" * 64)

    if args.json_out:
        summary = {
            "model": str(model_path),
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "images": ev.images,
            "total_predictions": ev.total_pred,
            "overall_recall": overall_recall,
            "buckets": {n: {"gt": ev.buckets[n].gt, "tp": ev.buckets[n].tp, "recall": ev.buckets[n].recall}
                        for n, _, _ in BUCKETS},
        }
        out = resolve(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote JSON summary: {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
