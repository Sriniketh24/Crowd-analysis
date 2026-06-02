"""Prepare a real labeled head-detection dataset in YOLO format.

Primary dataset: **RPEE-Heads** (Railway Platforms and Event Entrances - Heads),
the on-domain railway-platform head-detection benchmark (CC BY-SA 4.0).
See ``docs/HEAD_DETECTION_RESEARCH_PLAN.md`` and ``docs/HEAD_DATASET_REPORT.md``.

What this script does
---------------------
* Discovers an *already-downloaded* raw dataset under ``--raw-dir`` (it does NOT
  download anything and it never invents annotations).
* Organizes images + labels into the YOLO layout::

      <out-dir>/images/train  <out-dir>/labels/train
      <out-dir>/images/val    <out-dir>/labels/val

* Forces a single class ``0 = head``.
* Validates every box (normalized 0-1, positive size); skips invalid boxes and
  logs them. Skips images whose label file is missing and logs them.
* Counts images / labels / head boxes per split.

Hard rules honored
------------------
* No synthetic data. No fake labels. If the raw files are missing the script
  prints clear instructions and exits non-zero -- it never fabricates a dataset.

Supported datasets
------------------
* ``rpee_heads`` -- labels already in YOLO format (``class x y w h`` normalized).
  Conversion is mostly file organization using the official train/val/test split.
* ``scut_head``  -- backup dataset. Pascal-VOC XML head boxes -> YOLO. Only runs
  if the raw SCUT-HEAD files are present.

Examples
--------
    python3 scripts/prepare_head_dataset.py --help
    python3 scripts/prepare_head_dataset.py --dataset rpee_heads
    python3 scripts/prepare_head_dataset.py --dataset rpee_heads --validate-only
    python3 scripts/prepare_head_dataset.py --dataset scut_head \\
        --raw-dir data/head_datasets/raw/scut_head \\
        --out-dir data/head_datasets/yolo/scut_head
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

HEAD_CLASS_ID = 0
HEAD_CLASS_NAME = "head"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = {
    "rpee_heads": {
        "raw": REPO_ROOT / "data/head_datasets/raw/rpee_heads",
        "out": REPO_ROOT / "data/head_datasets/yolo/rpee_heads",
    },
    "scut_head": {
        "raw": REPO_ROOT / "data/head_datasets/raw/scut_head",
        "out": REPO_ROOT / "data/head_datasets/yolo/scut_head",
    },
}


# --------------------------------------------------------------------------- #
# Result bookkeeping
# --------------------------------------------------------------------------- #


@dataclass
class SplitStats:
    images: int = 0
    labels: int = 0
    boxes: int = 0


@dataclass
class PrepResult:
    dataset: str
    out_dir: Path
    splits: dict[str, SplitStats] = field(default_factory=dict)
    skipped_missing_label: list[str] = field(default_factory=list)
    skipped_invalid_box: list[str] = field(default_factory=list)
    skipped_unreadable: list[str] = field(default_factory=list)

    def split(self, name: str) -> SplitStats:
        return self.splits.setdefault(name, SplitStats())

    def print_report(self) -> None:
        print("\n" + "=" * 60)
        print(f"Dataset prepared: {self.dataset}")
        print(f"Output (YOLO):    {self.out_dir}")
        print("-" * 60)
        total_imgs = total_lbls = total_boxes = 0
        for name in ("train", "val"):
            s = self.splits.get(name, SplitStats())
            print(
                f"  {name:<5} images={s.images:<6} labels={s.labels:<6} head_boxes={s.boxes}"
            )
            total_imgs += s.images
            total_lbls += s.labels
            total_boxes += s.boxes
        print("-" * 60)
        print(f"  TOTAL images={total_imgs}  labels={total_lbls}  head_boxes={total_boxes}")
        print(
            f"  skipped: missing_label={len(self.skipped_missing_label)} "
            f"invalid_box={len(self.skipped_invalid_box)} "
            f"unreadable_image={len(self.skipped_unreadable)}"
        )
        print("=" * 60)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def _classify_split(path: Path) -> str | None:
    """Infer train/val/test from any path component (case-insensitive)."""
    parts = [p.lower() for p in path.parts]
    joined = "/".join(parts)
    if any(p in ("train", "training") for p in parts) or "train" in joined:
        return "train"
    if any(p in ("val", "valid", "validation", "valset") for p in parts) or "val" in joined:
        return "val"
    if any(p in ("test", "testing") for p in parts) or "test" in joined:
        return "test"
    return None


def _find_label_file(image: Path, raw_root: Path) -> Path | None:
    """Find the YOLO .txt label that matches ``image``.

    Handles two common layouts:
    * label sits next to the image (same dir, same stem)
    * label sits in a parallel ``labels/`` dir mirroring an ``images/`` dir
    """
    stem_txt = image.with_suffix(".txt")
    if stem_txt.exists():
        return stem_txt

    # parallel images/ -> labels/ mirroring
    parts = list(image.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() in ("images", "image", "imgs", "jpegimages"):
            mirror = list(parts)
            mirror[i] = "labels"
            candidate = Path(*mirror).with_suffix(".txt")
            if candidate.exists():
                return candidate
            mirror[i] = "annotations"
            candidate = Path(*mirror).with_suffix(".txt")
            if candidate.exists():
                return candidate

    # last resort: search the tree for a uniquely-named label
    matches = list(raw_root.rglob(stem_txt.name))
    if len(matches) == 1:
        return matches[0]
    return None


def _validate_yolo_line(line: str) -> tuple[float, float, float, float] | None:
    """Return normalized (x, y, w, h) if the line is a valid head box, else None."""
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        _cls = float(parts[0])
        x, y, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
    except ValueError:
        return None
    # normalized coordinates must be within [0, 1]; size must be positive
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return None
    if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        return None
    # box must stay (mostly) inside the frame
    if x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6:
        return None
    if y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6:
        return None
    return x, y, w, h


def _clean_label_text(src: Path, result: PrepResult) -> tuple[str, int]:
    """Read a YOLO label file, drop invalid boxes, force class 0.

    Returns (cleaned_text, n_valid_boxes).
    """
    kept: list[str] = []
    for raw_line in src.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        box = _validate_yolo_line(line)
        if box is None:
            result.skipped_invalid_box.append(f"{src}: {line!r}")
            continue
        x, y, w, h = box
        kept.append(f"{HEAD_CLASS_ID} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return ("\n".join(kept) + ("\n" if kept else "")), len(kept)


def _place_image(image: Path, dest_dir: Path, link: bool) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / image.name
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if link:
        # relative symlink keeps the YOLO tree small (images can be ~1 GB)
        rel = os.path.relpath(image.resolve(), dest_dir.resolve())
        os.symlink(rel, dest)
    else:
        shutil.copy2(image, dest)


def _reset_out_dir(out_dir: Path) -> None:
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# RPEE-Heads (already YOLO format)
# --------------------------------------------------------------------------- #


def prepare_rpee_heads(
    raw_dir: Path,
    out_dir: Path,
    link: bool = True,
    test_into_val: bool = False,
) -> PrepResult:
    """Organize the RPEE-Heads release into the project YOLO layout.

    RPEE-Heads labels are *already* YOLO (``class x y w h`` normalized), so this
    is file organization + validation, not re-encoding. The official
    train/val/test split is honored via path components. ``test`` is held out by
    default (kept only in raw) unless ``test_into_val`` is set.
    """
    result = PrepResult(dataset="rpee_heads", out_dir=out_dir)

    if not raw_dir.exists() or not any(raw_dir.rglob("*")):
        _raise_missing_rpee(raw_dir)

    images = list(_iter_images(raw_dir))
    if not images:
        _raise_missing_rpee(raw_dir)

    _reset_out_dir(out_dir)

    unsplit = 0
    for image in images:
        split = _classify_split(image.relative_to(raw_dir))
        if split == "test":
            if test_into_val:
                split = "val"
            else:
                continue  # held out
        if split is None:
            # No split hint in the path -> default to train so nothing real is lost.
            split = "train"
            unsplit += 1

        label = _find_label_file(image, raw_dir)
        if label is None:
            result.skipped_missing_label.append(str(image))
            continue

        cleaned, n_boxes = _clean_label_text(label, result)

        _place_image(image, out_dir / "images" / split, link=link)
        (out_dir / "labels" / split / f"{image.stem}.txt").write_text(
            cleaned, encoding="utf-8"
        )

        st = result.split(split)
        st.images += 1
        st.labels += 1
        st.boxes += n_boxes

    if unsplit:
        print(
            f"[warn] {unsplit} image(s) had no train/val/test hint in their path "
            f"and were placed in 'train'. Verify the raw layout if this is unexpected."
        )
    return result


def _raise_missing_rpee(raw_dir: Path) -> None:
    raise SystemExit(
        "ERROR: RPEE-Heads raw data not found (no images under "
        f"{raw_dir}).\n\n"
        "This script never fabricates labels. Download the real dataset first:\n\n"
        "  Manual download steps\n"
        "  ---------------------\n"
        "  1. Open the dataset page:\n"
        "       https://ped.fz-juelich.de/da/doku.php?id=rpee_heads\n"
        "     (DOI: https://doi.org/10.34735/ped.2024.2 , license CC BY-SA 4.0)\n"
        "  2. Download the dataset archive (~1.1 GB):\n"
        "       https://ped.fz-juelich.de/data/machine_learning/"
        "2024_11_Recognition_In_Field_Studies/data/2024rpee_heads_dataset.zip\n"
        "  3. Unzip it into:\n"
        f"       {raw_dir}\n"
        "  4. Re-run:\n"
        "       python3 scripts/prepare_head_dataset.py --dataset rpee_heads\n"
    )


# --------------------------------------------------------------------------- #
# SCUT-HEAD (backup; Pascal-VOC XML -> YOLO)
# --------------------------------------------------------------------------- #


def _voc_to_yolo_boxes(xml_path: Path) -> tuple[list[str], int, int] | None:
    """Parse a Pascal-VOC XML into YOLO lines. Returns (lines, n_skipped) or None."""
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return None
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        return None
    try:
        img_w = float(size.findtext("width", "0"))
        img_h = float(size.findtext("height", "0"))
    except ValueError:
        return None
    if img_w <= 0 or img_h <= 0:
        return None

    lines: list[str] = []
    skipped = 0
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            skipped += 1
            continue
        try:
            xmin = float(bb.findtext("xmin", "nan"))
            ymin = float(bb.findtext("ymin", "nan"))
            xmax = float(bb.findtext("xmax", "nan"))
            ymax = float(bb.findtext("ymax", "nan"))
        except ValueError:
            skipped += 1
            continue
        if not (xmax > xmin and ymax > ymin):
            skipped += 1
            continue
        xc = ((xmin + xmax) / 2.0) / img_w
        yc = ((ymin + ymax) / 2.0) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
            skipped += 1
            continue
        lines.append(f"{HEAD_CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines, skipped, len(lines)


def prepare_scut_head(raw_dir: Path, out_dir: Path, link: bool = True) -> PrepResult:
    """Convert SCUT-HEAD (VOC XML) to YOLO. Backup dataset only.

    Expected raw layout (SCUT-HEAD Part A / Part B):
        <raw>/**/JPEGImages/*.jpg
        <raw>/**/Annotations/*.xml
        <raw>/**/ImageSets/Main/{train,val,test}.txt   (optional split lists)
    Without split lists, a deterministic 85/15 train/val split is used.
    """
    result = PrepResult(dataset="scut_head", out_dir=out_dir)

    if not raw_dir.exists() or not any(raw_dir.rglob("*.xml")):
        raise SystemExit(
            "ERROR: SCUT-HEAD raw data not found (no .xml annotations under "
            f"{raw_dir}).\n\n"
            "This is the BACKUP dataset. To use it:\n"
            "  1. Download from: https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release\n"
            "  2. Unzip Part A / Part B into:\n"
            f"       {raw_dir}\n"
            "  3. Re-run:\n"
            "       python3 scripts/prepare_head_dataset.py --dataset scut_head \\\n"
            f"           --raw-dir {raw_dir} --out-dir {out_dir}\n"
        )

    _reset_out_dir(out_dir)
    images = list(_iter_images(raw_dir))

    # deterministic split: every 7th image -> val
    for idx, image in enumerate(images):
        # locate the matching XML
        xml = None
        cand = image.parent.parent / "Annotations" / f"{image.stem}.xml"
        if cand.exists():
            xml = cand
        else:
            matches = list(raw_dir.rglob(f"{image.stem}.xml"))
            if len(matches) == 1:
                xml = matches[0]
        if xml is None:
            result.skipped_missing_label.append(str(image))
            continue

        parsed = _voc_to_yolo_boxes(xml)
        if parsed is None:
            result.skipped_unreadable.append(str(xml))
            continue
        lines, skipped, n_boxes = parsed
        if skipped:
            result.skipped_invalid_box.append(f"{xml}: {skipped} box(es)")

        split = "val" if idx % 7 == 0 else "train"
        _place_image(image, out_dir / "images" / split, link=link)
        text = "\n".join(lines) + ("\n" if lines else "")
        (out_dir / "labels" / split / f"{image.stem}.txt").write_text(text, encoding="utf-8")

        st = result.split(split)
        st.images += 1
        st.labels += 1
        st.boxes += n_boxes

    return result


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate_yolo_dataset(out_dir: Path) -> PrepResult:
    """Re-read an already-prepared YOLO dataset and report counts + any problems."""
    result = PrepResult(dataset=f"validate:{out_dir.name}", out_dir=out_dir)
    for split in ("train", "val"):
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        if not img_dir.exists():
            continue
        st = result.split(split)
        for image in _iter_images(img_dir):
            # resolve symlink target for readability check
            target = image.resolve()
            if not target.exists():
                result.skipped_unreadable.append(str(image))
                continue
            st.images += 1
            lbl = lbl_dir / f"{image.stem}.txt"
            if not lbl.exists():
                result.skipped_missing_label.append(str(image))
                continue
            st.labels += 1
            for raw_line in lbl.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if _validate_yolo_line(line) is None:
                    result.skipped_invalid_box.append(f"{lbl}: {line!r}")
                else:
                    st.boxes += 1
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a real labeled head dataset (RPEE-Heads primary) in YOLO format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Primary dataset: RPEE-Heads (railway platforms + event entrances, "
            "CC BY-SA 4.0).\nThis script never creates fake labels; if raw data is "
            "missing it prints download steps and exits."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DEFAULTS.keys()),
        default="rpee_heads",
        help="Which raw dataset to convert (default: rpee_heads).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Raw (downloaded/extracted) dataset dir. Defaults per --dataset.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="YOLO output dir. Defaults per --dataset.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy images instead of symlinking (uses more disk).",
    )
    parser.add_argument(
        "--test-into-val",
        action="store_true",
        help="RPEE only: also include the official test split inside val "
        "(default: test is held out and left in raw).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Do not convert; just validate an already-prepared YOLO out-dir.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir or DEFAULTS[args.dataset]["raw"]
    out_dir = args.out_dir or DEFAULTS[args.dataset]["out"]

    if args.validate_only:
        if not out_dir.exists():
            print(f"ERROR: nothing to validate, {out_dir} does not exist.", file=sys.stderr)
            return 2
        result = validate_yolo_dataset(out_dir)
        result.print_report()
        return 0

    if args.dataset == "rpee_heads":
        result = prepare_rpee_heads(
            raw_dir, out_dir, link=not args.copy, test_into_val=args.test_into_val
        )
    elif args.dataset == "scut_head":
        result = prepare_scut_head(raw_dir, out_dir, link=not args.copy)
    else:  # pragma: no cover - argparse restricts choices
        print(f"ERROR: unsupported dataset {args.dataset}", file=sys.stderr)
        return 2

    result.print_report()

    # surface a few skipped entries so problems are visible (never silent)
    for label, items in (
        ("missing-label images", result.skipped_missing_label),
        ("invalid boxes/files", result.skipped_invalid_box),
        ("unreadable images/xml", result.skipped_unreadable),
    ):
        if items:
            print(f"\n[skipped] {label} ({len(items)}); first few:")
            for entry in items[:5]:
                print(f"  - {entry}")

    total_imgs = sum(s.images for s in result.splits.values())
    total_lbls = sum(s.labels for s in result.splits.values())
    if total_imgs == 0 or total_lbls == 0:
        print("\nERROR: no valid image/label pairs were produced.", file=sys.stderr)
        return 1
    print(f"\nDone. YOLO dataset ready at: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
