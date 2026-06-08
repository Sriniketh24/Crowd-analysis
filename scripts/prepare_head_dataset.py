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
* ``crowdhuman_heads`` -- supplemental dataset. CrowdHuman ODGT head boxes
  (``hbox``) -> YOLO. Only runs if the raw CrowdHuman files are present.
* ``scut_head`` -- supplemental dataset. Pascal-VOC XML head boxes -> YOLO.
  Only runs if the raw SCUT-HEAD files are present.

Examples
--------
    python3 scripts/prepare_head_dataset.py --help
    python3 scripts/prepare_head_dataset.py --dataset rpee_heads
    python3 scripts/prepare_head_dataset.py --dataset rpee_heads --validate-only
    python3 scripts/prepare_head_dataset.py --dataset crowdhuman_heads \\
        --raw-dir data/head_datasets/raw/crowdhuman \\
        --out-dir data/head_datasets/yolo/crowdhuman_heads
    python3 scripts/prepare_head_dataset.py --dataset scut_head \\
        --raw-dir data/head_datasets/raw/scut_head \\
        --out-dir data/head_datasets/yolo/scut_head
"""

from __future__ import annotations

import argparse
import json
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
    "crowdhuman_heads": {
        "raw": REPO_ROOT / "data/head_datasets/raw/crowdhuman",
        "out": REPO_ROOT / "data/head_datasets/yolo/crowdhuman_heads",
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


def _pixel_xyxy_to_yolo(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    img_w: float,
    img_h: float,
) -> tuple[float, float, float, float] | None:
    """Convert pixel ``xyxy`` to normalized YOLO, clipping to image bounds."""
    if img_w <= 0 or img_h <= 0:
        return None
    xmin = max(0.0, min(float(xmin), img_w))
    ymin = max(0.0, min(float(ymin), img_h))
    xmax = max(0.0, min(float(xmax), img_w))
    ymax = max(0.0, min(float(ymax), img_h))
    if not (xmax > xmin and ymax > ymin):
        return None

    xc = ((xmin + xmax) / 2.0) / img_w
    yc = ((ymin + ymax) / 2.0) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
        return None
    return xc, yc, w, h


def _pixel_xywh_to_yolo(
    x: float,
    y: float,
    w: float,
    h: float,
    img_w: float,
    img_h: float,
) -> tuple[float, float, float, float] | None:
    """Convert pixel ``xywh`` to normalized YOLO, clipping to image bounds."""
    if w <= 0 or h <= 0:
        return None
    return _pixel_xyxy_to_yolo(x, y, x + w, y + h, img_w, img_h)


def _format_yolo_box(box: tuple[float, float, float, float]) -> str:
    x, y, w, h = box
    return f"{HEAD_CLASS_ID} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


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
        kept.append(_format_yolo_box(box))
    return ("\n".join(kept) + ("\n" if kept else "")), len(kept)


def _safe_output_stem(image: Path, raw_dir: Path) -> str:
    rel = image.relative_to(raw_dir).with_suffix("")
    return "__".join(rel.parts)


def _place_image(image: Path, dest_dir: Path, link: bool, dest_name: str | None = None) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / (dest_name or image.name)
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
# CrowdHuman (supplemental; ODGT hbox -> YOLO)
# --------------------------------------------------------------------------- #


def _build_image_index(raw_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for image in _iter_images(raw_dir):
        index.setdefault(image.stem, []).append(image)
    return index


def _find_indexed_image(
    image_id: str, image_index: dict[str, list[Path]], preferred_split: str
) -> Path | None:
    matches = image_index.get(image_id, []) or image_index.get(Path(image_id).stem, [])
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    split_hints = {
        "train": ("train", "training"),
        "val": ("val", "valid", "validation"),
    }.get(preferred_split, (preferred_split,))
    for image in matches:
        parts = {part.lower() for part in image.parts}
        if any(hint in parts for hint in split_hints):
            return image
    return matches[0]


def _read_image_size(image: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image

        with Image.open(image) as img:
            return img.size
    except Exception:
        pass

    try:
        import cv2

        frame = cv2.imread(str(image))
        if frame is None:
            return None
        h, w = frame.shape[:2]
        return w, h
    except Exception:
        return None


def _crowdhuman_ann_files(raw_dir: Path) -> list[tuple[str, Path]]:
    ann_files: list[tuple[str, Path]] = []
    for path in sorted(raw_dir.rglob("*.odgt")):
        split = _classify_split(path.relative_to(raw_dir))
        if split == "test":
            continue
        if split is None:
            name = path.name.lower()
            if "train" in name:
                split = "train"
            elif "val" in name:
                split = "val"
        if split in ("train", "val"):
            ann_files.append((split, path))
    return ann_files


def _crowdhuman_record_to_yolo_lines(
    record: dict,
    img_w: float,
    img_h: float,
) -> tuple[list[str], int]:
    """Convert one CrowdHuman ODGT record's ``hbox`` entries to YOLO lines."""
    lines: list[str] = []
    skipped = 0
    for gtbox in record.get("gtboxes", []):
        if not isinstance(gtbox, dict):
            skipped += 1
            continue
        extra = gtbox.get("extra") or {}
        if isinstance(extra, dict):
            try:
                if int(extra.get("ignore", 0) or 0) == 1:
                    continue
            except (TypeError, ValueError):
                skipped += 1
                continue
        hbox = gtbox.get("hbox")
        if not isinstance(hbox, (list, tuple)) or len(hbox) != 4:
            skipped += 1
            continue
        try:
            x, y, w, h = (float(hbox[0]), float(hbox[1]), float(hbox[2]), float(hbox[3]))
        except (TypeError, ValueError):
            skipped += 1
            continue
        box = _pixel_xywh_to_yolo(x, y, w, h, img_w, img_h)
        if box is None:
            skipped += 1
            continue
        lines.append(_format_yolo_box(box))
    return lines, skipped


def prepare_crowdhuman_heads(raw_dir: Path, out_dir: Path, link: bool = True) -> PrepResult:
    """Convert CrowdHuman ``hbox`` head boxes to YOLO.

    CrowdHuman is a supplemental occlusion/crowding dataset, not the primary
    railway-platform dataset. Expected local files include ``annotation_train.odgt``
    and/or ``annotation_val.odgt`` plus matching images named ``<ID>.<ext>``.
    No files are downloaded by this script.
    """
    result = PrepResult(dataset="crowdhuman_heads", out_dir=out_dir)

    if not raw_dir.exists():
        _raise_missing_crowdhuman(raw_dir, out_dir)
    ann_files = _crowdhuman_ann_files(raw_dir)
    if not ann_files:
        _raise_missing_crowdhuman(raw_dir, out_dir)

    _reset_out_dir(out_dir)
    image_index = _build_image_index(raw_dir)

    for split, ann_path in ann_files:
        for line_no, raw_line in enumerate(
            ann_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                result.skipped_unreadable.append(f"{ann_path}:{line_no}")
                continue

            image_id = str(record.get("ID") or record.get("id") or "").strip()
            if not image_id:
                result.skipped_unreadable.append(f"{ann_path}:{line_no}: missing ID")
                continue
            image = _find_indexed_image(image_id, image_index, split)
            if image is None:
                result.skipped_missing_label.append(f"{ann_path}:{line_no}: image {image_id}")
                continue

            img_w = record.get("width") or record.get("img_width")
            img_h = record.get("height") or record.get("img_height")
            if img_w is None or img_h is None:
                size = _read_image_size(image)
                if size is None:
                    result.skipped_unreadable.append(str(image))
                    continue
                img_w, img_h = size

            try:
                img_w_f, img_h_f = float(img_w), float(img_h)
            except (TypeError, ValueError):
                result.skipped_unreadable.append(f"{ann_path}:{line_no}: invalid image size")
                continue

            lines, skipped = _crowdhuman_record_to_yolo_lines(record, img_w_f, img_h_f)
            if skipped:
                result.skipped_invalid_box.append(f"{ann_path}:{line_no}: {skipped} box(es)")

            out_stem = _safe_output_stem(image, raw_dir)
            _place_image(
                image,
                out_dir / "images" / split,
                link=link,
                dest_name=f"{out_stem}{image.suffix.lower()}",
            )
            text = "\n".join(lines) + ("\n" if lines else "")
            (out_dir / "labels" / split / f"{out_stem}.txt").write_text(
                text, encoding="utf-8"
            )

            st = result.split(split)
            st.images += 1
            st.labels += 1
            st.boxes += len(lines)

    return result


def _raise_missing_crowdhuman(raw_dir: Path, out_dir: Path) -> None:
    raise SystemExit(
        "ERROR: CrowdHuman raw data not found (no .odgt annotations under "
        f"{raw_dir}).\n\n"
        "CrowdHuman is SUPPLEMENTAL only; RPEE-Heads remains the primary dataset.\n"
        "This script does not download data. To use CrowdHuman head boxes:\n"
        "  1. Review the official source and terms:\n"
        "       https://www.crowdhuman.org/\n"
        "     CrowdHuman is intended for academic/research use; verify terms before use.\n"
        "  2. Place annotation_train.odgt / annotation_val.odgt and images under:\n"
        f"       {raw_dir}\n"
        "  3. Re-run:\n"
        "       python3 scripts/prepare_head_dataset.py --dataset crowdhuman_heads \\\n"
        f"           --raw-dir {raw_dir} --out-dir {out_dir}\n"
    )


# --------------------------------------------------------------------------- #
# SCUT-HEAD (supplemental; Pascal-VOC XML -> YOLO)
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
        box = _pixel_xyxy_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
        if box is None:
            skipped += 1
            continue
        lines.append(_format_yolo_box(box))
    return lines, skipped, len(lines)


def _load_scut_split_map(raw_dir: Path) -> dict[str, str]:
    """Read SCUT/VOC ImageSets split files when available.

    VOC test files are mapped to YOLO ``val`` because this script only prepares
    train/val folders. If no split files exist the caller falls back to a
    deterministic split.
    """
    split_map: dict[str, str] = {}
    aliases = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "test": "val",
        "testing": "val",
    }
    for split_file in sorted(raw_dir.rglob("ImageSets/Main/*.txt")):
        split = aliases.get(split_file.stem.lower())
        if split is None:
            continue
        for raw_line in split_file.read_text(encoding="utf-8", errors="replace").splitlines():
            stem = raw_line.strip().split()[0] if raw_line.strip() else ""
            if stem:
                split_map[Path(stem).stem] = split
    return split_map


def prepare_scut_head(raw_dir: Path, out_dir: Path, link: bool = True) -> PrepResult:
    """Convert SCUT-HEAD (VOC XML) to YOLO. Supplemental dataset only.

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
            "SCUT-HEAD is SUPPLEMENTAL only; RPEE-Heads remains the primary dataset.\n"
            "To use it:\n"
            "  1. Download from: https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release\n"
            "     SCUT-HEAD is free for academic research use only.\n"
            "  2. Unzip Part A / Part B into:\n"
            f"       {raw_dir}\n"
            "  3. Re-run:\n"
            "       python3 scripts/prepare_head_dataset.py --dataset scut_head \\\n"
            f"           --raw-dir {raw_dir} --out-dir {out_dir}\n"
        )

    _reset_out_dir(out_dir)
    images = list(_iter_images(raw_dir))
    split_map = _load_scut_split_map(raw_dir)

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

        split = split_map.get(image.stem)
        if split is None:
            inferred = _classify_split(image.relative_to(raw_dir))
            split = "val" if inferred in ("val", "test") else inferred
        if split not in ("train", "val"):
            split = "val" if idx % 7 == 0 else "train"

        out_stem = _safe_output_stem(image, raw_dir)
        _place_image(
            image,
            out_dir / "images" / split,
            link=link,
            dest_name=f"{out_stem}{image.suffix.lower()}",
        )
        text = "\n".join(lines) + ("\n" if lines else "")
        (out_dir / "labels" / split / f"{out_stem}.txt").write_text(
            text, encoding="utf-8"
        )

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
        description=(
            "Prepare real labeled head datasets in YOLO format "
            "(RPEE-Heads primary; CrowdHuman/SCUT supplemental)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Primary dataset: RPEE-Heads (railway platforms + event entrances, "
            "CC BY-SA 4.0).\n"
            "Supplemental only: CrowdHuman heads and SCUT-HEAD.\n"
            "This script never creates fake labels; if raw data is "
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
    elif args.dataset == "crowdhuman_heads":
        result = prepare_crowdhuman_heads(raw_dir, out_dir, link=not args.copy)
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
