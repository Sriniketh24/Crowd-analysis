"""Tiny-fixture tests for head dataset conversion helpers."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_head_dataset import (
    _crowdhuman_record_to_yolo_lines,
    _pixel_xywh_to_yolo,
    _voc_to_yolo_boxes,
    prepare_crowdhuman_heads,
    prepare_scut_head,
)


def test_pixel_xywh_to_yolo_clips_to_image_bounds() -> None:
    """Boxes partly outside the image should be clipped, not written out of range."""
    box = _pixel_xywh_to_yolo(-10, 20, 30, 40, img_w=100, img_h=200)
    assert box == (0.1, 0.2, 0.2, 0.2)

    assert _pixel_xywh_to_yolo(10, 20, 0, 40, img_w=100, img_h=200) is None


def test_crowdhuman_record_to_yolo_lines_uses_hbox_and_skips_ignored() -> None:
    record = {
        "ID": "sample",
        "gtboxes": [
            {"tag": "person", "hbox": [10, 20, 30, 40], "extra": {"ignore": 0}},
            {"tag": "person", "hbox": [1, 2, 3, 4], "extra": {"ignore": 1}},
            {"tag": "person", "hbox": [5, 5, 0, 10], "extra": {"ignore": 0}},
        ],
    }

    lines, skipped = _crowdhuman_record_to_yolo_lines(record, img_w=100, img_h=200)

    assert lines == ["0 0.250000 0.200000 0.300000 0.200000"]
    assert skipped == 1


def test_prepare_crowdhuman_heads_from_odgt_fixture(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "yolo"
    image_dir = raw_dir / "Images"
    image_dir.mkdir(parents=True)
    (image_dir / "crowd_001.jpg").write_bytes(b"fake image bytes")

    record = {
        "ID": "crowd_001",
        "width": 100,
        "height": 200,
        "gtboxes": [
            {"tag": "person", "hbox": [10, 20, 30, 40], "extra": {"ignore": 0}},
            {"tag": "person", "hbox": [0, 0, -1, 5], "extra": {"ignore": 0}},
        ],
    }
    (raw_dir / "annotation_train.odgt").write_text(json.dumps(record) + "\n", encoding="utf-8")

    result = prepare_crowdhuman_heads(raw_dir, out_dir, link=False)

    assert result.splits["train"].images == 1
    assert result.splits["train"].boxes == 1
    label = out_dir / "labels/train/Images__crowd_001.txt"
    assert label.read_text(encoding="utf-8") == "0 0.250000 0.200000 0.300000 0.200000\n"


def test_voc_to_yolo_boxes_fixture(tmp_path: Path) -> None:
    xml_path = tmp_path / "sample.xml"
    xml_path.write_text(
        """
        <annotation>
          <size><width>100</width><height>200</height></size>
          <object><name>head</name><bndbox>
            <xmin>10</xmin><ymin>20</ymin><xmax>40</xmax><ymax>60</ymax>
          </bndbox></object>
          <object><name>head</name><bndbox>
            <xmin>5</xmin><ymin>5</ymin><xmax>5</xmax><ymax>10</ymax>
          </bndbox></object>
        </annotation>
        """,
        encoding="utf-8",
    )

    parsed = _voc_to_yolo_boxes(xml_path)

    assert parsed is not None
    lines, skipped, n_boxes = parsed
    assert lines == ["0 0.250000 0.200000 0.300000 0.200000"]
    assert skipped == 1
    assert n_boxes == 1


def test_prepare_scut_head_uses_voc_imagesets_split(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    out_dir = tmp_path / "yolo"
    image_dir = raw_dir / "PartA/JPEGImages"
    ann_dir = raw_dir / "PartA/Annotations"
    split_dir = raw_dir / "PartA/ImageSets/Main"
    image_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)

    (image_dir / "scut_001.jpg").write_bytes(b"fake image bytes")
    (ann_dir / "scut_001.xml").write_text(
        """
        <annotation>
          <size><width>100</width><height>200</height></size>
          <object><name>head</name><bndbox>
            <xmin>10</xmin><ymin>20</ymin><xmax>40</xmax><ymax>60</ymax>
          </bndbox></object>
        </annotation>
        """,
        encoding="utf-8",
    )
    (split_dir / "test.txt").write_text("scut_001\n", encoding="utf-8")

    result = prepare_scut_head(raw_dir, out_dir, link=False)

    assert result.splits["val"].images == 1
    assert result.splits["val"].boxes == 1
    label = out_dir / "labels/val/PartA__JPEGImages__scut_001.txt"
    assert label.read_text(encoding="utf-8") == "0 0.250000 0.200000 0.300000 0.200000\n"
