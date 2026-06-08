# Body vs Head Detection Integration Report

## Summary

The project now supports two passenger detection modes in the same video analytics
pipeline:

- `body`: pretrained YOLO full-body/person detection
- `head`: Google Colab fine-tuned YOLO head detection

Training was completed in Google Colab, not locally. No local training was run during
this integration.

## Colab Artifacts

- Colab zip found: `artifacts/colab/head_detector_colab_outputs.zip`
- Imported `best.pt` found: `artifacts/colab/best.pt`
- Canonical head detector path: `models/fine_tuned/head_detector/weights/best.pt`
- Existing canonical file status: present
- SHA-256 verification: `artifacts/colab/best.pt` matches
  `models/fine_tuned/head_detector/weights/best.pt`
- Colab zip contents include:
  - `content/crowd-analysis/models/fine_tuned/head_detector/weights/best.pt`
  - `models/fine_tuned/head_detector/weights/last.pt`
  - `models/fine_tuned/head_detector/results.csv`
  - `models/fine_tuned/head_detector/args.yaml`
  - `docs/HEAD_MODEL_TRAINING_REPORT.md`
  - `data/outputs/head_demo.mp4`

The canonical `best.pt` was already present, so it was used in place. The original
artifact was not deleted or overwritten.

## Dataset and Training

The available project docs and Colab training metadata identify the dataset as
RPEE-Heads. The model was trained as a single-class YOLO head detector (`0 = head`) with
`configs/head_dataset.yaml`.

This report does not claim production readiness. RPEE-Heads is railway-platform-relevant
but not Indian Railway CCTV, so domain validation on approved Indian platform footage is
still required.

## Code Changes

- Added `--detector-mode body|head` to `scripts/run_video_demo.py`.
- Added `scripts/run_comparison_demo.py` to run both modes on the same source.
- Added `detector_mode` to normalized detection/tracking records.
- Added mode-aware labels:
  - body: `person/passenger`
  - head: `head/passenger`
- Added mode-aware point strategies:
  - body: bottom-center of full-body box
  - head: center of head box
- Added `detector_mode` to SQLite/report outputs where practical:
  - `run_sessions`
  - `frames_processed`
  - `zone_occupancy`
  - `line_crossing_events`
  - `crowd_alerts`
- Added SQLite additive migrations so older databases receive `detector_mode` with
  default `body`.
- Updated annotation overlays with mode, camera/source, track counts, zones, lines, and
  alert status.

## Body Mode

Body mode uses the existing configured/pretrained YOLO person detector. By default this
is `yolo11n.pt` from `configs/app.yaml`, with the existing fallback behavior preserved.
Full-body zone and line analytics use the bottom-center of each tracked person box.

Run:

```bash
python3 scripts/run_video_demo.py \
  --source data/input_videos/sample.mp4 \
  --output data/outputs/body_demo.mp4 \
  --zones-config configs/zones.example.json \
  --db data/outputs/body_analytics.db \
  --detector-mode body
```

## Head Mode

Head mode uses the imported Colab-trained model:

```text
models/fine_tuned/head_detector/weights/best.pt
```

Head zone and line analytics use the center of each tracked head box. If the model file
is missing, the script prints the expected path and exits with a clear message instead of
raising an unclear traceback.

Run:

```bash
python3 scripts/run_video_demo.py \
  --source data/input_videos/sample.mp4 \
  --output data/outputs/head_demo.mp4 \
  --zones-config configs/zones.example.json \
  --db data/outputs/head_analytics.db \
  --detector-mode head \
  --model models/fine_tuned/head_detector/weights/best.pt
```

## Comparison Mode

The comparison script runs body first and then head on the same input, zones, and
confidence threshold. If the head model is missing, it still attempts body mode and then
prints where `best.pt` should be placed.

Run:

```bash
python3 scripts/run_comparison_demo.py \
  --source data/input_videos/sample.mp4 \
  --zones-config configs/zones.example.json
```

Optional:

```bash
python3 scripts/run_comparison_demo.py \
  --source data/input_videos/sample.mp4 \
  --zones-config configs/zones.example.json \
  --body-model yolo11n.pt \
  --head-model models/fine_tuned/head_detector/weights/best.pt \
  --confidence 0.35
```

## Report Export

Export reports separately:

```bash
python3 scripts/export_report.py \
  --db data/outputs/body_analytics.db \
  --out data/outputs/body_report.csv

python3 scripts/export_report.py \
  --db data/outputs/head_analytics.db \
  --out data/outputs/head_report.csv
```

The combined CSV includes `detector_mode` where that table stores it.

## Limitations

- The head model was trained on RPEE-Heads, not Indian Railway CCTV.
- No production accuracy claim is made.
- Body and head counts are not guaranteed to match because they detect different visual
  targets and use different reference points.
- Real CCTV use must avoid storing personally identifying footage unless explicitly
  approved.
- Existing zone/line geometry may need recalibration per camera and per mode, especially
  when head boxes are much smaller than full-body boxes.
