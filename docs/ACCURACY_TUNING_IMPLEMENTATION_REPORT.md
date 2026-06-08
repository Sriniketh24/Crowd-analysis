# Accuracy Tuning Implementation Report

## Summary

The system now supports accuracy tuning and comparison for both detector modes:

- `body`: full-body/person detector
- `head`: Colab-trained head detector at `models/fine_tuned/head_detector/weights/best.pt`

No local retraining was done. No synthetic data was used. These changes add practical
configuration, sweep, overlay, dashboard, and manual-count comparison tools so settings
can be tested honestly.

## Settings Added

The shared video pipeline now supports these controls through CLI/config where practical:

| Setting | Why it matters |
|---|---|
| `--confidence` | Lower values can recover weak small-head detections, but may increase false positives. |
| `--iou` | Controls non-max suppression. Dense crowds may need different overlap handling. |
| `--imgsz` | Larger inference image sizes help small or distant heads, but reduce FPS. |
| `--augment` / `--no-augment` | Enables Ultralytics test-time augmentation when supported. It is slower and should be measured. |
| `--max-det` | Raises the maximum boxes per frame for crowded scenes. |
| `--tracker bytetrack|botsort` | Allows tracker comparison for ID stability. |
| `--device` | Allows CPU, CUDA, MPS, or other supported Ultralytics devices. |

Config additions in `configs/app.yaml` include:

- `model.augment`
- `model.max_det`
- `model.head_confidence`
- `model.head_imgsz`
- `model.head_max_det`
- `tracker.head_config_overrides`

Head mode now uses practical configurable defaults for small/far heads:

- confidence: `0.25`
- image size: `1280`
- max detections/frame: `1000`
- lower tracker thresholds and larger track buffer through `head_config_overrides`

These are defaults, not hardcoded accuracy claims. Change them in config or override them
from the CLI.

## Commands

Run a tuned head demo:

```bash
python3 scripts/run_video_demo.py \
  --source data/input_videos/sample.mp4 \
  --zones-config configs/zones.cctv_platform.example.json \
  --detector-mode head \
  --model models/fine_tuned/head_detector/weights/best.pt \
  --confidence 0.25 \
  --iou 0.50 \
  --imgsz 1280 \
  --max-det 1000 \
  --tracker bytetrack
```

Run body vs head with selected settings:

```bash
python3 scripts/run_comparison_demo.py \
  --source data/input_videos/sample.mp4 \
  --zones-config configs/zones.cctv_platform.example.json \
  --confidence 0.25 \
  --iou 0.50 \
  --imgsz 1280 \
  --max-det 1000 \
  --tracker bytetrack
```

Run a quick tuning sweep:

```bash
python3 scripts/tune_detection_settings.py \
  --source data/input_videos/sample.mp4 \
  --zones-config configs/zones.cctv_platform.example.json \
  --detector-mode head \
  --model models/fine_tuned/head_detector/weights/best.pt \
  --quick
```

Run a larger sweep:

```bash
python3 scripts/tune_detection_settings.py \
  --source data/input_videos/sample.mp4 \
  --zones-config configs/zones.cctv_platform.example.json \
  --detector-mode both \
  --confidence-values 0.15,0.20,0.25,0.30,0.40 \
  --imgsz-values 640,960,1280 \
  --iou-values 0.45,0.50,0.60 \
  --trackers bytetrack,botsort
```

## Tuning Results CSV

The sweep writes:

```text
data/outputs/tuning_results.csv
```

Columns:

- `detector_mode`
- `model`
- `confidence`
- `imgsz`
- `iou`
- `tracker`
- `frames_processed`
- `total_detections`
- `unique_tracks`
- `avg_detections_per_frame`
- `processing_fps`
- `output_video_path`

Interpretation:

- Higher `total_detections` or `avg_detections_per_frame` may mean better recall, but may also mean more false positives.
- Higher `unique_tracks` may mean more people were tracked, but may also mean ID switching.
- Lower `processing_fps` is expected when using larger `imgsz` or `--augment`.
- `output_video_path` is populated only when `--save-videos` is used.

## Why Tuning Results Are Not Real Accuracy

`tuning_results.csv` is not an accuracy report because the sample video does not have
ground-truth labels. It only records detector/tracker behavior under different settings.

True accuracy requires at least one of:

- manually counted passengers for selected frame/time ranges
- labelled bounding boxes for representative frames
- approved validation CCTV with ground-truth annotations

Without ground truth, accuracy can only be estimated visually or through manual counts.

## Manual Ground Truth

Template:

```text
data/manual_ground_truth/example_counts.csv
```

Fill `manual_count` after reviewing the selected frame/time ranges by eye. Use either:

- `start_frame` and `end_frame`
- `start_time_sec` and `end_time_sec`

Then run:

```bash
python3 scripts/compare_manual_counts.py \
  --manual-csv data/manual_ground_truth/example_counts.csv \
  --body-db data/outputs/body_analytics.db \
  --head-db data/outputs/head_analytics.db
```

Output:

```text
data/outputs/manual_count_comparison.csv
```

This comparison reports body count, head count, manual count, absolute error, and
percentage error for each manual segment.

## Dashboard Update

The Streamlit app now includes an **Accuracy Tuning / Model Settings** section with:

- confidence threshold slider
- image size selector
- IoU slider
- tracker selector
- max detections input
- test-time augmentation checkbox
- plain-English explanation of each setting
- button to rerun body/head comparison with selected settings
- table preview/download for `data/outputs/tuning_results.csv`
- warning that true accuracy requires manually labelled or manually counted ground truth

## Honest Status

The system now supports accuracy tuning and comparison. It does not prove that accuracy
has improved yet. The next step is manual ground-truth comparison or labelled Indian
CCTV validation.
