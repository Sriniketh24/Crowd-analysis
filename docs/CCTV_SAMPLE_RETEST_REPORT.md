# CCTV Sample Retest Report

## Why The Sample Was Changed

The previous demo video was not representative enough because it looked closer to phone-angle footage. A railway deployment will normally use fixed platform CCTV cameras with a higher, wider, and more static view. The retest uses a real railway platform clip with a CCTV-like elevated/static angle so the body-vs-head comparison is tested under conditions closer to platform monitoring.

## New Video Used

- source/path: `data/input_videos/cctv_platform_sample.mp4`
- canonical demo path: `data/input_videos/sample.mp4`
- source page: `https://www.pexels.com/video/people-on-platform-on-train-station-12049569/`
- source website: Pexels
- whether real footage: yes, it appears to be real public railway platform footage
- whether CCTV-like: partially; it is elevated and static-looking, but it is not confirmed operational CCTV
- resolution: 1280 x 720
- FPS: 25.000
- duration: 32.12 seconds
- frame count: 803
- extracted calibration frames: `data/sample_frames/current_cctv_sample/`
- limitations: not Indian Railway-specific, not confirmed CCTV, stock-footage provenance, and not a substitute for supervisor-approved station footage

`data/input_videos/sample.mp4` and `data/input_videos/cctv_platform_sample.mp4` have the same SHA-256 hash at retest time. The pre-existing `sample.mp4` was also copied to `data/input_videos/previous_sample_backup.mp4` before regeneration steps.

## Zone Configuration

- zones config path: `configs/zones.cctv_platform.example.json`
- app/demo camera config: `configs/cameras.yaml` now points `demo_platform` to this zone config
- threshold config: `configs/thresholds.yaml` includes explicit demo thresholds for the new zone IDs

Defined zones:

- `main_platform_waiting`: central visible waiting/crowd area where passengers gather near the train and pillars
- `platform_movement_path`: diagonal walking path through the platform
- `train_side_edge`: narrow train-side/platform-edge region where crowding near doors is most relevant

Defined counting line:

- `central_flow_line`: diagonal line separating the central walking path from the train-side area, labeled `TOWARD_TRAIN` and `AWAY_FROM_TRAIN`

The polygons were calibrated from `data/sample_frames/current_cctv_sample/preview.jpg` and kept within the visible platform surface. Demo warning thresholds were kept low enough to trigger on moderate crowding in this clip without inventing zones outside the scene.

## Body Detection Result

Command run:

```bash
python3 scripts/run_video_demo.py --source data/input_videos/sample.mp4 --output data/outputs/body_demo.mp4 --zones-config configs/zones.cctv_platform.example.json --db data/outputs/body_analytics.db --detector-mode body
```

Comparison wrapper also reran body mode:

```bash
python3 scripts/run_comparison_demo.py --source data/input_videos/sample.mp4 --zones-config configs/zones.cctv_platform.example.json
```

- output path: `data/outputs/body_demo.mp4`
- DB path: `data/outputs/body_analytics.db`
- report path: `data/outputs/body_report.csv`
- latest session: `id=2`
- frames processed: 803
- unique tracked passengers: 108
- max zone occupancy: `main_platform_waiting=8`, `platform_movement_path=3`, `train_side_edge=3`
- line crossings: `TOWARD_TRAIN=4`, `AWAY_FROM_TRAIN=3`
- alerts in latest session: `WARNING=88`, `CRITICAL=17`, `NORMAL=72`

Observations: full-body mode completed successfully on the new CCTV-angle sample and produced visible platform-zone analytics, line crossings, and crowd alerts with the tuned demo thresholds.

## Head Detection Result

Original command run:

```bash
python3 scripts/run_video_demo.py --source data/input_videos/sample.mp4 --output data/outputs/head_demo.mp4 --zones-config configs/zones.cctv_platform.example.json --db data/outputs/head_analytics.db --detector-mode head --model models/fine_tuned/head_detector/weights/best.pt
```

Corrected head-mode command after visual inspection showed 640px inference was too small for CCTV-angle heads:

```bash
python3 scripts/run_video_demo.py --source data/input_videos/sample.mp4 --output data/outputs/head_demo.mp4 --zones-config configs/zones.cctv_platform.example.json --db data/outputs/head_analytics.db --detector-mode head --model models/fine_tuned/head_detector/weights/best.pt --confidence 0.15 --imgsz 1536
```

The comparison wrapper now also supports head-specific defaults:

```bash
python3 scripts/run_comparison_demo.py --source data/input_videos/sample.mp4 --zones-config configs/zones.cctv_platform.example.json
```

- output path: `data/outputs/head_demo.mp4`
- DB path: `data/outputs/head_analytics.db`
- report path: `data/outputs/head_report.csv`
- latest session: `id=1` in the regenerated head DB
- frames processed: 803
- unique tracked head IDs: 294
- max zone occupancy: `main_platform_waiting=17`, `platform_movement_path=6`, `train_side_edge=2`
- line crossings: `TOWARD_TRAIN=12`
- Colab-trained `best.pt` used: yes, `models/fine_tuned/head_detector/weights/best.pt`

Observations: head mode completed successfully with the Colab-trained weights. The initial 640px inference settings missed many small heads in the 1280x720 CCTV-angle clip. Running head mode at `imgsz=1536` and `confidence=0.15` produced many more actual head detections. A head-specific detection filter now removes train-front and track-region false positives while keeping visible platform heads. The high unique-ID number should still be treated carefully because small head boxes cause more tracker ID switches than full-body boxes.

## Web App Update

Updated `src/dashboard/streamlit_app.py` so the boss/demo view shows:

- current CCTV-angle platform sample as the default selected video
- previous sample backup as a selectable option when present
- original input video player
- full-body output video
- head output video
- side-by-side body/head output comparison
- body analytics and head analytics
- report previews/downloads
- a visible section titled `Why this CCTV-angle test matters`
- head detection commands using `--confidence 0.15 --imgsz 1536`
- CCTV-sample head detection filters that ignore train-front/track regions

What the boss will see: the dashboard now makes it clear that this retest uses a fixed/elevated platform-style sample rather than the earlier phone-like angle, and it presents body-vs-head outputs and analytics from the same video.

## Limitations

- The video is not necessarily Indian-specific.
- The clip is real public platform footage but not confirmed operational CCTV.
- Video quality, camera height, compression, and crowd behavior may still differ from real railway CCTV.
- Head detector behavior needs further validation and calibration on approved station footage.
- Final testing needs supervisor-provided or formally approved railway CCTV/platform footage.
- Real CCTV use must avoid storing personally identifying footage unless explicitly approved.

## Checks

Passed:

```bash
python3 -m compileall src scripts
pytest
python3 scripts/run_video_demo.py --help
python3 scripts/run_comparison_demo.py --help
```
