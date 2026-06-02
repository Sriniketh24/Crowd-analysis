# Senior Review Report

Prepared: 2026-06-02

## Verdict

The current version is **boss-demo-ready as an MVP** for showing passenger detection,
tracking, zone occupancy, line-crossing counts, crowded-zone alerts, SQLite logging,
CSV export, API reads, and dashboard visualization on the public Mumbai platform
sample video.

It is **not production-ready** and must not be presented as validated live Indian
Railway CCTV. It has not been tested on approved station RTSP feeds, has no measured
accuracy against ground truth, and has no fine-tuned Indian platform model.

## What Was Reviewed

- Required docs: `AGENTS.md`, `README.md`, `docs/PRODUCTION_RESEARCH_ROADMAP.md`,
  `docs/IMPLEMENTATION_CHANGE_PLAN.md`, `docs/REAL_DATA_SOURCE_REPORT.md`, and
  `docs/IMPLEMENTATION_SUMMARY.md`.
- Runtime/config: `configs/app.yaml`, `configs/cameras.yaml`,
  `configs/thresholds.yaml`, `configs/zones.example.json`,
  `configs/zones.indian_platform.example.json`, and `.env.example`.
- Detection/tracking: `src/vision/detector.py`, `src/vision/tracker.py`,
  `src/video/frame_processor.py`, and `scripts/run_video_demo.py`.
- Analytics: zone counting, line counting, crowd alerts, SQLite models/event logging,
  report export, and metrics helpers.
- Interfaces: FastAPI routes/schemas and Streamlit dashboard.
- Training workflow: frame extraction/training scripts and dataset config.
- Security/privacy: secrets, unsafe execution, RTSP credential handling, raw frame
  storage, output behavior, and accidental private paths in docs.

## Issues Found

- RTSP URLs containing `user:password@host` could be logged into `run_sessions.source`
  and then surfaced through reports/API/dashboard session views.
- `configs/cameras.yaml` exposed `privacy.store_output_video`, but
  `scripts/run_video_demo.py` ignored it and always wrote annotated MP4 output.
- Passing an explicit `--source` without `--camera-id` still inherited the first
  enabled camera config, causing ad-hoc runs to be logged as the demo camera.
- Passing an unknown `--camera-id` silently fell back instead of failing clearly.
- Selected threshold profiles were weaker than generic zone overrides because merge
  precedence was reversed.
- `.env.example` still pointed to `yolov8n.pt` and `sqlite:///data/analytics.db`,
  not the current YOLO11/demo output defaults.
- Dashboard/help text used credential-looking RTSP examples.
- Several docs had stale or over-strong claims: YOLOv8 as current model, old Roboflow
  sample as current demo, “counts every passenger,” RTSP as ready by just pasting a
  URL, and privacy wording that ignored annotated video output.
- One absolute private local path remained in docs.

## Issues Fixed

- Added `redact_source_uri()` and `infer_camera_id()` in `src/config.py`.
- Updated `scripts/run_video_demo.py` to redact logged source URLs, infer ad-hoc
  camera IDs from explicit sources, fail clearly for missing camera IDs, honor
  `store_output_video: false`, and apply camera threshold profiles correctly.
- Added regression coverage for RTSP credential redaction and camera ID inference.
- Updated `.env.example` to current YOLO11 and `data/outputs/analytics.db` defaults,
  with an empty `CAM1_RTSP_URL` placeholder.
- Removed credential-looking RTSP placeholder text from the dashboard.
- Updated README and docs to avoid production, fine-tuning, live-CCTV, accuracy, and
  privacy overclaims.
- Marked older QA/demo reports and the change plan as historical where they conflict
  with the current implementation.
- Removed the remaining absolute local path from docs.

## Verification Run

All commands were run from the project root.

| Command | Result |
|---|---|
| `python3 -m compileall src scripts` | Pass |
| `pytest` | Pass, 48 tests |
| `python3 scripts/run_video_demo.py --help` | Pass |
| `python3 scripts/export_report.py --help` | Pass |
| `python3 scripts/run_video_demo.py --source data/input_videos/sample.mp4 --output data/outputs/demo.mp4 --zones-config configs/zones.example.json --db data/outputs/analytics.db` | Pass, processed 600 frames |
| `python3 scripts/export_report.py --db data/outputs/analytics.db --out data/outputs/report.csv` | Pass, wrote combined CSV |

Latest verified run metadata: session `6`, camera `sample`, source
`data/input_videos/sample.mp4`, model `yolo11n.pt`, tracker `bytetrack`,
600 processed frames, 140 unique ephemeral track IDs observed. These are pipeline
outputs, not validated accuracy metrics.

## What Remains Incomplete

- No approved live Indian Railway CCTV feed has been tested.
- No ground-truth validation exists for precision/recall, count MAE/RMSE, ID switches,
  or alert correctness on real station CCTV.
- No Indian platform fine-tuning has been completed; training is intentionally blocked
  until YOLO-format labels exist.
- Multi-camera orchestration is not implemented. `configs/cameras.yaml` can describe
  cameras, but there is no worker supervisor for 4-20 concurrent RTSP streams.
- RTSP support is initial only. OpenCV input, basic retry counters, FPS cap, and stride
  exist, but there is no proven long-running reconnect/backoff SLA, watchdog, or stream
  health service.
- API processing is synchronous batch-style and not a production background job queue.
- API/dashboard have no auth, RBAC, audit logging, or network exposure controls.
- Zone and line configs are examples. Real platform cameras require per-camera
  calibration from frames.
- Annotated MP4 output is acceptable for public demo video, but private CCTV output
  must remain disabled unless explicitly approved.
- Dense-crowd handling is still detection/tracking only; no density-estimation or
  head-detection fallback is implemented.

## Boss Demo Readiness

Use this version for a boss demo if the framing is: **working MVP on a public Indian
platform sample video, showing the full analytics pipeline and the path to live CCTV**.

Do not claim:

- production readiness,
- completed fine-tuning,
- validated live Indian Railway CCTV,
- measured passenger-counting accuracy,
- full multi-camera deployment support.

## Needed For Real Indian Railway Deployment

- Written approval for analytics on live station CCTV and a privacy/retention policy.
- Representative approved clips from multiple station cameras: platform edge,
  stair/footbridge, gate, waiting area, peak/off-peak, day/night, low-light, and rain.
- Manual ground truth for per-frame counts, zone occupancy, line crossings, and sample
  track continuity.
- A labeled YOLO-format Indian platform dataset and held-out validation/test split.
- Evaluation report comparing pretrained `yolo11n.pt`/`yolo11s.pt` and any fine-tuned
  model on accuracy and FPS.
- GPU/edge deployment profile, likely `yolo11s.pt` first, with CUDA/TensorRT evaluation.
- Long-running RTSP worker with reconnect, health status, frame dropping policy,
  latency metrics, and restart supervision.
- Multi-camera worker orchestration and camera-aware API/dashboard live views.
- Auth/RBAC, secure RTSP secret handling, logs without credentials, and no raw CCTV
  storage by default.
