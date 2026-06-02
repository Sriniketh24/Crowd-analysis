# Implementation Change Plan

> **Status note (2026-06-02 senior review):** This document is a historical
> change plan written before the Agent 4 implementation pass. Several items
> below have since been implemented or partially implemented. Treat
> `docs/IMPLEMENTATION_SUMMARY.md` and `docs/SENIOR_REVIEW_REPORT.md` as the
> current status sources.

## Pre-Agent-4 Baseline System Summary

Before the Indian Railway CCTV upgrade pass, the codebase was a working single-stream MVP for passenger/person analytics on sample video. The primary demo path was `scripts/run_video_demo.py`, which wired:

1. `src/video/stream_reader.py` for OpenCV frame ingestion.
2. `src/vision/tracker.py` or `src/vision/detector.py` for YOLO inference.
3. `src/vision/zone_manager.py`, `src/vision/line_counter.py`, and `src/vision/crowd_analyzer.py` for analytics.
4. `src/vision/annotator.py` and `src/video/video_writer.py` for annotated MP4 output.
5. `src/analytics/event_logger.py` and `src/analytics/database.py` for SQLite logging.

What worked in that baseline:

- YOLO person detection works through Ultralytics.
- Current configured model is `yolov8n.pt` in `configs/app.yaml`, `README.md`, `scripts/run_video_demo.py`, and `src/api/routes.py`.
- `src/vision/detector.py` and `src/vision/tracker.py` have fallback candidates `("yolo11n.pt", "yolov8n.pt")` when no explicit weights are supplied.
- Current tracker is Ultralytics tracking with `bytetrack` by default from `configs/app.yaml`; `botsort` is supported by `src/vision/tracker.py`.
- Video file input works through `StreamReader` and is validated by existing `data/input_videos/sample.mp4` and `data/outputs/demo.mp4`.
- Webcam input is supported by numeric source parsing in `StreamReader` and `scripts/run_video_demo.py`.
- RTSP/HTTP URL input is accepted by `StreamReader.source_exists()` and passed to OpenCV, but there is no reconnect, backoff, health state, FPS limiting, or stream timeout handling.
- Zone counting works with bottom-center point-in-polygon logic in `src/vision/zone_manager.py`.
- Line counting works with side-change logic in `src/vision/line_counter.py`.
- Crowd alerts work as immediate `NORMAL`, `WARNING`, `CRITICAL` threshold checks in `src/vision/crowd_analyzer.py`.
- SQLite logging works. Existing `data/outputs/analytics.db` has `run_sessions`, `frames_processed`, `zone_occupancy`, `line_crossing_events`, and `crowd_alerts`.
- CSV export works through `scripts/export_report.py` and `src/analytics/report_generator.py`.
- FastAPI works for `/health`, `/metrics/latest`, `/metrics/zones`, `/alerts`, `/sessions`, and synchronous `/process-video`.
- Streamlit works as a local dashboard that shells out to `scripts/run_video_demo.py`, then reads SQLite.
- Tests pass: `pytest` currently reports 40 passing tests.
- `python3 -m compileall src scripts` currently passes.
- `python3 scripts/run_video_demo.py --help` and `python3 scripts/export_report.py --help` currently pass.
- `python3 scripts/benchmark_fps.py --help` exits successfully but only prints `benchmark_fps placeholder`; it does not provide real help or benchmarking.

Hardcoded or weakly configurable behavior:

- `scripts/run_video_demo.py` defaults to `yolov8n.pt` even though production roadmap recommends YOLO11.
- `src/api/routes.py` hardcodes `model="yolov8n.pt"`, `confidence=0.35`, `snapshot_interval=5.0`, and only accepts local video paths for `/process-video`.
- `scripts/run_video_demo.py` loads only `configs/app.yaml` and `configs/thresholds.yaml`; it does not consume `configs/cameras.yaml`.
- `configs/cameras.yaml` points to `data/input_videos/sample_station.mp4`, which does not currently exist.
- `configs/zones.example.json` is calibrated for a 1920x1080 public concourse sample, not an Indian railway platform CCTV view.
- `src/video/video_writer.py` defaults to `fps=20.0`, not source FPS.
- `src/vision/line_counter.py` has no stable line ID field; downstream code uses line name as `line_id`.
- `src/analytics/event_logger.py` logs line crossing deltas without `track_id`.
- `src/vision/crowd_analyzer.py` does not implement dwell time or clear hysteresis, despite `configs/thresholds.yaml` containing dwell/clear settings.
- `src/video/frame_processor.py` accepts `resize_width`, but the CLI does not expose it and zone/line coordinates are not rescaled.
- API and dashboard are DB/demo oriented, not attached to a long-running live stream worker.

Missing for production-style CCTV use:

- Indian railway platform video validation.
- Labeled Indian railway platform dataset.
- Fine-tuned model support and training/evaluation scripts.
- RTSP reconnect and per-camera health reporting.
- Multi-camera orchestration from `configs/cameras.yaml`.
- Per-camera zone templates for platform edge, footbridge/stairs, concourse, gate, waiting area, and boarding area.
- Secret-safe RTSP credential handling.
- FPS caps, frame skipping, source FPS handling, and latency metrics.
- GPU/CUDA configuration and model-size selection by deployment profile.
- Stream status, last frame time, and dropped/reconnected frame metrics.
- Live API endpoints by camera ID.
- Dashboard flow for selecting/uploading sample video, running analysis, comparing original vs annotated output, and downloading CSV.
- Explicit privacy controls for avoiding raw CCTV storage.
- Deployment runbooks for local, Docker, and GPU server/edge pilot.

## Required Changes for Indian Railway CCTV Use

### Model

- Change the default model from `yolov8n.pt` to `yolo11s.pt` for pilot evaluation.
- Keep `yolo11n.pt` as the CPU/demo fallback.
- Keep `yolov8n.pt` as a legacy fallback for offline environments where it already exists.
- Add config fields for `imgsz`, `half`, `device`, `model_profile`, and `weights`.
- Add a model resolution policy:
  - Use explicit path if supplied.
  - Use `models/fine_tuned/best.pt` if enabled and present.
  - Use `models/pretrained/yolo11s.pt` or model name `yolo11s.pt`.
  - Fall back to `yolo11n.pt` for CPU.
- Do not claim fine-tuning unless labeled images and training outputs exist.

### Tracker

- Keep ByteTrack as the default tracker for initial static-CCTV pilot.
- Keep BoT-SORT selectable.
- Add tracker parameters from config instead of only `tracker_type`.
- Add a testable wrapper interface where detection can be mocked without loading YOLO.
- Add optional ID-switch diagnostics for future evaluation clips.

### RTSP/live stream handling

- Extend `StreamReader` to support source types: `file`, `webcam`, `rtsp`, `http`.
- Add reconnect loop with max retries, exponential backoff, and camera health state.
- Add FPS limiting and frame skipping.
- Add read timeout handling.
- Add graceful disconnect behavior: log stream health event, keep process alive if configured, and return a clear API/dashboard state.
- Track `camera_id`, source type, frame index, source timestamp, dropped frames, reconnect count, and last successful frame time.

### Zone templates

- Replace the current generic sample zone file with Indian railway platform examples.
- Add template zone groups:
  - `platform_waiting_area`
  - `platform_edge_yellow_line`
  - `stairs_or_footbridge_entry`
  - `ticket_gate_or_entry`
  - `boarding_area`
  - `concourse_flow_area`
- Keep `configs/zones.example.json` demo-safe and calibrated to existing `data/input_videos/sample.mp4`.
- Add additional example config files rather than breaking the current demo.

### Configuration

- Make `configs/cameras.yaml` the source of truth for camera source, resolution, lines, zones config path, and thresholds.
- Keep `configs/app.yaml` for global model/tracker/runtime/API/dashboard settings.
- Keep `configs/thresholds.yaml` for default and per-zone thresholds.
- Validate all configs with clear errors.
- Avoid storing RTSP passwords in committed YAML. Use environment variables such as `${CAM1_RTSP_URL}`.

### Database

- Add camera registry and stream health tables.
- Add model/tracker metadata to each run session.
- Add line crossing `track_id` when available.
- Add direction labels and stable line IDs.
- Add alert lifecycle fields such as `started_at`, `ended_at`, `duration_seconds`, or keep transition rows but make lifecycle semantics explicit.
- Add optional report summary tables or query functions for daily/hourly counts.
- Keep storing analytics only, not footage or identifying attributes.

### Dashboard

- Redesign as a clear demo/operator flow:
  - Choose sample video or configured camera.
  - Click `Run Analysis`.
  - Show original video.
  - Show annotated processed video.
  - Show passenger counts.
  - Show crowded zones.
  - Show line crossing counts.
  - Show plain-English explanation.
  - Show CSV/report download.
- Add live stream status when source is RTSP/webcam.
- Avoid requiring users to know internal DB paths for the main demo flow.

### API

- Add camera-aware endpoints:
  - `GET /cameras`
  - `GET /cameras/{camera_id}/health`
  - `GET /cameras/{camera_id}/latest`
  - `GET /cameras/{camera_id}/counts`
  - `GET /cameras/{camera_id}/occupancy`
  - `GET /cameras/{camera_id}/alerts`
  - `POST /analysis/run`
  - `GET /reports/csv`
- Keep `/health` and existing endpoints backward compatible or documented as legacy.
- Do not run long video jobs synchronously in a request for production mode; use a background worker or CLI-driven batch path.

### Reporting

- Keep current CSV export.
- Add summary exports by session, camera, line, zone, and time window.
- Add report metadata: model, tracker, source type, run start/end, total frames, processed FPS, alert counts.
- Add missing-file and empty-DB behavior tests.

### Testing

- Expand tests for config loading, source parsing, reconnect behavior, CSV filtering, API camera endpoints, dashboard fallback helpers, and model/tracker wrappers.
- Keep CV/model-heavy tests mocked or optional.
- Add a small fixture video only if it is public/sample and small enough for the repo.

### Documentation

- Update README commands and configuration docs after implementation.
- Add a dataset/labeling workflow doc.
- Add a production privacy note specific to Indian railway CCTV.
- Add a deployment runbook.
- Document that pretrained counts are estimates until validated on approved footage.

### Deployment

- Keep local demo and Docker paths.
- Add deployment profiles:
  - CPU demo: `yolo11n.pt`, low FPS.
  - GPU pilot: `yolo11s.pt`, CUDA, multi-camera.
  - Accuracy evaluation: `yolo11m.pt`, offline validation.
- Add `.env.example` for RTSP URLs and DB path.
- Add health checks for API and stream workers.

## File-by-File Change Plan

### `configs/app.yaml`

- What needs to change:
  - Set default `model.weights` to `yolo11s.pt`.
  - Add `model.imgsz`, `model.half`, and `model.profile`.
  - Add `runtime.frame_skip`, `runtime.max_fps`, `runtime.reconnect_backoff_seconds`, `runtime.max_reconnect_attempts`, and `runtime.stream_read_timeout_seconds`.
  - Add `privacy.store_raw_frames: false`.
  - Add report/output defaults under `outputs`.
- Why:
  - Production CCTV needs explicit performance and privacy behavior.
- Acceptance criteria:
  - Config loads without breaking current scripts.
  - Default model can be overridden by CLI.
  - Runtime values are consumed by `scripts/run_video_demo.py` or the new runner.

### `configs/cameras.yaml`

- What needs to change:
  - Replace missing `data/input_videos/sample_station.mp4` with existing `data/input_videos/sample.mp4` for demo camera.
  - Add fields: `source_type`, `enabled`, `zones_config`, `threshold_profile`, `fps_limit`, `frame_stride`, `privacy.store_output_video`, and `description`.
  - Add example RTSP camera using environment variable placeholder, not real credentials.
  - Add stable line IDs and zone IDs per camera.
- Why:
  - The file exists but is not production-ready and points to a nonexistent sample.
- Acceptance criteria:
  - A config loader can list enabled cameras.
  - Demo camera works out of the box.
  - RTSP example contains no secrets.

### `configs/thresholds.yaml`

- What needs to change:
  - Make warning, critical, dwell, and clear thresholds explicit for all example zones.
  - Add platform-specific profiles: `platform_edge`, `stairs`, `waiting_area`, `concourse`.
  - Use consistent field names: `warning_count`, `critical_count`, `dwell_seconds`, `clear_below_count`.
- Why:
  - Current code ignores dwell and clear values, and default field names are inconsistent.
- Acceptance criteria:
  - Threshold parser validates all profiles.
  - Crowd analyzer tests cover dwell and hysteresis.

### `configs/zones.example.json`

- What needs to change:
  - Keep current sample-compatible zones for `data/input_videos/sample.mp4`.
  - Add `frame_width`, `frame_height`, `camera_id`, and stable line `id` fields.
  - Add clear comments via JSON-safe `_note` fields explaining calibration.
  - Add `direction_hint` or direction documentation for each line.
- Why:
  - Coordinates must be tied to a reference resolution and camera.
- Acceptance criteria:
  - Existing demo still works.
  - Zone/line loader rejects malformed geometry with clear errors.

### `configs/zones.indian_platform.example.json`

- What needs to change:
  - Add this new file.
  - Include template polygons for platform waiting area, platform edge, footbridge/stairs, and boarding area using clearly fake/example coordinates.
- Why:
  - Indian railway use needs platform-specific templates without breaking current demo calibration.
- Acceptance criteria:
  - File is documented as an example requiring calibration.
  - Loader can parse it.

### `scripts/run_video_demo.py`

- What needs to change:
  - Default `--model` should come from `configs/app.yaml`, not hardcoded `yolov8n.pt`.
  - Add `--config`, `--camera-id`, and `--camera-config` support that can load `configs/cameras.yaml`.
  - Add `--source-type`, `--max-fps`, `--frame-stride`, `--resize-width`, and `--device` overrides.
  - Pass runtime options into `StreamReader`, `FrameProcessor`, `VideoWriter`, and `AnalyticsLogger`.
  - Use stable line IDs, not line names, when building line managers.
  - Print model, tracker, source type, FPS, reconnect count, and DB path at the end.
- Why:
  - This is the main demo runner and must become config-driven for CCTV use.
- Acceptance criteria:
  - Existing command in README still works.
  - New command can run by camera ID from `configs/cameras.yaml`.
  - `--help` documents all production-relevant options.

### `scripts/benchmark_fps.py`

- What needs to change:
  - Replace placeholder with real CLI.
  - Arguments: `--source`, `--model`, `--device`, `--frames`, `--confidence`, `--tracker`, `--imgsz`, `--frame-stride`.
  - Report decode FPS, inference FPS, end-to-end FPS, average latency, and processed frame count.
  - Support no-DB benchmarking.
- Why:
  - Production planning requires measured throughput on target hardware.
- Acceptance criteria:
  - `python3 scripts/benchmark_fps.py --help` shows argparse help.
  - Benchmark can run on `data/input_videos/sample.mp4`.
  - It exits gracefully if the model cannot load.

### `scripts/export_report.py`

- What needs to change:
  - Add `--from`, `--to`, `--format csv`, and `--summary` options.
  - Add a session summary export option.
  - Include stream/model metadata once database schema supports it.
- Why:
  - Operations teams need time-windowed reports, not only raw table dumps.
- Acceptance criteria:
  - Existing `--db --out` behavior remains.
  - Filtered CSV exports are tested.

### `scripts/extract_sample_frames.py`

- What needs to change:
  - Add support for `data/indian_railway_videos/`.
  - Add `--every-seconds`, `--start`, `--end`, and `--camera-id`.
  - Write an index CSV with frame path, source video, frame index, timestamp, and intended split.
- Why:
  - Fine-tuning requires frame extraction before labeling.
- Acceptance criteria:
  - Works on existing `data/input_videos/sample.mp4`.
  - Does not imply labels exist.

### `scripts/create_zones_from_frame.py`

- What needs to change:
  - Replace placeholder with a calibration helper.
  - At minimum, load an image, let user click polygon/line points through OpenCV, and write JSON.
  - Include non-interactive validation mode for CI.
- Why:
  - Every CCTV view needs calibrated zones and lines.
- Acceptance criteria:
  - `--help` exists.
  - Can validate an existing zones JSON without GUI.

### `scripts/train_yolo.py`

- What needs to change:
  - Add this new script only after dataset structure is defined.
  - Wrap Ultralytics training with args for `--data`, `--weights`, `--epochs`, `--imgsz`, `--device`, `--project`, and `--name`.
  - Refuse to run if dataset YAML or label folders are missing.
- Why:
  - Fine-tuning must be repeatable and must not pretend unlabeled videos are training data.
- Acceptance criteria:
  - `--help` works.
  - Missing labels produce a clear error.

### `scripts/evaluate_counts.py`

- What needs to change:
  - Add this new script.
  - Compare auto line counts and zone snapshots with a manual ground-truth CSV.
  - Output MAE, percent error, alert precision/recall, and optional ID-switch notes.
- Why:
  - Production claims require measured accuracy.
- Acceptance criteria:
  - Runs on synthetic/manual fixture CSV in tests.

### `src/config/schema.py`

- What needs to change:
  - Add this new module with typed config models.
  - Models should cover app, model, tracker, runtime, camera, line, zone, thresholds, persistence, and outputs.
- Why:
  - Current config parsing is scattered and weakly validated.
- Acceptance criteria:
  - Invalid geometry, missing IDs, and bad thresholds raise clear exceptions.

### `src/config/loader.py`

- What needs to change:
  - Add this new module.
  - Load YAML/JSON configs, expand environment variables, resolve paths relative to project root, and validate schema.
- Why:
  - Scripts/API/dashboard should share one config path.
- Acceptance criteria:
  - Unit tests cover app config, cameras config, zones JSON, thresholds YAML, and env-var source expansion.

### `src/vision/detector.py`

- What needs to change:
  - Update default candidates to prefer `yolo11s.pt`, then `yolo11n.pt`, then `yolov8n.pt`.
  - Add `imgsz` and `half` options.
  - Store `weights_path` and resolved model name for logging.
  - Add a simple fake/mock detector seam for tests without Ultralytics.
- Why:
  - Production model selection must be explicit and testable.
- Acceptance criteria:
  - Existing detector tests or compile still pass.
  - Model metadata can be written to `RunSession`.

### `src/vision/tracker.py`

- What needs to change:
  - Accept tracker parameters from config.
  - Add stable tracker metadata.
  - Keep ByteTrack default and BoT-SORT support.
  - Return normalized tracks with `track_id=None` instead of `-1` when unavailable, or document `-1` clearly and filter from counting.
- Why:
  - Counting and logging should not treat missing IDs as real people.
- Acceptance criteria:
  - Tracker wrapper can be instantiated from config.
  - Unit tests verify unsupported tracker rejection and missing-ID behavior.

### `src/vision/zone_manager.py`

- What needs to change:
  - Add optional zone area metadata and density calculation support.
  - Validate coordinates against configured frame size.
  - Support zone IDs and names without requiring warning/critical thresholds inside the zone JSON.
  - Add resolution-scaling helper if processing frames are resized.
- Why:
  - Thresholds should live in thresholds config, and CCTV resolutions vary.
- Acceptance criteria:
  - Existing zone tests pass.
  - New tests cover boundary points, invalid polygons, resized frame coordinate scaling, and density.

### `src/vision/line_counter.py`

- What needs to change:
  - Add `id` to `LineConfig`.
  - Return crossing events with `line_id`, `direction`, `track_id`, and `frame_index`, not only cumulative counts.
  - Keep cumulative counts for dashboard display.
  - Add direction documentation and tests for both crossing directions.
- Why:
  - SQLite and reports need stable event records.
- Acceptance criteria:
  - Existing line tests pass or are intentionally updated.
  - Event logger no longer infers events only from count deltas when direct crossing events are available.

### `src/vision/crowd_analyzer.py`

- What needs to change:
  - Implement dwell time before raising warning/critical alerts.
  - Implement clear hysteresis using `clear_below_count`.
  - Return alert transition events in addition to current levels.
  - Include thresholds per zone from config.
- Why:
  - Production alerts should not flicker due to one-frame detection noise.
- Acceptance criteria:
  - Tests cover NORMAL to WARNING, WARNING to CRITICAL, delayed dwell, and clearing below threshold.

### `src/vision/annotator.py`

- What needs to change:
  - Draw stable line IDs/names and direction arrows.
  - Improve overlay layout for dense CCTV frames.
  - Add source/camera ID, alert status, FPS, and reconnect indicator overlays.
  - Ensure overlays do not hide platform-edge zones.
- Why:
  - Demo and operator review require understandable annotated video.
- Acceptance criteria:
  - Annotated output still writes for sample video.
  - Unit-level smoke test can call annotation with empty detections/zones.

### `src/video/stream_reader.py`

- What needs to change:
  - Add `StreamReaderConfig`.
  - Support source type parsing for file, webcam, RTSP, and HTTP.
  - Add reconnect/backoff for live sources.
  - Add FPS limiting, frame stride, max frames, and timeout handling.
  - Emit stream health metadata.
  - Preserve source video FPS and dimensions when available.
- Why:
  - This is the main blocker for live CCTV use.
- Acceptance criteria:
  - Tests cover source parsing, local missing file, webcam index parsing, RTSP URL classification, frame stride, and reconnect behavior with mocked `cv2.VideoCapture`.

### `src/video/frame_processor.py`

- What needs to change:
  - Accept model runtime config and camera geometry config.
  - Correctly scale zones/lines if frames are resized before inference.
  - Return processing latency and optional crossing/alert event lists.
  - Avoid running both detector and tracker models separately; current tracking path uses only tracker model, which is acceptable but should be documented.
- Why:
  - Production telemetry and correct geometry depend on processor output.
- Acceptance criteria:
  - Tests cover no detections, tracking disabled, resized frame geometry, and event output.

### `src/video/video_writer.py`

- What needs to change:
  - Use source FPS by default when available.
  - Add `enabled` option so live CCTV can avoid storing annotated footage unless approved.
  - Add codec/fps validation and clear errors.
- Why:
  - Privacy rules discourage storing CCTV footage by default.
- Acceptance criteria:
  - Existing MP4 output works.
  - Writer disabled mode performs no file write and does not crash.

### `src/analytics/database.py`

- What needs to change:
  - Add `Camera` table.
  - Add `StreamHealthEvent` table.
  - Add model/tracker fields to `RunSession`: `model_weights`, `tracker_type`, `device`, `source_type`, `config_hash`.
  - Add nullable `track_id` to `LineCrossingEvent`.
  - Consider `processed_fps` and `average_latency_ms` summary fields.
- Why:
  - Reports must identify which model, camera, and stream state produced each count.
- Acceptance criteria:
  - Migrations are not present, so schema changes must be compatible with fresh DBs and tests.
  - Existing tests update for new nullable fields.

### `src/analytics/event_logger.py`

- What needs to change:
  - Log direct line crossing events when provided by `LineManager`.
  - Log stream health events.
  - Log model/tracker/session metadata.
  - Support batch commits or configurable frame logging interval to reduce SQLite load.
  - Do not log a `NORMAL` alert as an alert unless transition semantics require it.
- Why:
  - Current every-frame frame logging and delta-derived crossing events are enough for MVP but weak for production.
- Acceptance criteria:
  - Tests verify line event `track_id`, alert transitions, stream events, and snapshot interval behavior.

### `src/analytics/report_generator.py`

- What needs to change:
  - Add summary generation functions for counts by camera/line/direction and occupancy by zone/time window.
  - Add date range filters.
  - Include new tables after schema expansion.
- Why:
  - CSV reports should be useful to station safety and passenger-flow review.
- Acceptance criteria:
  - Existing raw table export remains backward compatible.
  - Summary export has tests with deterministic rows.

### `src/analytics/metrics.py`

- What needs to change:
  - Add helpers for rolling flow rate, average occupancy, max occupancy, alert duration, processed FPS, and latency percentiles.
- Why:
  - Dashboard/API should not duplicate metric calculations.
- Acceptance criteria:
  - Pure unit tests cover each metric helper.

### `src/api/app.py`

- What needs to change:
  - Keep app factory.
  - Add startup config loading if API is used as a live service.
  - Add CORS only if dashboard/API separation requires it.
- Why:
  - API needs a camera-aware runtime context.
- Acceptance criteria:
  - `/health` still returns 200.

### `src/api/routes.py`

- What needs to change:
  - Replace hardcoded processing defaults with config-driven defaults.
  - Add camera endpoints and report endpoint.
  - Add source validation for RTSP/webcam if background processing is supported.
  - Avoid synchronous long-running `/process-video` for live mode; keep a batch endpoint for local video demos.
  - Add query parameters for `camera_id`, `from`, `to`, `line_id`, `zone_id`, `session_id`.
- Why:
  - Current API is useful for demo DB reads but not for CCTV operations.
- Acceptance criteria:
  - Existing tests still pass or are updated for documented response behavior.
  - New tests cover empty DB, invalid camera, and populated session queries.

### `src/api/schemas.py`

- What needs to change:
  - Add schemas for camera config, stream health, run request/response, report metadata, line events, zone history, and alert lifecycle.
- Why:
  - Camera-aware API responses need typed models.
- Acceptance criteria:
  - Pydantic validation catches invalid requests.

### `src/dashboard/streamlit_app.py`

- What needs to change:
  - Remove reliance on users typing DB/output paths for normal demo.
  - Add a first-screen workflow:
    - Select sample video or camera.
    - Click `Run Analysis`.
    - Show original video.
    - Show annotated processed video.
    - Show counts, crowded zones, line crossing totals, and plain-English summary.
    - Provide report download.
  - Add RTSP/webcam status panel.
  - Add safe fallback when no video, no DB, or no model exists.
  - Keep subprocess batch mode for demos unless a background worker is implemented.
- Why:
  - The dashboard should be understandable to railway stakeholders and demo users.
- Acceptance criteria:
  - Dashboard starts with no existing DB.
  - Dashboard can process `data/input_videos/sample.mp4`.
  - Missing model/source errors are shown in UI.

### `src/main.py`

- What needs to change:
  - Replace placeholder with a real CLI entrypoint or remove it from documented paths.
  - Support `run`, `api`, `dashboard`, and possibly `worker` subcommands.
- Why:
  - A production-style project needs a clear top-level command path.
- Acceptance criteria:
  - `python3 -m src.main --help` shows useful commands.

### `tests/`

- What needs to change:
  - Add or update tests:
    - `tests/test_config_loader.py`
    - `tests/test_detector.py`
    - `tests/test_tracker.py`
    - `tests/test_stream_reader.py`
    - `tests/test_frame_processor.py`
    - `tests/test_report_generator.py`
    - `tests/test_api.py`
    - `tests/test_dashboard_helpers.py`
    - Existing `test_zone_manager.py`, `test_line_counter.py`, `test_crowd_analyzer.py`, and `test_analytics_db.py`.
- Why:
  - Production CCTV changes touch config, streaming, persistence, and reporting.
- Acceptance criteria:
  - Tests use mocks for YOLO/OpenCV live streams.
  - Tests do not require private CCTV, GPU, or network.

### `README.md`

- What needs to change:
  - Update default model to YOLO11 plan.
  - Document camera config flow.
  - Document RTSP limitations and privacy stance.
  - Document new benchmark, calibration, frame extraction, report, API, and dashboard commands.
- Why:
  - Users must not confuse demo readiness with production validation.
- Acceptance criteria:
  - All documented commands are executable.

### `docs/PRODUCTION_RESEARCH_ROADMAP.md`

- What needs to change:
  - Keep current warnings.
  - Update implementation status after Agent 4 completes changes.
- Why:
  - Roadmap should reflect actual implementation state.
- Acceptance criteria:
  - No claims of Indian railway validation unless measured.

### `docs/SYSTEM_ARCHITECTURE.md`

- What needs to change:
  - Align architecture with actual package names (`src/vision`, `src/video`, `src/analytics`) or explicitly propose a migration.
  - Add live-stream worker and config-loader architecture.
- Why:
  - Current architecture doc describes planned packages that differ from implemented packages.
- Acceptance criteria:
  - New agents can follow the doc without guessing module names.

### `docs/DATASET_AND_LABELING.md`

- What needs to change:
  - Add this new doc.
  - Explain approved footage intake, frame extraction, labeling with CVAT/Label Studio/Roboflow, YOLO export, split strategy, and privacy rules.
- Why:
  - Fine-tuning cannot happen from unlabeled videos alone.
- Acceptance criteria:
  - Document states that unlabeled videos only support demo, calibration, and frame extraction.

### `docs/DEPLOYMENT.md`

- What needs to change:
  - Add this new doc.
  - Cover local CPU demo, Docker demo, GPU pilot, environment variables, RTSP secrets, DB path, and health checks.
- Why:
  - CCTV use needs reproducible run instructions.
- Acceptance criteria:
  - Contains exact commands and expected ports.

## Model and Fine-Tuning Plan

Current model:

- Config and scripts default to `yolov8n.pt`.
- A local `yolov8n.pt` exists in the project root.
- Detector/tracker fallback candidates include `yolo11n.pt` then `yolov8n.pt` when no explicit model is passed.

Recommended model:

- Use `yolo11s.pt` as the pilot default when GPU or sufficient CPU budget exists.
- Use `yolo11n.pt` for CPU-only demos and stress tests.
- Keep `yolov8n.pt` as a backward-compatible fallback.
- Evaluate `yolo11m.pt` only for accuracy-focused offline validation or dedicated GPU cameras.

Support pretrained model now:

- Add model config fields in `configs/app.yaml`.
- Update `Detector` and `Tracker` to accept `imgsz`, `half`, `device`, and metadata.
- Keep pretrained COCO person class (`person_class_id: 0`) as the default.
- Do not require a custom dataset for demo use.

Support fine-tuned model later:

- Add `models/fine_tuned/` support in model resolution.
- Add `scripts/train_yolo.py`.
- Add `scripts/evaluate_counts.py`.
- Add `configs/training/indian_platform_yolo.yaml` after data folders exist.
- Add documentation for labeling and train/val/test split.

Training files/scripts/configs to add:

- `scripts/train_yolo.py`
- `scripts/evaluate_detector.py`
- `scripts/evaluate_counts.py`
- `configs/training/indian_platform_yolo.yaml`
- `docs/DATASET_AND_LABELING.md`
- Optional `src/training/dataset_validator.py`

Data folder structure for training:

```text
data/datasets/indian_platform_yolo/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  data.yaml
```

If no labeled dataset exists:

- Do not run fine-tuning.
- Use pretrained YOLO only.
- Use unlabeled videos for:
  - pipeline demo,
  - extracting sample frames,
  - zone/line calibration,
  - manual count evaluation,
  - identifying failure cases.
- Agent 4 must extract frames and prepare a labeling workflow if only unlabeled videos are found.
- Agent 4 must not create fake labels or claim training was completed.

## Data Folder Plan

Required folder structure:

```text
data/
  input_videos/
    sample.mp4
  indian_railway_videos/
    README.md
  sample_frames/
  annotations/
    README.md
  datasets/
    indian_platform_yolo/
      images/
        train/
        val/
        test/
      labels/
        train/
        val/
        test/
      data.yaml
  outputs/
    demo.mp4
    analytics.db
    report.csv
models/
  pretrained/
    README.md
  fine_tuned/
    README.md
```

Folder rules:

- `data/input_videos/` is for public/sample demo videos.
- `data/indian_railway_videos/` is for approved railway footage only; do not commit private CCTV.
- `data/sample_frames/` is for extracted calibration/labeling frames.
- `data/annotations/` is for exported labels or annotation project metadata.
- `data/datasets/` is for YOLO-ready train/val/test data.
- `data/outputs/` is for generated annotated videos, SQLite DBs, reports, and metrics.
- `models/pretrained/` is for downloaded pretrained weights if stored locally.
- `models/fine_tuned/` is for actual trained weights, never placeholder claims.

## RTSP/Live CCTV Plan

Source type handling:

- `file`: validate local path exists; stop at EOF.
- `webcam`: parse numeric index; continue until stopped or read fails.
- `rtsp`: treat as live; support reconnect/backoff.
- `http`: support HTTP video streams if OpenCV can open them.

Reconnect logic:

- On failed read from live source, mark camera health as `DEGRADED`.
- Release capture.
- Sleep with configurable backoff.
- Retry up to `max_reconnect_attempts` or indefinitely if configured.
- On reconnect success, mark camera health as `RUNNING`.
- On final failure, mark camera health as `DISCONNECTED`, log event, and exit gracefully for that camera.

Frame skipping:

- `frame_stride`: process every Nth decoded frame.
- `max_fps`: throttle processing for live sources.
- Store decoded frame count and processed frame count separately.

FPS limiting:

- Use monotonic time to enforce max processing FPS.
- Report actual end-to-end FPS in logs and DB run metadata.

Camera ID tracking:

- Every frame, event, alert, and report row must include `camera_id`.
- Derive camera ID from config, not from source basename, when camera config is used.

Per-camera configs:

- Each camera entry should define source, source type, resolution, zone config, threshold profile, frame stride, FPS cap, and enabled flag.

Graceful failure:

- Missing file returns clear message.
- Invalid RTSP URL marks camera disconnected and does not crash API/dashboard.
- Stream disconnect logs a stream health row and exposes health through API/dashboard.

## Dashboard Upgrade Plan

The dashboard should present a clear local web app/demo flow:

1. Show a source selector:
   - sample video,
   - configured camera,
   - manual local path,
   - webcam index,
   - RTSP/HTTP URL.
2. Show selected source details and zone config.
3. Provide a primary `Run Analysis` action.
4. Show the original video when source is a local video file.
5. Run the analysis with visible progress and clear errors.
6. Show the annotated processed video.
7. Show passenger/person counts:
   - latest detections,
   - total line IN/OUT counts,
   - per-zone occupancy,
   - warning/critical alert count.
8. Show crowded zones with status and plain-English explanation, for example:
   - `Central Crossing Area is CRITICAL because 14 people were detected and the configured critical threshold is 10.`
9. Show line crossing counts by line and direction.
10. Show report download:
   - combined CSV,
   - optional table-specific CSV,
   - summary CSV.
11. Show limitations:
   - pretrained model estimates,
   - no Indian railway validation unless actual validation data is loaded,
   - no raw CCTV storage by default.

## Testing Plan

Required tests:

- Detection wrapper:
  - model resolution order,
  - person-class filtering,
  - missing Ultralytics error path,
  - mocked model output normalization.
- Tracker wrapper:
  - ByteTrack default,
  - BoT-SORT selection,
  - unsupported tracker rejection,
  - missing track ID behavior.
- RTSP/source parsing:
  - file path,
  - missing file,
  - webcam index,
  - RTSP URL,
  - HTTP URL,
  - env-var source expansion.
- Stream reader:
  - frame stride,
  - max frame limit,
  - live reconnect with mocked capture,
  - graceful disconnect state.
- Zone logic:
  - invalid polygon,
  - boundary inclusion,
  - resized frame coordinate scaling,
  - density if area is configured.
- Line crossing:
  - both directions,
  - no duplicate count,
  - event payload includes `line_id`, `direction`, `track_id`, `frame_index`.
- Crowd alerts:
  - warning,
  - critical,
  - dwell delay,
  - clear hysteresis,
  - no flicker on one-frame spikes.
- Database logging:
  - session metadata,
  - frame intervals,
  - direct line events,
  - alert transitions,
  - stream health events.
- CSV export:
  - all tables,
  - single table,
  - camera filter,
  - session filter,
  - time range filter,
  - summary export,
  - empty DB headers.
- API health:
  - `/health`,
  - empty DB behavior,
  - camera list,
  - latest metrics by camera,
  - invalid camera,
  - report endpoint.
- Dashboard fallback behavior:
  - no DB,
  - missing video,
  - missing model,
  - no zone config,
  - report download helper.
- Missing file behavior:
  - missing source video,
  - missing zones config,
  - missing training labels,
  - invalid DB extension.

## Acceptance Criteria

The following commands must pass:

```bash
python3 -m compileall src scripts
pytest
python3 scripts/run_video_demo.py --help
python3 scripts/export_report.py --help
python3 scripts/benchmark_fps.py --help
```

If a real sample video exists, these must pass:

```bash
python3 scripts/run_video_demo.py \
  --source data/input_videos/sample.mp4 \
  --output data/outputs/demo.mp4 \
  --zones-config configs/zones.example.json \
  --db data/outputs/analytics.db

python3 scripts/export_report.py \
  --db data/outputs/analytics.db \
  --out data/outputs/report.csv
```

Additional acceptance criteria for Agent 4:

- No private CCTV footage is committed.
- No fake labels or fake fine-tuning results are created.
- Existing demo path remains functional.
- New config-driven camera path works for at least `cam1` on `data/input_videos/sample.mp4`.
- Dashboard can run with no existing DB and guide the user to run analysis.
- API returns clean JSON errors, not 500s, for empty/missing data.

## Implementation Order

Agent 4 should implement changes in this order:

1. Add `src/config/schema.py` and `src/config/loader.py`.
2. Fix `configs/cameras.yaml` to use existing `data/input_videos/sample.mp4` and add production-safe fields.
3. Update `configs/app.yaml`, `configs/thresholds.yaml`, and `configs/zones.example.json` while preserving current demo compatibility.
4. Add `configs/zones.indian_platform.example.json`.
5. Update `src/video/stream_reader.py` for source parsing, frame stride, FPS cap, and mocked reconnect behavior.
6. Update `src/vision/line_counter.py` to include stable line IDs and direct crossing events.
7. Update `src/vision/crowd_analyzer.py` to implement dwell and hysteresis.
8. Update `src/vision/zone_manager.py` for frame resolution metadata and optional scaling/density.
9. Update `src/vision/detector.py` and `src/vision/tracker.py` for YOLO11 defaults, runtime options, and metadata.
10. Update `src/video/frame_processor.py` to return latency, crossing events, and alert transition events.
11. Update `src/analytics/database.py` schema for camera/session metadata, stream health, and line `track_id`.
12. Update `src/analytics/event_logger.py` to log direct events, stream health, and metadata.
13. Update `src/analytics/report_generator.py` and `scripts/export_report.py` for time-window and summary reports.
14. Replace `scripts/benchmark_fps.py` placeholder with a real benchmark CLI.
15. Replace `scripts/create_zones_from_frame.py` placeholder with calibration/validation functionality.
16. Extend `scripts/extract_sample_frames.py` for labeling workflow metadata.
17. Update `scripts/run_video_demo.py` to consume config and camera IDs while preserving legacy flags.
18. Update `src/api/schemas.py` and `src/api/routes.py` for camera-aware endpoints and config-driven processing.
19. Update `src/dashboard/streamlit_app.py` for the guided demo/operator flow.
20. Replace or complete `src/main.py` with a useful top-level CLI.
21. Add `scripts/train_yolo.py`, `scripts/evaluate_detector.py`, and `scripts/evaluate_counts.py` with guarded behavior for missing labeled data.
22. Add and update tests in the order of changed modules.
23. Update `README.md`, `docs/SYSTEM_ARCHITECTURE.md`, `docs/DATASET_AND_LABELING.md`, and `docs/DEPLOYMENT.md`.
24. Run all acceptance commands.
25. If `data/input_videos/sample.mp4` exists, run the full demo and CSV export commands.

## Top 10 Implementation Changes Needed

1. Add a shared config loader/schema and make `configs/cameras.yaml` actually drive the pipeline.
2. Harden `StreamReader` for RTSP/live CCTV with reconnect, health, FPS cap, and frame skipping.
3. Switch default model policy to YOLO11, preferably `yolo11s.pt` for pilot and `yolo11n.pt` for CPU demos.
4. Add stable line IDs and direct crossing event payloads including `track_id`.
5. Implement dwell time and hysteresis in crowd alerts.
6. Expand SQLite schema for camera metadata, stream health, model/tracker metadata, and line event track IDs.
7. Replace placeholder `benchmark_fps.py` and `create_zones_from_frame.py` with real tools.
8. Upgrade FastAPI to camera-aware live/historical endpoints and avoid production-style long jobs inside synchronous requests.
9. Redesign Streamlit into a guided demo/operator flow with original video, annotated output, counts, alerts, explanation, and report download.
10. Add a real dataset/fine-tuning workflow that extracts frames and prepares labeling, while refusing to claim fine-tuning without labeled images.
