# Railway Crowd Analytics — Indian Platform Passenger Counting (MVP)

> **This is an MVP prototype, not a production-ready system.** It demonstrates the
> analytics pipeline on sample/public video and has **not** been validated on live
> Indian Railway platform CCTV. See `docs/PRODUCTION_RESEARCH_ROADMAP.md` for the
> path to pilot and production deployment.

A working computer-vision system that turns railway-station CCTV (or any video
feed) into real-time passenger-flow and crowd-safety analytics.

It detects people, tracks them with stable IDs, counts directional line
crossings, measures zone occupancy, raises warning/critical crowd alerts, and
stores everything in SQLite — surfaced through a FastAPI backend and a Streamlit
dashboard.

> Built on **pretrained YOLO** — no custom dataset required for demos. It runs today
> on sample video and webcam sources, and accepts RTSP/HTTP URLs through OpenCV for
> initial testing. Production accuracy and reliable live CCTV operation require
> approved footage, labeled data, fine-tuning, stream hardening, and station-specific
> calibration (see production roadmap).

---

## What It Does

- **Person detection** — YOLO (Ultralytics) detects passengers frame by frame.
- **Tracking** — ByteTrack/BoT-SORT assign each person a stable ID across frames.
- **Line crossing counts** — directional IN/OUT counters for passenger flow.
- **Zone occupancy** — live headcount inside configured polygon zones.
- **Crowd alerts** — `NORMAL → WARNING → CRITICAL` based on per-zone thresholds.
- **Persistence** — sessions, frames, occupancy snapshots, crossings, and alert
  transitions are written to SQLite.
- **API + dashboard** — query metrics over HTTP or watch them on a live dashboard.
- **Annotated video** — bounding boxes, zone overlays, and counters burned into an
  output MP4 for incident reports.

## Pipeline

```
video / RTSP  →  detect + track  →  zones + lines + crowd alerts  →  SQLite
                       (YOLO)              (analytics)                  │
                                                                       ├─→ FastAPI  (/metrics, /alerts, /sessions)
                                                                       ├─→ Streamlit dashboard
                                                                       └─→ CSV report export
```

## Tech Stack

- Python 3.11+ (developed/tested on 3.11–3.12)
- Ultralytics YOLO · OpenCV · Supervision
- FastAPI · Streamlit
- SQLite + SQLAlchemy
- Pytest

## Project Layout

```
src/
  video/        frame reading, processing, writing
  vision/       detector, tracker, line counter, zone manager, crowd analyzer, annotator
  analytics/    SQLAlchemy models, event logger, metrics, CSV report generator
  api/          FastAPI app, routes, schemas
  dashboard/    Streamlit app
scripts/        run_video_demo.py, export_report.py, calibration helpers
configs/        app.yaml, cameras.yaml, thresholds.yaml, zones.example.json, zones.indian_platform.example.json
tests/          unit tests for zones, lines, crowd alerts, metrics, persistence
docs/           demo script, architecture, production research roadmap
data/           input_videos/, outputs/ (annotated video + analytics.db)
```

---

## Quick Start

### 1) Create and activate a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** A virtual environment is tied to its absolute path. If you rename or
> move the project folder, recreate it (`rm -rf .venv && python3.11 -m venv .venv`)
> — a moved `.venv` will fail to activate.

### 2) Add a demo video (no private CCTV needed)

Place any short, public/sample crowd clip at:

```
data/input_videos/sample.mp4
```

The system is designed to demo on **sample/public video only**. You can also point
it at a webcam (`--source 0`) or an RTSP/HTTP URL instead of a file. If the source
is missing, the demo prints clear guidance instead of crashing.

### 3) Run the video demo

```bash
python scripts/run_video_demo.py \
  --source data/input_videos/sample.mp4 \
  --output data/outputs/demo.mp4 \
  --zones-config configs/zones.example.json \
  --db data/outputs/analytics.db
```

This produces an annotated `demo.mp4` and writes analytics to `analytics.db`.

### 4) Launch the dashboard

```bash
streamlit run src/dashboard/streamlit_app.py
```

Open <http://localhost:8501>. Pick a source in the sidebar and click
**▶ Process Video**, or just view analytics from a previous run.

### 5) Launch the API

```bash
uvicorn src.api.app:app --reload
```

Open <http://localhost:8000/docs> for interactive API documentation.

### 6) Export a report

Export everything into a single CSV file (each row tagged with its source table):

```bash
python scripts/export_report.py \
  --db data/outputs/analytics.db \
  --out data/outputs/report.csv
```

Open it with your default app:

```bash
open data/outputs/report.csv      # macOS
```

> If `--out` ends in `.csv` you get one file. If it has no `.csv` suffix it is
> treated as a directory and one CSV per table is written (e.g.
> `--out data/outputs/reports/`). Use `--table <name>` to export a single table.
> If the database does not exist yet, the script prints the exact command to
> generate one. An empty database still produces a valid CSV with headers.

---

## Command Reference

Every command below is supported and verified. Run them from the project root
with the virtual environment activated.

| Purpose | Command |
|---|---|
| Byte-compile all source (sanity check) | `python -m compileall src scripts` |
| Run the test suite | `pytest` |
| Video demo help | `python scripts/run_video_demo.py --help` |
| Export report help | `python scripts/export_report.py --help` |
| Start the API (auto-reload) | `uvicorn src.api.app:app --reload` |
| Start the dashboard | `streamlit run src/dashboard/streamlit_app.py` |

### `run_video_demo.py` options

| Flag | Default | Description |
|---|---|---|
| `--source` | `data/input_videos/sample.mp4` | File path, webcam index (e.g. `0`), or RTSP/HTTP URL |
| `--output` | `data/outputs/demo.mp4` | Annotated output video path |
| `--zones-config` | `configs/zones.example.json` | Zone/line JSON config |
| `--model` | `yolo11n.pt` | YOLO weights path or model name (`yolo11s.pt` recommended when hardware allows) |
| `--confidence` | `0.35` | Detection confidence threshold |
| `--show` | off | Show a live preview window while processing |
| `--db` | _none_ | SQLite analytics DB path (enables logging) |
| `--camera-id` | source basename for ad-hoc sources; configured camera ID when using `configs/cameras.yaml` | Camera identifier stored in the DB |
| `--snapshot-interval` | `5.0` | Seconds between zone-occupancy DB snapshots |

### `export_report.py` options

| Flag | Description |
|---|---|
| `--db` (required) | Path to the analytics SQLite database |
| `--out` (required) | `.csv` path → single file (all tables combined, or one table with `--table`); no `.csv` suffix → directory with one CSV per table |
| `--table` | One of `run_sessions`, `frames_processed`, `zone_occupancy`, `line_crossing_events`, `crowd_alerts` |
| `--camera` | Filter rows by `camera_id` |
| `--session` | Filter rows by `run_session_id` |

---

## Configuration

The system is fully **config-driven** — no code changes needed to add a camera,
zone, line, or threshold.

- `configs/app.yaml` — model, tracker, persistence, API/dashboard defaults.
- `configs/cameras.yaml` — camera sources, resolutions, lines, and zones.
- `configs/thresholds.yaml` — per-zone warning/critical crowd thresholds.
- `configs/zones.example.json` — zone polygons and counting lines used by the demo.
- `configs/zones.indian_platform.example.json` — template zones for a platform-side CCTV view.

Example zone + line (`zones.example.json`):

```json
{
  "zones": [
    {
      "id": "platform_zone",
      "name": "Platform Waiting Area",
      "polygon": [[100, 200], [1180, 200], [1180, 680], [100, 680]],
      "warning_threshold": 20,
      "critical_threshold": 30
    }
  ],
  "lines": [
    {
      "id": "entry_gate", "name": "Entry Gate",
      "start": [200, 600], "end": [1100, 600],
      "in_label": "IN", "out_label": "OUT"
    }
  ]
}
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/cameras` | Configured/persisted camera registry |
| GET | `/cameras/{camera_id}/health` | Latest camera stream health |
| GET | `/cameras/{camera_id}/latest` | Latest metrics for one camera |
| GET | `/cameras/{camera_id}/counts` | Aggregate counts for one camera |
| GET | `/cameras/{camera_id}/occupancy` | Latest zone occupancy for one camera |
| GET | `/cameras/{camera_id}/alerts` | Recent alerts for one camera |
| GET | `/metrics/latest` | Latest frame metrics + per-zone occupancy + line counts |
| GET | `/metrics/zones` | Latest occupancy per zone |
| GET | `/alerts` | Recent crowd alerts (newest first) |
| GET | `/sessions` | Recent processing sessions |
| POST | `/analysis/run` | Run the pipeline on a source and log analytics |
| POST | `/process-video` | Run the pipeline on a video and log analytics |

## Docker (Local Demo)

```bash
docker compose up --build
```

- API: <http://localhost:8000>
- Dashboard: <http://localhost:8501>

---

## Privacy

- The SQLite database stores **counts, ephemeral tracker IDs, line-crossing events,
  stream health, and alert events** only — no face crops, biometrics, raw frames,
  or identifying records.
- The demo can write an annotated MP4 under `data/outputs/`. For private/live CCTV,
  disable annotated video output unless storage has been explicitly approved.
- Use **sample/public video** for demos. Any real CCTV use must avoid storing
  personally identifying footage unless explicitly approved.
- Keep camera lines, zones, and thresholds in config files (already the default).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `command not found: python` after `source .venv/bin/activate` | The `.venv` was created at a different path (folder moved/renamed). Recreate it: `rm -rf .venv && python3.11 -m venv .venv && pip install -r requirements.txt`. |
| `Unable to load an Ultralytics YOLO model` | Pass an existing local weights file (`--model yolo11n.pt`) or allow network access so weights can download on first run. |

## Benchmarking And Training Prep

Benchmark model options on the sample clip:

```bash
python scripts/benchmark_models.py \
  --source data/input_videos/sample.mp4 \
  --frames 120 \
  --output-csv data/outputs/model_benchmark.csv
```

Prepare real Indian-platform frames for annotation:

```bash
python scripts/extract_training_frames.py \
  --source data/indian_railway_videos/pexels_crowded_train_station_6023186.mp4 \
  --out data/training_frames/raw \
  --num 24
```

The repo does not ship a labeled Indian railway dataset. Training is intentionally blocked until
you create YOLO-format labels. See `docs/ANNOTATION_WORKFLOW.md`.
| `Unable to open source` | Provide a valid video path, webcam index (`--source 0`), or RTSP/HTTP URL. |
| Dashboard shows "No analytics data found" | Run the video demo with `--db data/outputs/analytics.db` first, or click **▶ Process Video** in the sidebar. |

---

## Final Checklist

Use this to confirm the system is demo-ready end to end:

- [ ] **Setup complete** — `pip install -r requirements.txt` succeeds and
      `python -m compileall src` reports no errors.
- [ ] **Run video demo** —
      `python scripts/run_video_demo.py --source data/input_videos/sample.mp4 --output data/outputs/demo.mp4 --zones-config configs/zones.example.json --db data/outputs/analytics.db`
      produces `data/outputs/demo.mp4`.
- [ ] **Run dashboard** — `streamlit run src/dashboard/streamlit_app.py` opens at
      <http://localhost:8501> and shows zones, lines, alerts, and the annotated video.
- [ ] **Export report** —
      `python scripts/export_report.py --db data/outputs/analytics.db --out data/outputs/report.csv`
      writes a single CSV you can open with `open data/outputs/report.csv`.
- [ ] **Run tests** — `pytest` passes.

> Tip: the API (`uvicorn src.api.app:app --reload`) is optional for the demo but
> useful to show programmatic access at <http://localhost:8000/docs>.
