# System Architecture — AI-Based Crowd Analysis and Passenger Counting in Railway Stations

> Concrete design that coding agents implement from. Defines modules, data flow, config
> file schema, API/dashboard plan, and the database schema. Pairs with
> `RESEARCH_NOTES.md` (rationale) and `PROJECT_PLAN.md` (scope/schedule).
>
> Stack (from `AGENTS.md`): Python 3.11, Ultralytics YOLO, OpenCV, Supervision, FastAPI,
> Streamlit, SQLite, SQLAlchemy, Pytest.
>
> Design rules: small single-responsibility modules; **config over hardcoding** (cameras,
> lines, zones, thresholds in config files); analytics logic kept pure and unit-testable,
> isolated from detection/tracking I/O.

---

## 1. High-Level Overview

```
              ┌──────────────────────────────────────────────────────────┐
              │                     Pipeline (per camera)                  │
 Video ──────▶│ Ingestion → Detection → Tracking → Analytics ─────────────│──▶ State
 (file/RTSP)  │   (OpenCV)   (YOLO)    (ByteTrack)  (lines/zones/crowd)    │    (in-memory)
              └──────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                                   Persistence (SQLAlchemy → SQLite)
                                          │
                          ┌───────────────┴───────────────┐
                          ▼                                ▼
                    FastAPI (REST)                  Streamlit dashboard
                  live + historical                (annotated frames,
                     analytics                       counts, alerts)
```

The pipeline is the heart of the MVP. **State** (current counts, occupancy, alerts) is
held in memory for the live API/dashboard and periodically written to SQLite. The API and
dashboard are read-oriented consumers.

---

## 2. Module Breakdown

Proposed package layout under `src/` (keep files < 500 lines; one responsibility each):

```
src/
  config/
    loader.py          # Load + validate YAML config into typed dataclasses/Pydantic models
    schema.py          # Config schema definitions (Camera, Line, Zone, Thresholds, App)
  ingestion/
    video_source.py    # Open file/RTSP/HTTP via OpenCV; yield frames; reconnect logic
  detection/
    detector.py        # YOLO wrapper: frame -> Detections (person class), conf/iou config
  tracking/
    tracker.py         # ByteTrack/BoT-SORT wrapper: Detections -> tracked Detections w/ IDs
  analytics/
    line_counter.py    # Virtual-line in/out counting (supervision LineZone) — PURE logic
    zone_counter.py    # Polygon-zone occupancy counting (supervision PolygonZone) — PURE
    crowd_monitor.py   # Threshold + dwell/hysteresis crowd detection & alert generation
    metrics.py         # Flow rate, occupancy snapshots, aggregation helpers
  pipeline/
    pipeline.py        # Orchestrates ingestion→detection→tracking→analytics per tick
    state.py           # In-memory live state (current counts/occupancy/active alerts)
    annotator.py       # Draw boxes, IDs, lines, zones, counters onto frames (demo/dashboard)
  persistence/
    models.py          # SQLAlchemy ORM models (see §6)
    db.py              # Engine/session setup, init_db()
    repository.py      # CRUD/query helpers (write events/snapshots/alerts, read history)
  api/
    main.py            # FastAPI app + routes (see §5)
    schemas.py         # Pydantic response/request models
  dashboard/
    app.py             # Streamlit app (reads via API or repository)
  app.py               # Entry point: load config, build + run pipeline (CLI args)
config/
  app.yaml             # Global app/runtime config
  cameras/
    cam1.yaml          # Per-camera config: source, lines, zones, thresholds
tests/
  test_line_counter.py
  test_zone_counter.py
  test_crowd_monitor.py
  conftest.py          # Synthetic-track fixtures
```

**Responsibilities**

- **config** — single source of truth; validates types/ranges; fails fast on bad config.
- **ingestion** — robust frame source (loop file, reconnect stream); decoupled from CV.
- **detection** — thin YOLO adapter returning a normalized `Detections` object.
- **tracking** — assigns/keeps stable track IDs; tracker choice from config.
- **analytics** — the *testable core*. `line_counter`, `zone_counter`, `crowd_monitor`
  take tracked detections + config and return counts/occupancy/alerts. **No I/O here.**
- **pipeline** — wires modules together on a per-frame/per-tick loop; updates `state`.
- **persistence** — ORM models + repository; the only place that touches SQLite.
- **api** — serves live `state` and historical DB queries.
- **dashboard** — visualization layer over the API/DB.

> **Testability boundary:** analytics modules must be importable and runnable with
> synthetic tracks (lists of boxes/IDs over frames) — no camera, model, or DB needed.
> This is what the required Pytest suites target.

---

## 3. Data Flow

### 3.1 Per-frame (live) flow
1. **Ingestion** yields a frame `(frame, frame_index, timestamp)`.
2. **Detection**: `detector.detect(frame)` → `Detections` filtered to `person`, above
   confidence threshold.
3. **Tracking**: `tracker.update(detections)` → detections annotated with stable
   `tracker_id`.
4. **Analytics** (per camera config):
   - `line_counter.update(tracked)` → increments in/out per line; emits crossing events.
   - `zone_counter.update(tracked)` → current occupancy per zone.
   - `crowd_monitor.update(occupancy)` → evaluates thresholds + dwell → alert on/off.
   - `metrics` → flow rate (crossings/min), rolling aggregates.
5. **State update**: `state` holds current counts, per-zone occupancy, active alerts,
   last frame, and last flow rates.
6. **Annotation** (optional/for dashboard): `annotator` draws boxes, IDs, lines, zones,
   and live counters on the frame.

### 3.2 Persistence flow (periodic / event-driven)
- **Events** (line crossings, alert raised/cleared) are written **as they occur**.
- **Occupancy snapshots** + **count summaries** are written on a configurable interval
  (e.g., every N seconds) — not every frame — to keep the DB lean.

### 3.3 Read flow (API / dashboard)
- **Live**: API reads in-memory `state` (or a shared store) for current counts/alerts.
- **Historical**: API queries SQLite via `repository` (time-range filters, per camera/
  line/zone).
- **Dashboard**: Streamlit calls the API (preferred) or repository directly for charts +
  annotated frame display.

### 3.4 Threading/process note (implementation guidance)
- MVP can run the pipeline in one process; the API/dashboard read shared state.
- Simplest robust pattern: **pipeline writes to SQLite + an in-memory state object**;
  API/dashboard read from those. Avoid premature async/multiprocess complexity — add only
  if FPS requires it (see stretch goals).

---

## 4. Configuration Files

**Format:** YAML, loaded and validated by `config/loader.py` into typed models
(`config/schema.py`). **All camera-specific geometry and thresholds live here** — never in
code. Coordinates are in **pixel space of that camera's frame** (document resolution).

### 4.1 `config/app.yaml` (global runtime)
```yaml
app:
  name: "railway-crowd-analytics"
  log_level: "INFO"

model:
  weights: "yolov8n.pt"      # pretrained; person class only used
  device: "cpu"              # "cpu" | "cuda:0"
  confidence: 0.35
  iou: 0.5
  person_class_id: 0         # COCO person
  frame_stride: 1            # process every Nth frame (perf tuning)

tracker:
  type: "bytetrack"          # "bytetrack" | "botsort"
  track_activation_threshold: 0.25
  lost_track_buffer: 30
  minimum_matching_threshold: 0.8

persistence:
  db_url: "sqlite:///data/analytics.db"
  snapshot_interval_seconds: 5

api:
  host: "0.0.0.0"
  port: 8000
```

### 4.2 `config/cameras/cam1.yaml` (per camera)
```yaml
camera:
  id: "cam1"
  name: "Platform 1 - East"
  source: "data/samples/station.mp4"   # file path or RTSP/HTTP URL
  frame_width: 1280                      # reference resolution for coordinates
  frame_height: 720

lines:                                   # virtual lines for in/out counting
  - id: "gate_line"
    name: "Entry Gate"
    start: [200, 600]                    # [x, y] pixel
    end: [1100, 600]
    # 'in' direction = crossing from side A->B (define convention in code/docstring)

zones:                                   # polygons for occupancy / crowd detection
  - id: "platform_zone"
    name: "Platform Waiting Area"
    polygon: [[100, 200], [1180, 200], [1180, 680], [100, 680]]
    crowd_threshold: 25                  # occupancy count that triggers an alert
    crowd_dwell_seconds: 3               # must stay above threshold this long to alert
    density_area_sqm: null               # optional: real-world area for density (people/m^2)
```

**Validation rules (loader must enforce):**
- IDs unique within a camera; polygons have ≥ 3 points; line has distinct start/end.
- Thresholds ≥ 0; intervals > 0; device/tracker/model values from allowed sets.
- Coordinates within `frame_width/height` (warn if out of bounds).
- Fail fast with a clear error message on invalid config.

---

## 5. API & Dashboard Plan

### 5.1 FastAPI (REST) — `src/api/main.py`
Read-oriented endpoints; JSON via Pydantic schemas (`src/api/schemas.py`).

| Method & Path | Purpose |
|---------------|---------|
| `GET /health` | Liveness/readiness probe |
| `GET /cameras` | List configured cameras + their lines/zones |
| `GET /cameras/{camera_id}/live` | Current live state: per-line in/out, per-zone occupancy, flow rates, active alerts |
| `GET /cameras/{camera_id}/counts` | Historical line-crossing counts (query: `from`, `to`, `line_id`) |
| `GET /cameras/{camera_id}/occupancy` | Historical zone occupancy snapshots (query: `from`, `to`, `zone_id`) |
| `GET /cameras/{camera_id}/alerts` | Crowd alerts (query: `from`, `to`, `active` flag) |
| `GET /cameras/{camera_id}/frame` | Latest annotated frame (JPEG) for the dashboard *(optional)* |

Notes:
- Time-range params are ISO timestamps; sensible defaults (e.g., last hour).
- Responses are typed; errors return structured JSON with status codes.
- **No auth in MVP** (note as stretch goal); bind to localhost for demos.

### 5.2 Streamlit dashboard — `src/dashboard/app.py`
Consumes the API (preferred) or repository. Panels:
- **Live view:** annotated frame/video, per-line in/out counters, per-zone occupancy.
- **Flow rate:** crossings/min per line (live + short trend).
- **Crowd status:** per-zone occupancy vs. threshold, **active alert banner** when crowded.
- **History:** time-series charts of counts/occupancy; table of recent alerts.
- **Camera selector** when multiple cameras exist (stretch).
- Auto-refresh on an interval for near-real-time feel.

---

## 6. Database Schema Plan

**SQLite** via **SQLAlchemy ORM** (`src/persistence/models.py`). Store **aggregates and
events only — never PII or raw footage** (`RESEARCH_NOTES.md` §7). Timestamps in UTC.

### 6.1 Tables

**`cameras`** — registry mirrored from config (optional convenience).
| Column | Type | Notes |
|--------|------|-------|
| `id` | TEXT (PK) | camera id from config (e.g., "cam1") |
| `name` | TEXT | display name |
| `source` | TEXT | file/stream identifier (no secrets) |
| `created_at` | DATETIME | |

**`line_crossings`** — one row per crossing event (or per interval summary).
| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER (PK, autoincr) | |
| `camera_id` | TEXT (FK→cameras.id, indexed) | |
| `line_id` | TEXT (indexed) | from config |
| `direction` | TEXT | "in" \| "out" |
| `track_id` | INTEGER (nullable) | ephemeral tracker id (not identity) |
| `timestamp` | DATETIME (indexed) | event time (UTC) |

**`occupancy_snapshots`** — periodic per-zone occupancy.
| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER (PK) | |
| `camera_id` | TEXT (FK, indexed) | |
| `zone_id` | TEXT (indexed) | from config |
| `count` | INTEGER | people in zone at snapshot |
| `density` | REAL (nullable) | people/m² if area configured |
| `timestamp` | DATETIME (indexed) | snapshot time (UTC) |

**`count_summaries`** — periodic rollups for fast trend queries (optional but recommended).
| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER (PK) | |
| `camera_id` | TEXT (FK, indexed) | |
| `line_id` | TEXT (indexed) | |
| `in_count` | INTEGER | crossings "in" during interval |
| `out_count` | INTEGER | crossings "out" during interval |
| `interval_start` | DATETIME (indexed) | |
| `interval_end` | DATETIME | |

**`alerts`** — crowd alert lifecycle.
| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER (PK) | |
| `camera_id` | TEXT (FK, indexed) | |
| `zone_id` | TEXT (indexed) | |
| `alert_type` | TEXT | e.g., "crowd_threshold_exceeded" |
| `severity` | TEXT | "info" \| "warning" \| "critical" |
| `occupancy_at_trigger` | INTEGER | count when raised |
| `threshold` | INTEGER | configured threshold |
| `raised_at` | DATETIME (indexed) | |
| `cleared_at` | DATETIME (nullable) | null while active |

### 6.2 Relationships & indexing
- `cameras (1) → (N) line_crossings / occupancy_snapshots / count_summaries / alerts`.
- Index `(camera_id, timestamp)` / `(camera_id, raised_at)` for time-range queries.
- Active alerts = rows where `cleared_at IS NULL`.

### 6.3 Retention (privacy)
- Provide a config-driven retention/purge for old `occupancy_snapshots` and
  `line_crossings` (data minimization). `alerts` may be kept longer for safety review.

---

## 7. Testing Hooks (maps to `AGENTS.md` required tests)

- **`test_line_counter.py`** — feed synthetic tracks crossing a line in each direction;
  assert in/out counts and that re-counting is prevented per track.
- **`test_zone_counter.py`** — place synthetic centroids inside/outside a polygon; assert
  occupancy counts, including boundary cases.
- **`test_crowd_monitor.py`** — drive occupancy above/below threshold over time; assert
  alert raises only after dwell, clears correctly, and no flicker (hysteresis).
- Fixtures in `conftest.py` provide deterministic synthetic detections/tracks so tests run
  **without** a model, GPU, video, or DB.

---

## 8. Implementation Order (for coding agents)

Follow `PROJECT_PLAN.md` weeks. Recommended build order:
1. `config/` (schema + loader + sample YAML) →
2. `ingestion/` →
3. `detection/` →
4. `tracking/` →
5. `analytics/` (+ tests, the testable core) →
6. `pipeline/` + `state` + `annotator` →
7. `persistence/` →
8. `api/` →
9. `dashboard/`.

Build each module small and runnable; verify with a sample video before moving on.
Do not hardcode camera geometry — read it from `config/`.
