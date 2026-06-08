# Full Project Explanation

> A complete, plain-English-then-technical walkthrough of the Railway Crowd Analytics
> project, written so you can explain it confidently to your boss and answer technical
> questions. Nothing in this document changes code — it only explains what is already here.
>
> **One honest framing to keep front of mind:** this is a **working MVP / prototype**, not a
> production system. Everything runs end to end on a public sample video. It has **not** been
> validated on real, approved Indian Railway CCTV. Say that out loud in the demo — it builds
> trust, and every doc in this repo says the same thing.

---

## 1. One-Sentence Summary

It is a computer-vision system that takes railway-platform video, detects and tracks every
passenger, counts how many are in each area and crossing each line, raises crowd-safety
alerts, and shows all of that through a web dashboard — and it can do the detection two ways
(whole-body or head-only) so you can compare which counts people better on CCTV-style
footage.

---

## 2. Simple Explanation (for a non-technical manager)

Imagine a CCTV camera pointing at a railway platform. Normally a human would have to watch
the screen and guess "that area looks too crowded." This system does that automatically.

It watches the video frame by frame, puts a box around each person, gives each person a
number so it doesn't count the same person twice, and then:

- counts how many people are standing in each marked area of the platform,
- counts how many people walk past an invisible line (and in which direction),
- turns the area **green → yellow → red** when it gets too crowded,
- saves all those numbers so you can open them later as a spreadsheet, and
- shows everything on a simple web page.

The new part: platform cameras are mounted high and far, so people at the far end are tiny
and often half-hidden behind others. A "whole body" detector misses them. So we added a
second detector that looks for **just heads** — because a head is usually still visible even
in a crowd. The web page runs **both** and shows them side by side so you can see which one
counts better.

No faces, no names, no identities are stored — only counts and anonymous numbers.

---

## 3. Why The Project Exists (the railway-platform problem)

Indian railway platforms are some of the most crowded public spaces in the world. The
operational problems this system targets:

- **Crowding / overcrowding.** When a platform exceeds safe capacity, the risk of crushes,
  falls onto the track, and stampedes rises sharply. Staff need to know *which* zone is
  overcrowded *before* it becomes dangerous — not after.
- **Passenger counting.** Stations want to know how many people are on a platform, entering,
  and exiting. Today this is mostly manual estimation. Automatic counting is more accurate
  and continuous.
- **Safety.** The platform edge / yellow line near the train doors is the highest-risk area.
  Automatic alerts let staff and RPF (Railway Protection Force) respond faster.
- **Flow management.** Knowing how crowds *build up and disperse* (IN vs OUT over time) helps
  manage gates, foot-over-bridges, and announcements during peak hours.
- **Operational planning.** Peak-hour occupancy data over days/weeks guides staffing,
  resource allocation, and infrastructure decisions (where to add exits, where bottlenecks
  form).
- **Why CCTV can help.** Stations already have CCTV cameras installed. This system reuses
  that existing infrastructure — no new sensors required — and turns raw video into numbers,
  alerts, and reports automatically, 24/7, without a human staring at a screen.

This is exactly what `AGENTS.md` defines as the project's core goal: *detect passengers,
track them, count line/zone crossings, detect crowded areas, generate real-time analytics
for station safety and passenger-flow management, and surface it via a dashboard and API.*

---

## 4. Why Body Detection Alone Is Not Enough

The original system detects **whole people** (the standard "person" object that pretrained
YOLO models know). That works well in some conditions and fails in others:

- **Full body works when the person is close and fully visible.** A clear head-to-feet
  rectangle is easy for the model to find, and the tracker can follow it smoothly.
- **It fails when people are far away.** Platform cameras are mounted high and cover a long
  platform. A passenger at the far end may be only a few dozen pixels tall — too small and
  blurry for a whole-body detector to register, so that passenger is simply **missed** and
  the count reads too low.
- **It fails when people are blocked (occlusion).** On a busy platform people stand
  shoulder-to-shoulder. Each person's torso and legs are hidden behind the person in front.
  A body detector needs to see most of the body; when it can't, it either misses the person
  or merges several people into one box.
- **Low-resolution footage hurts full-body detection.** Compressed CCTV streams lose fine
  body detail. Seated passengers, people behind railings, and people carrying large
  luggage/trolleys also break the "upright full body" shape the model expects.
- **Railway platforms have exactly the bad combination:** long views (distant, tiny people) +
  heavy occlusion (dense crowds) + variable posture (sitting, leaning, luggage). This is the
  worst case for whole-body detection, and it's the *normal* case on an Indian platform at
  rush hour.

This reasoning is laid out in detail in `docs/HEAD_DETECTION_RESEARCH_PLAN.md`.

---

## 5. Why Head Detection Was Added

The key insight: **in almost every failure case above, the head is still visible.** Heads
stick up above the crowd, are rarely fully hidden, and keep a stable round/oval shape
regardless of distance, posture, or luggage. Counting heads is the standard academic approach
to crowd counting precisely because **one head ≈ one person**, even when bodies overlap.

So a head detector:

- **catches passengers the body detector misses** in dense and distant views,
- **gives a more accurate count when bodies overlap**, and
- **degrades more gracefully** as the scene gets more crowded.

Important facts about *our* head model (verified against the repo):

- **The boss specifically wanted a fine-tuned head model** — not a generic one — so it would
  be tuned to the head-detection task rather than guessing from a general object detector.
- **It was trained in Google Colab, not on this laptop.** Training used a cloud GPU (an
  NVIDIA A100). The completed model artifacts were then copied into this project. (See
  `docs/HEAD_MODEL_TRAINING_REPORT.md`.)
- **`best.pt` was imported locally.** The trained weights file landed at
  `artifacts/colab/best.pt` and the canonical path the system uses is
  `models/fine_tuned/head_detector/weights/best.pt`. Their SHA-256 hashes match, proving the
  imported file is exactly the Colab-trained one.
- **It still needs real Indian Railway CCTV validation.** It was trained on a public dataset
  (RPEE-Heads — a German railway-platform + event-entrance head dataset), which is
  platform-*relevant* but **not Indian**. The accuracy numbers we have are measured on that
  dataset's own validation images, **not** on Indian platform footage. Real validation
  requires approved Indian CCTV clips.

We did **not** delete the body detector. The system runs **both** and compares them honestly
— showing where head detection helps and where it doesn't.

---

## 6. Big Picture Architecture

The full pipeline, with what happens at each arrow:

```
Video input
   │   A camera file, webcam, or RTSP/HTTP stream is opened.
   ▼
Frame reader  (src/video/stream_reader.py — StreamReader)
   │   Decodes the video into individual frames (images), one at a time.
   │   Handles live-stream reconnects, frame-skipping, and FPS caps.
   ▼
YOLO detector / tracker  (src/vision/detector.py + tracker.py)
   │   A YOLO neural network looks at each frame and draws boxes around
   │   either PERSONS (body mode) or HEADS (head mode). The tracker
   │   (ByteTrack) gives each box a stable ID so the same person isn't
   │   counted twice across frames.
   ▼
(optional) detection filter  (src/vision/detection_filter.py)
   │   In head mode on the CCTV sample, removes false positives outside
   │   the platform (e.g. train front, signage) by size and region rules.
   ▼
Zone counter  (src/vision/zone_manager.py — ZoneManager)
   │   For each tracked box, checks whether its reference point is inside
   │   each marked platform polygon, and counts how many are inside.
   ▼
Line-crossing counter  (src/vision/line_counter.py — LineManager)
   │   Detects when a track crosses a virtual line and in which direction,
   │   incrementing IN / OUT counts (each track counted once per direction).
   ▼
Crowd-alert analyzer  (src/vision/crowd_analyzer.py — CrowdAnalyzer)
   │   Compares each zone's count to its warning/critical thresholds and
   │   produces NORMAL / WARNING / CRITICAL (with dwell + hysteresis logic).
   ▼
Annotator  (src/vision/annotator.py)
   │   Draws all of the above onto the frame: boxes, IDs, zone polygons
   │   (green/yellow/red), counting lines, IN/OUT counters, alert banner.
   │   The annotated frames are written to an output MP4 (video_writer.py).
   ▼
Database logger  (src/analytics/event_logger.py → SQLite)
   │   Writes sessions, per-frame metrics, periodic zone snapshots, line
   │   crossing events, and alert transitions into a SQLite database.
   ▼
CSV report  (src/analytics/report_generator.py + scripts/export_report.py)
   │   Exports the database tables to CSV for spreadsheets / sharing.
   ▼
API + Web app  (src/api/* FastAPI, src/dashboard/streamlit_app.py)
       The dashboard and API read the database back and present metrics,
       comparisons, videos, and downloadable reports.
```

The orchestration happens in two places:
- **`src/video/frame_processor.py` (`FrameProcessor`)** glues detection/tracking → zones →
  lines → alerts together for one frame, returning a `FrameProcessResult`.
- **`scripts/run_video_demo.py` (`run_pipeline`)** is the top-level driver: it reads config,
  builds the models, loops over every frame, annotates, writes video, and logs analytics.

---

## 7. Folder Structure

| Folder | What it does |
|---|---|
| `configs/` | All the knobs, in files (not code): `app.yaml` (model/tracker/DB defaults), `cameras.yaml` (camera sources + which zone file each uses), `thresholds.yaml` (per-zone crowd thresholds), and the `zones.*.json` files (zone polygons + counting lines). |
| `data/input_videos/` | Input videos. `sample.mp4` (the canonical demo clip) and `cctv_platform_sample.mp4` (a preserved copy of it), plus `previous_sample_backup.mp4` and `.source.json` provenance files. |
| `data/outputs/` | Everything the pipeline produces: annotated videos (`body_demo.mp4`, `head_demo.mp4`), SQLite databases (`body_analytics.db`, `head_analytics.db`), and CSV reports (`body_report.csv`, `head_report.csv`). |
| `data/sample_frames/` | Still frames extracted from the videos, used to *calibrate* zone/line polygons (you draw zones by looking at a representative frame). |
| `data/head_datasets/` | The RPEE-Heads head-detection dataset in YOLO format (`yolo/rpee_heads/`) and raw form (`raw/`). Used only if you re-train the head model. |
| `models/fine_tuned/` | Fine-tuned model weights. The head detector lives at `fine_tuned/head_detector/weights/best.pt` (the Colab-trained file), with `last.pt`, `results.csv`, and training images alongside. |
| `scripts/` | Runnable command-line tools: the video demo, the body-vs-head comparison, report export, training, evaluation, dataset prep, benchmarks, and calibration helpers. |
| `src/vision/` | The "seeing" code: `detector.py`, `tracker.py`, `zone_manager.py`, `line_counter.py`, `crowd_analyzer.py`, `annotator.py`, `detection_filter.py`. |
| `src/video/` | The "video plumbing" code: `stream_reader.py` (read frames), `frame_processor.py` (run one frame through the pipeline), `video_writer.py` (write annotated frames out). |
| `src/analytics/` | The "memory" code: `database.py` (SQLAlchemy tables), `event_logger.py` (writes analytics), `report_generator.py` (CSV export), `metrics.py` (metric dataclasses). |
| `src/api/` | The FastAPI web service: `app.py` (app factory), `routes.py` (endpoints), `schemas.py` (request/response shapes). |
| `src/dashboard/` | The Streamlit web app (`streamlit_app.py`) — the visual demo your boss sees. |
| `docs/` | All reports and plans, including this file, the training report, the body-vs-head integration report, the CCTV retest report, and the production roadmap. |
| `tests/` | Pytest unit tests for zones, lines, crowd alerts, metrics, database persistence, detector modes, the API, and the stream reader. (54 tests passed in the last verification.) |

---

## 8. Important Scripts

All commands are run from the project root with the virtual environment activated
(`source .venv/bin/activate`).

### `scripts/run_video_demo.py` — the main pipeline runner
- **What it does:** runs the full pipeline (read → detect/track → zones/lines/alerts →
  annotate → log) on one source, in either body or head mode.
- **When to use it:** any time you want to process one video in one mode and get an annotated
  output video + analytics database.
- **Example:**
  ```bash
  python scripts/run_video_demo.py \
    --source data/input_videos/sample.mp4 \
    --output data/outputs/body_demo.mp4 \
    --zones-config configs/zones.cctv_platform.example.json \
    --db data/outputs/body_analytics.db \
    --detector-mode body
  ```
- **Expected output:** `body_demo.mp4` (annotated video) and `body_analytics.db` (analytics),
  plus a console line like `Processed 803 frames.` In head mode it auto-uses higher inference
  resolution (`imgsz≥1536`) and lower confidence (≤0.25) because heads are small.

### `scripts/run_comparison_demo.py` — run both modes on the same video
- **What it does:** a wrapper that runs body mode first, then head mode, on the *same* source
  and zones, writing two outputs and two databases. This is the script behind the dashboard's
  "Run Body vs Head Comparison" button.
- **When to use it:** to generate the side-by-side comparison before a demo.
- **Example:**
  ```bash
  python scripts/run_comparison_demo.py \
    --source data/input_videos/sample.mp4 \
    --zones-config configs/zones.cctv_platform.example.json
  ```
- **Expected output:** `body_demo.mp4` + `body_analytics.db` *and* `head_demo.mp4` +
  `head_analytics.db`. Head mode defaults to `--confidence 0.15 --imgsz 1536`. If the head
  model is missing, it still runs body mode and tells you exactly where to put `best.pt`.

### `scripts/export_report.py` — turn the database into CSV
- **What it does:** exports the SQLite analytics tables to CSV (one combined file, or one file
  per table, or a single table — with optional camera/session filters).
- **When to use it:** to hand a spreadsheet to someone, or to attach to a report.
- **Example:**
  ```bash
  python scripts/export_report.py \
    --db data/outputs/body_analytics.db \
    --out data/outputs/body_report.csv
  ```
- **Expected output:** `body_report.csv`. If `--out` ends in `.csv` you get one file; if it's
  a directory you get one CSV per table. Empty DBs still produce a valid header-only CSV.

### `scripts/train_head_detector.py` — fine-tune the head model (rarely run)
- **What it does:** validates the labeled head dataset and runs Ultralytics YOLO training to
  produce a head detector. It **refuses to train** unless real YOLO-format labels (class
  `0 = head`) exist on disk — it never fakes training.
- **When to use it:** only when you want to re-train the head model locally. **You normally
  don't** — the production model was trained in Colab and imported. Use `--help` to see
  options; running it for real needs `data/head_datasets/yolo/rpee_heads/` present.
- **Example (help only, safe):**
  ```bash
  python scripts/train_head_detector.py --help
  ```
- **Expected output (if actually trained):** a `best.pt` under
  `models/fine_tuned/head_detector/weights/`.

### `scripts/evaluate_head_detector.py` — quick head-model sanity check
- **What it does:** loads the head `best.pt`, runs it on one dataset image (prints head count)
  and on a sample video (prints frames, average heads/frame, max heads/frame, FPS), and writes
  an annotated video. This is a *raw detector* check, separate from the full analytics
  pipeline.
- **When to use it:** to confirm the head model loads and detects heads at all.
- **Example:**
  ```bash
  python scripts/evaluate_head_detector.py \
    --model models/fine_tuned/head_detector/weights/best.pt \
    --source data/input_videos/sample.mp4 \
    --output data/outputs/head_demo.mp4
  ```
- **Expected output:** console stats (e.g. "average heads/frame: 7.31, max: 27") and an
  annotated `head_demo.mp4`.

### `scripts/prepare_head_dataset.py` — build the training dataset (one-time)
- **What it does:** organizes the downloaded RPEE-Heads dataset into the YOLO folder layout,
  validates every label box, forces single-class `0 = head`, and writes the dataset config.
  It produced the counts documented in `docs/HEAD_DATASET_REPORT.md` (1,346 train / 246 val
  images; ~94,614 head boxes kept; 14 degenerate boxes skipped).
- **When to use it:** only once, before re-training. You don't need it for demos.

> There are also helper scripts you'll rarely touch: `benchmark_models.py` /
> `benchmark_fps.py` (compare model speed/accuracy), `extract_training_frames.py` /
> `extract_sample_frames.py` (pull frames for labeling/calibration),
> `create_zones_from_frame.py` (draw zones on a frame), and `train_yolo_indian_platform.py`
> (a *separate* guardrailed body-training entrypoint, intentionally blocked until labeled
> Indian data exists).

---

## 9. Detector System

**What YOLO is, in simple language.** YOLO ("You Only Look Once") is a neural network that
looks at an image once and instantly draws boxes around the objects it recognizes, each with
a label ("person", "head") and a confidence score. It's fast enough to run on video frames.
We use the **Ultralytics YOLO11** family (`yolo11n` = nano = smallest/fastest, `yolo11s` =
small = more accurate).

**Body mode.** Uses a **pretrained** YOLO person detector. It only keeps the "person" class
(`person_class_id = 0`). No training needed — it works out of the box. Default weights:
`yolo11n.pt` (from `configs/app.yaml`). The reference point for counting is the
**bottom-center** of the body box (roughly where the feet touch the ground).

**Head mode.** Uses the **fine-tuned** head model. It's a single-class detector where class
`0 = head`. The reference point for counting is the **center** of the head box.

**What `best.pt` is.** In YOLO/Ultralytics, training produces two weight files: `last.pt`
(the final epoch) and `best.pt` (the epoch that scored best on validation). `best.pt` is the
one we use. Ours is the Colab-trained head detector, living at
`models/fine_tuned/head_detector/weights/best.pt` (≈5.5 MB, single class `head`).

**Why the model path matters.** The code picks weights in a specific order (see
`resolve_model_candidates` in `detector.py`). In body mode it can fall back through
`yolo11s → yolo11n → yolov8n` or auto-use a fine-tuned body model if present. In head mode,
the path **must** point to the head `best.pt` — if it's missing, the runner prints a clear
message telling you the exact expected path and exits cleanly (no confusing crash). That's
why you pass `--model models/fine_tuned/head_detector/weights/best.pt` for head runs.

**What a confidence threshold means.** Every detection comes with a confidence (0–1: "how
sure am I this is a head?"). The threshold is the cutoff — boxes below it are discarded.
- Body default: `0.35`.
- Head default is **lower** (the comparison script uses `0.15`, the runner caps head mode at
  `≤0.25`) because heads are small CCTV targets and naturally score lower confidence; a high
  threshold would throw away real heads. Lower confidence finds more heads but risks more
  false positives — which is why head mode also uses the detection filter.

**What bounding boxes are.** A bounding box is the rectangle `(x1, y1, x2, y2)` the model
draws around each detected object. Everything downstream (zones, lines, tracking) operates on
these rectangles, so swapping "body boxes" for "head boxes" needs no rewrite of the analytics
— that's the whole reason head mode was easy to add.

---

## 10. Tracking System

**Why detection alone is not enough.** Detection is per-frame and has no memory. If you only
detected, the same person standing still for 100 frames would be "100 people." You'd also have
no way to know a person crossed a line (which needs comparing where they were vs where they
are now).

**What a tracker does.** A tracker links detections across frames: "the box here in frame 5 is
the same object as the box there in frame 6." It gives each object a persistent identity.

**What tracking IDs mean.** Each tracked object gets a stable integer `track_id` (e.g. `#42`).
The same passenger keeps `#42` as they move across the platform, frame after frame.

**Why tracking prevents double counting.** Because each person has one ID, zone occupancy
counts *unique IDs inside the zone* (not raw boxes), and a line crossing is counted **once per
direction per ID**. Without IDs, you'd recount the same person every frame.

**Why IDs can switch in crowded footage.** When two boxes get close, overlap, or briefly
disappear behind someone, the tracker can lose the thread and assign a **new** ID to the same
person ("ID switch"). This inflates the unique-passenger count. It's worse for **small head
boxes** than for large body boxes — which is exactly what we saw: on the same clip, body mode
found ~108 unique tracks while head mode reported ~294 unique head IDs. The higher head number
is partly real extra detections and partly ID-switch inflation, so treat it cautiously.

**How body boxes and head boxes are tracked the same way.** Both modes call Ultralytics
`.track()` with the same tracker (**ByteTrack** by default, **BoT-SORT** optional — set in
`configs/app.yaml`). The tracker doesn't care whether the box is a body or a head; it's the
same algorithm on the same `(x1,y1,x2,y2)` rectangles. The only differences between modes are
the *model that produces the boxes* and the *reference point* used for counting.

---

## 11. Zone Counting

**What zones are.** A zone is an area of the platform you care about, drawn as a polygon
(a shape with corner points) on the video. Examples in our CCTV config: "Main Platform Waiting
Area," "Platform Movement Path," "Train-Side Platform Edge."

**How zones are defined in JSON.** In files like `configs/zones.cctv_platform.example.json`,
each zone has an `id`, a `name`, a `polygon` (list of `[x, y]` pixel coordinates), and
`warning_threshold` / `critical_threshold` counts. Lines are defined in the same file. The
polygons are calibrated by looking at a representative frame from the actual video — that's
why there's a `frame_width`/`frame_height` reference, so the runner can rescale the polygons
if the video resolution differs.

**How the system decides if a passenger is inside a zone.** For each tracked box, it computes
a single **reference point** and runs a point-in-polygon test (`point_in_polygon` in
`zone_manager.py`, a ray-casting algorithm). If the point is inside the polygon, that track
counts toward the zone. Occupancy = number of unique track IDs whose reference point is inside.

**Body mode uses the bottom-center of the body box.** That approximates where the person's
feet are on the ground — the right spot to decide which floor area they're standing in.

**Head mode uses the center of the head box.** A head has no meaningful "feet," so the box
center (the head's position in the scene) is used instead.

**Why this difference matters.** If you used bottom-center for a head box, you'd be testing a
point floating in mid-air *below* the head, which could fall in the wrong zone or on the wrong
side of a line — biasing the counts. Using the head center keeps head counting correct. This
is why the runner automatically sets `point_strategy = "center"` for head mode and
`"bottom_center"` for body mode. It also means **body and head counts are not expected to
match exactly** — they detect different things and measure from different points.

---

## 12. Line Crossing

**What virtual lines are.** A virtual line is an invisible boundary you draw across the video
(two endpoints). It's used to count directional passenger flow — e.g. a line near the train
doors, or across the platform entrance.

**How IN/OUT counting works.** For each tracked person, the system computes which **side** of
the line their reference point is on (a sign-of-cross-product test in `line_counter.py`). When
a track's side flips from one frame to the next, that's a crossing. The direction of the flip
determines whether it's labeled IN or OUT (configurable labels — the CCTV config uses
`TOWARD_TRAIN` / `AWAY_FROM_TRAIN`). It increments the matching counter.

**Why it helps passenger flow management.** IN vs OUT over time tells you how a crowd is
building or dispersing: lots of IN before a train arrives, lots of OUT after it leaves. That's
directly useful for gate control, announcements, and staffing.

**How it avoids counting the same track repeatedly.** Each `track_id` records which directions
it has already been counted for (`counted_directions`). Once a person is counted as IN, they
won't be counted IN again — so jitter near the line, or a person loitering on the boundary,
doesn't inflate the count. (Frames where the point sits exactly on the line are ignored until
it clearly moves to one side.)

---

## 13. Crowd Alerts

Handled by `CrowdAnalyzer` in `src/vision/crowd_analyzer.py`, with thresholds from
`configs/thresholds.yaml` (or per-zone values in the zone JSON).

**Warning threshold.** When a zone's occupancy reaches its `warning_count`, the zone goes to
**WARNING** (yellow) — "getting crowded, keep an eye on it."

**Critical threshold.** When occupancy reaches `critical_count`, the zone goes to **CRITICAL**
(red) — "over safe capacity, act now."

**How alerts are triggered.** Each frame, the analyzer compares every zone's occupancy to its
thresholds and returns `NORMAL` / `WARNING` / `CRITICAL`. It also supports two refinements:
- **Dwell time** (`dwell_seconds`): the count must stay high for N seconds before escalating,
  so a one-frame spike doesn't trigger a red alert.
- **Hysteresis / clear-below** (`clear_below_count`): the alert only clears once occupancy
  drops *below* a lower number, preventing flickering green↔yellow right at the threshold.

The database stores an alert row only when the level **changes** (a transition), not every
frame — so the alert log is a clean timeline of "zone X went WARNING at frame 120."

**How this supports station safety.** Staff/RPF get an automatic, objective signal of *which*
zone is dangerous and *when*, instead of relying on someone watching a wall of monitors. In
the CCTV retest, the body run produced 88 WARNING and 17 CRITICAL alert states on the sample
clip with the tuned demo thresholds — proof the alerting path works end to end.

---

## 14. Database and Reports

**What SQLite is.** SQLite is a tiny, file-based database — the whole database is a single
`.db` file, no separate server to install or run. Perfect for a local prototype. We use it
through SQLAlchemy (a Python library that maps database tables to Python classes).

**Why analytics are saved.** So you can review the run later, build time-series reports,
power the dashboard/API, and export spreadsheets — without re-processing the video.

**What gets logged** (tables in `src/analytics/database.py`):
- `camera_registry` — the cameras/sources known to the system.
- `run_sessions` — one row per pipeline run (source, model, detector_mode, tracker, start/end
  time, total frames, total unique passengers).
- `frames_processed` — per-frame metrics (detections, tracked objects, unique-seen, FPS,
  stream health counters).
- `zone_occupancy` — periodic per-zone occupancy snapshots (every ~5 s by default) with the
  alert level at that moment.
- `line_crossing_events` — one row per directional crossing (which line, which direction,
  which track, which frame).
- `crowd_alerts` — one row per alert-level *transition*.
- `stream_health_events` — stream lifecycle/health (started, streaming, reconnects, completed).

Every analytics table carries a `detector_mode` column (`body` or `head`) so body and head
records never get confused. (Older databases are auto-migrated to add this column.) Crucially,
**no faces, biometrics, names, or raw frames are stored** — only counts, anonymous track IDs,
and events.

**What `report.csv` contains.** `export_report.py` flattens the database into CSV. The
combined CSV tags each row with its source `table` and unions all columns, so one file
contains sessions, frames, zone snapshots, crossings, and alerts together — openable in Excel.

**Difference between `body_analytics.db` and `head_analytics.db`.** They are the *same schema*
but hold the results of the two different detection modes. Body mode writes to
`body_analytics.db`; head mode writes to `head_analytics.db`. Keeping them in separate files
makes the side-by-side comparison clean (the dashboard reads each independently). The
`run_sessions.detector_mode` value will be `body` vs `head` accordingly.

**Difference between `body_report.csv` and `head_report.csv`.** Same idea, one step later:
they are the CSV exports of `body_analytics.db` and `head_analytics.db` respectively. Every
row in `body_report.csv` has `detector_mode = body`; every row in `head_report.csv` has
`detector_mode = head`. They let you compare the two modes' numbers in a spreadsheet.

---

## 15. Web App (Streamlit)

`src/dashboard/streamlit_app.py` is the visual demo — a single web page (run with
`streamlit run src/dashboard/streamlit_app.py`, opens at http://localhost:8501). It is written
for a **non-technical** viewer and is organized into 13 numbered sections:

- **Original video** (Section 3) — plays the raw input clip ("this is what the CCTV sees").
  The sidebar lets you pick the CCTV-angle sample, the canonical `sample.mp4`, the previous
  backup, or a custom path/RTSP URL.
- **Body output** (Section 6, left) — the annotated full-body detection video.
- **Head output** (Section 6, right) — the annotated head detection video, side by side.
- **Analytics comparison** (Section 8) — for each mode: unique passengers, frames processed,
  zone occupancy (with green/yellow/red status), line crossings (IN/OUT), and crowd alerts —
  laid out in two columns so body and head sit next to each other.
- **Report download** (Section 9) — previews `body_report.csv` / `head_report.csv` and offers
  download buttons.
- **Explanation sections** — Section 1 (body-vs-head comparison table), Section 2 (plain-
  English pipeline), "Why this CCTV-angle test matters," Section 4 (model status: confirms the
  head `best.pt` is found and was Colab-trained), Section 7 (how to read the overlays),
  Section 10 (what data is collected — operational only), Section 11 (why it matters for
  Indian Railways), Section 12 (limitations), Section 13 (a built-in ~10-minute boss demo
  guide).
- **Run buttons** (Section 5) — "Run Full-Body Detection," "Run Head Detection" (disabled if
  the model is missing), "Run Body vs Head Comparison," and CSV export buttons. These just
  call the same scripts you'd run on the command line, streaming their output live.

**What the boss should understand from it:** that the system turns CCTV into live, anonymous
crowd analytics; that head detection picks up passengers body detection misses on CCTV-style
footage; that everything is reproducible (the buttons re-run it live); and that it's an honest
prototype with clearly stated limitations — not a finished product.

---

## 16. API (FastAPI)

`src/api/` is a small FastAPI service (run with `uvicorn src.api.app:app --reload`, docs at
http://localhost:8000/docs). It exposes the analytics in the database over HTTP/JSON so other
software (not just the dashboard) can read them. Key endpoints:

- **`GET /health`** — liveness check; returns `{"status":"ok"}`. Used to confirm the service
  is up.
- **`GET /metrics/latest`** — the latest processed frame's metrics for the most recent session:
  total detections, unique passengers, FPS, per-zone occupancy + alert level, and line counts.
- **`GET /metrics/zones`** — the latest occupancy per zone (with alert level) for the most
  recent session.
- **`GET /alerts`** — recent crowd alerts, newest first (with a `limit`).
- Plus per-camera variants (`/cameras`, `/cameras/{id}/latest`, `/counts`, `/occupancy`,
  `/alerts`, `/health`), `/sessions`, and `POST /process-video` (a.k.a. `/analysis/run`) which
  triggers a pipeline run programmatically.
- By default the API reads `data/outputs/analytics.db`; pass `?db_path=...body_analytics.db`
  or `...head_analytics.db` to point it at a specific mode's database.

**When the API matters vs when Streamlit matters.** **Streamlit** is for *humans* — the live
visual demo you show your boss, with videos, tables, and explanations. The **API** is for
*other systems* — a control-room dashboard, a mobile alert app, or an integration that wants
to *query the numbers* programmatically (e.g. "give me the current occupancy of zone X every
10 seconds"). For the demo you'll mostly use Streamlit; the API is what you'd build on for a
real operational integration.

---

## 17. CCTV-Angle Test Video

`docs/CCTV_SAMPLE_RETEST_REPORT.md` exists, so here's what happened and why it matters:

**Why the old phone-angle video was bad.** The earlier demo clip looked like phone footage —
a low, close, slightly hand-held-looking angle. Real railway deployment uses **fixed CCTV
cameras mounted high and wide**, so that old clip didn't test the conditions that actually
matter (small, distant, partially-blocked passengers).

**What new CCTV-angle video was chosen.** The Pexels clip **"People on Platform on Train
Station"** (https://www.pexels.com/video/people-on-platform-on-train-station-12049569/). It's
1280×720, 25 fps, ~32 seconds, 803 frames. It's stored as the canonical
`data/input_videos/sample.mp4` with a preserved copy at
`data/input_videos/cctv_platform_sample.mp4`. (`docs/CCTV_PLATFORM_VIDEO_SEARCH_REPORT.md`
documents the 10 candidates that were compared before picking it.)

**Why it is better.** It's *real railway platform* footage with an **elevated, static-looking
angle**, a train at the platform, and passengers at multiple distances with moderate
crowding/occlusion — exactly the conditions needed to fairly test body vs head detection on
CCTV-style video. New zones were calibrated for it
(`configs/zones.cctv_platform.example.json`), and head mode was corrected to run at
`imgsz=1536, confidence=0.15` after the initial 640px setting missed small heads. On this
clip: body mode = 803 frames, 108 unique tracks; head mode = 803 frames, 294 unique head IDs.

**What limitations remain.** It is **not** Indian-Railway-specific, **not** confirmed
operational CCTV (it's stock footage with a CCTV-*like* angle), and its resolution, camera
height, compression, and crowd behavior may still differ from real station CCTV. The correct
phrasing for the boss is: *"real railway-platform stock footage with a CCTV-like elevated
angle — a good test proxy, but not a substitute for approved Indian Railway CCTV."* Final
validation still needs supervisor-approved Indian footage.

---

## 18. Exact Demo Flow

Run these from the project root. (The dashboard buttons do the same thing if you prefer
clicking — but running the comparison in advance means the videos are ready when you present.)

```bash
# 1) Activate the environment
cd /Users/sriniketh/crowd-analysis
source .venv/bin/activate

# 2) Run the body-vs-head comparison on the CCTV-angle sample
#    (generates both annotated videos + both analytics databases)
python scripts/run_comparison_demo.py \
  --source data/input_videos/sample.mp4 \
  --zones-config configs/zones.cctv_platform.example.json

# (optional) export the CSV reports so Section 9 is populated
python scripts/export_report.py --db data/outputs/body_analytics.db --out data/outputs/body_report.csv
python scripts/export_report.py --db data/outputs/head_analytics.db --out data/outputs/head_report.csv

# 3) Launch the web app
streamlit run src/dashboard/streamlit_app.py
#    open http://localhost:8501
```

Then, in the browser, walk through:
4. **Show the original video** — Section 3: "this is what the platform camera sees."
5. **Show body detection** — Section 6 (left): boxes around whole people.
6. **Show head detection** — Section 6 (right): boxes around heads; point out heads picked up
   in the distant/crowded areas where bodies are hard to see.
7. **Show the analytics** — Section 8: unique passengers, zone occupancy (green/yellow/red),
   line crossings, and alerts for each mode, side by side; Section 9 for the downloadable CSV.
8. **Explain the limitations** — Section 12: prototype, trained on a public head dataset
   (RPEE-Heads), counts not yet directly comparable, needs station-specific calibration.
9. **Ask for real Indian Railway CCTV footage** — the single most valuable next input. Even a
   short approved clip lets us validate properly and re-tune for real conditions.

> The talking point on the head model: *"We fine-tuned this head model in Google Colab on a
> railway-platform head dataset, then imported the trained `best.pt` into the project. It runs
> alongside the body detector today. The next step is validating it on actual approved Indian
> Railway platform footage."*

---

## 19. Common Questions Your Boss Might Ask (Q&A)

**Why head detection?**
Because on platform CCTV, passengers far down the platform or packed in a crowd are too small
or too blocked for whole-body detection — but their heads are usually still visible. Head
detection catches people body detection misses, giving a more complete count in exactly the
crowded/distant cases that matter most for safety.

**Is this production-ready?**
No. It's a working MVP/prototype. It runs end to end on a public sample video, but it has not
been validated on real, approved Indian Railway CCTV, and production needs live-stream
hardening, deployment, monitoring, and privacy approval. We're transparent about that.

**How accurate is it?**
The head model scores well on its *training dataset's* validation images (precision ≈ 0.87,
recall ≈ 0.72, mAP@0.5 ≈ 0.79 on RPEE-Heads). But those numbers are **not** measured on Indian
footage or even on our sample video — they don't translate directly to real-world accuracy. We
have no production accuracy claim until we validate on approved Indian CCTV.

**Can it work live on CCTV?**
The pipeline already accepts RTSP/HTTP camera streams and has reconnect/health logic, so it can
*technically* run on a live feed. But reliable 24/7 operation needs additional infrastructure
(stream hardening, hardware, monitoring) and all legal/privacy approvals first.

**What data does it store?**
Only operational analytics: detector mode, timestamps/frame numbers, anonymous tracking IDs,
zone occupancy counts, line-crossing events, alert levels, and camera/source IDs. **No faces,
no names, no biometrics, no raw identifying footage.**

**Does it identify people?**
No. The tracking ID is just a temporary number to avoid double-counting within one run; it has
no link to any person's identity and is meaningless after the run.

**What happens in dense crowds?**
Body detection degrades (people merge or get missed); head detection holds up better but isn't
perfect — back-row heads fully hidden behind front-row heads are still missed, and tiny heads
cause more tracking ID switches (which can inflate the unique count). Head detection *reduces*
undercounting, it doesn't eliminate it.

**What is needed to improve it?**
Most of all: real, approved Indian Railway CCTV clips to validate and re-tune on. Then
threshold tuning, per-camera zone calibration, labeling Indian footage, and optionally
re-fine-tuning the head model on Indian heads.

**What hardware is needed?**
It runs on a CPU laptop for the demo (slower). For real-time live CCTV, a GPU machine is
recommended — heads are tiny and benefit from higher-resolution inference, which is heavier.
Each camera stream adds processing load.

**Can it scale to multiple cameras?**
The design is multi-camera-aware (cameras are configured in `configs/cameras.yaml`, every
record carries a `camera_id`, and the API has per-camera endpoints). But running many live
streams at once needs proportional hardware and a deployment/orchestration layer we haven't
built yet.

**Why not only use full-body detection?**
Because it misses distant and occluded passengers on platform CCTV — undercounting exactly
when crowding (and risk) is highest.

**Why not only use head detection?**
Because head boxes are small, causing more tracking ID switches, more potential false positives
(round objects mistaken for heads), and the zones/lines were originally drawn for body-scale
boxes. Body detection is more stable when people are close and clearly visible. Running both
and comparing is the honest approach until we validate on real footage.

---

## 20. Current Limitations (be honest)

- **The model needs real Indian Railway CCTV validation.** All accuracy numbers so far are on
  a public German/event dataset, not Indian platforms.
- **Public footage may not match actual stations.** The sample clip is CCTV-*like* stock
  footage, not confirmed operational CCTV, and not Indian.
- **The head model may produce false positives.** Round/dark objects (bags, helmets, signage,
  lights) can be mistaken for heads; the detection filter mitigates but doesn't fully solve
  this.
- **The full-body model may miss far passengers.** Distant, tiny, or occluded people are
  undercounted in body mode.
- **Tracking IDs may switch**, especially for small head boxes in dense crowds — inflating the
  unique-passenger count.
- **Zones need manual setup.** Each camera view requires polygons drawn by hand on a
  representative frame.
- **Thresholds need station-specific tuning.** The default warning/critical counts are demo
  values, not real platform capacities.
- **Production needs more.** Live RTSP at scale, deployment, monitoring/observability, and
  **privacy/legal approval** are all required before any real CCTV deployment.
- **Body and head counts are not directly comparable yet** — different targets, different
  reference points, and zone/line geometry drawn for body-scale boxes; they need per-mode
  calibration before the numbers line up.

---

## 21. What To Improve Next (prioritized)

1. **Get real Indian Railway CCTV clips from the boss** (even short, approved ones) — the
   single highest-value input.
2. **Test body vs head on that actual CCTV** to see which counts better under real conditions.
3. **Tune confidence thresholds** for the real footage (body and head separately).
4. **Adjust zones per camera** — redraw polygons and lines for each real camera view.
5. **Label Indian footage** (heads and/or bodies) using the annotation workflow.
6. **Re-fine-tune on Indian platform heads** once labeled data exists, to close the domain gap.
7. **Add RTSP live streaming** hardening for continuous real-time operation.
8. **Improve the dashboard for multiple cameras** (a control-room view across platforms).
9. **Add a deployment plan** — hardware, hosting, monitoring, and privacy/governance.

---

## 22. Glossary

- **YOLO** — "You Only Look Once," a fast neural network that detects objects in an image and
  draws labeled boxes around them. We use Ultralytics YOLO11.
- **Bounding box** — the rectangle `(x1, y1, x2, y2)` drawn around a detected object (a person
  or a head).
- **Confidence** — the model's certainty (0–1) that a box is correct; boxes below the
  threshold are discarded.
- **Fine-tuning** — taking a pretrained model and training it further on a specific dataset
  (here, head images) so it specializes in that task.
- **best.pt** — the saved weights file from the best-scoring training epoch; the head model we
  actually use, trained in Colab and imported.
- **Tracker** — the component that links detections across frames so the same object keeps one
  identity (we use ByteTrack; BoT-SORT optional).
- **Tracking ID** — the persistent number assigned to one tracked object; prevents
  double-counting; anonymous and temporary.
- **Zone** — a marked polygon area of the platform whose occupancy is counted.
- **Line crossing** — counting people who cross a virtual line, with direction (IN/OUT).
- **Crowd alert** — the NORMAL/WARNING/CRITICAL status raised when a zone's occupancy crosses
  its thresholds.
- **SQLite** — a lightweight, single-file database used to store all analytics.
- **CSV** — a plain "comma-separated values" spreadsheet file you can open in Excel; the export
  format for reports.
- **API** — Application Programming Interface; here, the FastAPI HTTP service that lets other
  software query the analytics.
- **Streamlit** — a Python library for quickly building web dashboards; powers the visual demo
  page.
- **RTSP** — "Real-Time Streaming Protocol," the standard URL format for live IP/CCTV camera
  streams; the pipeline can read these.
- **CCTV** — Closed-Circuit Television; the fixed surveillance cameras already installed in
  stations that this system is designed to use.

---

*End of full project explanation. For deeper detail, see `README.md`,
`docs/HEAD_DETECTION_RESEARCH_PLAN.md`, `docs/HEAD_MODEL_TRAINING_REPORT.md`,
`docs/BODY_VS_HEAD_INTEGRATION_REPORT.md`, `docs/CCTV_SAMPLE_RETEST_REPORT.md`,
`docs/HEAD_UPGRADE_FINAL_TEST_REPORT.md`, and `docs/PRODUCTION_RESEARCH_ROADMAP.md`.*
