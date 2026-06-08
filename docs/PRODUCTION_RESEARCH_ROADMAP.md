# Production Research Roadmap: Indian Railway Passenger Counting

_Last updated: 2026-06-02_

This document defines the production-oriented research direction for moving from the current MVP toward a pilot-ready Indian Railway platform CCTV passenger counting system. It is based on inspection of the current codebase, published benchmarks and papers, and honest assessment of what has and has not been validated on real Indian Railway footage.

---

## Updated Problem Definition

The target system is an **AI-based passenger counting and crowd analytics platform for Indian railway station platforms**, not a generic crowd demo.

### Operational context

Indian railway platforms present a distinctive visual and operational environment:

- **Fixed CCTV viewpoints** — overhead, oblique, or long platform shots; often low resolution (576×768 to 1280×720), heavy H.264 compression, and 10–25 FPS.
- **Platform clutter** — trains at platform edge, benches, pillars, roof shade, signage, luggage, trolleys, and seated passengers who may be partially occluded or appear as static blobs.
- **Passenger states** — standing, walking, sitting, boarding/deboarding, waiting near train doors, moving on footbridges and staircases.
- **Density variation** — sparse off-peak periods vs. dense rush-hour crowds with heavy occlusion.
- **Safety-critical zones** — platform edge (yellow line), entry/exit paths, ticket/queue areas, footbridge landings, and boarding areas.

### Required capabilities (production target)

| Capability | Purpose |
|------------|---------|
| Person detection | Locate passengers in each frame |
| Multi-object tracking | Assign stable ephemeral IDs for counting logic |
| Zone occupancy | Headcount inside configured platform polygons |
| Line crossing counts | Directional IN/OUT flow at gates, stairs, platform edges |
| Crowded-area alerts | WARNING/CRITICAL when occupancy exceeds thresholds |
| Real-time analytics | Live counts, flow rates, alert state for operators |
| Operational planning | Historical trends for staffing and crowd management |
| Station safety | Early warning before dangerous overcrowding |

### Explicit non-goals

- Facial recognition, biometric identification, or cross-camera re-identification
- Fare evasion or individual passenger profiling
- Claiming production readiness without measured validation on approved CCTV

---

## Current MVP Assessment

### What the prototype does today

The project codebase implements a **working single-stream MVP**:

| Component | Status | Notes |
|-----------|--------|-------|
| Person detection | Working | Ultralytics YOLO (`yolo11n.pt` CPU/demo default; `yolo11s.pt` available for higher-accuracy evaluation; `yolov8n.pt` legacy fallback) |
| Tracking | Working | ByteTrack default; BoT-SORT switchable via `configs/app.yaml` |
| Zone occupancy | Working | Polygon zones via `ZoneManager` + `configs/zones.example.json` |
| Line crossing | Working | Directional IN/OUT via `LineManager` |
| Crowd alerts | Working | NORMAL → WARNING → CRITICAL with configurable thresholds |
| SQLite persistence | Working | Sessions, frames, occupancy snapshots, crossings, alerts |
| FastAPI | Working | `/health`, `/metrics/latest`, `/metrics/zones`, `/alerts`, `/sessions` |
| Streamlit dashboard | Working | KPIs, zone table, line counts, alerts, occupancy chart, video preview |
| CSV export | Working | `scripts/export_report.py` |
| RTSP URL support | Initial | `StreamReader` accepts RTSP/HTTP URLs via OpenCV with basic retry/FPS/stride controls; **not validated as a long-running live CCTV worker** |
| Multi-camera | Not implemented | Single source per run; `cameras.yaml` exists but is not wired to a live orchestrator |
| Annotated video output | Working | `demo.mp4` with boxes, IDs, zones, lines, alert overlays |

### What was actually validated

Current verification on 2026-06-02:

- The canonical sample is now a **real public Pexels railway platform clip** (`data/input_videos/sample.mp4`, 1280×720, 32.13 s, 803 frames) with a CCTV-like elevated/static angle. It is not confirmed CCTV and not Indian Railway-specific.
- 48 unit tests passed; analytics logic is deterministic and tested with synthetic tracks.
- The verified demo used pretrained `yolo11n.pt` on CPU. This demonstrates pipeline mechanics, not production accuracy.
- Zone occupancy, line-crossing events, crowd alerts, SQLite logging, report export, API reads, and dashboard flow all work on the sample clip.

### What the MVP does **not** prove

| Gap | Why it matters |
|-----|----------------|
| **No approved live Indian Railway CCTV tested** | Demo video is public Pexels railway platform footage, not an authorized RTSP feed from a deployed station CCTV system |
| **No measured accuracy on platform footage** | No precision/recall, counting MAE, or ID-switch metrics on approved station video |
| **No fine-tuning performed** | Pretrained COCO weights only; no labeled Indian platform dataset exists in-repo |
| **No production RTSP hardening** | Basic OpenCV retry/FPS controls exist, but there is no proven long-running worker, stream SLA, or multi-camera health supervisor |
| **No multi-camera orchestration** | Cannot run 4–20 platform cameras concurrently |
| **No GPU/TensorRT deployment** | CPU-only demo; not representative of edge/server production performance |
| **No density estimation fallback** | Pure detection+tracking; will degrade in very dense standing crowds |
| **No production auth/RBAC** | API and dashboard are open on localhost |
| **No operational SLAs** | Uptime, latency, and alert reliability are unmeasured |

**Verdict:** The MVP is **demo-ready** for showing pipeline mechanics to stakeholders. It is **not production-ready** and must not be described as validated for Indian Railway deployment.

---

## Model Recommendation

### Keep YOLO — yes

YOLO remains the right detection backbone for this project because:

1. **Real-time single-stage detection** suits live CCTV with configurable model sizes.
2. **Pretrained COCO `person` class** enables immediate MVP without labeled data.
3. **Native Ultralytics tracking integration** (ByteTrack/BoT-SORT) reduces glue code.
4. **Clean upgrade path** — swap weights, export ONNX/TensorRT, or fine-tune without redesigning the pipeline.
5. **DeepStream community support** — YOLO11/YOLOv8 ONNX export is well documented for NVIDIA deployment.

Alternatives (RT-DETR, Faster R-CNN, two-stage detectors) offer marginal accuracy gains but add latency and complexity unsuitable for a first production pilot on edge hardware.

### YOLO11 over YOLOv8 — yes, for new work

Ultralytics YOLO11 (released September 2024) improves on YOLOv8 on the COCO benchmark:

| Model | COCO mAP (val) | Params (M) | Role |
|-------|----------------|------------|------|
| YOLO11n | 39.5 | 2.6 | Fastest; edge/CPU demos |
| YOLO11s | 47.0 | 9.4 | **Recommended pilot default** |
| YOLO11m | 51.5 | 20.1 | Higher accuracy when GPU available |
| YOLOv8n | 37.3 | 3.2 | Legacy fallback only |
| YOLOv8s | 44.9 | 11.2 | Acceptable fallback |

YOLO11 achieves higher mAP with fewer parameters than the equivalent YOLOv8 variant. The codebase now defaults to `yolo11n.pt` for CPU/demo runs and keeps `yolo11s.pt` available for higher-accuracy pilot evaluation.

**Important:** COCO mAP is on general objects, not Indian platform CCTV. Gains on real station footage must be measured after fine-tuning; do not assume COCO rankings transfer directly.

### When to use yolo11n, yolo11s, yolo11m

| Variant | Use when | Hardware | Tradeoff |
|---------|----------|----------|----------|
| **yolo11n** | CPU-only demos, Jetson Nano-class edge, >15 cameras on one GPU with DeepStream | Low compute | Fastest; misses small/distant/occluded passengers |
| **yolo11s** | **Pilot default** — single-server or Jetson Orin with 2–8 cameras | Mid GPU or Orin | Best speed/accuracy balance for platform CCTV |
| **yolo11m** | Accuracy-critical zones (platform edge safety, dense waiting areas) with dedicated GPU | RTX-class GPU or Orin with headroom | ~2× compute vs. 11s; better small-object recall |

**Recommendation for next implementation phase:**

1. Keep `yolo11n.pt` as the CPU/demo default.
2. Evaluate `yolo11s.pt` as the GPU pilot default once target hardware is available.
3. Evaluate `yolo11m.pt` only after GPU deployment is wired and baseline metrics exist.

### Fine-tuning — required before production claims

Pretrained COCO weights are sufficient for **demos** but insufficient for **production accuracy claims** on Indian platform CCTV because:

- COCO training data lacks seated passengers on benches, luggage-heavy scenes, train-side occlusions, and Indian clothing/density patterns.
- Low-resolution CCTV produces smaller bounding boxes than COCO-scale training images.
- Platform-specific false positives (posters, reflections, staff uniforms) need domain adaptation.

**Fine-tune when:** approved labeled platform footage exists (see Dataset section). Until then, report all counts as **estimates** with known limits.

---

## Tracking Recommendation

### ByteTrack vs BoT-SORT

| Criterion | ByteTrack | BoT-SORT |
|-----------|-----------|----------|
| Speed | Faster (no appearance model) | Slower (optional ReID + GMC) |
| Occlusion handling | Good — uses low-confidence detections | Better — appearance + camera motion compensation |
| ID switches | Higher in dense crowds / long occlusions | Lower when ReID enabled |
| Static CCTV | Excellent fit | Overhead may be unnecessary |
| Moving camera | Weak | Better (global motion compensation) |
| Compute | Lightweight | Moderate (+ ReID model if enabled) |
| Implementation | Already default in `configs/app.yaml` | Switchable via `tracker.type: botsort` |

Published comparisons (e.g., MOT benchmarks with YOLOv8) consistently show BoT-SORT with higher MOTA/MOTP at lower FPS; ByteTrack wins on throughput.

### Recommendation: **ByteTrack for initial production pilot**

**Rationale for Indian platform CCTV:**

1. **Fixed cameras** — Indian station platform CCTV is overwhelmingly static-mount; camera motion compensation (BoT-SORT's key advantage) adds little value.
2. **Throughput priority** — A pilot must process multiple RTSP streams near real-time; ByteTrack's lower overhead preserves FPS budget for detection.
3. **Counting semantics** — Line crossing and zone occupancy count **events**, not long-term identity. Ephemeral ID switches affect flow counts less than detection misses.
4. **Already integrated** — Current MVP uses ByteTrack; zero migration risk for pilot.

**When to switch to BoT-SORT:**

- Measured ID-switch rate (IDs/min) exceeds acceptable threshold on dense rush-hour clips.
- Dwell-time analytics (time spent in zone per track) becomes a requirement — needs longer ID persistence.
- Single high-value camera (e.g., platform edge safety zone) where accuracy outweighs throughput.

**DeepSORT:** Not recommended for initial pilot. It is slower, requires more tuning, and Ultralytics/supervision ecosystem has moved to ByteTrack/BoT-SORT. Consider only if custom ReID on Indian platform appearance becomes a measured requirement.

---

## Dataset and Fine-Tuning Strategy

### What data is needed

| Data type | Purpose | Required for |
|-----------|---------|--------------|
| **Unlabeled platform video clips** | Demo, frame extraction, threshold tuning, failure analysis | Pilot prep; **not** supervised training |
| **Labeled bounding-box images (YOLO format)** | Supervised fine-tuning | Production accuracy |
| **Manual count ground truth** | Evaluate line crossing and zone occupancy accuracy | Validation |
| **Per-camera calibration metadata** | Zone/line pixel coordinates | All deployment phases |

### Label types needed

For YOLO person-detection fine-tuning:

- **Primary:** `person` full-body bounding boxes (class 0)
- **Optional secondary:** `person_head` boxes for dense crowd scenarios (CrowdHuman-style)
- **Ignore regions:** Areas with posters, reflections, or permanent platform fixtures that cause false positives
- **Do not label:** Faces as identity; boxes are for detection geometry only

For tracking/counting evaluation (separate from detection labels):

- Virtual line crossing events (manual count per direction per clip)
- Zone occupancy at sampled timestamps (manual head count)
- Track continuity segments for ID-switch analysis

### How much data for a pilot

| Phase | Images/clips | Annotation effort | Expected outcome |
|-------|--------------|-------------------|------------------|
| **Baseline eval** | 3–5 clips (5–15 min each), no labels | Manual count only | Establish error rates with pretrained model |
| **Pilot fine-tune** | 2,000–5,000 labeled frames | ~40–80 hours annotation | Measurable improvement on platform-specific scenes |
| **Production fine-tune** | 10,000–20,000+ labeled frames | Ongoing annotation pipeline | Stable accuracy across cameras, lighting, seasons |

Frame extraction strategy: sample 1 frame every 2–5 seconds across peak/off-peak, day/night, and monsoon/clear conditions to maximize diversity. Use `scripts/extract_sample_frames.py` (already implemented) as the starting point.

### How to label people in Indian platform footage

1. **Tool:** CVAT, Label Studio, or Roboflow (export YOLO format).
2. **Guidelines:**
   - Draw tight full-body boxes where >30% of body is visible.
   - Include seated passengers, partial bodies at frame edge, and people behind bench rails.
   - Mark heavily occluded individuals only if head+shoulders are identifiable.
   - Create `ignore` regions over platform posters and TV screens.
   - Do **not** record passenger names, faces, or identifying attributes in annotation metadata.
3. **Quality control:** Dual-annotate 10% of frames; resolve IoU disagreements before training.

### Train / validation / test split

| Split | Ratio | Stratification |
|-------|-------|----------------|
| Train | 70% | Balance by camera, time-of-day, crowd density |
| Validation | 15% | Used for early stopping and threshold tuning |
| Test | 15% | **Held out completely** — never used during training or config tuning |

Split by **clip/camera-day**, not by individual frames, to prevent data leakage from adjacent frames in the same video.

### Metrics to track during fine-tuning

| Metric | Target direction | Tool |
|--------|------------------|------|
| mAP@0.5 (person) | Higher | Ultralytics val |
| mAP@0.5:0.95 (person) | Higher | Ultralytics val |
| Precision / Recall at operating conf | Balance for counting | Custom eval script |
| Small-object recall (bbox area < 32² px) | Higher | Custom — critical for distant CCTV subjects |
| False positive rate on empty platform | Lower | Custom — trains leaving, no passengers |

### Transfer learning strategy

1. **Pretrain exposure:** Optionally train on **CrowdHuman** or **WiderPerson** (open, non-commercial license for WiderPerson — verify before use) to improve occlusion handling, then fine-tune on Indian platform data.
2. **Direct fine-tune:** Start from `yolo11s.pt` COCO weights; fine-tune on platform labels with reduced LR (e.g., `lr0=0.001`, 50–100 epochs, early stopping).
3. **Do not claim** fine-tuning results until training actually runs and test-set metrics are recorded.

### Why unlabeled videos alone are not enough

Supervised fine-tuning requires **input–label pairs**. Unlabeled CCTV clips are valuable for:

- Stakeholder demos and pipeline integration testing
- Frame extraction and annotation queue population
- Threshold and zone calibration

But they **cannot** improve detector weights without bounding-box annotations. Semi-supervised or self-supervised approaches exist but are out of MVP scope and require research-grade investment.

### Available open datasets (not Indian-specific)

| Dataset | Size | Relevance | License/notes |
|---------|------|-----------|---------------|
| **CrowdHuman** | 15K train / 4.3K val images; ~23 persons/image | Dense occlusion, head+body boxes | Research use; strong transfer for crowds |
| **WiderPerson** | 13K images; 400K annotations | Includes station scenes among diverse environments | Non-commercial |
| **MOT17 / MOT20** | Video sequences | Tracking evaluation (pedestrian) | Academic; not platform-specific |
| **PAMELA-UANDES** | 348 boarding/alighting sequences | Train door counting; CCTV-style | Simulated metro carriage; public |
| **Metro Platform** (Li et al.) | 627 images, 9,243 head annotations | Metro platform surveillance | Small; useful for transfer experiments |
| **MOT-RPCH / RailwayPlatformCrowdHead** | 27 sequences, 89K head boxes | Railway platform crowd (train-mounted camera) | Recent (2026); YouTube-sourced; verify availability |
| **Indian Railway-specific** | **None identified as publicly available** | — | Requires approved footage from Railway authority |

**No substitute exists for approved Indian Railway platform CCTV** for final production validation.

---

## Live CCTV Architecture

Production-oriented design for multi-camera RTSP ingestion and analytics.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Station Edge Server / NVR GPU                  │
│                                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐       ┌──────────┐          │
│  │ Camera 1 │  │ Camera 2 │  │ Camera N │  ...  │ Camera N │          │
│  │ RTSP     │  │ RTSP     │  │ RTSP     │       │ RTSP     │          │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘       └────┬─────┘          │
│       │             │             │                   │                  │
│       ▼             ▼             ▼                   ▼                  │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │              Per-Camera Worker Process / Thread              │        │
│  │  StreamReader → FrameSampler → Detector → Tracker → Analytics│        │
│  │  (reconnect)    (FPS cap)                                     │        │
│  └──────────────────────────┬──────────────────────────────────┘        │
│                             │                                            │
│                             ▼                                            │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │           Analytics Event Bus / Shared State                 │        │
│  │  zone occupancy │ line crossings │ alerts │ health metrics  │        │
│  └──────────────────────────┬──────────────────────────────────┘        │
│                             │                                            │
│              ┌──────────────┼──────────────┐                             │
│              ▼              ▼              ▼                             │
│         SQLite/PostgreSQL  FastAPI    Alert Dispatcher                   │
│         (time-series)      (REST)     (webhook/SMS/email)               │
└─────────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
                    Streamlit / Grafana Dashboard
                    (operator-facing, on-prem VPN)
```

### RTSP streams

- Ingest via OpenCV `VideoCapture` (pilot) or GStreamer/DeepStream (production).
- Support `rtsp://` with credentials stored in environment/secrets, not committed configs.
- Configure per-camera: URL, expected resolution, transport (TCP preferred for stability).

### Per-camera workers

- One worker process (or async task) per camera for fault isolation.
- Worker loads camera-specific config: zones, lines, thresholds from `configs/cameras/<id>.yaml`.
- Crash of one worker must not stop others; supervisor (systemd/Docker restart policy) relaunches.

### Frame sampling / FPS control

- CCTV often streams at 25 FPS; processing every frame may be unnecessary.
- Recommended starting point: **process 5–10 FPS** (every 3rd–5th frame at 25 FPS source).
- Config key: `frame_stride` (already in `configs/app.yaml` schema).
- Line-crossing logic must account for skipped frames (tracker `persist=True` helps).

### Reconnect logic

Current `StreamReader` reads until failure with no retry. Production requires:

```
on read failure:
  log warning with camera_id + timestamp
  release capture
  wait backoff (1s → 2s → 5s → 30s cap)
  retry open (max N attempts or infinite with alert)
  on success: reset backoff, log recovery, continue frame index or mark gap in DB
```

Emit **camera_health** events: `ONLINE`, `DEGRADED`, `OFFLINE`.

### Camera configuration

Extend `configs/cameras.yaml` (or per-file `configs/cameras/<id>.yaml`):

```yaml
camera:
  id: platform_1_overhead
  name: Platform 1 — Overhead East
  source: ${RTSP_PLATFORM_1}   # env var, not hardcoded URL
  frame_width: 1280
  frame_height: 720
  target_fps: 8
  frame_stride: 3
  reconnect:
    max_backoff_seconds: 30
    alert_after_seconds: 120
zones: [...]
lines: [...]
thresholds: [...]
```

### Per-camera zones and lines

- All geometry in **pixel coordinates** for that camera's native resolution.
- Provide a calibration UI or script (extend `scripts/extract_sample_frames.py`) to draw polygons on extracted frames.
- Indian platform zone types to configure:
  - Platform waiting area
  - Train-side platform edge (yellow line safety zone)
  - Entry/exit gate path
  - Footbridge/staircase landing
  - Ticket counter queue area
  - Boarding/deboarding buffer near train doors

### Analytics database

- **Pilot:** SQLite (current) — sufficient for 1–4 cameras, short retention.
- **Production:** PostgreSQL or TimescaleDB for concurrent writes, retention policies, and multi-station aggregation.
- Store: counts, events, alerts, health — never raw frames by default.

### Dashboard / API

- FastAPI (current) for programmatic access and dashboard backend.
- Streamlit (current) for pilot operator UI; consider Grafana for production NOC-style monitoring.
- WebSocket or SSE for live metric push (currently dashboard polls DB / reruns).

### Alerting

- Extend `CrowdAnalyzer` output to dispatch:
  - WARNING → operator dashboard highlight
  - CRITICAL → webhook/email/SMS to station control (config-driven)
- Alert deduplication and cooldown to prevent notification storms.

### Logging and health checks

- Structured JSON logs: `{camera_id, event, fps, detections, latency_ms, status}`.
- `/health` per worker + aggregate `/health/cameras`.
- Metrics: Prometheus-compatible counters for ops (optional pilot stretch).

---

## Deployment Architecture

### Tier 1 — Current prototype (development / demo)

| Aspect | Detail |
|--------|--------|
| Stack | Python 3.11, OpenCV, Ultralytics YOLO, Supervision, FastAPI, Streamlit, SQLite |
| Input | Video file, webcam, or RTSP URL (single stream) |
| Hardware | Developer laptop / CPU |
| Throughput | ~6–7 FPS HD on CPU with yolov8n (measured in demo report) |
| Status | **Working MVP — not production** |

### Tier 2 — Pilot (target next phase)

| Aspect | Detail |
|--------|--------|
| Stack | Same Python stack; default `yolo11s.pt`; hardened `StreamReader` |
| Input | 2–8 live RTSP cameras per GPU server |
| Hardware | NVIDIA GPU server (e.g., T4/L4) or Jetson Orin NX/AGX |
| Optimizations | CUDA inference; `frame_stride`; per-camera workers; PostgreSQL optional |
| Validation | Measured metrics on approved platform clips |
| Status | **Target for next implementation phase** |

### Tier 3 — Production (station deployment)

| Aspect | Detail |
|--------|--------|
| Stack | NVIDIA DeepStream 7+ / GStreamer pipeline |
| Model | YOLO11 ONNX → TensorRT engine (FP16 or INT8 with calibration) |
| Input | 10–30+ RTSP streams per GPU via `nvmultiurisrcbin` + `nvstreammux` |
| Tracking | DeepStream `nvtracker` (NvDCF or custom ByteTrack plugin) |
| Hardware | Edge: Jetson Orin AGX per station; Central: RTX/L4 GPU server for large hubs |
| Analytics | Custom C++/Python probe or message broker → existing event schema / DB |
| Status | **Future — after pilot metrics justify investment** |

### Hardware assumptions

| Deployment | Cameras | Hardware | Notes |
|------------|---------|----------|-------|
| Small station pilot | 2–4 | Jetson Orin NX 16GB | yolo11s FP16 TensorRT; 5–8 FPS per camera |
| Medium station | 8–16 | GPU server (L4/T4) | DeepStream batching; INT8 optional |
| Major hub | 20–40+ | Multi-GPU server or distributed edge | DeepStream multi-stream; central aggregation DB |

TensorRT engines are **device-specific** — build on the target Jetson/GPU, not on a dev laptop.

### Scaling to multiple cameras

1. **Vertical:** DeepStream batch inference on one GPU (preferred production path).
2. **Horizontal:** One worker per camera on multiple CPU cores (pilot path); aggregate events in shared DB.
3. **Hybrid:** Edge Jetson per platform section; central API aggregates station-wide counts.

---

## Safety and Privacy

### Data minimization (mandatory)

| Store | Do not store (unless explicitly approved) |
|-------|---------------------------------------------|
| Zone occupancy counts | Raw CCTV video |
| Line crossing events (timestamp, direction) | Face crops or embeddings |
| Anonymized ephemeral track IDs | Passenger identity linkage |
| Alert level + timestamp | Audio from cameras |
| Camera ID + session metadata | Unencrypted RTSP credentials in repos |

### Retention policy (recommended starting point)

| Data type | Retention | Rationale |
|-----------|-----------|-----------|
| Raw frames/video | 0 days (default) | Privacy |
| Per-frame detections | 0 days | Not needed; aggregates suffice |
| Occupancy snapshots | 90 days | Operational trends |
| Line crossing events | 1 year | Flow planning |
| Crowd alerts | 2 years | Safety audit |
| Session health logs | 30 days | Debugging |

Implement configurable purge jobs; document in deployment runbook.

### Access control (production)

- Dashboard and API behind VPN or station LAN only.
- Role-based access: `operator` (view), `admin` (config), `auditor` (read logs).
- Audit log for config changes (zone/threshold edits).
- RTSP credentials in secrets manager or `.env` excluded from git.

### Legal and operational

- Requires Indian Railway / station authority **written approval** for live CCTV analytics.
- Signage informing passengers of automated crowd monitoring (no facial recognition).
- Human oversight for safety-critical decisions — alerts inform staff, not automated platform barriers.
- Comply with IT Act 2000, DPDP Act 2023 principles (purpose limitation, data minimization).

---

## Evaluation Plan

All metrics must be measured on **approved platform footage** with documented ground truth. Do not publish numbers until measured.

### Detection quality

| Metric | Definition | How to measure |
|--------|------------|----------------|
| Precision | TP / (TP + FP) | Labeled test frames |
| Recall | TP / (TP + FN) | Labeled test frames |
| mAP@0.5 | COCO-style on person class | Ultralytics val on test split |
| Small-object recall | Recall for bbox area < 32² px | Custom script |

### Counting accuracy

| Metric | Definition | How to measure |
|--------|------------|----------------|
| Line-count MAE | \|auto − manual\| per clip per direction | Manual count from video |
| Line-count % error | MAE / manual × 100 | Per clip, report mean |
| Zone occupancy MAE | \|auto − manual\| at sampled timestamps | Pause video; count heads in zone |
| Cumulative flow error | End-of-hour IN+OUT vs manual | Longer clips |

### Tracking quality

| Metric | Definition | Acceptable pilot target (TBD on data) |
|--------|------------|--------------------------------------|
| ID switches | Track ID changes / minute | Lower is better; measure, don't guess |
| MOTA | Multi-object tracking accuracy | Optional; requires track-level GT |
| Fragmentation | Short broken tracks | Count for seated passenger scenarios |

### Performance

| Metric | Definition | Pilot target (indicative, not validated) |
|--------|------------|----------------------------------------|
| Processing FPS | Frames inferred per second | ≥ source FPS on GPU for 720p |
| End-to-end latency | Frame capture → DB write | < 500 ms per camera |
| GPU utilization | % under N cameras | < 80% sustained |

### Alert quality

| Metric | Definition |
|--------|------------|
| Alert precision | Fraction of WARNING/CRITICAL that match human judgment |
| Alert recall | Fraction of genuinely crowded periods that triggered alert |
| False crowd alerts | Alerts when manual count below warning threshold |
| Alert latency | Time from threshold breach to alert in dashboard |

### Reliability

| Metric | Definition |
|--------|------------|
| Uptime | % time camera worker is ONLINE |
| Reconnect success | % of stream drops recovered within 60 s |
| Data gap rate | Missing snapshot intervals / expected intervals |

---

## Production Readiness Gap

What is missing before Indian Railway Department-style deployment:

| # | Gap | Severity |
|---|-----|----------|
| 1 | Validated accuracy on approved Indian platform CCTV | **Blocker** |
| 2 | Labeled fine-tuning dataset from platform footage | **Blocker** |
| 3 | Live RTSP reconnect, health monitoring, FPS control | **High** |
| 4 | Multi-camera concurrent processing | **High** |
| 5 | GPU / TensorRT inference path | **High** |
| 6 | Production database (PostgreSQL) with retention jobs | **Medium** |
| 7 | Authentication and access control on API/dashboard | **Medium** |
| 8 | Alert dispatch (webhook/SMS) to operators | **Medium** |
| 9 | Per-camera config loader wired to live orchestrator | **Medium** |
| 10 | Calibration tooling for zone/line setup on real cameras | **Medium** |
| 11 | Seated passenger and small-object detection robustness | **Medium** |
| 12 | Density-estimation fallback for extreme crowds | **Low** (post-pilot) |
| 13 | DeepStream production pipeline | **Low** (scale phase) |
| 14 | Formal security audit and privacy impact assessment | **Blocker** (organizational) |
| 15 | Operational runbook, on-call, and SLA definition | **High** (organizational) |

---

## Recommended Next Steps

Numbered roadmap to move as close to production-ready as possible.

### Phase 0 — Stakeholder alignment (1 week)

1. Obtain **written approval** and 3–5 representative platform CCTV clips from Railway authority ( varied cameras, peak/off-peak, day/night).
2. Confirm privacy constraints: no frame storage, retention limits, access control requirements.
3. Document camera inventory: mounting angles, resolutions, RTSP access method.

### Phase 1 — Baseline evaluation (2 weeks)

4. Run pretrained `yolo11s.pt` and `yolo11n.pt` on approved clips; record detection counts vs manual counts.
5. Measure line-count and zone occupancy error with hand-calibrated zones (use frame extraction + calibration script).
6. Benchmark FPS on target GPU hardware (not CPU-only).
7. Document baseline metrics in a validation report (honest numbers only).

### Phase 2 — Pilot hardening (3–4 weeks)

8. Harden RTSP reconnect, FPS throttling, and camera health events into a long-running stream worker with measured uptime.
9. Wire `configs/cameras.yaml` to a multi-camera worker orchestrator (one process per camera).
10. Evaluate `yolo11s.pt` as the pilot default on GPU; keep `yolo11n.pt` for CPU/demo and stress tests.
11. Add Prometheus-style or structured health logging.
12. Build zone/line calibration helper on extracted platform frames.

### Phase 3 — Fine-tuning (4–6 weeks, requires labeled data)

13. Extract 3,000–5,000 diverse frames from approved clips.
14. Annotate person bounding boxes (YOLO format); QA 10% dual-labeled.
15. Fine-tune `yolo11s.pt` on platform dataset; evaluate on held-out test split.
16. Compare pretrained vs fine-tuned on counting metrics; deploy better weights to pilot.

### Phase 4 — Pilot deployment (4 weeks)

17. Deploy on station edge GPU with 2–4 live cameras.
18. PostgreSQL + retention jobs; secure API behind VPN.
19. Operator dashboard with live alerts; webhook for CRITICAL events.
20. Run 2-week pilot; collect uptime, accuracy, and operator feedback.

### Phase 5 — Production scale (ongoing)

21. Export fine-tuned model to ONNX → TensorRT INT8 (calibration on platform frames).
22. Migrate inference to DeepStream for 10+ camera scale.
23. Add density-map fallback module for extreme crowd zones (optional).
24. Security audit, DPIA, and Railway authority sign-off before network-wide rollout.

---

## Appendix: Crowd Density Estimation vs Object Detection

For most Indian platform zones, **detection + tracking + zone counting** is the correct primary approach because:

- Operators need **spatial** information (which zone is crowded), not just a total count.
- Line crossing requires **individual trajectories**.
- Platform crowds are often moderate density where detection works.

**Density estimation** (predicting a heatmap whose integral equals headcount) becomes relevant when:

- Counting standing passengers in extremely dense patches (mela/rush hour) where >50% occlusion makes boxing impossible.
- Validating total platform headcount against detection-based sums.

A hybrid architecture is the production ideal:

- **Zones with moderate density:** YOLO + ByteTrack + polygon counting (current approach).
- **Ultra-dense patches:** Add a density-estimation model (e.g., CSRNet-class) as a secondary signal for total count sanity check and heatmap visualization.

Do not replace detection with density estimation for line-crossing or safety-zone logic — density maps do not provide individual trajectories.

---

## Appendix: Key References

- Ultralytics YOLO11 docs: https://docs.ultralytics.com/models/yolo11/
- YOLO11 vs YOLOv8 comparison: https://docs.ultralytics.com/compare/yolo11-vs-yolov8/
- CrowdHuman dataset: https://www.crowdhuman.org/
- WiderPerson dataset: http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/
- NVIDIA DeepStream SDK: https://docs.nvidia.com/metropolis/deepstream/
- DeepStream-Yolo (community): https://github.com/marcoslucianops/DeepStream-Yolo/
- Ultralytics DeepStream guide: https://docs.ultralytics.com/guides/deepstream-nvidia-jetson/
- PAMELA-UANDES boarding/alighting dataset (Sensors, 2020)
- Metro Platform head-count dataset (627 images, metro surveillance)
- Current project demo report: `docs/FINAL_DEMO_REPORT.md`
- Current architecture: `docs/SYSTEM_ARCHITECTURE.md`
