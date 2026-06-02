# Project Plan — AI-Based Crowd Analysis and Passenger Counting in Railway Stations

> Scope, schedule, metrics, and demo plan for the MVP. Reads alongside
> `RESEARCH_NOTES.md` (the *why*) and `SYSTEM_ARCHITECTURE.md` (the *how*).
> Guiding rule from `AGENTS.md`: **build a working MVP first**, config over hardcoding,
> small testable modules, pretrained YOLO before any custom training.

---

## 1. MVP Scope (must-have)

The MVP is a **single-machine, offline-and-live-capable** crowd-analytics system that
runs on **sample/public railway-station video** (no private CCTV).

**In scope**
- **Video ingestion** from a file or a stream URL (RTSP/HTTP) for one camera at a time.
- **Person detection** with pretrained Ultralytics YOLO (COCO `person` class).
- **Multi-object tracking** with ByteTrack (BoT-SORT switchable via config).
- **Line-crossing counting** — directional in/out counts per configured virtual line.
- **Zone occupancy counting** — live count of people inside each configured polygon zone.
- **Crowd detection & alerts** — threshold-based alert when a zone exceeds its occupancy/
  density limit (with dwell/hysteresis to avoid flicker).
- **Config-driven setup** — cameras, source, lines, zones, and thresholds live in config
  files; **nothing camera-specific is hardcoded**.
- **Persistence** — counts, zone occupancy snapshots, line-crossing events, and alerts
  stored in **SQLite** via **SQLAlchemy**.
- **API** — **FastAPI** endpoints exposing live state and historical analytics.
- **Dashboard** — **Streamlit** app showing the annotated video/frames, live counts,
  occupancy, flow rate, and active alerts.
- **Tests** — **Pytest** unit tests for zone counting, line crossing, and alert logic
  (deterministic, using synthetic tracks; no GPU required).
- **Annotated output** — overlay of boxes, IDs, lines, zones, and counters for the demo.

**Out of scope for MVP** (see Stretch Goals / future): multi-camera fusion, person
re-identification, facial recognition, custom-trained models, NVIDIA DeepStream,
cloud deployment, authentication/RBAC, density heatmaps.

**Definition of Done (MVP)**
- Runs end-to-end on a sample video with a documented config.
- Produces stable in/out and occupancy numbers and fires a crowd alert on a busy clip.
- Dashboard + API both reflect live and stored analytics.
- All counting/alert unit tests pass; build/run instructions in README.

---

## 2. Stretch Goals (nice-to-have, post-MVP)

Prioritized, roughly in order of value:

1. **Multi-camera support** — process several feeds concurrently; per-camera analytics.
2. **Crowd density heatmap** — spatial occupancy visualization over the frame.
3. **BoT-SORT + ReID tuning** — reduce ID switches in dense/occluded scenes.
4. **Dwell-time analytics** — average time spent per zone (waiting-area insight).
5. **Flow-rate trends & forecasting** — short-horizon prediction for proactive staffing.
6. **Configurable alert delivery** — webhook/email/Slack on crowd alerts.
7. **Detector fine-tuning** — train/adapt YOLO on station-like footage for accuracy.
8. **Performance hardening** — frame sampling, batching, GPU/TensorRT, async pipeline.
9. **Auth & roles** on the dashboard/API for operational deployment.
10. **NVIDIA DeepStream production pipeline** — GPU multi-stream edge deployment
    (see `RESEARCH_NOTES.md` §6).

---

## 3. Week-by-Week Development Plan

A suggested **6-week** plan (compress/expand to fit the internship). Each week ends with
a runnable, demoable increment.

### Week 1 — Foundations & detection
- Set up repo structure, Python 3.11 env, dependencies, and `config/` skeleton.
- Implement video ingestion (file + stream URL) and a frame loop.
- Integrate pretrained YOLO; render detection boxes on sample video.
- **Deliverable:** "video in → boxes out" demo on a sample clip.

### Week 2 — Tracking & geometry config
- Add ByteTrack via `supervision`; render stable track IDs.
- Define config schema for cameras, lines, and zones; load and validate it.
- Build a small helper to draw lines/zones from config onto frames for calibration.
- **Deliverable:** tracked IDs + configured lines/zones overlaid on video.

### Week 3 — Counting & crowd logic (core analytics)
- Implement line-crossing in/out counters (`supervision` LineZone).
- Implement zone occupancy counting (`supervision` PolygonZone).
- Implement crowd-detection thresholds with dwell/hysteresis and alert generation.
- Write **Pytest** unit tests for counting, line crossing, and alerts using synthetic
  tracks. **(Tests are required this week, per `AGENTS.md`.)**
- **Deliverable:** correct counts + alerts, all analytics unit tests green.

### Week 4 — Persistence & API
- Define SQLAlchemy models + SQLite schema (events, occupancy snapshots, alerts).
- Persist analytics on the tick loop; add a repository/service layer.
- Build FastAPI endpoints for live state and historical queries.
- **Deliverable:** running API serving live + stored analytics; DB populated from a run.

### Week 5 — Dashboard & integration
- Build the Streamlit dashboard: annotated frames, live counters, occupancy, flow rate,
  active alerts, and simple historical charts (reading from the API/DB).
- End-to-end wiring: ingestion → analytics → DB → API → dashboard.
- Tune thresholds/model size on the sample video.
- **Deliverable:** full MVP demo path working on one camera.

### Week 6 — Hardening, docs & demo
- Robustness pass (stream reconnect, config validation errors, empty-frame handling).
- Performance tuning (frame sampling, model size) and accuracy spot-check vs. manual count.
- Finalize README/run docs; record/prepare the demo (§5).
- Optional: begin one stretch goal (e.g., multi-camera or heatmap) if time allows.
- **Deliverable:** polished MVP, passing tests, demo-ready, documented.

> If the internship is shorter, the minimum viable slice is **Weeks 1–3 + a thin
> dashboard**: detect → track → count/occupancy/alerts with overlay. Persistence, API,
> and the full dashboard can follow.

---

## 4. Evaluation Metrics

Measured on **sample/public video** with manually labeled ground truth for short clips.

### 4.1 Accuracy
- **Counting accuracy (line crossing):** compare automated in/out totals to a manual
  hand-count on a clip. Report **MAE** and **% error** per direction.
- **Occupancy accuracy:** at sampled timestamps, compare zone count to manual count;
  report mean absolute occupancy error.
- **Alert correctness:** for staged crowded/uncrowded clips, report whether alerts fire
  at the right times — **precision/recall** of alert events (false alarms vs. misses).
- **Detection sanity (optional):** spot-check detection precision/recall on sampled frames.
- **Tracking stability (optional):** count **ID switches** per minute as a tracker-quality
  proxy (lower is better).

### 4.2 Performance
- **Throughput (FPS):** processed frames per second on the target dev hardware (note CPU
  vs. GPU and YOLO model size).
- **Latency:** time from frame capture to updated analytics/alert.
- **Resource use:** CPU/GPU/RAM during a sustained run.

### 4.3 Software quality
- **Unit-test pass rate** for counting, line-crossing, and alert logic (must be green).
- **Test coverage** of the analytics modules.
- **Config-driven check:** confirm zero camera-specific hardcoded values (lines/zones/
  thresholds all from config).

### 4.4 Targets (initial, tune as needed)
| Metric | MVP target |
|--------|-----------|
| Line-count error (uncrowded clip) | ≤ 10% |
| Occupancy error (sampled) | ≤ 15% |
| Alert precision / recall (staged clips) | ≥ 0.9 / ≥ 0.9 |
| Throughput (dev hardware) | near real-time on sampled frames |
| Analytics unit tests | 100% pass |

> Targets are MVP guidance, not production SLAs; accuracy will be lower in dense/occluded
> footage (see `RESEARCH_NOTES.md` §3). Report numbers honestly with the clip conditions.

---

## 5. Demo Plan

**Goal:** show, in a few minutes, that the system detects, tracks, counts, detects
crowding, and surfaces real-time analytics on a railway-station-style video.

**Setup**
- Use a **public/sample** station or pedestrian video (per `AGENTS.md` — no private CCTV).
- Prepare a `config/` with one camera, 1–2 virtual lines (entry/exit), 1–2 zones
  (platform/waiting area), and a crowd threshold tuned to trigger on a busy segment.

**Demo flow**
1. **Live annotated video** — boxes + track IDs + drawn lines/zones + on-frame counters.
2. **Line counting** — narrate in/out counts incrementing as people cross the gate line.
3. **Zone occupancy** — show the live count inside a zone changing as people enter/leave.
4. **Crowd alert** — play a busy segment; show the alert firing when the threshold is
   crossed (and clearing afterward).
5. **Dashboard** — switch to Streamlit: live counters, occupancy, flow rate, active
   alerts, and a historical chart.
6. **API** — hit a FastAPI endpoint (e.g., live state + recent alerts) to show
   integration-readiness.
7. **Persistence** — show rows written to SQLite (events, occupancy snapshots, alerts).
8. **Tests** — run `pytest` to show counting/line/alert logic is verified.

**Backups & framing**
- Pre-record a clean run as a fallback in case of live failure.
- State limitations honestly (dense-crowd accuracy, single camera) and the upgrade path
  (BoT-SORT, multi-camera, DeepStream) from `RESEARCH_NOTES.md`.
- Emphasize the **privacy stance**: anonymous counting only, no PII stored, sample video.

**Deliverables for the demo**
- Sample video(s), a committed demo `config`, a populated SQLite DB, the running
  API + dashboard, and a short README with exact run commands.

---

## 6. Indian Railway CCTV Production Roadmap

> Full research, gap analysis, and phased plan: **`docs/PRODUCTION_RESEARCH_ROADMAP.md`**

This project targets **Indian railway platform CCTV** for passenger counting, crowd
monitoring, and station safety — not a generic crowd demo. The MVP is validated on
public sample video only; **it is not production-ready** for Railway Department deployment.

### Production target (summary)

| Phase | Goal | Key deliverables |
|-------|------|------------------|
| **MVP (current)** | Prove pipeline mechanics | Detection, tracking, zones, lines, alerts, API, dashboard |
| **Baseline eval** | Measure on approved platform CCTV | Manual vs auto counts; FPS on GPU; validation report |
| **Pilot hardening** | Live multi-camera RTSP | Reconnect, health checks, per-camera workers, `yolo11s` |
| **Fine-tuning** | Domain-adapted detector | 3K–5K labeled platform frames; held-out test metrics |
| **Pilot deployment** | 2–4 live cameras at one station | Secure API, alert dispatch, 2-week operator trial |
| **Production scale** | 10–30+ cameras | TensorRT/DeepStream, PostgreSQL, organizational sign-off |

### Model and tracking direction

- **Detection:** YOLO11 (migrate default from `yolov8n` → `yolo11s` for pilot).
- **Tracking:** ByteTrack for initial pilot (static CCTV, throughput priority); BoT-SORT
  if ID-switch rate is unacceptable on dense clips.
- **Fine-tuning:** Required before production accuracy claims; needs labeled bounding
  boxes from approved Indian platform footage (unlabeled video alone is insufficient).

### Data required from Railway authority

1. Written approval for analytics on live CCTV (no facial recognition).
2. 3–5 representative platform clips (peak/off-peak, day/night, multiple camera angles).
3. RTSP access details for 2–4 pilot cameras (credentials via secrets, not git).
4. Station camera inventory: resolution, mount type, coverage map.
5. Operator alert workflow (who receives CRITICAL crowd alerts, how).

### Production gaps (blockers)

- No validated accuracy on Indian platform CCTV.
- No labeled fine-tuning dataset.
- No live RTSP reconnect / multi-camera orchestration.
- No GPU/TensorRT deployment path.
- No organizational privacy sign-off or operational SLA.

### Next implementation agent focus

1. Switch default model to `yolo11s.pt`; add CUDA device path.
2. Harden `StreamReader`: reconnect, backoff, FPS cap, health events.
3. Multi-camera worker from `configs/cameras.yaml`.
4. Baseline evaluation script (manual count CSV → auto count comparison).
5. Frame extraction + annotation workflow docs for boss-provided footage.

---

## 7. Head Detection Upgrade Using RPEE-Heads

> Full research, dataset review, model strategy, and fine-tuning plan:
> **[`docs/HEAD_DETECTION_RESEARCH_PLAN.md`](HEAD_DETECTION_RESEARCH_PLAN.md)**
>
> **Status: research/planning only — no training has happened, no source code changed.**
> This is **not** a production-readiness claim.

### Why (boss feedback)

Full-body passenger detection becomes unreliable on Indian railway platform CCTV when
passengers are **far away, partially occluded, low-resolution, or in dense crowds** — the
whole body is often not visible. **Heads usually remain visible** above the crowd, so a
head detector should detect, track, and count passengers more reliably in these views.
The final web app will **compare both**: current full-body person detection vs. a
fine-tuned head detector.

### Primary dataset: RPEE-Heads

**Railway Platforms and Event Entrances – Heads** — the best available public match for
this problem.

| Property | Value |
|---|---|
| Images / heads / videos | **1,886 / 109,913 / 66** |
| Scenes | Railway platforms (Düsseldorf), music-concert + event entrances |
| Density | avg **56.2** heads/image, max **270** |
| Annotation format | **Already YOLO** (`<class x y w h>` normalized) |
| Split | official ~70/15/15 (train 1,346 / val 246 / test 294) |
| License | **CC BY-SA 4.0** |
| Download | https://doi.org/10.34735/ped.2024.2 · paper https://arxiv.org/abs/2411.18164 |

**Limitation:** railway-platform-relevant but **not Indian-railway-specific** — there is a
domain gap; Indian footage is still required for final validation.

### Dataset strategy

- **Primary:** RPEE-Heads (on-domain, YOLO-ready, properly licensed).
- **Supplement only if a weakness appears:** **CrowdHuman** head boxes (occlusion),
  **SCUT-HEAD** (extra head variety), **CroHD** (head-tracking eval), **Brainwash**
  (surveillance heads — only if a legit mirror exists). Avoid point-only crowd-counting
  sets for detector training. **No synthetic data.**

### Model & tracking direction

- **Baseline (compare against):** current **YOLO11n** full-body `person` pipeline.
- **Head model — laptop/demo:** fine-tune **YOLO11n** on RPEE-Heads → `head_best_n.pt`.
- **Head model — accuracy:** fine-tune **YOLO11s** (m on GPU) at larger `imgsz` (960–1280)
  for tiny/distant heads → `head_best_s.pt`.
- **Stay on YOLO11** (keep `yolov8n` only as fallback).
- **Tracker:** ByteTrack for the demo; **BoT-SORT** if head ID-switch rate is too high.

### Metrics

Detector: mAP@0.5, mAP@0.5:0.95, precision/recall/F1, **AP by head size**, FPS (RPEE-Heads
test split). Application (body vs head, same clips): counting MAE/% error vs manual count,
alert precision/recall, tracking ID switches/min, passengers recovered that the body model
misses.

### Web app comparison

Same clip, side by side: (1) original video, (2) body-detection output, (3) head-detection
output, (4) counts/analytics table vs manual ground truth, (5) plain-English explanation of
where head detection helps. Both pipelines reuse the **existing SQLite schema** (different
session/model tag), so no new storage design is needed.

### Required code changes (later — not done yet)

1. RPEE-Heads download docs + `scripts/prepare_rpee_heads.py` converter into a YOLO layout.
2. `configs/train_rpee_heads.yaml` (single class `head`); reuse the train guard in
   `scripts/train_yolo_indian_platform.py` (still refuses to train without real labels).
3. Head **reference-point = box center** option in `ZoneManager` / `LineManager`
   (bodies stay bottom-center).
4. Second detector/tracker instance + dashboard comparison panel.

### Blockers / honesty

- No Indian-platform validation data yet → RPEE-Heads numbers are **proxy** numbers.
- Head false positives (bags/helmets/signage) and missed back-row heads remain.
- **Do not fake fine-tuning.** If no labeled head data is accessible, fall back to the
  annotation workflow and document the dataset as pending.
