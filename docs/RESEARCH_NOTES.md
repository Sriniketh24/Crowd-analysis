# Research Notes — AI-Based Crowd Analysis and Passenger Counting in Railway Stations

> Background research and technical justification for the MVP. This document explains
> *what* we are building, *why* the chosen approach makes sense, and the real-world
> constraints implementers must respect. Read this before `PROJECT_PLAN.md` and
> `SYSTEM_ARCHITECTURE.md`.

---

## 1. What This Project Entails

We are building a **practical computer-vision MVP** that turns ordinary railway-station
CCTV feeds into actionable operational intelligence. From one or more video streams the
system must:

- **Detect** every person in each frame.
- **Track** each person across frames with a stable ID (so the same person is not
  recounted).
- **Count** passengers crossing **virtual lines** (e.g., platform entry/exit, gate lines)
  to measure directional flow (in vs. out).
- **Count** passengers inside configured **zones** (e.g., a platform segment, a waiting
  area, a concourse) to measure occupancy.
- **Detect crowding** when a zone's occupancy (or density) crosses a configurable
  threshold, and raise **alerts**.
- **Generate real-time analytics** — live counts, flow rates, occupancy trends, dwell
  patterns — surfaced via a **dashboard** and an **API**, and persisted to **SQLite**.

The output supports three stakeholder needs called out in the project goal:

| Need | What the system provides |
|------|--------------------------|
| **Passenger flow management** | Live in/out counts, directional flow rates, congestion alerts |
| **Operational planning** | Historical occupancy and flow trends for staffing/scheduling |
| **Station safety** | Real-time crowd-density alerts to prevent dangerous overcrowding |

**Explicit non-goals for the MVP:** facial recognition, individual re-identification
across cameras, biometric identity, fare evasion detection, or any form of person
identification. The system counts and tracks *anonymous* moving objects only.

---

## 2. Core Computer Vision Tasks

The pipeline decomposes into four well-bounded CV problems. Keeping them as separate,
testable modules (per `AGENTS.md`) is deliberate.

### 2.1 Person Detection
Locate people in each frame as bounding boxes. We use a **pretrained YOLO model**
(`person` / COCO class 0) — no custom training for the MVP. Detection runs per frame (or
on a sampled subset of frames for performance) and emits `[x1, y1, x2, y2, confidence]`
per detection.

### 2.2 Multi-Object Tracking (MOT)
Associate detections across consecutive frames so each person receives a **persistent
track ID**. Tracking is what makes *counting* correct: without stable IDs we would
recount the same person every frame. We use **ByteTrack** (default) or **BoT-SORT**
(when appearance features help through occlusion) — both ship inside the `supervision`
library and integrate cleanly with Ultralytics YOLO.

### 2.3 Line-Crossing Counting
A **virtual line** is a configured 2D segment. When a tracked person's trajectory crosses
it, we increment a directional counter (in/out determined by which side of the line the
centroid moved from→to). This gives **flow** (people per minute through a gate/door).

### 2.4 Zone Occupancy & Crowd Detection
A **zone** is a configured polygon. We count how many active track centroids fall inside
it at each tick. Crowding is declared when occupancy (or occupancy ÷ zone area = density)
exceeds a configured threshold, optionally with hysteresis/dwell time to avoid flickering
alerts. This gives **density** and **safety alerts**.

> Detection answers "where are people?", tracking answers "which person is which over
> time?", and lines/zones convert tracks into the business metrics operators care about.

---

## 3. Real-World Railway Station Challenges

Railway CCTV is one of the harder environments for crowd CV. Implementers must design for:

- **Dense crowds & heavy occlusion** — peak hours produce overlapping bodies; detectors
  miss heavily occluded people and trackers drop/swap IDs. Counting accuracy degrades
  fastest exactly when it matters most.
- **Variable camera angles & heights** — overhead, oblique, and near-horizontal mounts
  all exist. A person's pixel size and the meaning of a "line" differ per camera, so
  **all geometry must be per-camera config**, never hardcoded.
- **Perspective & scale variation** — people far from the camera are tiny and low
  confidence; people near it are large. Small-object detection and per-zone calibration
  matter.
- **Lighting & environment** — day/night cycles, tunnel-mouth glare, fluorescent flicker,
  shadows, reflections on wet/polished floors, sun through skylights.
- **Motion blur & low frame rate** — older CCTV may be low-FPS or compressed, hurting
  tracking continuity.
- **Visual clutter** — luggage, trolleys, pillars, signage, benches, and trains entering/
  leaving the frame create false positives and occlusions.
- **Camera artifacts** — fisheye/wide-angle distortion, fixed text/timestamp overlays,
  bad white balance.
- **Throughput at the edge** — stations have many cameras; production must process many
  streams in near real time, which strongly shapes the future architecture (see §6).
- **Class confusion** — staff, reflections, and posters of people can be misdetected;
  tuning confidence thresholds and zones reduces this.

**Implication:** the MVP must be *robust and configurable*, not perfectly accurate. We
optimize for a working, tunable system on sample video first, then harden.

---

## 4. Recommended Approach

A staged, MVP-first strategy that matches `AGENTS.md`:

1. **Start with a pretrained YOLO person detector.** No dataset collection, no training —
   immediate results on public/sample railway-station videos.
2. **Add ByteTrack/BoT-SORT tracking** via `supervision` for stable IDs.
3. **Layer `supervision` line and polygon-zone primitives** for counting and occupancy,
   driven entirely by **config files** (cameras, lines, zones, thresholds).
4. **Compute analytics** (in/out counts, occupancy, flow rate, crowd state) on a tick loop.
5. **Persist** events, periodic counts, and alerts to **SQLite** via **SQLAlchemy**.
6. **Expose** live state and history through a **FastAPI** service and a **Streamlit**
   dashboard.
7. **Test** counting, line-crossing, and alert logic with **Pytest** using synthetic/
   recorded tracks (deterministic, no GPU needed).
8. **Iterate**: tune thresholds and detector size per camera; only consider custom
   training or DeepStream (§6) once the MVP is validated.

Design principles throughout: small, single-responsibility modules; config over
hardcoding; deterministic, unit-testable analytics separated from the (harder-to-test)
detection/tracking layer.

---

## 5. Why YOLO + ByteTrack/BoT-SORT Is a Good MVP Choice

**YOLO (Ultralytics) for detection**
- **Real-time** single-stage detector — strong speed/accuracy trade-off suitable for
  live video, even on modest hardware (smaller variants run on CPU; `n`/`s` scale up to
  `m`/`l`/`x` on GPU).
- **Pretrained on COCO**, which includes the `person` class — *zero training* to get a
  working passenger detector, exactly the MVP-first rule in `AGENTS.md`.
- **First-class `supervision` integration** — detections flow directly into trackers,
  line counters, and zone primitives with minimal glue code.
- **Easily upgradable** — swap model size or weights without changing pipeline structure;
  later fine-tune on station footage if needed.

**ByteTrack / BoT-SORT for tracking**
- **ByteTrack** associates *both* high- and low-confidence detections, recovering people
  during partial occlusion — a major win in dense station crowds. It is **detection-only**
  (no appearance model), so it is **fast and lightweight**, ideal for an MVP.
- **BoT-SORT** adds appearance (ReID) features and camera-motion compensation, improving
  ID stability through longer occlusions at higher compute cost — a drop-in upgrade when
  ByteTrack's ID switches become the bottleneck.
- Both are **bundled/supported in `supervision`**, so we can start with ByteTrack and
  switch to BoT-SORT via config without rewriting the pipeline.

**Why this combo for an MVP specifically:** maximum capability for minimum setup — no
training data, no labeling, no bespoke infrastructure — while leaving clean upgrade paths
(bigger model, BoT-SORT, fine-tuning, DeepStream) for later.

---

## 6. Why NVIDIA DeepStream Is a Strong Future Production Upgrade

The MVP (Python + OpenCV + Ultralytics) is perfect for development and single-stream
demos but is not optimized for running **dozens of station cameras** continuously. For
production, **NVIDIA DeepStream** is the natural upgrade:

- **GPU-accelerated, multi-stream pipeline** built on GStreamer + TensorRT — decode,
  infer, track, and analyze **many concurrent camera feeds** on one GPU.
- **Hardware-accelerated video decode** (NVDEC) frees CPU and raises throughput.
- **TensorRT-optimized inference** — the same YOLO model, quantized and optimized, runs
  far faster.
- **Built-in tracker plugins** (including NvDCF and others) and analytics primitives
  (lines/zones) mirror what we build manually in the MVP, easing migration.
- **Edge-ready** — runs on NVIDIA Jetson devices for on-site deployment, avoiding
  streaming raw footage off-premises (a privacy and bandwidth win).
- **Scales horizontally** for station- or network-wide rollout.

**Migration path:** the MVP's *config schema and analytics semantics* (cameras, lines,
zones, thresholds, event/DB model) are designed to carry over. DeepStream replaces the
detection/decode/tracking core; our analytics, storage, API, and dashboard concepts
remain. We deliberately do **not** adopt DeepStream now — it adds GPU/SDK complexity that
would slow the MVP — but we keep the door open.

---

## 7. Privacy & Safety Considerations

Railway CCTV involves the public, so privacy is a first-class requirement (and is
mandated by `AGENTS.md`):

- **Count, don't identify.** The system performs anonymous person detection and tracking
  only. **No facial recognition, no biometrics, no cross-camera re-identification, no
  identity linkage.** Track IDs are ephemeral per-camera handles, not identities.
- **No PII storage by default.** Persist only **aggregate analytics** (counts, occupancy,
  flow rates, alerts) — never personally identifying footage. Do not store raw frames or
  crops of individuals unless **explicitly approved** for a specific, time-boxed purpose.
- **No private CCTV / no invented data.** Per `AGENTS.md`, use only **public or sample
  videos** for demos; do not assume access to live station cameras or fabricate datasets.
- **Data minimization & retention.** Store the minimum needed for analytics; define a
  retention/auto-purge policy for event rows. Prefer **on-premises/edge** processing
  (reinforced by the DeepStream/Jetson path) so video need not leave the station.
- **Transparency & lawful basis.** Real deployments require signage, a documented lawful
  basis, and compliance with local data-protection law and railway authority policy.
- **Safety framing, not surveillance.** Position outputs as crowd-safety and flow tools.
  Avoid features enabling individual tracking or profiling.
- **Bias & accuracy honesty.** Detection accuracy drops in dense/occluded/low-light
  scenes; report counts as **estimates** with known limits, and never use them as the
  sole basis for safety-critical automated action without human oversight.
- **Security.** Restrict dashboard/API access; do not expose raw streams publicly; keep
  config (camera locations/URLs) out of public repos.

---

## 8. Key References / Terms (for implementers)

- **YOLO (Ultralytics)** — real-time object detector; COCO `person` = class 0.
- **ByteTrack** — detection-association multi-object tracker; uses low-confidence boxes.
- **BoT-SORT** — appearance-aware tracker with camera-motion compensation.
- **`supervision`** — Roboflow library providing Detections, trackers, `LineZone`
  (line counting), and `PolygonZone` (zone occupancy) utilities.
- **MOT** — Multi-Object Tracking.
- **NVIDIA DeepStream** — GPU/GStreamer SDK for scalable multi-stream video analytics.
- **TensorRT** — NVIDIA inference optimizer/runtime.

See `PROJECT_PLAN.md` for scope/schedule and `SYSTEM_ARCHITECTURE.md` for the concrete
module, config, API, and database design.
