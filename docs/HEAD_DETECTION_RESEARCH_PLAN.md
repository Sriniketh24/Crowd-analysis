# Head Detection Research Plan

> Research and production-planning document for upgrading the railway crowd-analytics
> system from **full-body passenger detection** to **head detection** for far-away,
> occluded, low-resolution, and crowded Indian railway platform CCTV.
>
> **Status:** research + planning only. **No training has happened.** No source code has
> been changed by this document. Fine-tuning requires a real labeled head dataset
> (primary: RPEE-Heads). This is **not** a production-readiness claim.
>
> Reads alongside [`PROJECT_PLAN.md`](PROJECT_PLAN.md) (scope/schedule),
> [`PRODUCTION_RESEARCH_ROADMAP.md`](PRODUCTION_RESEARCH_ROADMAP.md) (Indian-railway gaps),
> [`../data/DATASET_NOTES.md`](../data/DATASET_NOTES.md) (data inventory + licenses), and
> [`ANNOTATION_WORKFLOW.md`](ANNOTATION_WORKFLOW.md) (labeling pipeline).

---

## Why Head Detection Is Needed

The current system detects **whole people** (the COCO `person` class). That works well
when passengers are **close to the camera and fully visible** — a clear head-to-foot
bounding box is easy for YOLO to find, and the tracker can follow it.

Indian railway platform CCTV is usually the opposite situation:

- **Cameras are mounted high and far away.** Passengers at the far end of a long platform
  are only a few dozen pixels tall. The full body is too small and blurry to detect
  reliably, so distant passengers get **missed entirely** and the count reads low.
- **Crowds occlude bodies.** On a busy platform, people stand shoulder-to-shoulder. Each
  person's torso and legs are hidden behind the person in front. A full-body detector
  needs to see most of the body; when it can't, it either misses the person or merges
  several people into one box.
- **People sit, lean, and carry luggage.** Seated passengers, people behind railings,
  trolleys, and bags break the "upright full body" shape the detector expects.
- **Low resolution and motion blur.** Compressed CCTV streams degrade fine body detail,
  but a head remains a compact, consistently-shaped blob.

The key insight: **in almost every one of these failure cases, the head is still
visible.** Heads stick up above the crowd, they are rarely fully occluded, and they keep
a stable round/oval shape regardless of distance, posture, or luggage. Counting heads is
the standard approach in academic **crowd counting** precisely because **one head ≈ one
person**, even when bodies overlap.

So the plan is to add a **head detector** that:
- finds passengers the full-body detector misses in dense/distant views,
- gives a more accurate count when bodies overlap,
- and degrades more gracefully as the scene gets crowded.

We are **not deleting** the full-body detector. The web app will run **both** and compare
them, so we can show honestly where head detection helps and where it doesn't.

---

## Current Full-Body Baseline

The existing pipeline (unchanged by this plan) is:

```
StreamReader → FrameProcessor → Detector/Tracker (YOLO person) → ZoneManager
            → LineManager → CrowdAnalyzer → AnalyticsLogger (SQLite) → Streamlit/FastAPI
```

Concretely, from the code:

- **Detection** — [`src/vision/detector.py`](../src/vision/detector.py). `Detector` wraps
  Ultralytics YOLO and filters to `person_class_id = 0` (COCO `person`). Default weights
  are `yolo11n.pt`, with `yolo11s.pt` as the accuracy option and `yolov8n.pt` as legacy
  fallback (see `DEFAULT_MODEL_CANDIDATES` and [`configs/app.yaml`](../configs/app.yaml)).
  It already supports a fine-tuned model at `models/fine_tuned/best.pt` via
  `use_fine_tuned_if_available`.
- **Tracking** — [`src/vision/tracker.py`](../src/vision/tracker.py). `Tracker` runs
  Ultralytics `.track()` with **ByteTrack** (default) or **BoT-SORT**, assigning stable
  `track_id`s, still filtered to the `person` class.
- **Frame orchestration** — [`src/video/frame_processor.py`](../src/video/frame_processor.py).
  `FrameProcessor.process()` runs detection/tracking, then feeds normalized detections
  into zones, lines, and the crowd analyzer, returning a `FrameProcessResult`.
- **Analytics**:
  - **Zones** — [`src/vision/zone_manager.py`](../src/vision/zone_manager.py) counts how
    many tracks fall inside each polygon zone, using the **bottom-center** of each bbox.
  - **Lines** — [`src/vision/line_counter.py`](../src/vision/line_counter.py) counts
    directional IN/OUT crossings per virtual line, also using bbox bottom-center.
  - **Crowd alerts** — `CrowdAnalyzer` raises NORMAL/WARNING/CRITICAL per zone from
    occupancy thresholds in [`configs/thresholds.yaml`](../configs/thresholds.yaml).
- **Persistence + UI** — `AnalyticsLogger` writes sessions, per-frame counts, zone
  occupancy, line events, and alerts to SQLite; the Streamlit app
  ([`src/dashboard/streamlit_app.py`](../src/dashboard/streamlit_app.py)) and FastAPI
  routes read it back.

**Why this is the right baseline to compare against:** it is the *current shipping
behavior*. The boss's question is "does head detection count better than what we have
now?" — so the honest comparison is head detection vs. **this exact full-body pipeline**,
on the **same** video, zones, and lines.

---

## Head-Based Detection Strategy

The head detector is a **drop-in swap of the detection target**, reusing the entire
analytics stack. Heads are just a different class of box; everything downstream already
works on generic bounding boxes.

How the new head detector will work:

1. **Train/fine-tune YOLO on head bounding boxes** (single class: `head`) using
   RPEE-Heads (see [Fine-Tuning Plan](#fine-tuning-plan)). Output: a head-detector weight
   file, e.g. `models/fine_tuned/head_best.pt`.
2. **Detect heads instead of full people.** The head model emits one box per visible head.
   In code terms this means a second detector/tracker configured with the head class
   (the `person_class_id` filter becomes the head class id, which for a single-class head
   model is `0`). No change to the `NormalizedDetection` schema is required — a head box
   is still `(x1, y1, x2, y2)` + confidence + track_id.
3. **Assign tracking IDs to heads.** Run the same Ultralytics `.track()` (ByteTrack /
   BoT-SORT) on head detections, so each head gets a stable `track_id` across frames —
   exactly as bodies do today.
4. **Count heads in zones.** `ZoneManager` already counts any boxes whose reference point
   is inside a polygon. For heads, the reference point should be the **box center**
   (head centroid) rather than bottom-center, because a head's "feet" are meaningless.
   This is a small, well-scoped change (a reference-point option), not a rewrite.
5. **Count head tracks crossing lines.** `LineManager` counts directional crossings of a
   track's reference point. Same reference-point note applies (use head center).
6. **Log the same analytics as before.** Zone occupancy, line IN/OUT, crowd alerts,
   unique-track counts, FPS — all flow through the existing `AnalyticsLogger` and DB
   schema unchanged. The only difference is the *source of the boxes*.

Net effect: heads ride the existing rails. The engineering work is (a) producing the head
model, and (b) a reference-point option + a second pipeline instance for comparison.

---

## Primary Dataset: RPEE-Heads

**RPEE-Heads** = **Railway Platforms and Event Entrances – Heads**.

A benchmark dataset for **pedestrian head detection in crowded videos**, purpose-built for
exactly the scenario the boss described: dense crowds where bodies are occluded but heads
are visible.

### What it is / why it is preferred

It is the **only** well-known public dataset that is **railway-platform-focused** *and*
ships **head bounding boxes** *and* is already in **YOLO annotation format**. That makes it
the best available proxy for Indian platform CCTV without any Indian-specific data being
public.

### Size and annotation details

| Property | Value |
|---|---|
| Images | **1,886** |
| Annotated heads | **109,913** |
| Video recordings | **66** |
| Avg heads / image | **56.2** (dense) |
| Max heads / image | **270** |
| Scene split | Railway platforms **314**, music-concert entrances **993**, controlled event-entrance experiments **579** |
| Resolution | High-res, ~1,920×1,080 up to 4,000×3,000 |
| Annotation format | Per-image **`.txt`, one row per head: `<class, x_center, y_center, w, h>` normalized 0–1** → **this is exactly YOLO format** |
| Official split | ~70/15/15 → train 1,346 imgs / val 246 / test 294 |
| License | **CC BY-SA 4.0** (attribution + share-alike) |

### Railway-platform relevance

The railway-platform images come from **Merkur Spiel-Arena / Messe Nord train station,
Düsseldorf, Germany** — real platform CCTV-style overhead crowd scenes with the same
"heads above a dense crowd, bodies occluded" structure as an Indian platform at rush hour.
The event-entrance scenes add even denser bottleneck crowds, which is useful for the
worst-case "fully packed platform" condition.

### Limitations (because it is not Indian-railway-specific)

- **Not Indian footage.** Crowd appearance, clothing, luggage (large bags/trolleys),
  headwear, hairstyles, platform geometry, signage, and lighting differ from Indian
  stations. A model trained only on RPEE-Heads will have a **domain gap**.
- **German/European station + concert crowds**, not Mumbai/Delhi rush-hour density
  patterns.
- **Camera angles/optics** (resolution, mount height, lens) differ from the specific
  Indian CCTV cameras eventually targeted.

→ RPEE-Heads gets us a **strong head detector** and a **fair body-vs-head comparison**.
It does **not** by itself prove accuracy on Indian Railway cameras — that still requires
**labeled Indian platform footage** for validation (see [Risks](#risks-and-limitations)).

### Download / access notes

- **Dataset DOI (primary download):** https://doi.org/10.34735/ped.2024.2 (PED data
  archive; dataset + pretrained models). Distributed under CC BY-SA 4.0.
- **Paper (arXiv):** https://arxiv.org/abs/2411.18164 ·
  HTML: https://arxiv.org/html/2411.18164v1
- **Paper (IEEE):** https://ieeexplore.ieee.org/document/10973050/
- If the DOI archive requires a manual/browser download (likely — it is a research data
  portal, not a CLI endpoint), **document the manual steps** and place the extracted data
  under `data/rpee_heads/` rather than faking an automatic fetch. A converter script
  (see [Fine-Tuning Plan](#fine-tuning-plan)) then arranges it into the project's
  `data/training_dataset/` YOLO layout.

---

## Backup and Supplemental Dataset Review

All rows below are **real, labeled** datasets unless noted. "Head boxes" = the dataset
ships actual head **bounding boxes** (not just dot/point annotations). Verify every
license on the official page before any redistribution or commercial use.

| Dataset | URL | Labeled head boxes? | Relevance to railway CCTV | License / access notes | Recommended use |
|---|---|---|---|---|---|
| **CrowdHuman** | https://www.crowdhuman.org/ · https://huggingface.co/datasets/sshao0516/CrowdHuman | **Yes** — head + visible-body + full-body boxes | High for **occlusion/crowding**; web images not CCTV, but very dense, heavy occlusion | Free for academic/research; verify terms before commercial use | **Primary supplement** — adds occlusion robustness; mix in head boxes |
| **SCUT-HEAD** | https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release | **Yes** — head boxes (PartA classrooms, PartB internet), ~4.4k imgs / ~111k heads | Medium — overhead classroom crowds resemble surveillance head views; not platforms | Research use; VOC XML (convert to YOLO) | **Backup supplement** — extra head variety, surveillance-ish angles |
| **Brainwash** | (original Stanford/MPI page; mirror via https://paperswithcode.com/dataset/brainwash ) | **Yes** — head boxes from a café webcam (time-lapse) | Medium — true **fixed surveillance** camera, busy scene; single indoor scene only | **Access caveat:** original host has been **taken down over ethics/consent concerns**; only use if a legitimate mirror is available and terms allow | **Optional backup** — surveillance-style heads if obtainable |
| **FDST** (Fudan-ShanghaiTech) | https://github.com/sweetyy83/Lstn_fdst_dataset | Partial — video crowd-counting; head **points**, boxes derivable but not native bbox benchmark | Medium — real surveillance video, moderate density | Research use | Optional — video/temporal head data; needs conversion |
| **JHU-CROWD++** | http://www.crowd-counting.com/ | **No** native boxes — head **points/dots** (+ approx box/blur labels) | Medium — very diverse crowd-counting imagery, some dense | Research use, non-commercial leanings — verify | Reference / optional point-to-box conversion only |
| **CroHD** (Crowd of Heads, HeadHunter) | https://motchallenge.net/data/Head_Tracking_21/ | **Yes** — head boxes **+ track IDs** (MOT-style) | **High for tracking** — designed for head *tracking* in crowds, elevated cameras | Research use (MOTChallenge terms) | **Supplement for tracker eval** — head-track ID-switch benchmarking |
| **NWPU-Crowd** | https://www.crowdbenchmark.com/ · https://github.com/gjy3035/NWPU-Crowd-Sample-Code | Partial — head **points + boxes** for a large crowd-counting set | Medium — extremely diverse/dense, some boxes | Research use; large download | Optional — scale/density variety if more data needed |
| **Hollywood Heads** | https://www.di.ens.fr/willow/research/headdetection/ | **Yes** — head boxes (VOC) from movies | **Low** — cinematic close-ups, not crowds or CCTV | Research use | Skip for this use case (wrong domain) |

---

## Recommended Dataset Strategy

**Primary: RPEE-Heads.** It is railway-platform-focused, has 109,913 real head boxes,
ships in YOLO format, and is CC BY-SA 4.0. Train/fine-tune the head model on it first and
report the body-vs-head comparison using its official train/val/test split.

**Supplement only if needed, in this order:**
1. **CrowdHuman head boxes** — best for hardening against **occlusion** in very dense
   crowds (the exact rush-hour failure mode). Mix a sampled subset of head boxes into
   training if RPEE-Heads alone under-detects in the densest clips.
2. **SCUT-HEAD** — cheap extra head-box variety with surveillance-like overhead angles.
3. **Brainwash** — only if a legitimately-hosted copy is available (original host removed
   over consent concerns); adds true fixed-camera surveillance heads.
4. **CroHD** — not for training the detector primarily, but as a **head-tracking**
   evaluation set to measure ID switches in dense scenes.

**Why this order:** start with the most on-domain, fully-formatted, properly-licensed set
(RPEE-Heads); add data only to fix an observed weakness, preferring sets that target the
specific failure mode (occlusion → CrowdHuman; tracking → CroHD). Avoid point-only
crowd-counting sets (JHU-CROWD++, NWPU-Crowd, ShanghaiTech) for *detector* training since
converting dots to boxes is lossy. **Do not** add a dataset just because it exists — every
addition widens the domain gap and the license surface.

> Reminder (hard rule): **no synthetic data.** Skip GCC/GTA5/MOTSynth-style CGI sets, as
> already stated in [`../data/DATASET_NOTES.md`](../data/DATASET_NOTES.md).

---

## Model Recommendation

| Role | Recommendation | Why |
|---|---|---|
| **Baseline body model** | **YOLO11n** (current default) → optionally **YOLO11s** for the accuracy column | It is the shipping baseline; comparing against it answers the boss's real question. Already in repo (`yolo11n.pt`, `yolo11s.pt`). |
| **Head model to fine-tune (laptop demo)** | **YOLO11n** fine-tuned on RPEE-Heads heads | Smallest/fastest; runs on a CPU laptop for the side-by-side demo. Matches the existing default so the comparison isolates *target* (body vs head), not *model size*. |
| **Head model for best accuracy** | **YOLO11s** fine-tuned (optionally **YOLO11m** if a GPU is available) | More capacity helps **tiny/distant heads**, the hardest case. RPEE-Heads paper shows large detectors (YOLOv9 / RT-DETR ≈ 90% mAP) win on heads — capacity matters. |
| **YOLOv8 vs YOLO11** | **Use YOLO11**, keep `yolov8n` only as legacy fallback | YOLO11 is newer, generally better accuracy/latency, and is already the project default. No reason to fine-tune on the older v8 line. |
| **Tracker** | **ByteTrack** for the demo; **BoT-SORT** if head ID-switches are too high | ByteTrack is fast and already default; dense tiny heads cause ID switches, so BoT-SORT (with ReID) is the fallback. Evaluate on CroHD. |

**Recommended decision:**
- **Laptop demo / default:** fine-tune **YOLO11n** on RPEE-Heads → `head_best_n.pt`.
- **Accuracy variant:** fine-tune **YOLO11s** (or m on GPU) → `head_best_s.pt`.
- **Compare against:** the **YOLO11n full-body person** detector (current production
  default), on the same clips/zones/lines.

### Metrics to report

Detector quality (on RPEE-Heads test split, and later on any labeled Indian frames):
- **mAP@0.5** and **mAP@0.5:0.95**, **precision**, **recall**, **F1** for the head class.
- **AP by head size** (small / medium / large) — small-head AP is the whole point.
- **Inference FPS** on the demo hardware (CPU laptop and, if used, GPU).

Counting/application quality (the number the boss cares about), body vs head, same clips:
- **Counting MAE / % error** vs. a manual hand-count (line IN/OUT totals, zone occupancy
  at sampled timestamps) — report **per pipeline** so the gap is visible.
- **Alert precision/recall** on staged crowded/uncrowded clips.
- **Tracking ID switches per minute** (ByteTrack vs BoT-SORT on dense head tracks).
- **Detections recovered:** count of passengers the head model finds that the body model
  misses in distant/occluded regions (qualitative + sampled-frame counts).

> Report honestly with clip conditions. Higher head-detector mAP on RPEE-Heads does **not**
> automatically mean better counts on Indian footage — state that explicitly.

---

## Fine-Tuning Plan

> **Guardrail:** [`scripts/train_yolo_indian_platform.py`](../scripts/train_yolo_indian_platform.py)
> already refuses to "train" unless real labels exist on disk and the config is not a
> placeholder. The head workflow must keep that guardrail. **Do not fake fine-tuning.**

### 1. Dataset conversion to YOLO format

Good news: **RPEE-Heads is already in YOLO `.txt` format** (`<class x y w h>` normalized),
so conversion is mostly **file organization**, not re-encoding:

- Download RPEE-Heads from https://doi.org/10.34735/ped.2024.2 into `data/rpee_heads/`
  (document manual steps if the portal has no CLI download).
- Add a converter script (e.g. `scripts/prepare_rpee_heads.py`) that:
  - copies/links images and their `.txt` label files into the project's expected layout,
  - forces the class id to `0` = `head` (single class),
  - uses the dataset's **official train/val/test split** if provided, else makes a
    **clip-disjoint** split (no frames from the same video in two splits — prevents
    leakage),
  - writes a head dataset YAML.
- If CrowdHuman/SCUT-HEAD are mixed in later, the converter normalizes their head boxes to
  the same single-class YOLO format (CrowdHuman is JSON-ODGT; SCUT-HEAD is VOC XML).

### 2. Train / val / test split

- **Primary:** RPEE-Heads official ~70/15/15 (train 1,346 / val 246 / test 294 images).
- Keep splits **video-disjoint** so the test set measures generalization, not memorization.
- Report final detector metrics on the **held-out test split** only.

### 3. Training command (head model)

A head-specific dataset YAML (single class `head`) plus the existing training entrypoint
pattern. Example (a new `configs/train_rpee_heads.yaml` + reuse of the train script,
generalized to accept a head model path/name):

```bash
# 1) Prepare data (no training, just organizes RPEE-Heads into YOLO layout)
python3 scripts/prepare_rpee_heads.py --src data/rpee_heads --out data/head_dataset

# 2) Fine-tune YOLO11n on heads (laptop/demo model)
python3 scripts/train_yolo_indian_platform.py \
    --data configs/train_rpee_heads.yaml \
    --model yolo11n.pt \
    --epochs 80 --imgsz 960

# 3) Accuracy variant (GPU recommended)
python3 scripts/train_yolo_indian_platform.py \
    --data configs/train_rpee_heads.yaml \
    --model yolo11s.pt \
    --epochs 100 --imgsz 1280
```

Notes:
- Use a **larger `imgsz`** (960–1280) than the body pipeline's 640 — heads are tiny and
  benefit from higher input resolution.
- `configs/train_rpee_heads.yaml` mirrors the existing placeholder but with
  `names: {0: head}` and the head dataset paths.
- The training-guard logic in the script must be reused (refuse if labels absent).

### 4. Output model path

- Demo model: `models/fine_tuned/head_best_n.pt`
- Accuracy model: `models/fine_tuned/head_best_s.pt`
- The detector already supports a fine-tuned path; the head model would be selected by a
  config/CLI flag rather than auto-overriding the body model, so **both stay runnable**
  for the comparison.

### 5. Evaluation plan

- Run Ultralytics `val` on the RPEE-Heads **test split** → mAP@0.5, mAP@0.5:0.95,
  precision, recall, AP-by-size, FPS.
- Run **both** pipelines (body, head) on the same demo clips → counting MAE, alert
  precision/recall, ID switches (see [Metrics](#metrics-to-report)).
- Save a short evaluation report (numbers + clip conditions) under `docs/`.

### 6. What to do if only unlabeled video is available

If RPEE-Heads is inaccessible **and** there is no other labeled head dataset:

- **Do not fabricate labels or claim training happened.**
- Fall back to the existing annotation pipeline
  ([`ANNOTATION_WORKFLOW.md`](ANNOTATION_WORKFLOW.md),
  [`scripts/extract_training_frames.py`](../scripts/extract_training_frames.py)):
  extract frames from the real (unlabeled) Indian clips, label **heads** in
  CVAT/Roboflow/Label Studio, export YOLO, then train.
- Document the blocker plainly: "head detector not trained yet — dataset pending."

---

## Web App Comparison Plan

The web app (Streamlit, extending [`src/dashboard/streamlit_app.py`](../src/dashboard/streamlit_app.py))
should let a non-technical viewer (the boss) see, on the **same clip**:

1. **Original video** — the raw input clip, no overlays.
2. **Body detection output** — current YOLO11 `person` pipeline: boxes + track IDs +
   zones + lines + counters (today's behavior).
3. **Head detection output** — fine-tuned head model: head boxes + head track IDs + the
   **same** zones/lines + counters.
4. **Counts & analytics side by side** — a comparison panel:

   | Metric | Body (person) | Head | Manual ground truth |
   |---|---|---|---|
   | Line IN / OUT total | … | … | … |
   | Peak zone occupancy | … | … | … |
   | Unique passengers | … | … | … |
   | Crowd alerts fired | … | … | … |
   | FPS | … | … | — |

   Both pipelines write to the **same SQLite schema** (different `run_session_id` /
   model tag), so the dashboard just queries two sessions and renders them together — no
   new storage design needed.
5. **Plain-English explanation** — a short text block the app shows, e.g.:
   > "On this distant, crowded clip the body detector counted **N** passengers and the
   > head detector counted **M**. Heads stay visible when bodies are hidden behind the
   > crowd, so the head count is closer to the manual count of **G**. On the close-up,
   > uncrowded clip both methods agree."

This keeps the comparison **honest and visual**: where head detection wins, it shows;
where it doesn't, it shows that too.

---

## Risks and Limitations

- **RPEE-Heads is railway-platform-relevant but NOT Indian-railway-specific.** A model
  trained on it has a real **domain gap** (clothing, luggage, headwear, station geometry,
  camera optics, lighting). Strong RPEE-Heads metrics ≠ proven Indian-platform accuracy.
- **Head false positives.** Round/dark objects (bags, helmets, signage, lights, pillars)
  can be mis-detected as heads, inflating counts. Needs threshold tuning and sampled-frame
  review.
- **Occluded heads can still be missed.** In extreme density, back-row heads hidden behind
  front-row heads are not visible at all — head detection reduces but does **not eliminate**
  undercounting.
- **Tiny heads cause tracking ID switches.** Distant heads are a few pixels wide; ByteTrack
  may swap IDs, corrupting line-crossing and unique-passenger counts. May require BoT-SORT
  + higher `imgsz`, and ID-switch measurement on CroHD.
- **Reference-point change is required.** Zone/line logic currently uses bbox bottom-center
  (correct for bodies/feet). Heads need **box-center**; using bottom-center for heads would
  bias which zone/line side a head falls on. Scoped change, but must be done before trusting
  head counts.
- **Real Indian Railway footage is still needed for validation.** Until labeled Indian
  platform frames exist, all head-detection accuracy numbers are **proxy numbers** on
  German/European + concert crowds. **This is not production-ready** and must not be
  presented as such.
- **Licensing.** RPEE-Heads is CC BY-SA 4.0 (attribute + share-alike). CrowdHuman / SCUT-HEAD
  / CroHD / others carry research-use terms — verify before any commercial deployment.
  Brainwash's original host was removed over consent concerns; treat as restricted.

---

## Sources

- RPEE-Heads paper (arXiv): https://arxiv.org/abs/2411.18164 — https://arxiv.org/html/2411.18164v1
- RPEE-Heads paper (IEEE Xplore): https://ieeexplore.ieee.org/document/10973050/
- RPEE-Heads dataset (DOI / download): https://doi.org/10.34735/ped.2024.2
- CrowdHuman: https://www.crowdhuman.org/
- SCUT-HEAD: https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release
- CroHD (Head Tracking 21): https://motchallenge.net/data/Head_Tracking_21/
- JHU-CROWD++: http://www.crowd-counting.com/
- NWPU-Crowd: https://www.crowdbenchmark.com/
- Hollywood Heads: https://www.di.ens.fr/willow/research/headdetection/
