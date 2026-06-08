# Accuracy Improvement Plan

> Diagnosis + planning document for the next coding agent ("Agent 3").
> **No source code was changed to produce this file.** No new test runs were
> executed; every number quoted below is traced to an existing report or config
> and labelled as either a **measured metric**, an **observational count**, or a
> **configured default**. Nothing here claims an accuracy improvement, because no
> new accuracy was measured.

---

## Boss Feedback

The boss reviewed the head-detection output on the current CCTV/platform-style
test video and asked for two things:

1. **"Increase the accuracy."** — improve how reliably the system detects and
   counts passengers (especially heads) on platform-style footage.
2. **"Brief the problem statement, the assumptions made, what is in scope and
   what is not, and the steps involved."** — a clear, non-technical explanation
   of the problem, assumptions, scope boundaries, and method.

This document addresses (1) with a diagnosis and an improvement plan, and
provides (2) below so the explanation can be reused in the boss package.

### Problem Statement (for the boss)

Indian Railway platforms need an automatic way to **count passengers and detect
crowding** from camera feeds, so staff can react to overcrowding near train
doors and on platform edges. This MVP processes a video, detects each passenger
(either by **full body** or by **head**), tracks them with stable IDs, counts
how many are inside defined **zones**, counts how many **cross virtual lines**,
raises **crowd alerts** when occupancy passes thresholds, logs everything to a
database, exports CSV reports, and shows a **web app comparison** of body vs head
detection.

### Assumptions Made

- The head detector was trained on **RPEE-Heads** (European railway-platform /
  event-entrance heads), assumed to be *platform-relevant* but **not** Indian
  Railway CCTV.
- The current test clip (Pexels "People on Platform on Train Station",
  1280×720, 25 fps, 803 frames) is assumed to be a **CCTV-like** elevated/static
  platform view — but it is **stock footage, not confirmed CCTV**, and not Indian.
- Zones, lines, and thresholds are assumed to be **per-camera** and were
  hand-calibrated to this one clip.
- "Accuracy" so far has been judged **visually**; there is **no labelled ground
  truth** on the test video, so no precision/recall has been measured on it.

### In Scope

- Body and head passenger detection on recorded video.
- Tracking IDs, zone occupancy, line crossing, crowd alerts.
- SQLite logging, CSV reports, Streamlit body-vs-head comparison.
- No-retrain tuning (confidence / image size / NMS / tracker / zones).

### Out of Scope (for now)

- Live RTSP CCTV ingestion in production.
- A measured production accuracy claim on Indian Railway CCTV.
- Local retraining (only done in Colab, and only when explicitly requested).
- Re-identification across cameras, face recognition, or any identity storage.
- Synthetic-video evaluation (explicitly disallowed).

### Steps Involved (method)

`video → detection (body or head) → tracking IDs → zone occupancy → line
crossing → crowd alerts → SQLite logging → CSV report → web app comparison.`

---

## Current Model Status

**Body model**
- Default weights: `yolo11n.pt` (from [configs/app.yaml](../configs/app.yaml#L7),
  `model.weights`).
- Accuracy upgrade weights available: `yolo11s.pt`; legacy fallback `yolov8n.pt`.
- Uses the COCO `person` class (`person_class_id: 0`).
- `use_fine_tuned_if_available: true` would prefer
  `models/fine_tuned/best.pt`, but **that path does not exist** (only the head
  model exists), so body mode falls back to `yolo11n.pt`.

**Head model**
- Weights: `models/fine_tuned/head_detector/weights/best.pt`
  (see [src/vision/detector.py:18](../src/vision/detector.py#L18)).
- Single class `0 = head`; base model `yolo11n.pt`; 25 epochs; imgsz 640;
  batch 32; trained on **RPEE-Heads** (1,346 train / 246 val images).
- **Training happened in Google Colab** on an A100, then the artifacts were
  copied into this repo. No local training was run. SHA-256 of
  `artifacts/colab/best.pt` matches the canonical `best.pt`.

**Where `best.pt` lives**
- Canonical: `models/fine_tuned/head_detector/weights/best.pt`
- Source copy: `artifacts/colab/best.pt` (hash-matched)
- `last.pt` and `results.csv` also present in the same folder.

**Local evaluation metrics available?**
- **Measured metrics exist only on the RPEE-Heads validation split**
  (from [docs/HEAD_MODEL_TRAINING_REPORT.md](HEAD_MODEL_TRAINING_REPORT.md) and
  `results.csv`, epoch 25):
  - precision(B) = **0.866**
  - recall(B) = **0.723**
  - mAP50(B) = **0.794**
  - mAP50-95(B) = **0.431**
- **There is NO measured accuracy on the test video or on Indian Railway CCTV.**
  The retest "counts" below are observational detection counts, **not** accuracy.

---

## Current Pipeline Summary

```
video file / stream
   │
   ▼
StreamReader (frame_stride, max_fps)           configs/app.yaml runtime.*
   │
   ▼
FrameProcessor._maybe_resize (resize_width=None by default → no resize)
   │
   ▼
Detector / Tracker  (Ultralytics YOLO)
   - conf, iou, classes=[0], imgsz, half, device                 src/vision/detector.py, tracker.py
   - body: yolo11n.pt   |   head: head_detector/best.pt
   │
   ▼
DetectionRegionFilter  (include/ignore polygons, size/aspect gates)   src/vision/detection_filter.py
   │
   ▼
ZoneManager (point-in-polygon occupancy)  +  LineManager (directional crossings)
   - point strategy: body=bottom_center, head=center
   │
   ▼
CrowdAnalyzer (warning/critical/dwell thresholds)              configs/thresholds.yaml
   │
   ├─► AnalyticsLogger → SQLite (run_sessions, frames_processed, zone_occupancy, …)
   ├─► annotate_frame → VideoWriter (body_demo.mp4 / head_demo.mp4)
   └─► export_report.py → CSV
                                   │
                                   ▼
                      Streamlit app (body vs head comparison)
```

Key configured defaults today:
- confidence: body `0.35`; head auto-lowered to `min(conf, 0.25)` when no CLI
  value, comparison script default head `0.15`
  ([scripts/run_video_demo.py:264](../scripts/run_video_demo.py#L264),
  [scripts/run_comparison_demo.py:52](../scripts/run_comparison_demo.py#L52)).
- imgsz: `640` default; head auto-bumped to `max(imgsz, 1536)` when no CLI value
  ([scripts/run_video_demo.py:269](../scripts/run_video_demo.py#L269)).
- iou (NMS): fixed `0.5` in config, **no CLI override exists**.
- tracker: `bytetrack`, `track_low_thresh: 0.10`, `fuse_score: true`.
- resize_width: empty → **frames are not downscaled**; YOLO letterboxes to imgsz.
- augment (TTA): **off** (not exposed anywhere).

---

## Accuracy Problem Diagnosis

Most likely contributors, ordered roughly by impact:

1. **Domain mismatch (highest impact).** The head model only ever saw
   RPEE-Heads. The test clip is European stock footage; the eventual target is
   Indian Railway CCTV. Camera height, lens, lighting, compression, clothing,
   head coverings, and crowd density all differ. mAP50-95 of 0.431 even
   *in-domain* shows tight localization is already imperfect.

2. **No ground truth on the actual footage.** "Increase accuracy" cannot be
   verified because there is **no labelled test set** for this video. Any change
   is currently judged by eye. This must be fixed first or improvements are
   unfalsifiable.

3. **Smallest model, short training.** Head detector is `yolo11n` (nano) trained
   25 epochs. Small heads at distance are exactly where nano models and few
   epochs struggle. Recall 0.723 means ~1 in 4 heads missed even in-domain.

4. **Small far-away heads vs image size.** At imgsz 640, heads in a 1280×720
   frame shrink below reliable size — the retest explicitly found 640 "too small"
   and switched to imgsz 1536. Anything below ~1280 will under-detect distant
   heads.

5. **Confidence set very low for heads (0.15).** This recovers small heads but
   invites **false positives** (bags, poles, train fixtures). The current fix is
   hand-drawn `ignore_polygons` in the zones config — which is **overfit to this
   one clip** and will not transfer.

6. **NMS / IoU not tunable.** IoU is fixed at 0.5. In dense crowds, overlapping
   heads can be over-suppressed (merged) or, in body mode, partially occluded
   people merged. No way to sweep it.

7. **Tracker tuned for bodies, not heads.** ByteTrack params suit large body
   boxes. Tiny head boxes → frequent **ID switches** → inflated unique counts
   (retest: 294 unique head IDs vs 108 body IDs on the same clip). This is a
   **counting-accuracy** problem distinct from detection.

8. **No test-time augmentation.** `augment` is never enabled; a flag could trade
   speed for recall on hard frames.

9. **Zone / line geometry not comparable across modes.** Body uses
   `bottom_center` (feet), head uses `center`. The shared line geometry was drawn
   for body-scale boxes, so body vs head line counts are not directly comparable
   (retest: 59 vs 2 crossings). This looks like "low accuracy" but is a
   calibration artifact.

10. **CPU inference budget.** `device: cpu` limits how far imgsz/TTA can be pushed
    before FPS becomes impractical, forcing accuracy/throughput trade-offs.

11. **Compressed stock footage.** Block/compression artifacts on small heads
    further hurt detection; real CCTV will have different (often worse) noise.

---

## Immediate No-Retrain Improvements

These can be measured and tuned without retraining. None of them should be
*claimed* as improvements until measured against ground truth (see Evaluation).

- **Confidence threshold sweep.** Try head conf ∈ {0.10, 0.15, 0.20, 0.25, 0.30}
  and body conf ∈ {0.25, 0.35, 0.45}; record detections kept vs visually obvious
  false positives. Pick the knee, don't hard-code 0.15.
- **Image size tuning.** Compare head imgsz ∈ {960, 1280, 1536} for the
  recall/FPS trade-off. 1280 is often the sweet spot on 720p CPU; 1536 only if
  FPS is acceptable.
- **NMS IoU tuning.** Expose IoU as a CLI flag and sweep ∈ {0.45, 0.5, 0.6, 0.7}.
  Crowded head scenes often prefer higher IoU (less merging of adjacent heads).
- **Tracker tuning for small heads.** Increase `lost_track_buffer`, lower
  `track_low_thresh`, adjust `minimum_matching_threshold`; measure ID-switch
  reduction. Consider `botsort` (used during Colab training) for comparison.
- **Frame sampling / FPS control.** Use `frame_stride`/`max_fps` to keep large
  imgsz practical on CPU during demos without dropping perceptible detections.
- **Safe preprocessing (optional, measured).** A mild contrast/CLAHE or light
  sharpening *before* inference can help small heads — but only keep it if it
  measurably helps; it can also amplify compression noise. Make it an opt-in flag.
- **Better zone configuration.** Re-calibrate polygons per camera; align body and
  head reference points/geometry so cross-mode comparisons are fair. Avoid
  clip-specific `ignore_polygons` as a substitute for a good confidence threshold.
- **Show confidence scores.** Already drawn in the overlay; surface a per-frame
  detection-count and mean-confidence overlay so the boss can see *why* counts
  change between settings.
- **Compare multiple thresholds side by side.** Generate the same clip at 2–3
  confidence/imgsz settings so reviewers can choose, instead of one opaque run.

---

## Model-Level Improvements

Require Colab retraining — **only when explicitly requested.**

- **Train longer.** 25 → 50–100 epochs with early stopping; current run used
  `patience: 100` so it never early-stopped — more epochs likely help.
- **Bigger backbone.** Train `yolo11s` (and benchmark `yolo11m` if the A100
  budget allows) instead of only `yolo11n`. `yolo11s.pt` is already in the repo
  for body mode, so the upgrade path is consistent.
- **Train at higher imgsz / multi-scale.** Train at 960–1280 (and/or enable
  `multi_scale`) so the model is matched to the inference resolution used for
  small heads.
- **More platform-like data.** Keep RPEE-Heads as the base, then **add Indian
  railway / platform CCTV frames once the boss provides footage.**
- **Label real Indian CCTV and fine-tune again.** Highest-leverage step for
  closing the domain gap (see Data-Level Improvements).
- **Evaluate on a held-out railway platform clip**, not just RPEE-Heads val —
  ideally the held-out RPEE-Heads *test* split plus any labelled Indian frames.

---

## Data-Level Improvements

- **Why real Indian Railway CCTV footage is needed.** The model's only measured
  competence is on RPEE-Heads. Indian platform CCTV differs in angle, density,
  lighting, attire (turbans, dupattas, luggage on heads), and compression. Without
  in-domain data we can neither train for it nor *measure* accuracy on it.
- **What labels are needed.** Single-class `head` bounding boxes in YOLO format
  (`class cx cy w h`, normalized) — matching the existing `0 = head` schema. For
  body validation, COCO `person` boxes.
- **What frames to extract.** Sample diverse frames across time, crowd density,
  lighting (day/night/artificial), and platform positions; avoid near-duplicate
  consecutive frames. A few hundred well-chosen labelled frames beats thousands of
  redundant ones.
- **How to label.** Use the existing `docs/ANNOTATION_WORKFLOW.md` and a YOLO-format
  tool (e.g. Label Studio / CVAT); export normalized boxes; one `.txt` per image.
- **Train/val/test split.** Split by **scene/camera/time**, not random frames, to
  avoid leakage from near-duplicate frames. Hold out a true test set never seen in
  training, for the final accuracy number.
- **Why unlabeled videos help testing but not supervised fine-tuning.** Unlabeled
  clips are great for *qualitative* review and for picking frames to label, but
  supervised fine-tuning and precision/recall **require labels**. Do not treat
  unlabeled stock footage as training data.

---

## Evaluation Plan

Define accuracy concretely **before** tuning, so changes are measurable.

- **Manual sample count comparison.** On N sampled frames, a human counts heads;
  compare to model counts. Report mean absolute count error and % error.
- **Precision / recall / F1** — only once a labelled mini test set exists (even
  50–100 labelled frames is enough to start). Report per confidence threshold.
- **False positives** — count detections with no real head (poles, bags, train
  parts), per setting.
- **False negatives** — count missed real heads, especially far/occluded.
- **Counting error** — predicted unique passengers vs human count over the clip.
- **Zone occupancy error** — predicted vs human occupancy at sampled timestamps.
- **Line-crossing error** — predicted vs human IN/OUT counts.
- **FPS / latency** — frames/sec at each imgsz on the demo device, so accuracy
  gains are reported with their cost.
- **Body vs head comparison** — same clip, same (re-aligned) geometry, report both
  modes' counts and errors side by side.

Record all of the above in a results table per configuration so the "best"
setting is chosen on evidence, not by eye.

---

## Specific Code Changes For Agent 3

File-by-file, concrete and minimal. Keep all knobs **configurable** and
defaulting to current behavior.

- **[configs/app.yaml](../configs/app.yaml)**
  - Add `model.augment: false` (TTA toggle) and keep `iou` here as the default.
  - Optionally add a `head:` sub-block for head-specific `confidence`, `imgsz`,
    `iou` defaults instead of hard-coding them in scripts.

- **[configs/thresholds.yaml](../configs/thresholds.yaml)**
  - No structural change required; document that thresholds are per-camera and add
    a head-mode profile if head occupancy scales differently from body.

- **[src/vision/detector.py](../src/vision/detector.py)**
  - Add an `augment: bool = False` field and pass `augment=self.augment` into
    `self.model.predict(...)`.
  - Optionally add `max_det` as a parameter (Ultralytics default 300 can clip very
    dense crowds).
  - Keep `iou` already plumbed; just ensure it is overridable from CLI.

- **[src/vision/tracker.py](../src/vision/tracker.py)**
  - Plumb the same `augment` / `max_det` through `model.track(...)`.
  - Expose tracker params (`track_low_thresh`, `lost_track_buffer`,
    `minimum_matching_threshold`) as overrides so head-mode ID switches can be
    reduced and measured.

- **[src/video/frame_processor.py](../src/video/frame_processor.py)**
  - Add an optional, **off-by-default** preprocessing hook (e.g. CLAHE / sharpen)
    behind a flag, applied in `_maybe_resize`'s neighborhood, so it can be A/B
    measured. Do not enable by default.

- **[src/vision/annotator.py](../src/vision/annotator.py)**
  - Add per-frame overlay of detection count and mean confidence (the overlay dict
    already supports arbitrary keys, so this is a small caller change in
    `run_video_demo.py`). Helps reviewers see the effect of threshold changes.

- **[scripts/run_video_demo.py](../scripts/run_video_demo.py)**
  - Add `--iou`, `--augment`, `--max-det` CLI flags (currently only `--confidence`
    and `--imgsz` exist). Wire them into `build_models`.
  - Surface `Detections` and `MeanConf` in the `overlays` dict passed to
    `annotate_frame`.

- **[scripts/run_comparison_demo.py](../scripts/run_comparison_demo.py)**
  - Pass through new `--iou` / `--augment` for both modes; stop hard-coding head
    `0.15`/`1536` and instead read head defaults from config so the comparison is
    reproducible.

- **[scripts/evaluate_head_detector.py](../scripts/evaluate_head_detector.py)**
  - This currently only prints avg/max detections per frame. **Add a true
    evaluation path**: if a labelled dataset/test split exists, run
    `model.val(data=..., conf=..., iou=..., imgsz=...)` and print
    precision/recall/mAP. Add a **threshold-sweep mode** (loop conf × imgsz × iou,
    write a CSV of detections/FPS, and metrics where labels exist).

- **New: `scripts/sweep_thresholds.py`** (or a `--sweep` flag on the evaluator)
  - Run the detector over a clip across conf × imgsz × iou, write a results CSV
    (detections kept, mean conf, FPS), and produce a small comparison report. This
    is the artifact the boss wants for "increase accuracy."

- **[src/dashboard/streamlit_app.py](../src/dashboard/streamlit_app.py)**
  - Add a short, plain-English explanation of **confidence** and **image size**
    ("lower confidence = more detections but more false alarms; larger image =
    catches smaller/farther heads but runs slower").
  - Optionally expose conf/imgsz selectors and show the multi-threshold outputs
    side by side.

- **docs/**
  - Add an **accuracy comparison report** (results of the sweep + evaluation) once
    runs exist. Update `WEB_APP_BODY_HEAD_COMPARISON_GUIDE.md` with the new flags.
  - Do **not** restate RPEE-Heads val metrics as test-video accuracy.

---

## Acceptance Criteria

Agent 3's work is complete when:

- Confidence threshold is configurable (already true) **and** swept/measured.
- Image size is configurable (already true) **and** compared across ≥3 values.
- IoU / NMS threshold is configurable via CLI and config (new).
- An optional test-time augmentation flag exists and is off by default (new).
- A threshold-sweep script exists and outputs a results CSV.
- An accuracy comparison report exists in `docs/`, with measured numbers (or
  clearly labelled observational counts where no ground truth exists yet).
- The web app explains confidence/threshold/image-size in plain language.
- Final output videos exist for **both** body and head modes on the agreed clip.
- No fabricated metrics; no synthetic video; no local retraining unless requested.

---

## Top 5 Most Important Accuracy Changes To Implement Next

1. **Create a labelled mini test set on real platform footage (ideally Indian).**
   Without ground truth, "increase accuracy" cannot be measured or proven. This is
   the single highest-leverage step and unblocks every other one.
2. **Add a threshold/imgsz/IoU sweep + real evaluation** (extend
   `evaluate_head_detector.py`, expose `--iou`/`--augment` in the demo scripts).
   Choose settings from measured precision/recall/FPS, not by eye.
3. **Retrain the head detector stronger in Colab** — `yolo11s` (benchmark
   `yolo11m`), 50–100 epochs, train at 960–1280 imgsz — to lift the 0.723 recall
   and 0.431 mAP50-95 that cap detection quality even in-domain.
4. **Fine-tune on labelled Indian Railway CCTV frames** once footage is provided,
   to close the RPEE-Heads → Indian-platform domain gap (the root cause of "low
   accuracy" on the target scenes).
5. **Tune the tracker for small head boxes and re-align body/head zone-line
   geometry**, so unique-passenger, zone, and line counts stop being inflated by
   ID switches and become directly comparable between modes.
```