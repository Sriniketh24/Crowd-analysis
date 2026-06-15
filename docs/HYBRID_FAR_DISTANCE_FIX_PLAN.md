# Hybrid Far-Distance Fix Plan

Status: DESIGN — implementation owned by `coder-hybrid` and `coder-params`.
Frame reference: **1280 x 720** (the only calibrated resolution; see
`configs/zones.hybrid_cctv_platform.example.json`).

## 1. Problem statement

Hybrid mode underperforms plain `body` mode at far distances.

Root cause (verified in code):

- `HybridDetector.detect` runs the body detector only on the crop bounding
  `roi_config.near_body_polygons` (`src/vision/hybrid_detector.py` lines
  173-182). With a populated `near_body_zone`, the body model is geometrically
  banned from the far platform.
- The far platform is served **only** by the fine-tuned head model, restricted
  to `far_head_polygons` (lines 183-192).
- Plain `body` mode runs the body detector full-frame, so it sees far people the
  hybrid body model never gets to see. Whatever the head model misses in the far
  region is simply lost in hybrid mode, while plain body mode still recovers
  some of those people.

Fusion (`src/vision/detection_fusion.py`) is **not** the bug: it already keeps
all bodies and only drops heads that collide with a body. The bug is that
hybrid produces no far bodies at all.

## 2. Design overview

Four workstreams. (1)-(3) are the immediate accuracy fix and are
code/config-only. (4) is an independent model-quality workstream.

### (1) Body runs FULL-FRAME in hybrid

**Change the data, not the algorithm.** `crop_bounds_from_polygons`
(`src/vision/hybrid_detector.py` lines 56-80) already returns `None` for an
empty polygon list, and `run_detector_in_rois` already treats `None` bounds as
"run the detector on the full frame with no polygon post-filter" (lines
113-122, 128-129). So the fix is: **make `near_body_polygons` empty** so the
body detector runs full-frame, while the head detector keeps running inside
`far_head_polygons`.

New `HybridDetector` behavior (this is the contract `coder-hybrid` must make
testable):

- Given `roi_config.near_body_polygons == []`:
  - The body detector is called exactly once on the **entire frame**
    (`offset_x == 0`, `offset_y == 0`, crop equals the input frame).
  - Every body detection returned by the body detector is kept (no polygon
    post-filter), translated by a zero offset (i.e. unchanged coordinates).
- Given `roi_config.far_head_polygons` non-empty:
  - The head detector is called on the crop bounding `far_head_polygons`
    (unchanged from today).
  - Head detections are translated back to full-frame coords and post-filtered
    so only heads whose `center` anchor lies inside a `far_head` polygon
    survive (unchanged from today, `HEAD_ANCHOR == "center"`).
- The two lists are fused via `fuse_body_head` with the detector's existing
  `upper_body_fraction` and `iou_threshold` (unchanged).

No signature changes are required. `HybridDetector.__init__`,
`run_detector_in_rois`, `crop_bounds_from_polygons`, and `detect` keep their
current shapes. The only behavioral switch is the ROI **content**.

**Where the empty `near_body_zone` comes from.** Two layers must agree:

- Config layer (owned by `coder-params`): the example zones file must define
  `near_body_zone` as an empty polygon list so `load_hybrid_roi_config` yields
  `near_body_polygons == []`. `_parse_polygons` already returns `[]` for an
  empty/absent value, and `load_hybrid_roi_config` reads
  `near_body_zone` from `hybrid_detection_rois` first, then top-level.
- Builder layer (owned by `coder-hybrid`): `build_hybrid_models` in
  `scripts/run_video_demo.py` (lines 465-512) must guarantee the body detector
  runs full-frame in hybrid even if a stale config still carries a
  `near_body_zone`. After `load_hybrid_roi_config(...)`, `coder-hybrid` should
  force `roi_config.near_body_polygons = []` before constructing
  `HybridDetector`. This makes full-frame body the invariant of hybrid mode and
  prevents regressions from old config files. `far_head_polygons` is passed
  through untouched.

**Fusion de-dups far heads against the new far bodies — confirmed.**
`fuse_body_head` (lines 101-122) seeds the output with every body, then for each
head calls `_head_matches_any_body` against **all** body boxes (not just near
ones). Now that bodies span the full frame, a far body and its head are tested
the same way as a near pair:

- `head_inside_upper_body`: the head center falls inside the top
  `upper_body_fraction` (default 0.45) of the body box → head dropped.
- `bbox_iou >= iou_threshold` (default 0.45) → head dropped.

For a correctly detected far person the body box is tall and its top 45% covers
the head, so the head is suppressed: **no double counting.** When the far body
is missed or poor, the head fails both tests and survives: **head acts as the
fallback**, which is exactly the desired far-distance behavior. Fusion code
needs no change.

Testable acceptance for (1):

- With `near_body_polygons=[]` and a fake body detector returning a box, the
  body detector is invoked with a frame equal to the input frame, and the
  returned body bbox is unchanged.
- A head whose center lies in the upper 45% of a returned body box is absent
  from the fused output; that body is present.
- A head with no overlapping body is present in the fused output.

### (2) Widen `far_head_zone` to the full far platform

The head model can only score people whose center lands inside a `far_head`
polygon (post-filter, lines 128-129). The current polygon is a narrow central
wedge and excludes the left/right platform edges and the seam with the (former)
near body zone, so far people at the margins are dropped even when detected.

`coder-params` must widen `far_head_zone` in
`configs/zones.hybrid_cctv_platform.example.json` so it covers the **entire far
platform**: full left and right edges, the top/back of the platform, and an
overlap downward across the seam where `near_body_zone` used to start (so the
handoff region is double-covered rather than gapped). Because body now runs
full-frame, head/body overlap in that seam is fine — fusion resolves it.

Target coverage (1280x720, treat as a spec, tune to footage):

- Left edge reaching to roughly `x≈135` at the platform's near extent.
- Right edge reaching to roughly `x≈1160`.
- Top following the platform vanishing region (`y≈85..110`).
- Bottom extended down to the old seam (`y≈470..540`) so the seam is covered.

Keep the existing `detection_filters.head` size/aspect and
`include_polygons`/`ignore_polygons` masks as the precision guardrail; widening
the zone increases recall, the filters keep precision. `coder-params` should
sanity-check that the widened `far_head_zone` does not extend into the
`ignore_polygons` (e.g. the train-side blackout rectangle near
`[875,270]..[1135,560]` and the right/top border strips).

### (3) Lower `head_confidence`, raise `head_imgsz`

Current defaults (`src/config.py` lines 119-120, and parsed at 231-232):

```python
head_confidence: float = 0.25
head_imgsz: int = 1280
```

Proposed new defaults (owned by `coder-params`):

```python
head_confidence: float = 0.15
head_imgsz: int = 1536
```

Rationale:

- **`head_confidence` 0.25 → 0.15.** Far heads are tiny and low-contrast; the
  head model's score on them is depressed, so 0.25 silently discards true far
  heads. 0.15 recovers low-score far detections. The precision cost is absorbed
  by three existing filters that the loosened threshold does **not** bypass: the
  `far_head_zone` spatial post-filter, the `detection_filters.head` size/aspect
  gate, and fusion's body suppression. Going below ~0.12 starts to surface
  background texture as heads on this footage, so 0.15 is the recall floor we
  keep. This is the immediate lever; do not lower further without re-checking
  false positives on `data/input_videos/sample.mp4`.
- **`head_imgsz` 1280 → 1536.** The head detector runs on the `far_head_zone`
  crop, not the full frame, so a far head can be only a handful of pixels at
  1280. Inferring at 1536 upsamples the crop and gives the network more pixels
  on each far head, directly improving small-object recall — the dominant
  failure mode here. 1536 is a multiple of 32 (stride-safe for YOLO) and is the
  practical accuracy/throughput knee for this single-camera demo; 1920 yields
  marginal recall for a large latency cost. If latency regresses unacceptably on
  the target device, 1536 is the value to revisit first, but for the demo
  workflow correctness outranks throughput.

These are `ModelSettings` defaults; the demo path threads them through
`effective["head_confidence"]` / `effective["head_imgsz"]` into the head
`Detector` in `build_hybrid_models` (lines 491, 494). No builder change needed
for (3) beyond consuming the new defaults, which already happens.

### (4) Head-model recall retraining (independent workstream)

Independent of (1)-(3); does not block them and is not owned by either coder
above. Captured here so the geometry/threshold fix is not mistaken for a model
fix.

Goal: raise the fine-tuned head model's recall on small, far, partly-occluded
heads so hybrid's far region stops relying on threshold-stretching.

- **Data:** mine far/dense frames from `data/input_videos/sample.mp4` and
  comparable platform footage; label small heads down to ~7px (matches the
  `detection_filters.head` `min_width/min_height` of 7). Oversample the far
  region and platform edges.
- **Training:** continue the existing Colab YOLO11s head path (see
  `docs/HYBRID_BODY_HEAD_DETECTION_PLAN.md` and the Colab notes referenced in
  recent commits). Train at `imgsz>=1280`, heavy small-object and mosaic
  augmentation, tuned anchors/strides for tiny objects.
- **Eval:** report recall at fixed precision specifically on the far region;
  gate promotion on far-region recall, not whole-frame mAP.
- **Promotion:** drop `best.pt` at the expected head-model path
  (`validate_head_model`, lines 515-527). Once recall improves, revisit (3) and
  consider raising `head_confidence` back toward 0.20 to recover precision.

## 3. File ownership map

| Owner          | Files (exact scope)                                                                                  |
|----------------|------------------------------------------------------------------------------------------------------|
| `coder-hybrid` | `src/vision/hybrid_detector.py`; `scripts/run_video_demo.py` (**`build_hybrid_models` only**)        |
| `coder-params` | `src/config.py`; `configs/zones.hybrid_cctv_platform.example.json`                                    |
| (unowned)      | head-model retraining workstream (4) — separate effort, no file overlap with the above               |

`src/vision/detection_fusion.py` is **unchanged** by anyone.

### coder-hybrid scope

1. In `build_hybrid_models` (`scripts/run_video_demo.py`, lines 465-512): after
   `load_hybrid_roi_config(...)`, force `roi_config.near_body_polygons = []` so
   the body detector always runs full-frame in hybrid; leave
   `far_head_polygons` untouched. Do not touch the head/body `Detector`
   construction args (those carry `coder-params`' new defaults).
2. In `src/vision/hybrid_detector.py`: confirm and, if needed, harden the
   documented full-frame behavior for empty `near_body_polygons`. Do not change
   public signatures; do not change fusion.
3. Make behavior in §2(1) testable per the acceptance list above
   (`tests/test_hybrid_detector.py`).

### coder-params scope

1. `src/config.py`: `head_confidence` 0.25 → **0.15**, `head_imgsz` 1280 →
   **1536** (both the dataclass defaults, lines 119-120, and the parser
   `.get(...)` defaults, lines 231-232, must match).
2. `configs/zones.hybrid_cctv_platform.example.json`:
   - Set `hybrid_detection_rois.near_body_zone.polygons` to `[]` (and update its
     description to "empty — body runs full-frame in hybrid").
   - Widen `hybrid_detection_rois.far_head_zone.polygons` to cover the full far
     platform per §2(2), staying clear of `detection_filters.head.ignore_polygons`.

## 4. Out of scope / do not change

- `src/vision/detection_fusion.py` (logic is correct as-is).
- Public signatures of `HybridDetector`, `run_detector_in_rois`,
  `crop_bounds_from_polygons`, `load_hybrid_roi_config`.
- Non-hybrid code paths in `scripts/run_video_demo.py`.
