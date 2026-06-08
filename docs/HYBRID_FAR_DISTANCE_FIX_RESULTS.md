# Hybrid Far-Distance Fix — Before/After Results

Generated: 2026-06-08. Clip: `data/input_videos/sample.mp4` (1280x720, public
CCTV-like railway platform footage; byte-identical to
`data/input_videos/cctv_platform_sample.mp4`, MD5 `0a1612311b92a0edf05d02f01bbc5a7d`).
Zones config: `configs/zones.hybrid_cctv_platform.example.json`.
Detector model (body): `yolo11n.pt`. Head model:
`models/fine_tuned/head_detector/weights/best.pt`. Tracker: bytetrack.

## What this measures (and what it does NOT)

- All numbers below are **detection volume** (`*_avg_count` / per-frame
  `total_detections`). Detection volume is an **operational observation, NOT
  accuracy**. A higher hybrid count can mean **either** better recall of real
  far/edge passengers **OR** more false positives — this clip cannot tell the
  two apart.
- **Approximate manual-count accuracy is now measured below.** The six
  `manual_count` values were filled by visual review of selected frames. They
  are useful for an MVP check, but they are not a substitute for a dense labeled
  validation set.
- **Clip caveat:** this is a single, sparse demo clip. It is NOT a
  dense/occluded crowd validation set. It proves the software path works and
  shows a behavioral shift; it does not certify counting accuracy. Real
  approved-CCTV validation with manual labels is still required (see
  `docs/HYBRID_BODY_HEAD_DETECTION_PLAN.md`).

## Follow-up rerun with live 0.15/1536 head settings

After updating `configs/app.yaml`, the comparison was re-run with the intended
head recall settings live (`head_confidence=0.15`, `head_imgsz=1536`):

```bash
.venv/bin/python scripts/run_comparison_demo.py \
  --source data/input_videos/sample.mp4 \
  --zones-config configs/zones.hybrid_cctv_platform.example.json
```

Latest completed sessions from this rerun:

| Mode | Session | Frames | Unique tracks | Avg total detections/frame | Max total detections |
|------|---------|--------|---------------|----------------------------|----------------------|
| body | 10 | 803 | 108 | 6.89 | 12 |
| head | 9 | 803 | 664 | 10.40 | 19 |
| hybrid | 7 | 803 | 402 | 20.54 | 28 |

Same-run far/edge zone (`train_side_edge`) average:

| Mode | train_side_edge avg | max | sampled zone rows |
|------|---------------------|-----|-------------------|
| body | 1.33 | 2 | 12 |
| head | 0.30 | 2 | 43 |
| hybrid | 2.16 | 4 | 56 |

Result: with the live recall settings, hybrid is **+0.83 detections/frame
(+62%) above body** in `train_side_edge` and has the highest whole-frame
detection volume. This is still detection volume, not accuracy by itself.

## The fix under test

Working-tree changes (uncommitted) re-run here:

1. `src/vision/hybrid_detector.py` + `scripts/run_video_demo.py`: in hybrid
   mode the **body detector now runs full-frame** (`near_body_polygons = []`),
   so passengers outside any near zone are still detected; fusion de-dupes
   overlapping far heads. The head detector stays restricted to
   `far_head_polygons`.
2. `configs/zones.hybrid_cctv_platform.example.json`: **widened `far_head_zone`
   ROI** (now spans down to y=720 and wider across the platform edge).
3. `src/config.py`: **code defaults** for head detection raised to
   `head_confidence=0.15`, `head_imgsz=1536`.

### Historical discrepancy — settings used in the first AFTER run

The expected far-head settings (`head_confidence 0.15`, `head_imgsz 1536`) were
**NOT the values applied** to the AFTER run. `scripts/run_comparison_demo.py`
was run without `--head-confidence` / `--head-imgsz`, so `run_video_demo.py`
loaded its default config `configs/app.yaml`, which still pins:

```
head_confidence: 0.25
head_imgsz: 1280
```

`configs/app.yaml` overrides the new `src/config.py` defaults. So the AFTER
hybrid/head used **head_confidence=0.25, head_imgsz=1280**, not 0.15/1536. The
full-frame-body change and the widened ROI WERE active (they live in code +
zones JSON). To exercise the intended 0.15/1536, either update `app.yaml` or
pass `--head-confidence 0.15 --head-imgsz 1536` on the demo command and re-run.
This discrepancy is resolved by the follow-up rerun above.

## Before/after per-zone detection volume (zone_occupancy `count`)

`n` = number of sampled frames written to `zone_occupancy` for that session.

| Run | Mode | train_side_edge (FAR/EDGE) avg | max | n | main_platform_waiting avg | platform_movement_path avg | per-frame total_detections avg | max |
|-----|------|-------------------------------|-----|---|---------------------------|----------------------------|--------------------------------|-----|
| BEFORE (pre-fix DBs) | body   | 4.753 | 8 | 77 | 19.987 | 4.987 | 25.20 | 31 |
| BEFORE (pre-fix DBs) | hybrid | 2.209 | 6 | 43 | 21.349 | 5.581 | 24.84 | 32 |
| AFTER (re-run)       | body   | 1.353 | 3 | 17 | 6.059  | 1.647 | 6.89  | 12 |
| AFTER (re-run)       | head   | 0.324 | 2 | 37 | 5.838  | 1.892 | 7.31  | 19 |
| AFTER (re-run)       | hybrid | 1.864 | 4 | 44 | 13.886 | 3.591 | 17.34 | 26 |

BEFORE = the pre-fix DBs captured and backed up (`/tmp/{body,head,hybrid}_before.db`)
before the re-run overwrote them. AFTER = current `data/outputs/*_analytics.db`.

## Headline: did hybrid's far-zone count move UP toward/above body?

Two readings, because the BEFORE body baseline is not stable (see caveat below).

1. **Same-run, apples-to-apples (most defensible — body, head, hybrid all from
   the AFTER re-run on identical video/code):**
   - train_side_edge: body **1.353** vs hybrid **1.864** → hybrid is **+0.51
     (+38%) ABOVE body**, and far above head-only (0.324). The fusion is adding
     far/edge detections that body-only misses.
   - This **reverses the pre-fix relationship**, where hybrid (2.209) sat
     *below* body (4.753) in the far zone. After the fix, hybrid >= body in
     train_side_edge. Directionally this is the intended effect: the full-frame
     body pass plus widened far-head ROI lift the hybrid far/edge count to or
     above the body-only count instead of dropping below it.
   - Whole-frame: hybrid 17.34/frame vs body 6.89 vs head 7.31 — hybrid recovers
     ~2.5x the body-only volume, consistent with fusing both sources.

2. **Literal before-vs-after of hybrid alone:** hybrid train_side_edge went
   2.209 -> 1.864 (a slight DROP). But this cross-run delta is **not reliable**
   because the BEFORE and AFTER body baselines differ massively (body per-frame
   25.20 -> 6.89) on the **same video and same body settings** — see below.

## Baseline-instability caveat (do not over-read cross-run deltas)

The BEFORE and AFTER runs use the **same video** (verified byte-identical MD5),
the **same body model** (`yolo11n.pt`), and the **same body settings** in the
current `app.yaml` (`confidence 0.35`, `imgsz 640`). Yet body-only per-frame
detections fell from **25.20 to 6.89** and body sampled-frame count (`n`) fell
77 -> 17. The committed/working-tree fix diffs touch **only hybrid and head**
paths, not body-only mode, so this body shift is **not explained** by the fix.
The likely cause is that the pre-fix BEFORE DBs (timestamped ~08:54 today) were
produced under a different transient `app.yaml` (e.g. a lower body confidence)
that has since been reverted; this cannot be confirmed from the saved artifacts.

Consequence: treat **BEFORE vs AFTER as not strictly comparable**. The reliable
evidence for the fix is the **AFTER same-run** comparison (reading 1), where
hybrid clearly exceeds body in the far/edge zone.

## Manual-count accuracy

`data/manual_ground_truth/review_counts.csv` has now been filled with six
approximate visual frame counts. The notes column explicitly marks these as
visual counts from occluded frames, not a dense labeled benchmark. The comparison
was then generated with:

```bash
.venv/bin/python scripts/compare_manual_counts.py \
  --manual-csv data/manual_ground_truth/review_counts.csv \
  --body-db data/outputs/body_analytics.db \
  --head-db data/outputs/head_analytics.db \
  --hybrid-db data/outputs/hybrid_analytics.db \
  --output-csv data/outputs/manual_count_comparison.csv
```

Per-segment result summary:

| Segment | Manual count | Body avg | Head avg | Hybrid avg | Body error % | Head error % | Hybrid error % |
|---------|--------------|----------|----------|------------|--------------|--------------|----------------|
| seg_001 | 45 | 7.43 | 12.71 | 22.40 | 83.48 | 71.75 | 50.22 |
| seg_002 | 46 | 6.08 | 11.04 | 20.58 | 86.79 | 76.01 | 55.26 |
| seg_003 | 41 | 6.88 | 7.41 | 19.67 | 83.22 | 81.94 | 52.03 |
| seg_004 | 43 | 5.45 | 7.42 | 20.72 | 87.32 | 82.73 | 51.81 |
| seg_005 | 42 | 7.71 | 10.34 | 18.93 | 81.64 | 75.38 | 54.92 |
| seg_006 | 40 | 7.77 | 13.35 | 20.89 | 80.58 | 66.63 | 47.77 |

Hybrid has the lowest error on all six approximate segments, but it still
undercounts substantially. Treat this as a useful MVP accuracy check, not a
production claim.

## Reproduction

```bash
.venv/bin/python scripts/run_comparison_demo.py \
  --source data/input_videos/sample.mp4 \
  --zones-config configs/zones.hybrid_cctv_platform.example.json
.venv/bin/python scripts/compare_manual_counts.py \
  --manual-csv data/manual_ground_truth/review_counts.csv
```

Latest live-settings session ids: body=10, head=9, hybrid=7 (in
`data/outputs/*_analytics.db`).
