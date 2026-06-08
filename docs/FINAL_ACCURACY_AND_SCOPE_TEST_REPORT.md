# Final Accuracy and Scope Test Report

_Test date: 2026-06-08 · Tester: final testing & comparison agent · Machine: macOS (Darwin 24.5.0), CPU/MPS inference, no GPU CUDA._

This report records exactly what was run and observed. It contains **no invented accuracy
numbers**. Where true accuracy cannot be measured, that is stated plainly.

## Boss Feedback Addressed

- **Increase accuracy** — tuning controls (confidence, image size, IoU, tracker, max detections,
  test-time augmentation) and a tuning-sweep script are implemented and were exercised on the
  current CCTV sample. These let settings be tuned and compared honestly. They do **not** by
  themselves prove accuracy improved — that requires manual ground truth (see below).
- **Brief problem statement, assumptions, scope, and steps** — `docs/PROBLEM_STATEMENT_SCOPE_AND_STEPS.md`
  exists, is detailed, and includes a one-page boss brief. It is now also downloadable from the
  Streamlit dashboard.

## Commands Run

| # | Command | Result |
|---|---|---|
| 1 | `python3 -m compileall src scripts` | **PASS** (exit 0) |
| 2 | `pytest` | **PASS** — 62 passed |
| 3 | `python3 scripts/run_video_demo.py --help` | **PASS** |
| 4 | `python3 scripts/run_comparison_demo.py --help` | **PASS** |
| 5 | `python3 scripts/tune_detection_settings.py --help` | **PASS** |
| 6 | `python3 scripts/compare_manual_counts.py --help` | **PASS** |
| 7 | `python3 scripts/export_report.py --help` | **PASS** (file exists; `--db` / `--out`) |
| 8 | Body demo (`--detector-mode body --confidence 0.25 --imgsz 960`) | **PASS** — 803 frames, ~99s |
| 9 | Head demo (`--detector-mode head --confidence 0.20 --imgsz 960`) | **PASS** — 803 frames, ~90s, imgsz 960 worked |
| 10 | Tuning sweep (`--detector-mode head --quick`) | **PASS** — wrote `tuning_results.csv` |
| 11 | `export_report.py` body → `body_tuned_report.csv` | **PASS** — 961 rows |
| 12 | `export_report.py` head → `head_tuned_report.csv` | **PASS** — 1006 rows |
| 13 | `compare_manual_counts.py` (manual ground truth) | **RAN, no usable ground truth** (template empty — by design, not faked) |
| 14 | `streamlit run streamlit_app.py` (headless smoke test) | **PASS** — serves HTTP 200, no errors |

### Manual Comparison Command

`compare_manual_counts.py` reads detector counts from the **SQLite databases**, not from exported
report CSVs. It compares manual counts against the latest completed body session and latest
completed head session (`ended_at IS NOT NULL` and `total_frames > 0`), ignoring unfinished runs.
The correct command is:

```bash
python3 scripts/compare_manual_counts.py \
  --manual-csv data/manual_ground_truth/example_counts.csv \
  --body-db data/outputs/body_tuned_analytics.db \
  --head-db data/outputs/head_tuned_analytics.db \
  --output-csv data/outputs/manual_count_comparison.csv
```

The output CSV is `data/outputs/manual_count_comparison.csv`. Its per-segment columns include
`manual_count`, `body_session_id`, `head_session_id`, `body_avg_count`, `head_avg_count`,
`body_absolute_error`, `body_percentage_error`, `head_absolute_error`, and
`head_percentage_error`.

## Required Files Verified

| File | Status |
|---|---|
| `data/input_videos/sample.mp4` | Present — 1280×720, 25 fps, 803 frames (~32 s), real Pexels railway-platform clip |
| `models/fine_tuned/head_detector/weights/best.pt` | Present |
| `configs/zones.cctv_platform.example.json` | Present (used for all runs) |
| `configs/zones.example.json` | Present (fallback, not needed) |
| `data/manual_ground_truth/example_counts.csv` | Present but **empty template** (`manual_count` blank) |

## Tuning Features Tested

| Feature | Status |
|---|---|
| Confidence (`--confidence`) | Works — used 0.25 (body), 0.20 (head); sweep tested 0.15 & 0.20 |
| Image size (`--imgsz`) | Works — 960 ran successfully on this machine for both modes |
| IoU / NMS (`--iou`) | Implemented (sweep default 0.50) |
| Max detections (`--max-det`) | Implemented |
| Test-time augmentation (`--augment`) | Implemented (not exercised; slower) |
| Tracker (`--tracker bytetrack\|botsort`) | Implemented (sweep default bytetrack) |
| Tuning sweep script | Works — wrote `data/outputs/tuning_results.csv` |

`imgsz 960` was **not** too slow and did not need downgrading to 640. Throughput on this machine
was roughly **8–10 frames/s** for both modes (full demo) and ~10.5 fps in the quick sweep.

## Body Detection Test

- **Command:**
  ```bash
  python3 scripts/run_video_demo.py --source data/input_videos/sample.mp4 \
    --output data/outputs/body_tuned_demo.mp4 \
    --zones-config configs/zones.cctv_platform.example.json \
    --db data/outputs/body_tuned_analytics.db \
    --detector-mode body --confidence 0.25 --imgsz 960
  ```
- **Outputs:** `data/outputs/body_tuned_demo.mp4`, `data/outputs/body_tuned_analytics.db`,
  `data/outputs/body_tuned_report.csv`
- **Observations:** 803/803 frames processed. Average **18.36** detections/frame, max **24**.

## Head Detection Test

- **Command:**
  ```bash
  python3 scripts/run_video_demo.py --source data/input_videos/sample.mp4 \
    --output data/outputs/head_tuned_demo.mp4 \
    --zones-config configs/zones.cctv_platform.example.json \
    --db data/outputs/head_tuned_analytics.db \
    --detector-mode head --model models/fine_tuned/head_detector/weights/best.pt \
    --confidence 0.20 --imgsz 960
  ```
- **Outputs:** `data/outputs/head_tuned_demo.mp4`, `data/outputs/head_tuned_analytics.db`,
  `data/outputs/head_tuned_report.csv`
- **Observations:** 803/803 frames processed. Average **3.17** detections/frame, max **11**.

### Honest body-vs-head observation (NOT an accuracy claim)

On this specific clip, **body detection produced far more detections than head detection**
(18.36 vs 3.17 average per frame). This is the opposite of the general expectation that head
detection recovers more passengers in dense/distant views. Likely reasons:

- this Pexels clip is only moderately crowded and many passengers are reasonably close, which
  favours the body detector;
- the head model was fine-tuned on RPEE-Heads, not Indian Railway CCTV, so domain mismatch can
  make it under-detect on this footage.

**Without ground truth we cannot say which count is "more correct."** Higher body counts could be
correct recall *or* false positives; lower head counts could be conservative precision *or* missed
heads. This must be resolved with manual counts before any accuracy conclusion is drawn.

## Tuning Results

- **File:** `data/outputs/tuning_results.csv`
- **Quick sweep contents (head, 75 frames each):**

  | conf | imgsz | iou | tracker | frames | total_det | unique_tracks | avg_det/frame | fps |
  |---|---|---|---|---|---|---|---|---|
  | 0.15 | 960 | 0.50 | bytetrack | 75 | 395 | 33 | 5.27 | 10.51 |
  | 0.20 | 960 | 0.50 | bytetrack | 75 | 395 | 33 | 5.27 | 10.62 |

- **Best-looking setting:** **None can be declared from this data.** Confidence 0.15 and 0.20
  produced *identical* detection/track counts on the 75-frame quick sweep, so there is no
  measurable separation and no quantitative or labelled basis to call one "best." A larger sweep
  plus manual ground truth is needed to choose a setting on accuracy rather than raw detection volume.
- **No fake accuracy claims are made from this CSV.** Per the script's own output, these are
  operational tuning signals, not accuracy.

## Manual Ground Truth Status

- **Available: NO.** `data/manual_ground_truth/example_counts.csv` is the shipped template with
  `manual_count` left blank. `compare_manual_counts.py` correctly reported
  *"No usable manual ground truth counts were found"* and did not produce a comparison.
- **No ground truth was invented.** No `manual_count_comparison.csv` with numbers exists.
- **Evaluation layer updated:** `scripts/compare_manual_counts.py` now selects only the latest
  completed body/head sessions and produces per-segment average-count and error metrics once
  real manual counts are supplied.
- **Tests added:** manual CSV parsing, completed-session selection, frame/time segment averaging,
  and absolute/percentage error math are covered in `tests/test_compare_manual_counts.py`.

### How to create real ground truth (required for true accuracy)

1. Open `data/outputs/body_tuned_demo.mp4` (or the raw `sample.mp4`).
2. Pick a few short segments and fill `data/manual_ground_truth/example_counts.csv`:
   - set either `start_frame`/`end_frame` **or** `start_time_sec`/`end_time_sec`,
   - count the passengers visible in that segment **by eye** and put the number in `manual_count`.
3. Re-run, using the **correct** flags:
   ```bash
   python3 scripts/compare_manual_counts.py \
     --manual-csv data/manual_ground_truth/example_counts.csv \
     --body-db data/outputs/body_tuned_analytics.db \
     --head-db data/outputs/head_tuned_analytics.db \
     --output-csv data/outputs/manual_count_comparison.csv
   ```
4. The output reports `manual_count`, `body_avg_count`, `head_avg_count`, and detector-specific
   absolute/% error per segment — that is the first real, defensible accuracy measurement.

## Problem Statement and Scope Document

- `docs/PROBLEM_STATEMENT_SCOPE_AND_STEPS.md` **exists and is boss-ready.** It covers: problem
  statement, objective, why head detection was added, assumptions, in-scope, out-of-scope, steps,
  data collected, accuracy improvement plan, risks/limitations, next steps, feedback needed, and a
  one-page boss brief.
- It honestly labels RPEE-Heads training metrics (P 0.866 / R 0.723 / mAP50 0.794 / mAP50-95 0.431)
  as model-training indicators, **not** Indian Railway CCTV accuracy.
- **Dashboard change made (small, low-risk):** the Streamlit app now has a "Problem statement,
  assumptions, scope, and steps (boss brief)" expander with an in-line preview and a
  **download button** for the document. Verified the app still starts (HTTP 200) after the change.

## Streamlit App Verification

Headless smoke test on ports 8599/8601: app starts, serves **HTTP 200**, no tracebacks.
By source inspection the app includes:

- **Tuning controls present:** confidence slider, image-size selector, IoU slider, tracker selector,
  max-detections input, test-time-augmentation checkbox (Section 5).
- **Plain-English explanation present** for confidence, image size, IoU, tracker, max-det, and TTA.
- **Body/head outputs display:** Section 7 shows body vs head videos side by side; Section 9 shows
  analytics; Section 10 has CSV report previews/downloads.
- **Tuning CSV:** preview + download of `data/outputs/tuning_results.csv`.
- **Scope document:** now referenced and downloadable (added in this pass).

**Practical demo note:** the dashboard reads the **default-named** files
(`body_demo.mp4`, `head_demo.mp4`, `body_analytics.db`, `head_analytics.db`), **not** the
`*_tuned_*` files produced in this test. For the boss demo, either (a) click the in-app
**Run Full-Body / Run Head** buttons so it regenerates the default-named outputs with the chosen
tuning settings, or (b) re-run the demos writing to the default names. Older default-named outputs
(from Jun 5) are present, so the dashboard will display *something* either way — just confirm it is
the run you intend to show.

## Remaining Limitations

- **Needs Indian Railway CCTV validation** — current footage is a public Pexels platform clip, not
  confirmed operational railway CCTV.
- **Needs manual ground truth for true accuracy** — none exists yet; all current numbers are
  detection/throughput counts, not accuracy.
- **Head model may underperform on this domain** — it under-detected vs body on this clip; may need
  more relevant training data and re-fine-tuning in Colab (do not train locally).
- **Accuracy depends on camera angle, resolution, lighting, occlusion, and crowd density**, plus
  correct per-camera zone/line placement and thresholds.
- **Body vs head counts are not directly comparable** — they detect different targets and use
  different reference points (body = bottom-centre/feet, head = box centre).

## Verdict

**Mostly ready, with limitations.**

The pipeline runs end to end, all 62 tests pass, tuning features work, both detectors produced
tuned outputs on the current CCTV sample, the dashboard starts and exposes the tuning controls and
the scope brief, and the problem-statement/scope document is boss-ready. The blocker for any
*accuracy* claim is the absence of manual ground truth — that is the single most important next
step before telling the boss accuracy has improved.
