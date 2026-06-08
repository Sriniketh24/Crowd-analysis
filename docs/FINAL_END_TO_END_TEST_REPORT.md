# Final End-to-End Test Report

**Project:** Railway Crowd Analytics — Indian Platform Passenger Counting (MVP)
**Test date:** 2026-06-02
**Tester:** Final end-to-end testing agent
**Environment:** macOS (Darwin 24.5.0), Python 3.12.5 (venv), project root `/Users/sriniketh/crowd-analysis`

> Current sample note: `data/input_videos/sample.mp4` has since been replaced with
> the selected public Pexels **People on Platform on Train Station** clip
> (1280×720, 32.13 s, 803 frames). Re-run the commands below before quoting fresh
> performance/count numbers.

---

## TL;DR Verdict

**The project runs end-to-end and is ready to show your boss.** Every documented
demo command works, the analysis pipeline processes real video, analytics persist
to SQLite, the report exports cleanly, the FastAPI API serves live data, and the
Streamlit dashboard starts and renders without errors.

One **critical environment bug was found and fixed** (broken virtual-env console
scripts caused by a folder rename). No code bugs were found — the application code,
configs, and tests are healthy.

---

## Commands Run & Pass/Fail Results

| # | Command | Result |
|---|---------|--------|
| 1 | `python3 --version` | ✅ PASS — Python 3.12.5 |
| 2 | `python3 -m compileall src scripts` | ✅ PASS — no compile errors |
| 3 | `pytest` (bare, documented command) | ✅ PASS — **48 passed**, 1 non-fatal warning, ~1.3 s |
| 4 | `python3 scripts/run_video_demo.py --help` | ✅ PASS — full usage printed |
| 5 | `python3 scripts/export_report.py --help` | ✅ PASS — full usage printed |
| 6 | `run_video_demo.py` on `data/input_videos/sample.mp4` | ✅ PASS — **600 frames processed in ~84 s**, session_id=11, `demo.mp4` written |
| 7 | `export_report.py --db … --out data/outputs/report.csv` | ✅ PASS — **6,654 rows** exported across 7 tables |
| 8 | API: `uvicorn src.api.app:app` + curl endpoints | ✅ PASS — all endpoints HTTP 200 (see below) |
| 9 | `streamlit run src/dashboard/streamlit_app.py` | ✅ PASS — starts clean, HTTP 200, script logic executes without errors |

### Output files (after a fresh pipeline run)

| File | Status | Detail |
|------|--------|--------|
| `data/input_videos/sample.mp4` | ✅ present | current canonical sample: 10 MB · h264 · 1280×720 · 803 frames |
| `data/outputs/demo.mp4` | ✅ present, non-empty | 12 MB · **h264 / High / yuv420p** (browser-playable in `st.video`) · 600 frames |
| `data/outputs/analytics.db` | ✅ present, non-empty | 952 KB · all 7 tables populated |
| `data/outputs/report.csv` | ✅ present, readable | 854 KB · 6,654 data rows · loads in pandas |

### Database contents (`analytics.db`)

```
camera_registry:        2
run_sessions:          11
frames_processed:    5,823
zone_occupancy:        198
line_crossing_events:  538
crowd_alerts:           66
stream_health_events:   16
```

### API endpoint results (server: `uvicorn src.api.app:app` on 127.0.0.1:8000)

| Endpoint | HTTP | Notes |
|----------|------|-------|
| `GET /health` | ✅ 200 | `{"status":"ok","service":"railway-crowd-analytics"}` |
| `GET /metrics/latest` | ✅ 200 | session 11 · 140 unique passengers · 2 zones · line IN=28/OUT=31 |
| `GET /metrics/zones` | ✅ 200 | `platform_zone` (occ 4, NORMAL), `concourse_zone` (occ 0, NORMAL) |
| `GET /alerts` | ✅ 200 | returns recent alert transitions newest-first |
| `GET /sessions` | ✅ 200 | session 11 = 600 frames, 140 unique passengers |
| `GET /cameras` | ✅ 200 | `demo_platform`, `sample` |
| `GET /cameras/{id}/latest` | ✅ 200 | per-camera metrics resolve |
| `GET /docs`, `GET /openapi.json` | ✅ 200 | interactive docs available |

No errors or tracebacks appeared in the uvicorn log; every request logged `200 OK`.

### Streamlit results

- Server starts headless and answers `GET /_stcore/health` → `ok` (HTTP 200); main page → HTTP 200.
- Startup log is clean (no import errors / tracebacks).
- Executing the script body directly (bare mode, so buttons/checkboxes return defaults and no side effects fire) exits **0** and runs **every** DB-query path (latest session, latest frame, zone occupancy, line crossings, alerts, occupancy history) plus the `report.csv` preview against the real database without raising.
- All dashboard sections are wired correctly: input video (§2), run-analysis button (§3), annotated output video (§4), analytics summary (§5), zones/lines/alerts tables (§6), report preview + download (§7). Missing-file states use helpful `st.warning`/`st.info` fallbacks rather than crashing.

---

## Bugs Found

### 🔴 BUG #1 — CRITICAL — Broken virtual-env console scripts after folder rename (FIXED)

**Symptom:** Every bare console command shipped in the venv failed:

```
/Users/sriniketh/crowd-analysis/.venv/bin/pytest: line 2:
  /Users/sriniketh/Crowd analysis/.venv/bin/python3.12: Not a directory
```

`pytest`, `streamlit`, `uvicorn`, `pip`, `yolo`, etc. all aborted.

**Root cause:** The project folder was renamed from `Crowd analysis` →
`crowd-analysis` after the venv was created. A venv's console scripts hard-code an
absolute interpreter path in their shebang/exec line, and `pyvenv.cfg` / the
`activate` scripts hard-code `VIRTUAL_ENV`. 33 files in `.venv/bin/` still pointed
at the old `/Users/sriniketh/Crowd analysis/...` path. (The `python`/`python3`
symlinks kept working because they resolve to the framework interpreter, which is
why `python3 -m <module>` worked while the bare commands did not.)

**Why it matters for the demo:** The README and demo guide tell the operator to run
`pytest`, `streamlit run …`, and `uvicorn …` directly. Those exact commands would
have failed in front of your boss — a misleading "it's broken" first impression
even though the application itself is fine.

### 🟡 Minor / non-blocking observations (NOT fixed — cosmetic only)

- **Streamlit `use_container_width` deprecation warnings.** Streamlit 1.58.0 still
  honors `use_container_width=True` (the app rendered correctly and exited 0); it
  only prints a deprecation notice in the server log, not in the browser. Cosmetic.
- **pytest warning:** `StarletteDeprecationWarning` about `httpx`/`TestClient`. Does
  not affect the 48 passing tests.
- **`_cleanup_backup/` directory at repo root** holds an older copy of
  `AGENTS.md`, `docs/`, `requirements.txt`, `src/`. It is not referenced by any
  code and is harmless, but could be deleted to avoid confusion.
- **`analytics.db-wal` / `analytics.db-shm`** sidecar files exist. These are normal
  SQLite WAL-mode artifacts and are read correctly by the API, dashboard, and
  export script.

---

## Bugs Fixed

### Fix for BUG #1

Rewrote the stale absolute path in every affected venv file (in place, no reinstall
required):

```bash
# Fix all 33 console/activation scripts under .venv/bin
for f in $(grep -rl "Crowd analysis" .venv/bin/); do
  sed -i '' 's|/Users/sriniketh/Crowd analysis/|/Users/sriniketh/crowd-analysis/|g' "$f"
done
# Fix the recorded venv command path
sed -i '' 's|/Users/sriniketh/Crowd analysis/|/Users/sriniketh/crowd-analysis/|g' .venv/pyvenv.cfg
```

**Verification after fix:**

```
pytest 9.0.3                     # bare pytest → 48 passed
pip 26.1.2                       # bare pip works
Running uvicorn 0.48.0 …         # bare uvicorn works
Streamlit, version 1.58.0        # bare streamlit works
yolo 8.4.58                      # bare yolo works
```

> **Note:** this is a local-environment fix. It will hold as long as the folder is
> not renamed/moved again. The equivalent (heavier) alternative documented in the
> README is to recreate the venv:
> `rm -rf .venv && python3.11 -m venv .venv && pip install -r requirements.txt`
> (requires network to re-download torch/ultralytics). The in-place fix was chosen
> because it is instant and reliable.

No application source code, config, or test was changed — none needed it.

---

## Remaining Limitations

These are inherent to the MVP scope and are **already disclosed** in the README and
dashboard (they are not defects):

1. **Pretrained, not fine-tuned.** Detection uses general-purpose YOLOv11n, not a
   model trained on Indian-platform CCTV. Accuracy in dense, cluttered scenes will
   improve with fine-tuning (`docs/ANNOTATION_WORKFLOW.md`).
2. **Recorded video, not live.** Demo runs on a sample clip. RTSP/HTTP sources are
   supported via `--source` but live CCTV needs stream hardening + infrastructure.
3. **Thresholds need calibration.** Zone WARNING/CRITICAL counts are demo defaults
   and must be tuned per station's real capacity.
4. **CPU performance.** ~10 FPS / ~84 s for 600 frames on this machine. Fine for a
   demo; production throughput needs GPU/TensorRT.
5. **Venv is path-bound.** Renaming/moving the project folder again will re-break
   the console scripts (recreate the venv if you do).

---

## Final Demo Readiness Verdict

| Capability | Status |
|------------|--------|
| Runs end-to-end (video → detect/track → zones/lines/alerts → SQLite) | ✅ Yes |
| Demo video (`demo.mp4`) produced, non-empty, browser-playable | ✅ Yes |
| `report.csv` exports and is readable | ✅ Yes |
| FastAPI API works (all endpoints 200, real data) | ✅ Yes |
| Streamlit dashboard works (starts, renders, no crash) | ✅ Yes |
| Unit tests pass | ✅ Yes (48/48) |
| **Ready to show your boss** | ✅ **YES** |

---

## Exact Commands to Run for the Demo

Run all of these **from the project root** `/Users/sriniketh/crowd-analysis`.
The virtual environment is already set up and fixed.

```bash
cd /Users/sriniketh/crowd-analysis
source .venv/bin/activate            # now works after the fix
```

**0) (Optional) sanity check — confirm everything is green**

```bash
python -m compileall src scripts
pytest
```

**1) Run the analysis pipeline on the sample video** (produces the annotated MP4 + analytics DB)

```bash
python scripts/run_video_demo.py \
  --source data/input_videos/sample.mp4 \
  --output data/outputs/demo.mp4 \
  --zones-config configs/zones.example.json \
  --db data/outputs/analytics.db
```

**2) Export the analytics report to CSV**

```bash
python scripts/export_report.py \
  --db data/outputs/analytics.db \
  --out data/outputs/report.csv
open data/outputs/report.csv          # opens in your default app (macOS)
```

**3) Launch the dashboard (primary thing to show the boss)**

```bash
streamlit run src/dashboard/streamlit_app.py
# open http://localhost:8501
```

In the dashboard you can either view the analytics from step 1, or click
**▶ Run Passenger Counting Analysis** to run it live (re-encodes output to H.264 via
ffmpeg, which is installed on this machine).

**4) (Optional) Launch the API to show programmatic access**

```bash
uvicorn src.api.app:app --reload
# open http://localhost:8000/docs
# quick check:
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/metrics/latest
```

> Tip: if a `python:` / `pytest:` / `streamlit:` "Not a directory" error ever
> reappears, the folder was renamed again — recreate the venv:
> `rm -rf .venv && python3.11 -m venv .venv && pip install -r requirements.txt`
