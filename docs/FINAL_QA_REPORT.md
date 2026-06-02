# Final QA Report — AI Railway Crowd Analytics

> **Historical report:** This QA report was generated on 2026-06-01 before the
> Indian platform sample, YOLO11 config, camera-aware API, and RTSP/privacy
> updates. Do not treat its model/sample/count numbers as current. See
> `docs/IMPLEMENTATION_SUMMARY.md` and `docs/SENIOR_REVIEW_REPORT.md` for the
> current 2026-06-02 status.

**Date:** 2026-06-01
**Reviewer role:** Final senior testing, security & QA pass
**Verdict:** ✅ **Demo-ready.** All core requirements verified end-to-end on the bundled `sample.mp4`.

---

## 1. Scope

Full final review of the codebase for reliability, security, completeness, and
3-minute boss-demo readiness. No architecture changes were made; fixes were
limited to a clear missing dependency, a requirement gap in report export,
defensive error handling, a deprecated UI call, and added test coverage.

---

## 2. Commands Run

All commands run from the project root with the project virtual environment
(`.venv`, Python 3.12.5).

| Command | Result |
|---|---|
| `python3 -m compileall src scripts` | ✅ Pass (no errors) |
| `pytest` | ✅ Pass — **40 passed** (was 30) |
| `python3 scripts/run_video_demo.py --help` | ✅ Shows all flags |
| `python3 scripts/export_report.py --help` | ✅ Shows updated `--out` help |
| `python3 scripts/benchmark_fps.py --help` | ⚠️ Placeholder (prints stub text) |
| `python3 scripts/create_zones_from_frame.py --help` | ⚠️ Placeholder (prints stub text) |
| `run_video_demo.py … --source <missing>` | ✅ Clear guidance, no crash, exit 1 |
| `run_video_demo.py … --source data/input_videos/sample.mp4 --db …` | ✅ **Processed 688 frames**, wrote `demo.mp4` + `analytics.db` |
| `export_report.py --db … --out data/outputs/report.csv` | ✅ Single CSV (2136 rows), opens cleanly |
| `export_report.py` on **missing** DB | ✅ Helpful message explaining how to generate it |
| `export_report.py` on **empty** DB | ✅ CSV written with header only, no crash |
| FastAPI endpoints (`/health`, `/metrics/latest`, `/metrics/zones`, `/alerts`, `/sessions`) | ✅ Valid JSON on both empty and populated DBs |
| `streamlit run src/dashboard/streamlit_app.py` | ✅ Starts and renders; no console deprecation spam after fix |

---

## 3. What Passed (verified)

- **Project structure** — all required root files/dirs present (`AGENTS.md`,
  `README.md`, `requirements.txt`, `.env.example`, `.gitignore`, `Dockerfile`,
  `docker-compose.yml`, `configs/`, `data/`, `docs/`, `models/`, `scripts/`,
  `src/`, `tests/`).
- **Detection + tracking** — YOLOv8n + ByteTrack run end-to-end on the sample
  video and assign stable IDs.
- **Zone occupancy, line crossing, crowd alerts** — covered by unit tests and
  exercised by the live pipeline (alerts persisted to DB).
- **Persistence** — SQLite via SQLAlchemy; WAL mode; sessions/frames/zones/
  crossings/alerts tables created idempotently.
- **CSV export** — single-file and per-table modes; empty DB safe.
- **FastAPI** — all documented endpoints exist and appear in Swagger
  (`/docs`); empty data returns clean JSON (404 with detail or empty list),
  never a 500.
- **Streamlit dashboard** — handles a missing/empty DB with a clear
  "run the demo first" message; shows KPIs, zone occupancy, line counts,
  alerts, historical chart, and the annotated video; uses project-relative
  paths.
- **Configs** — `app.yaml`, `cameras.yaml`, `thresholds.yaml` valid YAML;
  `zones.example.json` valid JSON; thresholds reasonable (warning 20 /
  critical 30 for the platform zone); all paths project-relative.
- **Security/privacy** — no hardcoded secrets/keys/tokens; no `eval`/`exec`/
  `os.system`/`shell=True` in project code; the only `subprocess` use
  (dashboard) is a safe argument-list call (no shell). Only counts and event
  metadata are persisted — **no face crops or identifying images stored**.
  Outputs are written inside `data/outputs/` only.

---

## 4. Issues Found & Fixed

| # | Severity | Issue | Fix |
|---|---|---|---|
| 1 | **Critical** | Pipeline crashed on a fresh/offline install: `ModuleNotFoundError: No module named 'lap'` (ByteTrack requires `lap`, which was not declared). | Added `lap` to `requirements.txt` and installed it. Pipeline now runs offline. |
| 2 | **High** | `export_report.py --out data/outputs/report.csv` (no `--table`) created a **directory** named `report.csv` containing per-table files, so `open data/outputs/report.csv` failed. | Added `export_combined_csv()`. When `--out` ends in `.csv`, a single combined CSV is written to the exact path; directory mode kept for non-`.csv` paths. Removed the stale `report.csv/` directory. |
| 3 | Medium | Missing-DB export printed a terse error with no recovery guidance. | Message now prints the exact `run_video_demo.py` command to generate the DB. |
| 4 | Medium | Streamlit emitted continuous `use_container_width` deprecation warnings (removed after 2025-12-31) in the console. | Migrated all 5 call sites to `width="stretch"` (supported in Streamlit 1.58). |
| 5 | Low | `CrowdAnalyzer.evaluate` raised `KeyError` if a zone had no threshold entry. | Now defaults unconfigured zones to `NORMAL` instead of crashing. |
| 6 | Low | `alembic` listed as a dependency but unused (no migrations). | Removed from `requirements.txt`. |
| 7 | Low | Test coverage gaps (API, combined export, empty DB, missing thresholds). | Added `tests/test_api.py` (6 tests) and new report/analyzer tests. Suite: 30 → 40 tests. |

---

## 5. Remaining Limitations (non-blocking)

- **Helper scripts are placeholders:** `benchmark_fps.py`,
  `create_zones_from_frame.py`, and `extract_sample_frames.py` print stub text
  and do not parse `--help`. They are not required for the demo.
- **First model load needs the weights present.** `yolov8n.pt` is bundled in the
  repo root, so the demo works offline. A different `--model` not present locally
  would require network access to download (a clear error is printed if so).
- **Tracking accuracy** depends on `lap` (now installed). On a brand-new machine,
  run `pip install -r requirements.txt` before the demo.
- **`_cleanup_backup/`** directory contains leftover duplicate files; harmless,
  safe to delete manually if you want a tidier root (left in place to avoid
  deleting anything unexpectedly).
- Harmless console noise: a Matplotlib font-cache message and a Starlette/httpx
  test-client deprecation warning. Neither affects functionality or the demo UI.

---

## 6. Exact Commands for the Final Demo

```bash
# 0) One-time setup (fresh machine)
python3.11 -m venv .venv          # or python3
source .venv/bin/activate
pip install -r requirements.txt   # now includes lap (required for tracking)

# 1) Sanity checks
python3 -m compileall src scripts
pytest

# 2) Run the pipeline on the sample video (≈1 min for 688 frames)
python3 scripts/run_video_demo.py \
  --source data/input_videos/sample.mp4 \
  --output data/outputs/demo.mp4 \
  --zones-config configs/zones.example.json \
  --db data/outputs/analytics.db

# 3) Export a single CSV report and open it
python3 scripts/export_report.py \
  --db data/outputs/analytics.db \
  --out data/outputs/report.csv
open data/outputs/report.csv

# 4) Dashboard (primary demo surface)
streamlit run src/dashboard/streamlit_app.py     # http://localhost:8501

# 5) API (optional, shows programmatic access)
uvicorn src.api.app:app --reload                 # http://localhost:8000/docs
```

> If `data/input_videos/sample.mp4` is missing, the pipeline prints clear
> instructions instead of crashing. Place any short public crowd clip there.

---

## 7. Final Status

**The project is demo-ready.** The full detect → track → zones/lines/alerts →
SQLite → CSV/API/dashboard flow was verified end-to-end on the bundled sample
video. The one blocking issue (missing `lap` dependency) and the report-export
requirement gap are fixed and re-verified.
