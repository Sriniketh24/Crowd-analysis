# Boss Update: Head Detection Upgrade

## Short Summary

The railway passenger-counting prototype now supports **two detection modes** instead of
one:

- **Full-Body mode** — the original system: detects whole people.
- **Head mode** — a new model that detects passengers' **heads**.

Both modes feed the exact same analytics (tracking IDs, zone occupancy, line-crossing
counts, crowd alerts, SQLite logging, CSV reports). The web dashboard now runs both on the
same platform video and shows them **side by side** so we can compare. The head model was
**fine-tuned in Google Colab** and imported into the project; nothing was trained on this
laptop.

---

## Why Head Detection Was Added

On railway platform CCTV, full-body detection misses passengers who are far down the
platform or partly hidden — blocked by crowds, benches, pillars, luggage, train doors, or
simply too small/low-resolution to register as a whole person.

A **head is usually still visible** even when the rest of the body is hidden. So a
head-based detector can pick up passengers that a full-body detector misses, giving a more
complete crowd picture — exactly the far-away / dense-crowd cases that matter most on a
platform.

---

## What Is Working Now

Verified by actually running the system on the sample platform video (2026-06-03):

- ✅ **Body detection** — produces annotated video + analytics database.
- ✅ **Head detection** using the **Colab-trained `best.pt`** — produces annotated video +
  analytics database.
- ✅ **Comparison run** — one command generates both body and head outputs side by side.
- ✅ **Tracking IDs** — assigned in both modes.
- ✅ **Zone occupancy counts** — recorded in both modes.
- ✅ **Line-crossing counts** — recorded in both modes.
- ✅ **Crowd alerts** — recorded in both modes.
- ✅ **CSV reports** — exported for both modes, each tagged with its detector mode.
- ✅ **Web app (Streamlit)** — starts cleanly and presents the full body-vs-head comparison.
- ✅ **API (FastAPI)** — health and metrics endpoints respond.

> Honest note: "working" means the software runs end to end and produces outputs. It does
> **not** mean the head model's accuracy has been validated on real Indian Railway CCTV —
> see Limitations.

---

## How The System Works in Plain English

1. The video is split into individual frames (images).
2. A **YOLO** model looks at each frame and draws boxes — around **people** (body mode) or
   around **heads** (head mode).
3. A **tracker** gives each box a stable ID number so the same passenger isn't counted
   twice as they move across frames.
4. **Zones** (areas we draw on the platform) count how many IDs are inside them.
5. **Counting lines** count how many IDs cross them, and in which direction (in/out).
6. **Alerts** trigger automatically when a zone's count passes a set threshold
   (Normal → Warning → Critical).
7. All of this is **saved to a database** and can be **exported to a CSV report**.

---

## What Data Is Collected

| Stored | Purpose |
|--------|---------|
| Detector mode (`body` / `head`) | Which pipeline produced the record |
| Timestamp + frame number | Time-series analysis |
| Anonymised tracking ID (a number) | Avoid double-counting |
| Zone occupancy count | Monitor crowding per area |
| Line-crossing event (in/out) | Passenger flow |
| Alert level (Normal / Warning / Critical) | Trigger staff response |
| Camera / source ID | Multi-camera support |

**No personal names, faces, or identity data are collected or needed** for this analytics
task. It stores counts and anonymous IDs only.

---

## What To Show In The Demo (≈3 minutes)

1. **Open the web app** — `streamlit run src/dashboard/streamlit_app.py` → open
   <http://localhost:8501>.
2. **Show the original sample video** (Section 3) — "this is what the platform camera sees."
3. **Show full-body detection** (Section 6, left) — boxes around whole people.
4. **Show head detection** (Section 6, right) — boxes around heads; point out heads picked
   up where bodies would be hard to see.
5. **Compare the two** — Section 8 shows unique passengers, zone occupancy, line crossings,
   and alerts for each mode next to each other.
6. **Show the analytics / report** — Section 9 previews the CSV; click download.
7. **Explain limitations and next steps** — Section 12: prototype, trained on a public head
   dataset, needs validation on real Indian Railway CCTV.

> Tip: in Section 5, click **Run Body vs Head Comparison** before the meeting so the videos
> are already generated, or run the comparison command in advance (see commands below).

---

## Remaining Limitations (being honest)

- **Not production-ready** — this is a working prototype / pilot.
- The head model was trained on a **public dataset (RPEE-Heads)**, not Indian Railway CCTV.
  Real platform footage will look different (angles, lighting, crowd density) and accuracy
  will differ.
- The accuracy numbers in the training report are measured on that public dataset's own
  validation images — **not** on our sample video and **not** on Indian Railway footage.
- Low resolution and heavy crowding still cause missed detections.
- Small head boxes cause more tracking **ID switches** than full bodies.
- Body and head **counts don't match directly** yet — they detect different things and the
  zone/counting lines were drawn for body-sized boxes. Lines/zones need calibration per mode.
- **Thresholds need station-specific tuning.**
- Live CCTV (RTSP) is supported by the pipeline but needs additional infrastructure and all
  legal/privacy approvals before any real deployment.

---

## Feedback Needed From You (Boss)

1. Should **head detection become the main counting method**, or stay as a comparison aid?
2. Can you provide **real Indian Railway platform CCTV footage** (even short, approved clips)
   so we can validate properly?
3. **Which camera angles** matter most (overhead, oblique, platform-end, entry/exit)?
4. What **accuracy target** would be acceptable for a pilot?
5. **Which zones** should be monitored (specific platform areas, gates, foot-over-bridges)?
6. **What crowd thresholds** should trigger Warning vs Critical alerts?
7. What **dashboard/report format** do you want for operations staff?
8. Where should the **next focus** be: live RTSP CCTV, model accuracy, dashboard, or
   deployment?

---

## Exact Commands (for reference)

```bash
# Body mode
python3 scripts/run_video_demo.py --source data/input_videos/sample.mp4 \
  --output data/outputs/body_demo.mp4 --zones-config configs/zones.example.json \
  --db data/outputs/body_analytics.db --detector-mode body

# Head mode (Colab-trained model)
python3 scripts/run_video_demo.py --source data/input_videos/sample.mp4 \
  --output data/outputs/head_demo.mp4 --zones-config configs/zones.example.json \
  --db data/outputs/head_analytics.db --detector-mode head \
  --model models/fine_tuned/head_detector/weights/best.pt

# Both at once (recommended for the demo)
python3 scripts/run_comparison_demo.py --source data/input_videos/sample.mp4 \
  --zones-config configs/zones.example.json

# Reports
python3 scripts/export_report.py --db data/outputs/body_analytics.db --out data/outputs/body_report.csv
python3 scripts/export_report.py --db data/outputs/head_analytics.db --out data/outputs/head_report.csv

# Web app
streamlit run src/dashboard/streamlit_app.py
```
