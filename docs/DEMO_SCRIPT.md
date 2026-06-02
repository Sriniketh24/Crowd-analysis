# 3-Minute Boss Demo Script

## What You're Showing

A working AI system that watches CCTV footage from a railway station and automatically:
- detects passengers and estimates counts in the frame
- tracks which zones are getting crowded
- fires warning and critical alerts before things get dangerous
- logs everything to a database so you can query historical trends

---

## Before You Start (2 minutes before the meeting)

```bash
# From the project root:
streamlit run src/dashboard/streamlit_app.py
```

Open http://localhost:8501 in a browser and leave it on screen.
Make sure `data/input_videos/sample.mp4` exists (any short crowd video works).

---

## The Script (3 minutes, timed)

### Minute 0:00 – 0:45 · Show the dashboard live

> "This is our real-time crowd analytics dashboard. It connects directly to the
> analytics database produced by our computer vision pipeline."

Point to the four KPI tiles at the top:

- **Total Passengers** — how many people the model detected in the latest frame
- **Camera** — which camera feed this came from
- **Warnings / Criticals** — how many active crowd alert transitions happened

> "These numbers update automatically, so the operations team sees the same view
> in real time — no manual camera watching needed."

---

### Minute 0:45 – 1:30 · Run the pipeline live

Scroll to the **sidebar** and show:

1. The **video source** field — point to `data/input_videos/sample.mp4`.
2. The **zones config** dropdown — select `zones.example.json`.

> "The system is fully config-driven. Zones, counting lines, and alert thresholds
> are all in YAML/JSON files — no code changes needed to add a new camera."

Click **▶ Process Video**.

> "I'm running the full pipeline now — YOLO detection, ByteTrack person tracking,
> zone occupancy counting, and line crossing detection — all in one click."

While it runs (30–60 seconds for a short video):

> "Underneath, this is pretrained YOLO detecting people frame by frame, then a tracker
> giving each person a stable ID across frames. The results go into SQLite so
> nothing is lost even if the dashboard restarts."

---

### Minute 1:30 – 2:30 · Walk through the results

When processing finishes the dashboard auto-refreshes. Walk through each section:

**Zone Occupancy table**
> "Each configured zone shows its current passenger count and alert status —
> green normal, yellow warning, red critical."

**Line Crossing Counts**
> "This is the entry gate line. IN and OUT counts give us net passenger flow —
> useful for platform capacity planning."

**Warning & Critical Alerts table**
> "Every time a zone crossed a threshold, we logged it with the exact frame and
> timestamp. That's the audit trail for safety incidents."

**Historical Occupancy chart**
> "This shows how occupancy in each zone changed across the entire video. You
> can see crowd build-up and dispersal patterns at a glance."

**Annotated Output Video** (bottom of page)
> "The pipeline also saves an annotated MP4 — bounding boxes, zone overlays,
> and live counts burned into the video — ready to attach to an incident report."

---

### Minute 2:30 – 3:00 · Close with the safety pitch

> "The key thing for station operations is that staff get an automatic alert
> *before* a platform becomes dangerously overcrowded — not after.
>
> Demo thresholds are low enough to show alerts on the sample clip; for example,
> `configs/thresholds.yaml` sets `platform_zone` to warning at 5 and critical at 10.
> These thresholds are config values, so operations can tune them without a developer.
>
> We built this on pretrained YOLO — no custom dataset needed for the demo.
> The runner accepts files, webcams, and RTSP/HTTP streams, but real station CCTV
> still needs approval, calibration, and live-stream hardening before deployment."

---

## Quick Answers to Likely Questions

| Question | Answer |
|---|---|
| Does it work on live CCTV? | Initial RTSP/HTTP input is supported through OpenCV, but it has not been validated on approved live Indian Railway CCTV and is not production-hardened. Use environment variables for RTSP credentials. |
| Can we add more zones? | Yes — edit `configs/zones.example.json` and restart processing. |
| Where is the data stored? | SQLite at `data/outputs/analytics.db`. Queryable with any SQL tool. |
| Can we export a report? | Yes — `python scripts/export_report.py --db data/outputs/analytics.db --out data/outputs/report.csv` writes one CSV, then `open data/outputs/report.csv`. |
| Does it store faces? | The database stores analytics metadata only. The demo can write an annotated MP4 of the public sample video; disable output video for private CCTV unless storage is explicitly approved. |
| Can operations query history? | Yes — the FastAPI backend exposes `/metrics/latest`, `/alerts`, `/sessions`. |

---

## Run Command (put this in your slide)

```bash
streamlit run src/dashboard/streamlit_app.py
```
