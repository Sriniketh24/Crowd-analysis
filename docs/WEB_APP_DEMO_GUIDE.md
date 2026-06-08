# Web App Demo Guide
# Indian Railway Passenger Counting System

Prepared: 2026-06-02

---

## How to Run the App

```bash
# From the project root, with the virtual environment activated:
streamlit run src/dashboard/streamlit_app.py
```

Then open **http://localhost:8501** in a browser.

No extra setup is needed — the app reads pre-existing output files automatically if they exist.

---

## What Each Section Means

### Section 1 — What does this system do?
Plain-English explanation of the detection → tracking → zone counting → alert pipeline.
Show this to your boss first. It requires no technical background.

### Section 2 — Input Video
Displays `data/input_videos/sample.mp4` — the selected Pexels **People on Platform on Train Station** clip.
Tell your boss: *"This is CCTV-like public railway platform footage before the AI processes it. It is not confirmed CCTV."*

### Section 3 — Run Passenger Counting Analysis
Click **▶ Run Passenger Counting Analysis** to process the video through the full AI pipeline.
The app shows each stage as it runs — no silent freezing.
Processing takes 1–3 minutes on a standard laptop CPU.

### Section 4 — Annotated Output Video
Displays `data/outputs/demo.mp4` — the AI-processed video with overlays.
Walk your boss through the overlay guide in the app:
- boxes = detected passengers
- IDs = tracked passengers (no double-counting)
- coloured zones = monitored platform areas
- counting line = passenger flow measurement
- counter = live occupancy numbers
- alert banner = crowding warning

### Section 5 — Real-Time Analytics Summary
Five KPI metrics from the database:
- Unique passengers seen in the session
- Camera ID
- Frames processed
- Warning count
- Critical alert count

These come from `data/outputs/analytics.db`.

### Section 6 — Zone Occupancy, Line Crossings & Alerts
Three tables:
1. **Zone occupancy** — latest headcount per platform zone
2. **Line crossing counts** — how many people crossed each counting line, by direction
3. **Alerts** — all WARNING and CRITICAL crowding events, with timestamps

Plus a **historical occupancy chart** showing how crowd levels changed over the session.

### Section 7 — Exported Report
Preview and download of `data/outputs/report.csv`.
If the file doesn't exist yet, a **Generate Report CSV** button runs `export_report.py` automatically.

### Section 8 — What This Means for Indian Railways
Business context: how the system helps with platform crowding, passenger flow management, safety response, and operational planning. Also lists current limitations clearly.

### Section 9 — Next Steps Toward Production
Prioritised roadmap: live RTSP integration, fine-tuning, multi-camera dashboard, alert notifications, TensorRT deployment.

---

## Files the App Uses

| File | Purpose | If missing |
|------|---------|-----------|
| `data/input_videos/sample.mp4` | Selected Pexels CCTV-like railway platform sample to display and process | Section 2 shows a warning with placement instructions |
| `data/outputs/demo.mp4` | Annotated output video | Section 4 shows a "run analysis" prompt |
| `data/outputs/analytics.db` | SQLite database of all analytics | Sections 5 & 6 show "run analysis first" message |
| `data/outputs/report.csv` | Exported analytics CSV | Section 7 shows Generate button |
| `configs/zones.example.json` | Zone and counting-line config used by the pipeline | Pipeline will fail if missing — file exists by default |

---

## How to Explain It to Your Boss

Suggested talking points:

1. **Start with Section 1.** Read out the table. It tells the story in 30 seconds.

2. **Show Section 2 (input video).** Say: *"This is real public railway platform footage with a CCTV-like elevated angle. The AI processes this automatically."*

3. **Show Section 4 (annotated output).** Say: *"Here is what the AI sees. Every person is detected, given an ID, and tracked across the platform. The zones turn red when they get too crowded."*

4. **Show Section 5 (KPI metrics).** Highlight unique passenger count and alert counts. Say: *"This is what the system would report to a control room in real time."*

5. **Show Section 6 (charts).** Point to the historical occupancy chart. Say: *"This shows crowd build-up over time — exactly what a station manager needs to plan their response."*

6. **Close with Section 8.** Say: *"The system works end-to-end today on the selected Pexels platform sample. The next step is connecting it to approved real CCTV and fine-tuning the AI on Indian platform footage."*

---

## What Still Needs Real CCTV and Fine-Tuning

| Limitation | Details |
|------------|---------|
| **Model accuracy** | YOLOv11n is a general person detector. Indian platforms have dense crowds, overhead camera angles, and cluttered backgrounds — fine-tuning is required for reliable production accuracy. |
| **Live stream integration** | The system already supports RTSP URLs via `--source`. However, latency, reconnection, and frame-drop handling need validation on real network streams. |
| **Zone calibration** | Current zone polygons and alert thresholds are examples. Each station needs zones drawn from its actual camera view and thresholds tuned to its real capacity. |
| **Multi-camera scale** | The dashboard currently shows one session at a time. Multi-camera views require additional UI development. |
| **Annotation dataset** | Fine-tuning requires labeled images from real Indian Railway CCTV. See `docs/ANNOTATION_WORKFLOW.md` for the labeling workflow. |
