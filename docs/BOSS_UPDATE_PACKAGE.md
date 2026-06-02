# Boss Update Package

## 1. Short Project Status

The current Indian Railway passenger counting project is a working MVP/pilot prototype. It processes a real public Mumbai suburban railway platform sample video, detects people, assigns tracking IDs, counts people in configured zones, counts directional line crossings, raises crowding alerts, logs analytics to SQLite, exports CSV reports, serves FastAPI endpoints, and provides a Streamlit web app for demo and review.

This should be presented as a validated local prototype, not as a production-ready railway CCTV deployment. The next major step is to test on approved Indian Railway CCTV clips, label representative frames, fine-tune/validate the model, and harden live RTSP operation.

## 2. What Is Working Now

- Real video input: verified on `data/input_videos/sample.mp4`, a real public Mumbai platform clip.
- Passenger/person detection: YOLO detects people frame by frame.
- Tracking IDs: tracked passengers are assigned ephemeral IDs for counting logic.
- Zone occupancy: configured platform/concourse polygons report current occupancy.
- Line crossing: configured virtual line reports IN/OUT movement counts.
- Crowd alerts: NORMAL/WARNING/CRITICAL alert transitions are generated from thresholds.
- Database logging: analytics are written to SQLite at `data/outputs/analytics.db`.
- CSV report: `data/outputs/report.csv` exports analytics from the database.
- FastAPI: local API endpoints were verified with HTTP 200 responses.
- Streamlit web app: local dashboard starts and renders the demo workflow.
- Real sample video demo: `data/outputs/demo.mp4` shows boxes, IDs, zones, counts, and alerts.

## 3. What Demo Files To Show

- `data/input_videos/sample.mp4` - original real public Mumbai platform sample video.
- `data/outputs/demo.mp4` - processed/annotated output video.
- `data/outputs/report.csv` - exported analytics report.
- Streamlit app - run locally at `http://localhost:8501`.
- API docs, if useful - run locally at `http://localhost:8000/docs`.

Useful demo commands:

```bash
cd /Users/sriniketh/crowd-analysis
source .venv/bin/activate
streamlit run src/dashboard/streamlit_app.py
```

Optional API demo:

```bash
cd /Users/sriniketh/crowd-analysis
source .venv/bin/activate
uvicorn src.api.app:app --reload
```

Optional rerun of the full analysis pipeline:

```bash
cd /Users/sriniketh/crowd-analysis
source .venv/bin/activate
python scripts/run_video_demo.py \
  --source data/input_videos/sample.mp4 \
  --output data/outputs/demo.mp4 \
  --zones-config configs/zones.example.json \
  --db data/outputs/analytics.db

python scripts/export_report.py \
  --db data/outputs/analytics.db \
  --out data/outputs/report.csv
```

## 4. How To Demo It In 3 Minutes

1. Open the web app at `http://localhost:8501`.
   Say: "This is the local dashboard for the passenger counting prototype."

2. Show the original video.
   Say: "This is the raw platform video input. For the MVP I am using a real public Mumbai railway platform clip, not private CCTV."

3. Click the run-analysis button if you want to process live during the meeting, or use the already generated output.
   Say: "The pipeline runs person detection, tracking, zone counting, line crossing, crowd alerting, and database logging."

4. Show the processed video.
   Say: "The boxes are detected passengers, the IDs are temporary tracking IDs, the colored areas are configured zones, and the line is used for IN/OUT movement counting."

5. Point to counts and alerts.
   Say: "The system estimates how many people are inside each zone and raises WARNING or CRITICAL alerts when thresholds are crossed. These thresholds are configurable per camera and zone."

6. Show the CSV/report section.
   Say: "The same analytics are saved to SQLite and exported into CSV so station teams can review sessions, occupancy, crossings, and alerts."

7. Close with next steps.
   Say: "This is demo-ready as an MVP. For Indian Railway deployment, the next step is approved real CCTV footage, accuracy validation, fine-tuning on platform data, live RTSP hardening, and multi-camera dashboard work."

## 5. What To Say About Technical Direction

YOLO plus tracking is a good foundation for this project because it supports real-time person detection, works with pretrained models for an MVP, and has a clear upgrade path to fine-tuned weights and GPU deployment.

The current model is a prototype baseline. It uses pretrained YOLO11 for person detection and tracking for short-term passenger IDs. These IDs are useful for counting and flow analytics, but they are not identity recognition and should not be presented that way.

For Indian Railway deployment, the model must be tested and fine-tuned on real platform CCTV footage. Indian platforms include occlusion, low camera angles, compression, seated passengers, luggage, train-side clutter, and peak-hour density that public demo footage cannot fully validate.

RTSP/live CCTV support should be prioritized next. The code has initial support for file, webcam, RTSP, and HTTP sources, but a production pilot needs long-running reconnect handling, camera health monitoring, latency tracking, secure credential handling, and a worker process per camera.

For production scaling, GPU acceleration should be evaluated. A practical path is YOLO11s or YOLO11m on CUDA/TensorRT for a pilot, and NVIDIA DeepStream/GStreamer for larger multi-camera deployments.

## 6. What To Ask My Boss

- Can you provide approved sample Indian railway CCTV clips for testing?
- What camera resolution and FPS should we expect?
- How many cameras are needed per platform and per station?
- Which zones matter most: platform edge, stairs, footbridge, entry/exit, ticket area, waiting area, or train doors?
- What accuracy target is acceptable for passenger counts and alerts?
- What crowding thresholds should define NORMAL, WARNING, and CRITICAL?
- Should alerts be real-time, daily reports, or both?
- What reporting format do you want: dashboard, CSV, PDF, API, email, or station-control-room display?
- What privacy and storage policy should be followed for CCTV video and annotated output?
- Should I prioritize accuracy, dashboard polish, RTSP integration, or deployment setup next?

## 7. Remaining Limitations

- The current version is not production-ready.
- Accuracy depends on camera angle, resolution, lighting, compression, crowd density, and occlusion.
- The pretrained model may miss occluded, seated, small, or distant passengers.
- Real approved Indian Railway CCTV data is needed before making accuracy or deployment claims.
- Manual zone and line setup is still needed for each camera view.
- Multi-camera re-identification is not implemented and may need future work if the requirement becomes tracking the same person across cameras.
- Current RTSP support is initial; it has not been proven as a long-running live CCTV service.
- API and dashboard are local demo interfaces, not secured production operator systems.

## 8. Suggested Next Sprint

1. Get approved real Indian Railway CCTV clips from representative platform cameras.
2. Test current model accuracy on those clips and record failure cases.
3. Label sample frames with passenger/person boxes and manual count ground truth.
4. Fine-tune YOLO11 on labeled Indian platform data and compare against the pretrained baseline.
5. Add and test RTSP live mode with reconnect, FPS control, stream health, and secure credentials.
6. Improve the dashboard for multiple cameras and station-level views.
7. Define alert thresholds with a station supervisor or operations stakeholder.

## 9. Email To Boss

Subject: Indian Railway Passenger Counting MVP - Demo Ready and Next Steps

Dear [Boss Name],

I have prepared the current version of the Indian Railway passenger counting system for review. It is a working MVP/pilot prototype that runs end-to-end on a real public Mumbai railway platform sample video.

At this stage, the system can detect passengers, assign temporary tracking IDs, count zone occupancy, count IN/OUT line crossings, identify crowded areas using configurable thresholds, save analytics to SQLite, export a CSV report, expose FastAPI endpoints, and show the workflow in a Streamlit dashboard.

I want to be clear that this is not production-ready yet. It has not been validated on approved live Indian Railway CCTV, and the current model is a pretrained YOLO baseline. The next major step is to test it on real platform CCTV footage, label representative frames, fine-tune/validate the model, and harden live RTSP/multi-camera operation.

For the next phase, I would like your guidance on:

- Whether you can provide approved sample CCTV clips from Indian railway platforms.
- Expected camera resolution, FPS, and number of cameras per platform/station.
- The most important zones to monitor.
- Acceptable accuracy targets and crowding thresholds.
- Whether alerts should be real-time, daily reports, or both.
- The preferred reporting format.
- Privacy and storage rules for video and annotated output.
- Whether I should prioritize model accuracy, RTSP integration, dashboard improvements, or deployment setup next.

I can demo the current MVP using the web app, processed video, analytics tables, and CSV report.

Regards,  
[Your Name]

## 10. Short WhatsApp/Teams Message

Hi [Boss Name], I have the Indian Railway passenger counting MVP ready for review. It runs end-to-end on a real public Mumbai platform sample video: passenger detection, tracking IDs, zone occupancy, line crossing counts, crowd alerts, SQLite logging, CSV export, API, and Streamlit dashboard. It is still a pilot prototype, not production-ready. The next key step is approved real railway CCTV footage for validation, labeling, fine-tuning, and RTSP/live-camera testing. Please let me know what you want me to prioritize next: accuracy, RTSP integration, dashboard, or deployment.

## 11. 3-Minute Verbal Demo Script

"This is the current MVP for Indian Railway passenger counting and crowd analytics. I am showing it as a pilot prototype, not as a production-ready system yet.

First, this is the raw input video. For this demo I am using a real public Mumbai suburban railway platform clip. The goal is to simulate what a platform CCTV stream would provide to the system.

When I run the analysis, the pipeline uses YOLO to detect passengers frame by frame. It then tracks detected people with temporary IDs so the system can reason about movement and avoid simple double-counting within the same scene.

Now this is the processed output video. The boxes show detected passengers. The labels show temporary tracking IDs. The colored polygons are configured platform zones, and the virtual line is used for passenger flow counting. When people move across that line, the system records IN and OUT counts.

The dashboard also shows occupancy inside each configured zone. If the occupancy crosses configured thresholds, the system raises WARNING or CRITICAL crowd alerts. These thresholds are not hardcoded; they can be set per camera and per zone after discussing station capacity and safety rules.

Behind the dashboard, the system stores analytics in SQLite. It logs sessions, frames, zone occupancy, line crossing events, and crowd alerts. The same data can be exported into a CSV report for review or operational planning.

There is also a FastAPI backend, so the same metrics can be accessed programmatically by another dashboard or station control system.

The honest status is: the MVP pipeline works end-to-end on sample footage. For an actual Indian Railway pilot, we need approved real CCTV clips, manual validation, labeled frames for fine-tuning, RTSP live stream hardening, and multi-camera support. My recommendation for the next sprint is to get real CCTV samples, measure baseline accuracy, label frames, fine-tune YOLO11, and define alert thresholds with operations input."

## 12. One-Page Status Report Summary

### Project

Indian Railway platform passenger counting and crowd analytics MVP.

### Current Status

Working local MVP/pilot prototype. Verified on 2026-06-02 with the full pipeline running on `data/input_videos/sample.mp4`, a real public Mumbai platform sample video. The latest end-to-end test report records 48 passing tests, 600 processed frames, a generated annotated video, populated SQLite database, CSV export, working FastAPI endpoints, and a working Streamlit dashboard.

### Capabilities Demonstrated

- Person/passenger detection using pretrained YOLO11.
- Temporary tracking IDs for movement/counting logic.
- Zone occupancy counting with configured polygons.
- Directional line crossing counts.
- Crowd alerts using configurable thresholds.
- SQLite persistence for sessions, frames, occupancy, crossings, and alerts.
- CSV analytics report export.
- Local Streamlit dashboard.
- Local FastAPI API and API docs.

### Demo Assets

- Input video: `data/input_videos/sample.mp4`
- Annotated output: `data/outputs/demo.mp4`
- Database: `data/outputs/analytics.db`
- CSV report: `data/outputs/report.csv`
- Dashboard: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`

### Honest Limitations

This is not production-ready. It has not been validated on approved live Indian Railway CCTV. It has no measured ground-truth accuracy on station footage. The current model is pretrained, not fine-tuned on Indian platform data. RTSP support is initial and not yet hardened for long-running live deployment. Multi-camera orchestration and secured operator access are not complete.

### Recommended Next Actions

Get approved CCTV clips, test baseline accuracy, label representative frames, fine-tune YOLO11, validate passenger count and alert accuracy, harden RTSP live mode, improve the dashboard for multiple cameras, and define crowd thresholds with station operations.

### Decisions Needed From Boss

Approval/access for CCTV samples, target camera specs, priority zones, acceptable accuracy, alert thresholds, reporting format, privacy/storage rules, and next priority between accuracy, RTSP, dashboard, and deployment.
