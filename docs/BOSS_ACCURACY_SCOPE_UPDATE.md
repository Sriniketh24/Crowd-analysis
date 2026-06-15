# Boss Update: Accuracy Improvements and Project Scope

## Short Summary

After the feedback to increase accuracy and brief the problem statement, assumptions, scope, and steps, the MVP was updated with accuracy tuning controls and clearer documentation.

The system now supports configurable body, head, and hybrid body + head detection runs, tuning sweeps, comparison outputs, CSV reports, and dashboard controls for explaining and testing detection settings. Hybrid mode runs body and head detection together, then removes likely duplicate detections before tracking. The update improves the ability to tune and compare results, but it does not claim final production accuracy because true accuracy requires manually counted or labeled ground truth.

## Problem Statement

Indian Railway platforms can become crowded during peak hours, train arrivals, train departures, and service disruptions. Manual monitoring from CCTV is difficult because staff must continuously watch multiple camera views and estimate passenger count, crowd density, and passenger movement by eye.

This project is an AI-based MVP for railway platform passenger counting and crowd analytics. It analyzes CCTV-style platform video, detects passengers, tracks them with anonymous IDs, counts people in configured zones, counts line crossings, raises crowd alerts, and stores analytics for review.

## Objective

The objective of this MVP is to demonstrate a working passenger counting and crowd analytics pipeline for railway platform video.

The current system compares three approaches:

- Full-body passenger detection using pretrained YOLO person detection.
- Head-based passenger detection using the Colab-trained `best.pt` model.
- Hybrid body + head detection using both detectors with duplicate removal.

All three modes feed the same analytics flow: tracking, zone occupancy, line crossing, alerts, SQLite logging, CSV export, and Streamlit dashboard comparison.

## Assumptions Made

- The camera is fixed or mostly fixed during analysis.
- Zones and virtual counting lines are configured separately for each camera.
- Crowd thresholds vary by station, platform, and camera angle.
- The MVP counts visible persons or heads; it does not identify individual passengers.
- Full-body detection is the baseline method using pretrained YOLO person detection.
- Head detection is used for distant, dense, or partially occluded scenes where full bodies may not be visible.
- Hybrid mode uses both body and head detection, then removes likely duplicate detections before tracking.
- The head detector was trained in Google Colab using the RPEE-Heads dataset, not Indian Railway CCTV.
- The current demo video is public CCTV-style railway platform footage, not confirmed operational Indian Railway CCTV.
- Real deployment requires approved CCTV access, privacy clearance, hardware planning, and camera-specific calibration.
- Real CCTV use should avoid storing personally identifying footage unless explicitly approved.

## In Scope

- Recorded video testing.
- CCTV-style platform sample testing.
- Full-body passenger detection.
- Head-based passenger detection using `models/fine_tuned/head_detector/weights/best.pt`.
- Hybrid body + head detection with duplicate removal.
- Body, head, and hybrid comparison on the same input video.
- Passenger tracking with anonymous numeric IDs.
- Zone occupancy counting.
- Virtual line crossing counting.
- Crowd warning and critical alert logic.
- Configurable zones, lines, thresholds, and detector settings.
- SQLite analytics logging.
- CSV report export.
- Streamlit dashboard demo.
- Basic FastAPI analytics access.
- Annotated output videos for review.
- Tests for zone counting, line crossing, alert logic, API, metrics, and persistence.

## Out of Scope

- Guaranteed production-level accuracy.
- Official safety certification.
- Live integration with the operational Indian Railway CCTV network.
- Alert delivery to railway control room systems.
- Facial recognition or passenger identity recognition.
- Long-term storage of raw CCTV footage.
- Multi-camera passenger re-identification.
- Automatic camera calibration.
- Fully automated zone generation.
- Production monitoring, failover, and uptime guarantees.
- Large-scale validation on Indian Railway CCTV without approved data.
- Custom training on private railway CCTV unless footage is provided and approved for use.

## Steps Involved

1. Select or collect a suitable railway platform video.
2. Configure platform zones, virtual lines, and crowd thresholds.
3. Run full-body YOLO passenger detection as the baseline.
4. Run the fine-tuned head detector for head-based counting.
5. Run hybrid body + head detection and remove likely duplicates.
6. Track detections across frames using stable tracking IDs.
7. Count current occupancy inside configured zones.
8. Count passenger movement across virtual lines.
9. Generate warning or critical alerts when thresholds are crossed.
10. Save analytics events and snapshots to SQLite.
11. Export CSV reports for review.
12. Generate annotated body, head, and hybrid demo videos.
13. Compare body, head, and hybrid results in the Streamlit dashboard.
14. Tune settings and validate against manual or labeled ground truth.

## Accuracy Improvements Added

- Configurable confidence threshold so weak detections can be included or filtered depending on the scene.
- Image size tuning through `--imgsz`, useful for smaller and more distant passengers or heads.
- IoU/NMS tuning through `--iou`, useful when passengers or heads overlap in dense scenes.
- Tracker selection through `--tracker bytetrack|botsort` for ID stability comparison.
- Maximum detections control through `--max-det` for crowded scenes.
- Optional test-time augmentation through `--augment`, which can improve detection at the cost of speed.
- Head-specific configurable defaults for confidence, image size, max detections, and tracker overrides.
- Tuning sweep script: `scripts/tune_detection_settings.py`.
- Body/head tuned comparison outputs:
  - `data/outputs/body_tuned_demo.mp4`
  - `data/outputs/head_tuned_demo.mp4`
  - `data/outputs/body_tuned_report.csv`
  - `data/outputs/head_tuned_report.csv`
- Hybrid comparison output support:
  - `data/outputs/hybrid_demo.mp4`
  - `data/outputs/hybrid_analytics.db`
  - `data/outputs/hybrid_report.csv`
- Streamlit dashboard controls and explanations for confidence, image size, IoU, tracker, max detections, and augmentation.
- Streamlit dashboard option for **Hybrid body + head**, including output video, metrics, zone occupancy, line counts, and CSV report preview.
- Manual ground-truth comparison workflow through `scripts/compare_manual_counts.py`, using `data/manual_ground_truth/example_counts.csv` as the input template.

## Current Results

The current results below come from `docs/FINAL_ACCURACY_AND_SCOPE_TEST_REPORT.md` only. These are test and output observations, not final accuracy percentages.

- Code sanity check: `python3 -m compileall src scripts` passed.
- Test suite: `pytest` passed with 55 tests and 1 warning.
- Body tuned demo completed on `data/input_videos/sample.mp4`.
  - Output video: `data/outputs/body_tuned_demo.mp4`
  - Database: `data/outputs/body_tuned_analytics.db`
  - Report: `data/outputs/body_tuned_report.csv`
  - Frames processed: 803/803
  - Average detections/frame: 18.36
  - Maximum detections/frame: 24
- Head tuned demo completed on the same sample video.
  - Output video: `data/outputs/head_tuned_demo.mp4`
  - Database: `data/outputs/head_tuned_analytics.db`
  - Report: `data/outputs/head_tuned_report.csv`
  - Frames processed: 803/803
  - Average detections/frame: 3.17
  - Maximum detections/frame: 11
- Tuning sweep completed and wrote:
  - `data/outputs/tuning_results.csv`
- The quick tuning sweep tested head mode for 75 frames at:
  - confidence 0.15, image size 960, IoU 0.50, ByteTrack
  - confidence 0.20, image size 960, IoU 0.50, ByteTrack
- Both quick sweep rows produced the same detector/tracker counts:
  - total detections: 395
  - unique tracks: 33
  - average detections/frame: 5.2667
- Streamlit dashboard smoke test passed with HTTP 200 and no errors.
- Manual ground-truth comparison was not produced because the manual count template is still empty.

Important body/head/hybrid observation: on this specific public sample clip, body detection previously produced more detections than head detection. This is not an accuracy conclusion. Body detection is better when the full person is visible. Head detection is used for far, packed, or occluded passengers where bodies are hidden. Hybrid mode uses both and removes likely duplicates, but it still requires manual ground truth before we can say whether it is more accurate on a given camera.

## Important Note About Accuracy

True accuracy requires ground-truth labeled frames or manually counted frame/time segments.

The current update improves the ability to tune, compare, and inspect body, head, and hybrid detection results. It does not prove final accuracy improvement because the current sample video does not have verified ground truth.

Final accuracy validation requires approved Indian Railway CCTV footage and manual or annotated ground truth. After that, we can report defensible metrics such as counting error, missed detections, false positives, and detector performance by camera angle.

## Remaining Limitations

- The current sample video is public railway platform footage, not approved Indian Railway CCTV.
- There is no manual ground truth yet for the current sample video.
- The head model was trained on RPEE-Heads, so there may be domain mismatch with Indian Railway camera angles, lighting, crowd density, clothing, compression, and station layout.
- Body and head counts are not directly comparable because they detect different visual targets and use different reference points.
- Hybrid counts depend on the quality of the body/head models, duplicate-removal rules, and camera-specific ROI configuration.
- Dense crowds, occlusion, low resolution, and far-away passengers can still cause missed detections.
- Tracking IDs can switch, especially for small head boxes.
- Zones, lines, and thresholds must be calibrated per camera.
- Live CCTV integration and production deployment still require infrastructure, privacy, legal, and operational approvals.

## What I Need From You

- Approved real Indian Railway CCTV clips for testing.
- Target camera angles to prioritize.
- Acceptable accuracy target for a pilot.
- Which platform zones should be monitored.
- Warning and critical crowd thresholds for each zone.
- Sample manual ground-truth counts, if available.
- Direction on whether the next priority should be more training, live CCTV integration, or dashboard polish.

## Next Steps

1. Get approved Indian Railway CCTV clips for validation.
2. Select representative camera angles and short test segments.
3. Create manual ground-truth counts for selected frames or time ranges.
4. Run body, head, and hybrid detection against the same segments.
5. Compare body/head/hybrid counts against manual ground truth.
6. Tune confidence, image size, IoU, tracker, and thresholds using measured error.
7. Label representative head/person frames if model retraining is approved.
8. Fine-tune the head detector again with more relevant railway footage.
9. Calibrate zones and lines per camera.
10. Decide the next pilot focus: accuracy improvement, live CCTV ingestion, or dashboard/reporting.

## Professional Email/Message To Boss

Subject: Update on Passenger Counting Accuracy and Project Scope

Sir,

I have addressed your feedback on increasing accuracy and preparing a clearer brief of the problem statement, assumptions, scope, and project steps.

The system now supports accuracy tuning controls such as confidence threshold, image size, IoU, tracker selection, max detections, and optional test-time augmentation. I also added a tuning sweep workflow and body/head/hybrid comparison outputs, including annotated videos and CSV reports. Hybrid mode uses the body detector where the full person is visible and the head detector where passengers are far, packed, or occluded, then removes likely duplicates before tracking.

I have also prepared a structured scope document covering the problem statement, assumptions, in-scope items, out-of-scope items, steps involved, limitations, and next steps.

One important point: the current update improves the ability to tune and compare the models, including the hybrid approach, but final accuracy cannot be claimed without ground truth. For true accuracy validation, we need approved Indian Railway CCTV clips and manual or labeled passenger counts for selected frames/time ranges.

Could you please confirm the next priority: further model training, live CCTV integration, or dashboard/report polish? Also, if possible, please provide sample CCTV clips, target camera angles, expected accuracy target, zones to monitor, and crowd thresholds.

Regards,  
Sriniketh

## Short WhatsApp/Teams Version

Sir, I have addressed your feedback. The project now includes accuracy tuning controls, a tuning sweep, body/head/hybrid outputs, and clearer documentation for the problem statement, assumptions, scope, out-of-scope items, steps, limitations, and next steps.

Hybrid mode uses body detection for visible full persons and head detection for far/packed/occluded passengers, then removes likely duplicates. Important note: I have not claimed final accuracy because true accuracy needs ground-truth manual counts or labeled Indian Railway CCTV frames. Please share approved CCTV clips, target camera angles, expected accuracy target, zones, and crowd thresholds. Also please confirm whether I should prioritize further training, live CCTV integration, or dashboard polish next.

## 2-Minute Verbal Explanation Script

Sir, after your feedback, I updated the project in two areas: accuracy improvement support and clearer project scope documentation.

The problem we are solving is passenger counting and crowd monitoring on railway platforms using CCTV-style video. The system detects passengers, tracks them with anonymous IDs, counts people inside configured platform zones, counts movement across virtual lines, raises warning or critical crowd alerts, and stores the analytics in SQLite with CSV reports and a Streamlit dashboard.

For accuracy, I added tuning controls instead of hardcoding one setting. We can now adjust confidence threshold, image size, IoU, tracker type, maximum detections, and optional test-time augmentation. I also added a tuning sweep script so we can compare settings systematically. The project now generates body detection, head detection, and hybrid body + head outputs for the same sample video.

The current comparison uses three approaches. Body detection uses pretrained YOLO person detection and is better when the full person is visible. Head detection uses the Colab-trained `best.pt` model and is used for far, packed, or occluded passengers where bodies are hidden. Hybrid mode uses both, then removes likely duplicates before tracking. I am not presenting any of these as final accuracy because we do not yet have manual ground truth.

The main limitation is that true accuracy can only be measured with approved Indian Railway CCTV footage and manually counted or labeled frames. Once we have that, I can calculate real count error and tune the settings properly.

For the next step, I need your guidance on priority: should I focus on more model training, live CCTV integration, or improving the dashboard? I also need sample CCTV clips, target camera angles, acceptable accuracy target, zones to monitor, and crowd thresholds.

## Checklist Of Files To Show

- Streamlit app: `src/dashboard/streamlit_app.py`
- Body tuned demo video: `data/outputs/body_tuned_demo.mp4`
- Head tuned demo video: `data/outputs/head_tuned_demo.mp4`
- Hybrid demo video: `data/outputs/hybrid_demo.mp4`
- Tuning CSV: `data/outputs/tuning_results.csv`
- Body tuned report: `data/outputs/body_tuned_report.csv`
- Head tuned report: `data/outputs/head_tuned_report.csv`
- Hybrid report: `data/outputs/hybrid_report.csv`
- Problem/scope document: `docs/PROBLEM_STATEMENT_SCOPE_AND_STEPS.md`
- Boss update document: `docs/BOSS_ACCURACY_SCOPE_UPDATE.md`
- Final test source document: `docs/FINAL_ACCURACY_AND_SCOPE_TEST_REPORT.md`
