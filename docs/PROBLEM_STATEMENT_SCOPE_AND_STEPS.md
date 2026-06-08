# Problem Statement, Assumptions, Scope, and Steps

## 1. Problem Statement

Indian railway platforms can become crowded, especially during peak hours, train arrivals, train departures, and disruption periods. Manual monitoring from CCTV feeds is difficult because staff must continuously watch multiple cameras and estimate passenger counts, crowd density, movement flow, and risk areas by eye.

The goal of this project is to build an AI-based MVP/prototype that can analyze railway-platform CCTV-style video, detect and count passengers, identify crowded zones, monitor movement across configured virtual lines, and generate analytics that can support station safety and operational planning.

This project should be treated as a working MVP/prototype, not a production-certified system. It has been demonstrated on public/sample platform footage and still requires validation on approved Indian Railway CCTV footage before production use.

## 2. Objective

The objective is to develop a computer vision MVP for Indian railway platform passenger counting and crowd analytics.

The current system compares two passenger detection approaches:

| Mode | What it detects | Main purpose |
|---|---|---|
| Full-body detection | Whole passengers/persons | Baseline passenger detection using pretrained YOLO person detection |
| Head detection | Passenger heads | Improved detection in distant, dense, or partially occluded CCTV views |

Both modes feed the same analytics pipeline:

- passenger tracking IDs
- zone occupancy counts
- line crossing counts
- crowd warning and critical alerts
- SQLite logging
- CSV export
- Streamlit dashboard comparison
- FastAPI access to stored metrics

## 3. Why Head Detection Was Added

Full-body detection can fail in railway CCTV views when the entire body is not visible. This can happen when passengers are far from the camera, partially hidden by other people, blocked by pillars, standing near train doors, or visible only at small scale in low-resolution footage.

Head detection was added because a passenger's head may remain visible even when the full body is hidden. In dense platform scenes, head detection can therefore provide a more useful counting signal than full-body detection alone.

The current head detector was fine-tuned in Google Colab using the RPEE-Heads dataset and imported into this project as:

```text
models/fine_tuned/head_detector/weights/best.pt
```

The available training metrics are from the RPEE-Heads validation split, not from Indian Railway CCTV. Therefore, these numbers should not be presented as production accuracy for Indian railway stations.

## 4. Assumptions Made

- The footage is from a fixed or mostly fixed camera.
- The camera angle is stable during analysis.
- Platform zones and counting lines can be manually configured per camera.
- Crowd thresholds are configurable and may differ by station, platform, and camera view.
- Passengers are counted based on visible body or head detections, not identity recognition.
- The system does not identify individuals.
- The system does not require passenger names, face recognition, ticket data, or personal identity information.
- The current body detector uses pretrained YOLO person detection as the MVP baseline.
- The current head detector uses a Colab-trained model based on available/public head-detection data.
- The head detector may not fully match actual Indian Railway CCTV conditions because of domain differences in camera angle, crowd density, resolution, lighting, compression, and station layout.
- The current sample video is CCTV-style railway platform footage for demonstration, but it is not confirmed operational Indian Railway CCTV.
- Real deployment would require approved RTSP/CCTV access, hardware details, privacy approval, legal clearance, and camera-specific calibration.
- Any real CCTV use must avoid storing personally identifying footage unless explicitly approved.

## 5. In Scope

The following items are within scope for the current MVP/prototype:

- video file testing
- CCTV-style sample video testing
- full-body passenger detection
- head detection using the Colab-trained `best.pt`
- body-vs-head comparison on the same video
- passenger tracking IDs
- zone occupancy counting
- virtual line crossing counting
- crowd warning and critical alert generation
- configurable zones, lines, cameras, and thresholds
- SQLite logging
- CSV report export
- Streamlit web app demo
- basic FastAPI endpoints for analytics access
- annotated output videos for review
- tests for line counting, zone counting, alert logic, API, metrics, and persistence

## 6. Out of Scope For Current MVP

The following items are not included in the current MVP:

- guaranteed production-level accuracy
- official safety certification
- facial recognition or passenger identification
- storing long-term raw CCTV footage
- real-time deployment to Indian Railway infrastructure
- integration with the actual railway CCTV network
- alert delivery to railway control room systems
- multi-camera passenger re-identification across cameras
- automatic camera calibration
- fully automated zone generation
- large-scale validation on Indian Railway CCTV unless approved data is provided
- production monitoring, failover, and uptime guarantees
- legal/privacy approval for live station deployment
- custom training on private railway CCTV data unless such footage is provided and approved for use

## 7. Steps Involved

The current process is:

1. Collect or select a suitable CCTV-style railway/platform test video.
2. Define platform zones and virtual counting lines in configuration files.
3. Run the full-body YOLO detector as the baseline.
4. Run the fine-tuned head detector for dense or distant passenger views.
5. Track detections across frames using stable tracking IDs.
6. Count occupancy inside configured platform zones.
7. Count movement across virtual lines and record direction.
8. Generate crowd alerts based on warning and critical thresholds.
9. Save analytics events and snapshots to SQLite.
10. Export CSV reports for review.
11. Display body-vs-head comparison in the Streamlit web app.
12. Review results visually and tune model/settings.
13. Improve accuracy using real station footage, manual labels, and additional fine-tuning.

## 8. Data Collected

The MVP stores operational analytics data, not passenger identity data.

Examples of data collected:

- timestamp
- camera/source ID
- detector mode (`body` or `head`)
- frame number
- anonymized tracking ID
- bounding box location
- confidence score when available
- zone occupancy counts
- line crossing events
- crossing direction
- alert level (`NORMAL`, `WARNING`, `CRITICAL`)
- model/tracker metadata for the run

The system does not require names, passenger identities, ticket details, or face recognition. Tracking IDs are numeric IDs used for counting movement across frames; they are not identity records.

## 9. Accuracy Improvement Plan Summary

The boss asked to increase accuracy. The practical improvement plan is:

- tune detection confidence thresholds separately for body and head modes
- test larger YOLO inference image sizes, especially for small distant heads
- tune non-max suppression and tracker settings for dense scenes
- improve video quality or preprocessing only if measured results improve
- recalibrate zones and lines per camera and per detector mode
- collect approved real Indian railway platform CCTV clips
- create manual ground-truth counts for short clips
- label head bounding boxes on representative frames
- retrain or fine-tune the head detector in Colab with more relevant data
- evaluate body vs head outputs against manually counted ground truth
- report counting error, missed detections, false positives, and ID switching instead of relying only on visual inspection

Current known training metrics for the imported head detector are from the RPEE-Heads validation set:

| Metric | Value |
|---|---:|
| Precision | 0.866 |
| Recall | 0.723 |
| mAP50 | 0.794 |
| mAP50-95 | 0.431 |

These are useful model-training indicators, but they are not final accuracy metrics for Indian Railway CCTV.

## 10. Risks and Limitations

- Passengers can be missed due to occlusion.
- Low resolution makes far-away passengers and heads difficult to detect.
- Small heads can cause false negatives or unstable tracking.
- Low confidence thresholds can increase false positives.
- Dense crowds can cause overlapping detections and ID switching.
- The head model has domain mismatch because it was trained on RPEE-Heads, not Indian Railway CCTV.
- Different stations and camera angles will need separate calibration.
- Body and head counts may not match directly because they detect different visual targets and use different reference points.
- Line crossing counts depend heavily on accurate line placement.
- Zone occupancy depends on correctly configured polygons.
- Public/sample footage is useful for demonstration but not enough for production validation.
- Real deployment requires approved CCTV access, infrastructure planning, privacy controls, and operational acceptance criteria.

## 11. Next Steps

Recommended next steps:

1. Tune detection settings on the current CCTV-style sample.
2. Create manual ground-truth counts for a short clip.
3. Compare body vs head counting error against the manual count.
4. Request approved real Indian Railway CCTV footage.
5. Prioritize the most important camera angles for testing.
6. Label sample frames with head boxes and passenger counts.
7. Fine-tune the head detector again using more relevant footage.
8. Test RTSP/live stream input under realistic hardware conditions.
9. Recalibrate zones, lines, and thresholds for each camera.
10. Prepare a pilot deployment plan with privacy and operational requirements.

## 12. Boss Feedback Needed

To move from demo MVP toward a practical pilot, the following feedback is needed:

- Can you provide approved real Indian Railway CCTV footage?
- Which camera angles should be prioritized?
- What accuracy target is acceptable for a pilot?
- Which platform zones matter most?
- What crowd thresholds should trigger warning and critical alerts?
- Should head detection become the primary method, or remain a comparison mode?
- Should the next priority be accuracy, live CCTV integration, dashboard/reporting, or deployment planning?

## One-Page Boss Brief

### Project Summary

This project is an MVP/prototype for Indian railway platform passenger counting and crowd analytics. It analyzes railway-platform CCTV-style video using YOLO-based computer vision, tracks passengers with anonymous IDs, counts zone occupancy, counts movement across virtual lines, generates crowd alerts, saves analytics to SQLite, exports CSV reports, and displays results in a Streamlit web app.

### Problem

Railway platforms can become crowded during peak periods. Manual monitoring from CCTV is difficult because staff must estimate passenger counts, crowd density, and movement flow across multiple camera feeds. The project aims to automate these analytics to support station safety and operational planning.

### Current Approach

The system compares two detection modes:

- Full-body detection: detects complete passengers using pretrained YOLO person detection.
- Head detection: detects passenger heads using a Colab-trained head detector.

Head detection was added because full-body detection can miss passengers in distant or crowded CCTV views where the whole body is not visible. Heads may remain visible even when bodies are blocked.

### What Is Working

- video file processing
- full-body detection
- head detection using imported `best.pt`
- tracking IDs
- zone occupancy
- line crossing counts
- warning and critical crowd alerts
- SQLite analytics logging
- CSV report export
- Streamlit dashboard comparison
- basic FastAPI analytics endpoints

### Key Assumptions

- The camera is fixed and stable.
- Zones, lines, and thresholds are manually configured per camera.
- The system counts visible bodies/heads, not passenger identities.
- No face recognition or personal identification is required.
- Current results are MVP/demo results and need validation on approved Indian Railway CCTV.

### Not In Current Scope

- production-level accuracy guarantee
- official safety certification
- facial recognition
- live railway CCTV deployment
- multi-camera re-identification
- automatic calibration
- integration with railway control room alert systems
- large-scale Indian Railway CCTV validation unless data is provided

### Accuracy Improvement Plan

Immediate improvements include tuning confidence thresholds, testing larger image sizes, adjusting tracker settings, recalibrating zones/lines, and comparing body vs head counts against manual ground truth. The biggest next improvement is to obtain approved Indian Railway CCTV footage, label representative frames, and fine-tune/evaluate the model on real station conditions.

### Main Risks

Accuracy can be affected by occlusion, low resolution, small distant heads, false positives, ID switching, different camera angles, and domain mismatch between public training data and real Indian Railway CCTV.

### Feedback Needed

The next decision is what to prioritize: better accuracy, live CCTV testing, dashboard/reporting, or pilot deployment planning. To improve accuracy properly, approved real CCTV footage, target camera angles, important zones, and acceptable accuracy thresholds are needed.
