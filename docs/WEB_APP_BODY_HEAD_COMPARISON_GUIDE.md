# Web App Body vs Head Comparison Guide

## How to run the app

```bash
streamlit run src/dashboard/streamlit_app.py
```

Open <http://localhost:8501> in your browser.

---

## What the app does

The dashboard compares two passenger detection modes on the same railway platform
video. The default input is the selected Pexels **People on Platform on Train Station**
clip at `data/input_videos/sample.mp4`, with a preserved CCTV-angle copy at
`data/input_videos/cctv_platform_sample.mp4`.

- **Full-Body Detection** — pretrained YOLO detects whole persons
- **Head Detection** — Colab fine-tuned YOLO detects heads only

Both modes produce the same analytics output (zone occupancy, line crossings, crowd
alerts, SQLite database, CSV report). The side-by-side view lets you compare how many
passengers each mode finds.

---

## Files the app uses

| File | Purpose |
|------|---------|
| `data/input_videos/sample.mp4` | Canonical selected Pexels CCTV-like railway platform sample shown in Section 3 |
| `data/input_videos/cctv_platform_sample.mp4` | Preserved copy of the current CCTV-angle platform sample |
| `data/input_videos/previous_sample_backup.mp4` | Previous sample backup shown in the selector when available |
| `data/outputs/body_demo.mp4` | Annotated body-mode output |
| `data/outputs/head_demo.mp4` | Annotated head-mode output |
| `data/outputs/body_analytics.db` | Body-mode analytics database |
| `data/outputs/head_analytics.db` | Head-mode analytics database |
| `data/outputs/body_report.csv` | Body-mode CSV report |
| `data/outputs/head_report.csv` | Head-mode CSV report |
| `models/fine_tuned/head_detector/weights/best.pt` | Colab-trained head detector weights |
| `configs/zones.cctv_platform.example.json` | CCTV-angle sample zone and line configuration |

---

## What each section means

| Section | Content |
|---------|---------|
| 1 | Comparison overview table — body vs head, when each works, why head matters |
| 2 | Step-by-step plain-English explanation of detection, tracking, zones, and alerts |
| 3 | Input video player — shows the source footage |
| 4 | Model status — confirms whether body and head models are available |
| 5 | Run buttons — triggers detection pipelines and report export |
| 6 | Side-by-side annotated output videos |
| 7 | Guide to reading the overlay annotations |
| 8 | Analytics comparison — unique passengers, zone occupancy, line crossings, alerts |
| 9 | CSV report previews with download buttons |
| 10 | What data is collected (operational analytics only, no personal identity) |
| 11 | Why this matters for Indian Railways |
| 12 | Limitations — transparent about what is not yet production-ready |
| 13 | Boss demo guide — recommended 10-minute walkthrough |

---

## What the buttons do

| Button | Command it runs |
|--------|----------------|
| Run Full-Body Detection | `python scripts/run_video_demo.py ... --detector-mode body` |
| Run Head Detection | `python scripts/run_video_demo.py ... --detector-mode head --model models/fine_tuned/head_detector/weights/best.pt` |
| Run Body vs Head Comparison | `python scripts/run_comparison_demo.py ...` |
| Export Body Report CSV | `python scripts/export_report.py --db data/outputs/body_analytics.db --out data/outputs/body_report.csv` |
| Export Head Report CSV | `python scripts/export_report.py --db data/outputs/head_analytics.db --out data/outputs/head_report.csv` |

All commands run with `cwd` set to the project root. Output is streamed live in the
browser via a Streamlit status block.

Head mode uses higher-resolution inference (`--imgsz 1536`), a lower confidence
threshold (`--confidence 0.15`), and CCTV-sample detection filters because passenger
heads are small after resizing and train/track regions can create false positives.

---

## What to show the boss

Recommended demo flow (approximately 10 minutes):

1. **Section 1** — explain the body vs head comparison table (2 minutes).
2. **Section 3** — show the input video ("this is what the CCTV sees").
3. **Section 4** — confirm both models are found.
4. **Section 5** — click **Run Body vs Head Comparison** and let it run.
5. **Section 6** — show the side-by-side output videos; point out detection differences.
6. **Section 8** — show the analytics comparison (unique passengers, zone occupancy, alerts).
7. **Section 10** — confirm no personal data is stored.
8. **Section 11** — operational benefits for Indian Railways.
9. **Section 12** — be transparent about limitations before production deployment.

---

## Limitations

- Not production-ready. No validation on live Indian Railway CCTV.
- Head model trained on RPEE-Heads — domain shift expected on real Indian platform footage.
- No claimed production accuracy metrics.
- Low resolution and occlusion still cause missed detections.
- Head tracking ID switches are more common than body tracking.
- Thresholds need station-specific calibration.
- Live RTSP CCTV integration requires additional infrastructure.
- All legal and regulatory approvals must be in place before real CCTV deployment.
