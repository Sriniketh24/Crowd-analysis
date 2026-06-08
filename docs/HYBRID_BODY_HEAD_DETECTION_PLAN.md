# Hybrid Body+Head Detection Plan

## Demo Scope

This hybrid demo is calibrated only for `data/input_videos/sample.mp4`, which is
1280x720 public railway platform footage. It is useful for explaining the system
behavior, but it is not final accuracy evidence and it is not approved Indian
Railway CCTV.

Run the full local comparison with:

```bash
python scripts/run_comparison_demo.py \
  --source data/input_videos/sample.mp4 \
  --zones-config configs/zones.hybrid_cctv_platform.example.json
```

Expected outputs:

| Mode | Video | SQLite DB |
|------|-------|-----------|
| Body only | `data/outputs/body_demo.mp4` | `data/outputs/body_analytics.db` |
| Head only | `data/outputs/head_demo.mp4` | `data/outputs/head_analytics.db` |
| Hybrid | `data/outputs/hybrid_demo.mp4` | `data/outputs/hybrid_analytics.db` |

## Why Use Body Detection Near the Camera

Near passengers usually occupy enough pixels for a pretrained YOLO person model
to see the full body. For these passengers, a full-body box is better for
tracking movement, zone occupancy, and line crossing because the bottom center of
the box is a reasonable approximation of the passenger's position on the floor.

Body detection also avoids counting nearby bags, posters, or platform objects as
passengers just because they look like small head-shaped regions.

## Why Use Head Detection Far from the Camera

Far passengers are smaller in the frame and are often partly hidden by other
passengers, benches, poles, or train-side structures. In those cases, the whole
body may not be visible enough for a person detector, but the head may still be
visible. A head detector can therefore recover passengers that the body detector
misses in distant or crowded parts of the platform.

For this sample, `configs/zones.hybrid_cctv_platform.example.json` separates the
view into `near_body_zone` and `far_head_zone` under `hybrid_detection_rois`.
Normal analytics zones and counting lines remain separate from these detection
ROIs.

## Why Duplicate Suppression Is Needed

At the boundary between near and far regions, the body detector and head detector
can see the same person. Without duplicate suppression, one passenger could be
counted twice: once as a body box and once as a head box.

The hybrid pipeline fuses detections after both models run. It keeps body
detections as the stronger representation when a head falls inside the upper
part of a matching body box, and keeps standalone head detections when no
matching body exists.

## What This Sample Cannot Prove

The current sample can prove that the software path works: loading config,
running body/head/hybrid detection, writing annotated videos, writing SQLite,
and showing the expected overlays. It cannot prove final passenger-counting
accuracy.

Reasons:

- The sample is one public clip, not a representative railway CCTV validation
  set.
- It has no official frame-level passenger count labels.
- Crowd density, camera height, lens angle, lighting, motion blur, and occlusion
  are different across real stations.
- A head model trained on a public head dataset can still have domain shift on
  Indian railway CCTV.

## Required Real CCTV Validation

Before making production accuracy claims, the project needs a controlled
validation set from real approved station CCTV. That validation would require:

- Permission to use the footage and a privacy policy that avoids storing
  personally identifying video unless explicitly approved.
- Multiple cameras, platforms, crowd densities, lighting conditions, and times
  of day.
- Manual ground-truth counts for zones, line crossings, and crowded-area alerts
  over defined time windows.
- Separate calibration per camera for zones, lines, thresholds, and hybrid ROIs.
- Metrics reported per camera and scenario, including precision/recall,
  counting error, ID-switch behavior, false crowd alerts, and missed crowd
  alerts.

Until that validation is complete, the hybrid demo should be presented as a
working MVP approach, not a certified accuracy result.
