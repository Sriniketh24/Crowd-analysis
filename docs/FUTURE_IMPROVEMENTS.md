# Future Improvements

This document captures the planned evolution of the Railway Crowd Analytics
system beyond the current pretrained-YOLO MVP. Items are ordered roughly by
expected impact-to-effort ratio. Nothing here is required for the demo to run.

---

## 1. Custom Fine-Tuning on Railway CCTV

The MVP uses pretrained YOLO weights (`yolov8n.pt`) trained on COCO. Real
station cameras differ from COCO imagery: overhead/oblique angles, motion blur,
low light, heavy occlusion, and small/distant people on long platforms.

**Plan**
- Collect and label a station-specific dataset (with explicit approval and
  privacy review — see item 5).
- Fine-tune a YOLO model on railway crowd footage; start from `yolov8n`/`yolov8s`
  and scale up only if accuracy demands it.
- Track precision/recall on a held-out station validation set; compare against
  the pretrained baseline before promoting any model.
- Version weights and keep the pretrained model as a fallback (`--model` already
  makes weights swappable with zero code changes).

**Why it matters:** higher detection recall in crowded, occluded scenes directly
improves occupancy counts and alert reliability.

---

## 2. Multi-Camera Re-Identification (Re-ID)

Today each camera is tracked independently and IDs are local to one stream.
A single passenger walking across overlapping or adjacent cameras is counted as
separate identities.

**Plan**
- Add an appearance-embedding Re-ID model (e.g. OSNet) to extract per-person
  feature vectors.
- Match embeddings across cameras with a gallery + similarity threshold to assign
  a station-wide global ID.
- Use camera topology / handoff zones to constrain matches and reduce false links.
- Persist global IDs alongside per-camera track IDs in the analytics DB.

**Why it matters:** enables true station-wide passenger journeys, accurate
platform-to-platform flow, and de-duplicated headcounts.

---

## 3. NVIDIA DeepStream Deployment

OpenCV + Ultralytics on CPU is ideal for a demo but does not scale to many
simultaneous high-FPS camera feeds.

**Plan**
- Port the detect → track → analytics pipeline to the NVIDIA DeepStream SDK
  (GStreamer-based) for hardware-accelerated decode + inference.
- Export the YOLO model to TensorRT (`.engine`) for GPU inference.
- Keep the same SQLite/analytics schema as the sink so the API and dashboard
  remain unchanged.
- Benchmark streams-per-GPU and end-to-end latency versus the Python pipeline.

**Why it matters:** DeepStream can run dozens of camera streams per GPU with
batched inference, which is required for a real multi-platform station.

---

## 4. Edge Deployment on NVIDIA Jetson

Sending raw CCTV to a central server is bandwidth-heavy and raises privacy
concerns. Running analytics at the camera ("on the edge") avoids shipping video.

**Plan**
- Target Jetson Orin Nano / Xavier NX class devices.
- Build a TensorRT-optimized model sized for the device's power/thermal budget.
- Run detection + analytics locally and transmit only aggregated metrics and
  alerts (not frames) upstream.
- Provide a slim container image and a power/throughput profile per device.

**Why it matters:** lower bandwidth, lower latency alerts, and a stronger privacy
posture because video never leaves the platform.

---

## 5. Privacy-Preserving Analytics

The system already stores only bounding boxes, counts, and alert events — never
faces or raw identifying footage. The next step is to make privacy guarantees
explicit and enforceable.

**Plan**
- Optional on-frame face/body blurring before any frame is written or displayed.
- Strict retention policy: auto-expire annotated video and raw frames; keep only
  aggregate metrics long-term.
- Configurable "metrics-only" mode that disables all video persistence.
- Document data flows and add a privacy/DPIA checklist; gate any footage storage
  behind explicit, logged approval.

**Why it matters:** railway CCTV is highly sensitive. Privacy-by-design is both a
legal requirement and a trust requirement for deployment.

---

## 6. Heatmaps

Per-zone counts answer "how many", but not "where exactly" crowds form.

**Plan**
- Accumulate person bottom-center positions over a time window into a 2D density
  grid per camera.
- Render a color heatmap overlay on the annotated video and as a dashboard panel.
- Persist periodic density grids so historical hotspots can be reviewed.
- Use heatmaps to recommend better zone/line placement during calibration.

**Why it matters:** spatial hotspots reveal bottlenecks (stairwells, gates,
boarding points) that aggregate zone counts hide.

---

## 7. Anomaly Detection

Threshold alerts catch known crowding limits, but not unusual *behavior*.

**Plan**
- Baseline normal occupancy/flow patterns per zone, per time-of-day, per camera.
- Flag statistical anomalies: sudden surges, unexpected counter-flow, a platform
  that empties or fills far faster than usual, or motion stopping (possible
  incident/fall).
- Start with simple, explainable statistical models (rolling z-score, EWMA) before
  introducing heavier ML, to keep alerts auditable.
- Route anomaly events into the existing `crowd_alerts` pathway so the dashboard
  and API surface them with no schema change.

**Why it matters:** moves the system from reactive threshold monitoring toward
proactive incident detection for station safety teams.
