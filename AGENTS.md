# Railway Crowd Analytics Project Instructions

Build a Python-based computer vision system for railway station crowd analysis and passenger counting from CCTV feeds.

Core goals:
- Detect passengers/persons from video streams.
- Track each person with stable IDs.
- Count people crossing virtual lines.
- Count people inside configured zones.
- Detect crowded areas based on occupancy thresholds.
- Generate real-time analytics for station safety and passenger flow management.
- Provide a local dashboard and API.
- Save analytics and alerts to SQLite.

Preferred stack:
- Python 3.11
- Ultralytics YOLO
- OpenCV
- Supervision
- FastAPI
- Streamlit
- SQLite
- SQLAlchemy
- Pytest

Important rules:
- Build a working MVP first.
- Use pretrained YOLO person detection before adding custom training.
- Keep modules small and testable.
- Avoid hardcoding camera-specific values.
- Put zones, lines, thresholds, and cameras in config files.
- Add docstrings and comments where logic is non-obvious.
- Create tests for zone counting, line crossing, and alert logic.
- Do not invent unavailable datasets or private CCTV access.
- Any real CCTV use must avoid storing personally identifying footage unless explicitly approved.
- Use sample/public videos for demo.