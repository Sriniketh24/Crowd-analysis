"""Dashboard paths and display constants."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

INPUT_VIDEO = ROOT / "data" / "input_videos" / "sample.mp4"
CCTV_VIDEO = ROOT / "data" / "input_videos" / "cctv_platform_sample.mp4"
PREVIOUS_VIDEO_BACKUP = ROOT / "data" / "input_videos" / "previous_sample_backup.mp4"
BODY_VIDEO = ROOT / "data" / "outputs" / "body_demo.mp4"
HEAD_VIDEO = ROOT / "data" / "outputs" / "head_demo.mp4"
HYBRID_VIDEO = ROOT / "data" / "outputs" / "hybrid_demo.mp4"
BODY_DB = ROOT / "data" / "outputs" / "body_analytics.db"
HEAD_DB = ROOT / "data" / "outputs" / "head_analytics.db"
HYBRID_DB = ROOT / "data" / "outputs" / "hybrid_analytics.db"
BODY_CSV = ROOT / "data" / "outputs" / "body_report.csv"
HEAD_CSV = ROOT / "data" / "outputs" / "head_report.csv"
HYBRID_CSV = ROOT / "data" / "outputs" / "hybrid_report.csv"
TUNING_CSV = ROOT / "data" / "outputs" / "tuning_results.csv"
HEAD_MODEL = ROOT / "models" / "fine_tuned" / "head_detector" / "weights" / "best.pt"
HYBRID_ZONES_CONFIG = ROOT / "configs" / "zones.hybrid_cctv_platform.example.json"
ZONES_CONFIG = (
    HYBRID_ZONES_CONFIG
    if HYBRID_ZONES_CONFIG.exists()
    else ROOT / "configs" / "zones.cctv_platform.example.json"
)
SCOPE_DOC = ROOT / "docs" / "PROBLEM_STATEMENT_SCOPE_AND_STEPS.md"

ALERT_ICON = {"NORMAL": "🟢", "WARNING": "🟡", "CRITICAL": "🔴"}
MODE_OPTIONS = {
    "Full body": {
        "mode": "body",
        "video": BODY_VIDEO,
        "db": BODY_DB,
        "csv": BODY_CSV,
        "label": "Full-Body Detection",
    },
    "Head": {
        "mode": "head",
        "video": HEAD_VIDEO,
        "db": HEAD_DB,
        "csv": HEAD_CSV,
        "label": "Head Detection",
    },
    "Hybrid body + head": {
        "mode": "hybrid",
        "video": HYBRID_VIDEO,
        "db": HYBRID_DB,
        "csv": HYBRID_CSV,
        "label": "Hybrid Body + Head Detection",
    },
}
