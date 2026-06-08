"""Streamlit app — Indian Railway Platform Passenger Counting: Full Body vs Head Detection.

Run with:
    streamlit run src/dashboard/streamlit_app.py
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from subprocess import PIPE, STDOUT

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

INPUT_VIDEO = ROOT / "data" / "input_videos" / "sample.mp4"
CCTV_VIDEO = ROOT / "data" / "input_videos" / "cctv_platform_sample.mp4"
PREVIOUS_VIDEO_BACKUP = ROOT / "data" / "input_videos" / "previous_sample_backup.mp4"
BODY_VIDEO = ROOT / "data" / "outputs" / "body_demo.mp4"
HEAD_VIDEO = ROOT / "data" / "outputs" / "head_demo.mp4"
BODY_DB = ROOT / "data" / "outputs" / "body_analytics.db"
HEAD_DB = ROOT / "data" / "outputs" / "head_analytics.db"
BODY_CSV = ROOT / "data" / "outputs" / "body_report.csv"
HEAD_CSV = ROOT / "data" / "outputs" / "head_report.csv"
TUNING_CSV = ROOT / "data" / "outputs" / "tuning_results.csv"
HEAD_MODEL = ROOT / "models" / "fine_tuned" / "head_detector" / "weights" / "best.pt"
ZONES_CONFIG = ROOT / "configs" / "zones.cctv_platform.example.json"
SCOPE_DOC = ROOT / "docs" / "PROBLEM_STATEMENT_SCOPE_AND_STEPS.md"

ALERT_ICON = {"NORMAL": "🟢", "WARNING": "🟡", "CRITICAL": "🔴"}

st.set_page_config(
    page_title="Indian Railway Passenger Counting: Full Body vs Head Detection",
    page_icon="🚉",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── DB helpers ────────────────────────────────────────────────────────────────

def _db_ok(db_path: Path) -> bool:
    return db_path.exists() and db_path.stat().st_size > 0


def _query(db_path: Path, sql: str, params: tuple = ()) -> pd.DataFrame:
    if not _db_ok(db_path):
        return pd.DataFrame()
    try:
        with sqlite3.connect(str(db_path), timeout=10) as conn:
            return pd.read_sql_query(sql, conn, params=params)
    except Exception:
        return pd.DataFrame()


def _latest_session(db_path: Path) -> dict | None:
    df = _query(
        db_path,
        "SELECT id, camera_id, source, started_at, total_frames, "
        "total_unique_passengers, model_weights, tracker_type "
        "FROM run_sessions "
        "WHERE ended_at IS NOT NULL AND total_frames IS NOT NULL "
        "ORDER BY started_at DESC, id DESC LIMIT 1",
    )
    if df.empty:
        df = _query(
            db_path,
            "SELECT id, camera_id, source, started_at, total_frames, "
            "total_unique_passengers, model_weights, tracker_type "
            "FROM run_sessions ORDER BY started_at DESC, id DESC LIMIT 1",
        )
    if df.empty:
        return None
    row = df.iloc[0]
    return {
        "id": int(row["id"]),
        "camera_id": str(row.get("camera_id") or ""),
        "source": str(row.get("source") or ""),
        "started_at": str(row.get("started_at") or ""),
        "total_frames": int(row.get("total_frames") or 0),
        "total_unique_passengers": (
            int(row["total_unique_passengers"])
            if row.get("total_unique_passengers") is not None
            else None
        ),
        "model_weights": str(row.get("model_weights") or ""),
        "tracker_type": str(row.get("tracker_type") or ""),
    }


def _zone_occupancy(db_path: Path, session_id: int) -> pd.DataFrame:
    df = _query(
        db_path,
        "SELECT zone_id, count, alert_level, frame_index FROM zone_occupancy "
        "WHERE run_session_id=? ORDER BY frame_index DESC, id DESC",
        (session_id,),
    )
    if df.empty:
        return pd.DataFrame()
    latest = df.drop_duplicates(subset=["zone_id"], keep="first").copy()
    latest["Status"] = latest["alert_level"].map(lambda v: f"{ALERT_ICON.get(v, '')} {v}")
    return latest[["zone_id", "count", "Status"]].rename(
        columns={"zone_id": "Zone", "count": "Count"}
    )


def _line_crossings(db_path: Path, session_id: int) -> pd.DataFrame:
    return _query(
        db_path,
        "SELECT line_id AS Line, direction AS Direction, COUNT(*) AS Count "
        "FROM line_crossing_events WHERE run_session_id=? GROUP BY line_id, direction",
        (session_id,),
    )


def _alerts(db_path: Path, session_id: int) -> pd.DataFrame:
    df = _query(
        db_path,
        "SELECT zone_id, alert_level, occupancy, frame_index "
        "FROM crowd_alerts WHERE run_session_id=? AND alert_level != 'NORMAL' "
        "ORDER BY frame_index DESC LIMIT 20",
        (session_id,),
    )
    if df.empty:
        return pd.DataFrame()
    df["Level"] = df["alert_level"].map(lambda v: f"{ALERT_ICON.get(v, '')} {v}")
    return df[["zone_id", "Level", "occupancy", "frame_index"]].rename(
        columns={"zone_id": "Zone", "occupancy": "Occupancy", "frame_index": "Frame"}
    )


def _detector_mode_label(db_path: Path, session_id: int) -> str:
    df = _query(db_path, "SELECT detector_mode FROM run_sessions WHERE id=?", (session_id,))
    if df.empty or "detector_mode" not in df.columns:
        return "—"
    return str(df.iloc[0]["detector_mode"] or "—")


def _run_subprocess(cmd: list[str], label: str) -> None:
    with st.status(f"Running {label}...", expanded=True) as status:
        try:
            process = subprocess.Popen(cmd, stdout=PIPE, stderr=STDOUT, text=True, cwd=str(ROOT))
            for line in process.stdout:  # type: ignore[union-attr]
                if line.strip():
                    st.write(f"`{line.strip()}`")
            process.wait()
            if process.returncode == 0:
                status.update(label=f"{label} complete!", state="complete")
                st.rerun()
            else:
                status.update(label=f"{label} failed — check output above.", state="error")
        except Exception as exc:
            status.update(label=f"Error: {exc}", state="error")


def _add_tuning_args(cmd: list[str]) -> list[str]:
    """Append current dashboard tuning controls to a subprocess command."""
    tuned = [
        *cmd,
        "--confidence", f"{tune_confidence:.2f}",
        "--iou", f"{tune_iou:.2f}",
        "--imgsz", str(tune_imgsz),
        "--tracker", tune_tracker,
        "--max-det", str(tune_max_det),
    ]
    if tune_augment:
        tuned.append("--augment")
    return tuned


def _relative_or_display(path: Path | str) -> str:
    """Return a project-relative path when possible for dashboard captions."""
    candidate = Path(path)
    try:
        return str(candidate.relative_to(ROOT))
    except ValueError:
        return str(path)


def _render_analytics(db_path: Path, label: str) -> None:
    if not _db_ok(db_path):
        st.info(f"No {label} database. Run {label} detection first.")
        return
    session = _latest_session(db_path)
    if session is None:
        st.info(f"{label} database has no sessions yet.")
        return
    sid = session["id"]
    st.caption(f"Mode: `{_detector_mode_label(db_path, sid)}` · Session #{sid} · {session['started_at']}")
    ca, cb = st.columns(2)
    ca.metric("Unique Passengers", session["total_unique_passengers"] or "—")
    cb.metric("Frames Processed", session["total_frames"] or "—")
    for title, df in [
        ("**Zone Occupancy**", _zone_occupancy(db_path, sid)),
        ("**Line Crossings**", _line_crossings(db_path, sid)),
    ]:
        st.markdown(title)
        if not df.empty:
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.caption("No data yet.")
    alert_df = _alerts(db_path, sid)
    st.markdown("**Crowd Alerts**")
    if not alert_df.empty:
        st.dataframe(alert_df, hide_index=True, use_container_width=True)
    else:
        st.success("No WARNING/CRITICAL alerts.")


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Advanced Options")
    video_options: dict[str, Path] = {}
    if CCTV_VIDEO.exists():
        video_options["Current CCTV-angle platform sample"] = CCTV_VIDEO
    if INPUT_VIDEO.exists() and INPUT_VIDEO != CCTV_VIDEO:
        video_options["Canonical sample.mp4"] = INPUT_VIDEO
    if PREVIOUS_VIDEO_BACKUP.exists():
        video_options["Previous sample backup"] = PREVIOUS_VIDEO_BACKUP
    if not video_options:
        video_options["Missing default sample"] = INPUT_VIDEO
    selected_video_label = st.selectbox("Demo video", list(video_options))
    selected_video_path = video_options[selected_video_label]
    custom_source = st.text_input("Custom video source", placeholder="Leave blank to use the selected sample")
    auto_refresh = st.checkbox("Auto-refresh every 10 s", value=False)
    st.caption(f"Head model: {'✅ Found' if HEAD_MODEL.exists() else '❌ Missing'}")
    st.caption(f"Zones: `{ZONES_CONFIG.relative_to(ROOT)}`")

if auto_refresh:
    time.sleep(10)
    st.rerun()

video_source = custom_source.strip() if custom_source.strip() else (
    str(selected_video_path) if selected_video_path.exists() else ""
)
display_video_path = Path(video_source) if video_source and "://" not in video_source else None

# ── Title ─────────────────────────────────────────────────────────────────────

st.title("🚉 Indian Railway Platform Passenger Counting: Full Body vs Head Detection")
st.caption(
    "AI-powered crowd monitoring · Compares full-body detection vs head-based detection · "
    "YOLO + ByteTrack · CCTV-angle platform retest · Prototype — not production-ready"
)

# ── Section 1: What this demo compares ───────────────────────────────────────

st.markdown("---")
with st.expander("**1. What is this demo comparing?**", expanded=True):
    st.markdown(
        """
This demo analyses railway platform CCTV-style video using two different passenger detection approaches:

| | Full-Body Detection | Head Detection |
|---|---|---|
| **What it detects** | The entire person | The passenger's head only |
| **Best when** | Passengers are close, unobstructed, and visible | Passengers are far away, partially blocked, or in a dense crowd |
| **Challenge** | Misses passengers hidden by crowds, luggage, pillars, or low resolution | Smaller targets — can cause more ID switches in tracking |
| **Model** | Pretrained YOLO (general-purpose person detector) | Fine-tuned YOLO (trained on head images in Google Colab) |

**Why head detection matters for Indian Railways:**
Platform CCTV cameras often cover large areas. Passengers near the far end of the platform may be too
small or too crowded for full-body detection. The head is often still visible even when the rest of the
body is blocked — so a head detector can catch passengers that a body detector misses.
"""
    )

st.markdown("---")
with st.expander("**Problem statement, assumptions, scope, and steps (boss brief)**", expanded=False):
    if SCOPE_DOC.exists():
        st.markdown(
            "Full written brief — problem statement, objective, assumptions, in/out of scope, "
            "steps, accuracy improvement plan, risks, and a one-page boss summary — is in "
            f"`{SCOPE_DOC.relative_to(ROOT)}`."
        )
        try:
            scope_text = SCOPE_DOC.read_text(encoding="utf-8")
            st.download_button(
                "⬇ Download PROBLEM_STATEMENT_SCOPE_AND_STEPS.md",
                scope_text,
                "PROBLEM_STATEMENT_SCOPE_AND_STEPS.md",
                "text/markdown",
            )
            with st.expander("Preview brief", expanded=False):
                st.markdown(scope_text)
        except OSError as exc:
            st.warning(f"Could not read scope document: {exc}")
    else:
        st.info(
            "Scope brief not found. Expected at "
            f"`{SCOPE_DOC.relative_to(ROOT)}`."
        )

st.markdown("---")
with st.expander("**Why this CCTV-angle test matters**", expanded=True):
    st.markdown(
        """
The earlier demo video was not ideal because it looked closer to a phone-like angle.
Railway deployment will use fixed CCTV cameras, usually from a higher or wider angle.
This new test video is closer to that setup, so it better tests whether full-body detection
or head detection works when passengers are smaller, farther away, or partially blocked.

The selected clip is real railway platform footage with a CCTV-like elevated/static angle.
It should still be described as demo footage, not confirmed operational CCTV and not final
Indian Railway validation footage.
"""
    )

# ── Section 2: What is the model doing? ──────────────────────────────────────

st.markdown("---")
with st.expander("**2. What is the model actually doing?**", expanded=False):
    st.markdown(
        """
1. **Frame extraction** — video split into individual images.
2. **YOLO detection** — neural network draws boxes: *persons* (body mode) or *heads* (head mode).
3. **ByteTrack tracking** — links the same box across frames with a persistent ID (e.g. `#42`) to avoid double-counting.
4. **Zone occupancy** — counts tracked IDs inside each platform zone per frame.
5. **Line crossings** — virtual line counts IDs crossing it (IN / OUT direction).
6. **Crowd alerts** — raises WARNING or CRITICAL when a zone exceeds its threshold.
7. **Database logging** — everything saved to SQLite for reporting.

Body mode uses the *bottom-centre* of the box (feet on ground). Head mode uses the *centre* of the head box.
"""
    )

# ── Section 3: Input video ────────────────────────────────────────────────────

st.markdown("---")
st.subheader("3. Input Video — CCTV-Angle Platform Footage")
if display_video_path and display_video_path.exists():
    st.video(str(display_video_path))
    st.caption(
        f"`{_relative_or_display(display_video_path)}` — selected CCTV-angle platform footage used as input for both modes."
    )
    if CCTV_VIDEO.exists() and INPUT_VIDEO.exists() and CCTV_VIDEO.stat().st_size == INPUT_VIDEO.stat().st_size:
        st.caption("`sample.mp4` and `cctv_platform_sample.mp4` are both available for the selected CCTV-angle clip.")
else:
    st.warning(
        "**No input video found.**\n\n"
        f"Place a real CCTV-like platform video at `{CCTV_VIDEO.relative_to(ROOT)}`."
    )

# ── Section 4: Model status ───────────────────────────────────────────────────

st.markdown("---")
st.subheader("4. Model Status")
col_ms1, col_ms2 = st.columns(2)
with col_ms1:
    st.success("**Full-Body Model**: Pretrained YOLO — ready. No additional setup needed.")
with col_ms2:
    if HEAD_MODEL.exists():
        st.success(
            f"**Head Model**: Found at `{HEAD_MODEL.relative_to(ROOT)}`\n\n"
            "Fine-tuned in Google Colab and imported into this project. Not trained locally."
        )
    else:
        st.error(
            "**Head Model**: Missing.\n\n"
            f"Place `best.pt` at `{HEAD_MODEL.relative_to(ROOT)}`\n\n"
            "Import the Colab-trained weights — do not train locally."
        )

# ── Section 5: Accuracy tuning ────────────────────────────────────────────────

st.markdown("---")
st.subheader("5. Accuracy Tuning / Model Settings")
st.warning(
    "These controls tune detection behavior. They do not prove true accuracy; true accuracy needs "
    "manual counts or labelled ground-truth frames."
)
tc1, tc2, tc3 = st.columns(3)
with tc1:
    tune_confidence = st.slider("Confidence threshold", 0.05, 0.60, 0.25, 0.05)
    tune_tracker = st.selectbox("Tracker", ["bytetrack", "botsort"], index=0)
with tc2:
    tune_imgsz = st.selectbox("Image size", [640, 960, 1280, 1536], index=2)
    tune_iou = st.slider("NMS IoU threshold", 0.35, 0.80, 0.50, 0.05)
with tc3:
    tune_max_det = st.number_input("Max detections/frame", min_value=50, max_value=3000, value=1000, step=50)
    tune_augment = st.checkbox("Test-time augmentation", value=False)

with st.expander("What these settings mean", expanded=False):
    st.markdown(
        "- **Confidence threshold:** lower values catch weaker detections but can add false positives.\n"
        "- **Image size:** larger values help small or distant heads, but reduce processing speed.\n"
        "- **NMS IoU threshold:** controls how overlapping boxes are merged; dense crowds may need tuning.\n"
        "- **Tracker:** ByteTrack is the default; BoT-SORT can be compared when the environment supports it.\n"
        "- **Max detections/frame:** should be high enough for crowded platform scenes.\n"
        "- **Test-time augmentation:** can improve recall in some cases, but is slower."
    )

run_tuned_compare = st.button(
    "Run Body vs Head Comparison With Selected Settings",
    use_container_width=True,
    disabled=not video_source or not HEAD_MODEL.exists(),
)
if run_tuned_compare:
    _run_subprocess(
        _add_tuning_args([
            sys.executable, "scripts/run_comparison_demo.py",
            "--source", video_source, "--zones-config", str(ZONES_CONFIG),
            "--head-confidence", f"{tune_confidence:.2f}",
        ]),
        "Tuned Body vs Head Comparison",
    )

st.markdown("**Latest tuning_results.csv**")
if TUNING_CSV.exists():
    try:
        tuning_df = pd.read_csv(TUNING_CSV)
        st.dataframe(tuning_df, hide_index=True, use_container_width=True)
        st.download_button(
            "Download tuning_results.csv",
            TUNING_CSV.read_bytes(),
            "tuning_results.csv",
            "text/csv",
        )
    except Exception as exc:
        st.warning(f"Could not read tuning results: {exc}")
else:
    st.info("No tuning results yet. Run `scripts/tune_detection_settings.py` from the terminal.")

# ── Section 6: Run detection ──────────────────────────────────────────────────

st.markdown("---")
st.subheader("6. Run Detection")
if not video_source:
    st.error(
        "No video source available. "
        "Place the selected Pexels platform clip at `data/input_videos/sample.mp4` or enter a custom path in the sidebar."
    )
else:
    head_ready = HEAD_MODEL.exists()
    c1, c2, c3 = st.columns(3)
    run_body = c1.button("▶ Run Full-Body Detection", type="primary", use_container_width=True)
    run_head = c2.button(
        "▶ Run Head Detection", type="secondary", use_container_width=True, disabled=not head_ready
    )
    run_compare = c3.button("▶ Run Body vs Head Comparison", use_container_width=True)
    c4, c5 = st.columns(2)
    export_body = c4.button(
        "📄 Export Body Report CSV", use_container_width=True, disabled=not BODY_DB.exists()
    )
    export_head = c5.button(
        "📄 Export Head Report CSV", use_container_width=True, disabled=not HEAD_DB.exists()
    )
    if not head_ready:
        st.caption("⚠️ Head Detection disabled — head model missing. See Section 4.")
    if run_body:
        _run_subprocess(
            _add_tuning_args([
                sys.executable, "scripts/run_video_demo.py",
                "--source", video_source, "--output", str(BODY_VIDEO),
                "--zones-config", str(ZONES_CONFIG), "--db", str(BODY_DB),
                "--detector-mode", "body",
            ]),
            "Full-Body Detection",
        )
    if run_head:
        _run_subprocess(_add_tuning_args([
            sys.executable, "scripts/run_video_demo.py",
            "--source", video_source, "--output", str(HEAD_VIDEO),
            "--zones-config", str(ZONES_CONFIG), "--db", str(HEAD_DB),
            "--detector-mode", "head", "--model", str(HEAD_MODEL),
        ]), "Head Detection")
    if run_compare:
        _run_subprocess(_add_tuning_args([
            sys.executable, "scripts/run_comparison_demo.py",
            "--source", video_source, "--zones-config", str(ZONES_CONFIG),
            "--head-confidence", f"{tune_confidence:.2f}",
        ]), "Body vs Head Comparison")
    if export_body:
        _run_subprocess(
            [sys.executable, "scripts/export_report.py", "--db", str(BODY_DB), "--out", str(BODY_CSV)],
            "Body Report Export")
    if export_head:
        _run_subprocess(
            [sys.executable, "scripts/export_report.py", "--db", str(HEAD_DB), "--out", str(HEAD_CSV)],
            "Head Report Export")

# ── Section 7: Side-by-side output videos ────────────────────────────────────

st.markdown("---")
st.subheader("7. Detection Output — Side by Side")
col_bv, col_hv = st.columns(2)
with col_bv:
    st.markdown("**Full-Body Detection Output**")
    if BODY_VIDEO.exists():
        st.video(str(BODY_VIDEO))
        st.caption(f"`{BODY_VIDEO.relative_to(ROOT)}`")
    else:
        st.info("No body output yet. Click **Run Full-Body Detection** in Section 6.")
with col_hv:
    st.markdown("**Head Detection Output**")
    if HEAD_VIDEO.exists():
        st.video(str(HEAD_VIDEO))
        st.caption(f"`{HEAD_VIDEO.relative_to(ROOT)}`")
    else:
        st.info("No head output yet. Click **Run Head Detection** or **Run Comparison** in Section 6.")

# ── Section 8: How to read the video ─────────────────────────────────────────

st.markdown("---")
with st.expander("**8. How to read the output video**", expanded=False):
    st.markdown(
        "| Overlay | Meaning |\n"
        "|---------|--------|\n"
        "| **Bounding boxes** | Each box = one detected passenger or head |\n"
        "| **ID numbers** (e.g. `#12`) | Tracker ID — same person across frames, prevents double-counting |\n"
        "| **Mode label** | Full-Body or Head-Based detection |\n"
        "| **Zone polygons** | 🟢 Normal → 🟡 Warning → 🔴 Critical |\n"
        "| **Counting line** | Virtual boundary — IN / OUT crossings |\n"
        "| **Counter overlay** | Detections, unique IDs, FPS |\n"
        "| **ALERT banner** | Zone over threshold — staff action needed |\n\n"
        "Body boxes = *person/passenger*. Head boxes = *head/passenger*."
    )

# ── Section 9: Analytics comparison ──────────────────────────────────────────

st.markdown("---")
st.subheader("9. Analytics Comparison")
col_ab, col_ah = st.columns(2)
with col_ab:
    st.markdown("### Full-Body Detection")
    _render_analytics(BODY_DB, "body")
with col_ah:
    st.markdown("### Head Detection")
    _render_analytics(HEAD_DB, "head")

# ── Section 10: Report previews ───────────────────────────────────────────────

st.markdown("---")
st.subheader("10. Report Previews")
col_rb, col_rh = st.columns(2)
with col_rb:
    st.markdown("**Body Report**")
    if BODY_CSV.exists():
        try:
            df_b = pd.read_csv(BODY_CSV, nrows=100)
            st.dataframe(df_b, hide_index=True, use_container_width=True)
            st.download_button(
                "⬇ Download body_report.csv", BODY_CSV.read_bytes(), "body_report.csv", "text/csv"
            )
        except Exception as exc:
            st.warning(f"Could not read body report: {exc}")
    else:
        st.info("Click **Export Body Report CSV** in Section 6.")
with col_rh:
    st.markdown("**Head Report**")
    if HEAD_CSV.exists():
        try:
            df_h = pd.read_csv(HEAD_CSV, nrows=100)
            st.dataframe(df_h, hide_index=True, use_container_width=True)
            st.download_button(
                "⬇ Download head_report.csv", HEAD_CSV.read_bytes(), "head_report.csv", "text/csv"
            )
        except Exception as exc:
            st.warning(f"Could not read head report: {exc}")
    else:
        st.info("Click **Export Head Report CSV** in Section 6.")

# ── Section 11: What data is collected ────────────────────────────────────────

st.markdown("---")
with st.expander("**11. What data is collected?**", expanded=False):
    st.markdown(
        "Stores **operational analytics only** — no personal identity data.\n\n"
        "| Stored | Purpose |\n"
        "|--------|---------|\n"
        "| Detector mode (`body`/`head`) | Know which pipeline produced the record |\n"
        "| Timestamp and frame number | Time-series analysis |\n"
        "| Anonymised tracking ID | Avoid double-counting — no name or identity |\n"
        "| Zone occupancy counts | Monitor crowding per area |\n"
        "| Line crossing events (IN / OUT) | Passenger flow |\n"
        "| Alert level (NORMAL / WARNING / CRITICAL) | Staff response trigger |\n"
        "| Camera / source ID | Multi-camera support |\n\n"
        "No face crops, names, biometric data, or personal identity information is stored."
    )

# ── Section 12: Why this matters ──────────────────────────────────────────────

st.markdown("---")
with st.expander("**12. Why this matters for Indian Railways**", expanded=False):
    st.markdown(
        """
- **Monitor crowding** — know which zones exceed safe capacity before an incident.
- **Identify bottlenecks** — pinpoint gates, exits, and platform ends under pressure.
- **Understand passenger flow** — IN/OUT counts reveal how crowds build and disperse.
- **Improve safety** — automated alerts enable faster staff and RPF response.
- **Support planning** — peak-hour analytics guide staffing and resource allocation.
- **Future RTSP pilot** — the pipeline already supports RTSP; enter a camera URL in the sidebar.

Head detection is especially relevant for Indian platforms where distant or occluded passengers
are missed by full-body detection. Both modes together give a more complete occupancy picture.
"""
    )

# ── Section 13: Limitations ───────────────────────────────────────────────────

st.markdown("---")
with st.expander("**13. Limitations (read before presenting to stakeholders)**", expanded=False):
    st.markdown(
        "- **Not production-ready.** Prototype only — no live Indian Railway CCTV validation.\n"
        "- **Head model trained on RPEE-Heads**, not Indian Railway CCTV. Domain shift expected.\n"
        "- **No production accuracy claims.** Training metrics are in `docs/HEAD_MODEL_TRAINING_REPORT.md`.\n"
        "- **Missed detections** still occur in low resolution and heavy occlusion.\n"
        "- **Head tracking ID switches** are more frequent than body tracking in dense crowds.\n"
        "- **Thresholds need station-specific calibration** — defaults may not match real capacity.\n"
        "- **Pre-recorded video only** in this demo. Live RTSP needs additional infrastructure.\n"
        "- **Privacy:** obtain all legal/regulatory approvals before deploying on real CCTV."
    )

# ── Section 14: Boss demo guide ───────────────────────────────────────────────

st.markdown("---")
with st.expander("**14. Boss Demo Guide**", expanded=False):
    st.markdown(
        """
**~10 minute flow:**
1. **Section 1** — explain body vs head comparison table.
2. **Section 3** — show the input video.
3. **Section 4** — confirm both models are available.
4. **Section 5** — click **Run Body vs Head Comparison**.
5. **Section 7** — side-by-side output videos.
6. **Section 9** — analytics comparison (unique passengers, zone occupancy, alerts).
7. **Section 11** — confirm no personal data is stored.
8. **Section 12** — operational benefits.
9. **Section 13** — be transparent about limitations.

**On the head model:** "We fine-tuned this model in Google Colab on a railway-platform
head dataset. It's already integrated alongside the body detector. Next step: validate
it on actual approved Indian Railway platform footage."
"""
    )

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Indian Railway Platform Passenger Counting · Full Body vs Head Detection Comparison · "
    "YOLO + ByteTrack · Prototype — requires validation on Indian Railway CCTV before production deployment."
)
