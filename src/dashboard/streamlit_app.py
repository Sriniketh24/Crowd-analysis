"""Streamlit demo — Indian Railway Passenger Counting System.

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
OUTPUT_VIDEO = ROOT / "data" / "outputs" / "demo.mp4"
DB_PATH = ROOT / "data" / "outputs" / "analytics.db"
REPORT_CSV = ROOT / "data" / "outputs" / "report.csv"
ZONES_CONFIG = ROOT / "configs" / "zones.example.json"

ALERT_ICON = {"NORMAL": "🟢", "WARNING": "🟡", "CRITICAL": "🔴"}

st.set_page_config(
    page_title="Indian Railway Passenger Counting — Demo",
    page_icon="🚉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── DB helpers (sqlite3 + pandas, no ORM) ────────────────────────────────────


def _db_ok() -> bool:
    return DB_PATH.exists() and DB_PATH.stat().st_size > 0


def _query(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Run a read-only query and return a DataFrame. Returns empty DF on any error."""
    if not _db_ok():
        return pd.DataFrame()
    try:
        with sqlite3.connect(str(DB_PATH), timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            return pd.read_sql_query(sql, conn, params=params)
    except Exception:
        return pd.DataFrame()


def _latest_session() -> dict | None:
    df = _query(
        "SELECT id, camera_id, source, started_at, ended_at, total_frames, "
        "total_unique_passengers, model_weights, tracker_type "
        "FROM run_sessions ORDER BY started_at DESC, id DESC LIMIT 1"
    )
    if df.empty:
        return None
    row = df.iloc[0]
    return {
        "id": int(row["id"]),
        "camera_id": str(row["camera_id"] or ""),
        "source": str(row["source"] or ""),
        "started_at": str(row["started_at"] or ""),
        "ended_at": str(row["ended_at"] or ""),
        "total_frames": int(row["total_frames"] or 0),
        "total_unique_passengers": int(row["total_unique_passengers"] or 0)
        if row["total_unique_passengers"] is not None else None,
        "model_weights": str(row["model_weights"] or ""),
        "tracker_type": str(row["tracker_type"] or ""),
    }


def _latest_frame(session_id: int) -> dict | None:
    df = _query(
        "SELECT total_detections, unique_passengers_seen, frame_index, processing_fps "
        "FROM frames_processed WHERE run_session_id=? "
        "ORDER BY frame_index DESC LIMIT 1",
        (session_id,),
    )
    if df.empty:
        return None
    row = df.iloc[0]
    return {
        "total_detections": int(row["total_detections"] or 0),
        "unique_passengers_seen": int(row["unique_passengers_seen"] or 0),
        "processing_fps": float(row["processing_fps"] or 0.0),
    }


def _zone_occupancy(session_id: int) -> pd.DataFrame:
    df = _query(
        "SELECT zone_id, count, alert_level, frame_index FROM zone_occupancy "
        "WHERE run_session_id=? ORDER BY frame_index DESC, id DESC",
        (session_id,),
    )
    if df.empty:
        return pd.DataFrame()
    # Latest snapshot per zone
    latest = df.drop_duplicates(subset=["zone_id"], keep="first").copy()
    latest["Status"] = latest["alert_level"].map(
        lambda v: f"{ALERT_ICON.get(v, '')} {v}"
    )
    return latest[["zone_id", "count", "Status", "frame_index"]].rename(
        columns={"zone_id": "Zone", "count": "Count", "frame_index": "At Frame"}
    )


def _line_crossings(session_id: int) -> pd.DataFrame:
    return _query(
        "SELECT line_id AS Line, direction AS Direction, COUNT(*) AS Count "
        "FROM line_crossing_events WHERE run_session_id=? "
        "GROUP BY line_id, direction",
        (session_id,),
    )


def _alerts(session_id: int) -> pd.DataFrame:
    df = _query(
        "SELECT zone_id, alert_level, occupancy, frame_index, recorded_at "
        "FROM crowd_alerts WHERE run_session_id=? AND alert_level != 'NORMAL' "
        "ORDER BY recorded_at DESC LIMIT 50",
        (session_id,),
    )
    if df.empty:
        return pd.DataFrame()
    df["Level"] = df["alert_level"].map(lambda v: f"{ALERT_ICON.get(v, '')} {v}")
    df["Time"] = df["recorded_at"].str[-8:]  # HH:MM:SS from datetime string
    return df[["zone_id", "Level", "occupancy", "frame_index", "Time"]].rename(
        columns={"zone_id": "Zone", "occupancy": "Occupancy", "frame_index": "Frame"}
    )


def _occupancy_history(session_id: int) -> pd.DataFrame:
    df = _query(
        "SELECT recorded_at AS time, zone_id AS zone, count FROM zone_occupancy "
        "WHERE run_session_id=? ORDER BY recorded_at ASC",
        (session_id,),
    )
    if df.empty:
        return pd.DataFrame()
    df["time"] = pd.to_datetime(df["time"])
    return df.pivot_table(index="time", columns="zone", values="count", aggfunc="mean")


# ── Sidebar: advanced options ─────────────────────────────────────────────────

with st.sidebar:
    st.title("Advanced Options")
    custom_source = st.text_input(
        "Custom video source",
        placeholder="Leave blank to use sample.mp4",
        help="File path, webcam index (0), or RTSP URL.",
    )
    auto_refresh = st.checkbox("Auto-refresh every 10 s", value=False)

if auto_refresh:
    time.sleep(10)
    st.rerun()

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🚉 Indian Railway Passenger Counting — Demo")
st.caption(
    "AI-powered crowd monitoring for railway platforms  |  "
    "Powered by YOLOv11 + ByteTrack  |  Built for Indian Railway platform safety"
)

# ── Section 1: What this does ─────────────────────────────────────────────────

st.markdown("---")
with st.expander("**1. What does this system do?**", expanded=True):
    st.markdown(
        """
This is an **AI-powered passenger counting and crowd monitoring system** for Indian railway platforms.

| Step | What the system does | Why it matters |
|------|----------------------|----------------|
| 📹 **Read video** | Ingests footage from a CCTV camera on a railway platform | Works with any standard IP/CCTV camera |
| 👤 **Detect passengers** | A YOLOv11 AI model draws a bounding box around every person in each frame | Identifies all passengers in view |
| 🔢 **Track passengers** | ByteTrack assigns a unique ID to each person and follows them frame-to-frame | Avoids double-counting the same person |
| 📍 **Zone occupancy** | Platform zones (e.g. "Gate Area", "Platform End") are monitored for live head counts | Pinpoints *which* area is getting crowded |
| ➡️ **Passenger flow** | A virtual counting line measures how many people cross — and in which direction (IN/OUT) | Tracks crowd flow across entry/exit points |
| 🚨 **Crowd alerts** | If a zone exceeds a set threshold, a WARNING or CRITICAL alert is raised | Enables fast safety response |
| 💾 **Save analytics** | All counts, crossings, and alerts are saved to a database | Supports shift reports, safety audits, and planning |

**Bottom line:** Station managers can see exactly how many people are in each zone, when crowding peaks, and which direction passengers are moving — all from existing CCTV footage.
"""
    )

# ── Section 2: Input video ────────────────────────────────────────────────────

st.markdown("---")
st.subheader("2. Input Video — Platform CCTV Footage")

if INPUT_VIDEO.exists():
    st.video(str(INPUT_VIDEO))
    st.caption(
        f"`{INPUT_VIDEO.relative_to(ROOT)}` — "
        "This represents CCTV-style footage from a railway/platform environment. "
        "The AI processes this footage frame by frame."
    )
else:
    st.warning(
        "**No input video found.**\n\n"
        f"Please place a platform video at:  `{INPUT_VIDEO.relative_to(ROOT)}`\n\n"
        "The file should be an MP4 from a railway platform CCTV or similar source."
    )

# ── Section 3: Run analysis ───────────────────────────────────────────────────

st.markdown("---")
st.subheader("3. Run Passenger Counting Analysis")

video_source = custom_source.strip() if custom_source.strip() else (
    str(INPUT_VIDEO) if INPUT_VIDEO.exists() else ""
)

if not video_source:
    st.error(
        "No video source available. "
        "Place a video at `data/input_videos/sample.mp4` or enter a custom path in the sidebar."
    )
else:
    col_btn, col_note = st.columns([1, 3])
    with col_btn:
        run_btn = st.button("▶ Run Passenger Counting Analysis", type="primary")
    with col_note:
        st.caption(
            "Runs the full AI pipeline: detection → tracking → zone counting → alerts → save. "
            "Takes ~1–3 minutes on CPU. Results update in sections 4–7 once done."
        )

    if run_btn:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_video_demo.py"),
            "--source", video_source,
            "--output", str(OUTPUT_VIDEO),
            "--zones-config", str(ZONES_CONFIG),
            "--db", str(DB_PATH),
        ]
        pipeline_stages = [
            "Loading video and initializing YOLOv11 model...",
            "Detecting passengers in video frames...",
            "Tracking passenger movement with ByteTrack...",
            "Counting zone occupancy per frame...",
            "Evaluating crowd alert thresholds...",
            "Saving analytics to database...",
            "Writing annotated output video...",
        ]
        with st.status("Running analysis pipeline...", expanded=True) as run_status:
            for stage in pipeline_stages:
                st.write(stage)
            try:
                process = subprocess.Popen(
                    cmd, stdout=PIPE, stderr=STDOUT, text=True, cwd=str(ROOT)
                )
                for line in process.stdout:  # type: ignore[union-attr]
                    stripped = line.strip()
                    if stripped:
                        st.write(f"`{stripped}`")
                process.wait()
                if process.returncode == 0:
                    # Re-encode output to H.264 for browser playback
                    if OUTPUT_VIDEO.exists():
                        tmp = OUTPUT_VIDEO.with_suffix(".tmp.mp4")
                        enc = subprocess.run(
                            [
                                "ffmpeg", "-y", "-i", str(OUTPUT_VIDEO),
                                "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                                "-movflags", "+faststart", "-pix_fmt", "yuv420p",
                                "-c:a", "copy", str(tmp),
                            ],
                            capture_output=True,
                        )
                        if enc.returncode == 0 and tmp.exists():
                            tmp.replace(OUTPUT_VIDEO)
                            st.write("`Output video converted to H.264 for browser playback.`")
                    run_status.update(
                        label="Analysis complete! Scroll down for results.", state="complete"
                    )
                    st.rerun()
                else:
                    run_status.update(
                        label="Pipeline failed — check the output above.", state="error"
                    )
            except Exception as exc:
                run_status.update(label=f"Error: {exc}", state="error")

# ── Section 4: Output video ───────────────────────────────────────────────────

st.markdown("---")
st.subheader("4. Output — Annotated Video with AI Overlays")

if OUTPUT_VIDEO.exists():
    st.video(str(OUTPUT_VIDEO))
    st.markdown(
        """
**Guide to what you see in the video:**

| Overlay | Meaning |
|---------|---------|
| **Coloured bounding boxes** | Each box = one detected passenger. The AI found a person here. |
| **ID numbers** (e.g. `#12`) | Unique tracking ID per passenger — stays with them across frames so nobody is double-counted |
| **Shaded zone polygons** | Monitored platform areas. 🟢 Normal → 🟡 Warning → 🔴 Critical as count rises |
| **Counting line** | A virtual boundary drawn across the platform. Every person crossing it is counted by direction (IN / OUT) |
| **Counter overlay** (top-left) | Live totals: current detections, unique passengers seen, processing FPS |
| **ALERT banner** | Shown when a zone crosses the crowding threshold — tells staff where to act |
"""
    )
    st.caption(f"Output: `{OUTPUT_VIDEO.relative_to(ROOT)}`")
else:
    st.info(
        "No annotated video yet.  "
        "Click **Run Passenger Counting Analysis** in Section 3 to generate it.\n\n"
        f"It will appear at `{OUTPUT_VIDEO.relative_to(ROOT)}`."
    )

# ── Section 5: Analytics summary ─────────────────────────────────────────────

st.markdown("---")
st.subheader("5. Real-Time Analytics Summary")

if not _db_ok():
    st.info(
        "No analytics database found yet. "
        "Run the analysis pipeline in Section 3 — the database will be created automatically.\n\n"
        f"Expected location: `{DB_PATH.relative_to(ROOT)}`"
    )
else:
    run_info = _latest_session()
    if run_info is None:
        st.info("Database exists but contains no sessions. Run the analysis pipeline in Section 3.")
    else:
        session_id = run_info["id"]
        frame_info = _latest_frame(session_id)
        alert_df = _alerts(session_id)

        unique_pax = run_info["total_unique_passengers"]
        if unique_pax is None and frame_info:
            unique_pax = frame_info["unique_passengers_seen"]
        fps = frame_info["processing_fps"] if frame_info else 0.0
        critical_count = int((alert_df["Level"].str.contains("CRITICAL")).sum()) if not alert_df.empty else 0
        warning_count = int((alert_df["Level"].str.contains("WARNING")).sum()) if not alert_df.empty else 0

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("👥 Unique Passengers", unique_pax if unique_pax is not None else "—")
        k2.metric("📷 Camera", run_info["camera_id"] or "—")
        k3.metric("🎞️ Frames Processed", run_info["total_frames"] or "—")
        k4.metric("🟡 Warnings", warning_count)
        k5.metric("🔴 Criticals", critical_count)

        model = run_info["model_weights"] or "—"
        tracker = run_info["tracker_type"] or "—"
        st.caption(
            f"Session **#{session_id}** · {run_info['started_at']} · "
            f"source: `{run_info['source']}` · "
            f"model: `{model}` · tracker: `{tracker}` · processing FPS: `{fps:.1f}`"
        )

        # ── Section 6: Tables and charts ─────────────────────────────────────

        st.markdown("---")
        st.subheader("6. Zone Occupancy, Line Crossings & Alerts")

        zone_df = _zone_occupancy(session_id)
        line_df = _line_crossings(session_id)

        col_zone, col_line = st.columns([2, 1])
        with col_zone:
            st.markdown("**Platform Zone Occupancy** (latest snapshot per zone)")
            if not zone_df.empty:
                st.dataframe(zone_df, hide_index=True, use_container_width=True)
            else:
                st.info("No zone data recorded yet.")
        with col_line:
            st.markdown("**Line Crossing Counts** (total by direction)")
            if not line_df.empty:
                st.dataframe(line_df, hide_index=True, use_container_width=True)
            else:
                st.info("No line crossings recorded yet.")

        st.markdown("**Crowd Alerts** (WARNING & CRITICAL events only)")
        if not alert_df.empty:
            def _row_color(row):
                if "CRITICAL" in str(row.get("Level", "")):
                    return ["background-color:#ffe0e0"] * len(row)
                if "WARNING" in str(row.get("Level", "")):
                    return ["background-color:#fff8e0"] * len(row)
                return [""] * len(row)

            st.dataframe(
                alert_df.style.apply(_row_color, axis=1),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.success("No WARNING or CRITICAL alerts in this session.")

        st.markdown("**Historical Zone Occupancy Over Time**")
        hist_df = _occupancy_history(session_id)
        if not hist_df.empty:
            st.line_chart(hist_df, use_container_width=True)
        else:
            st.info("No occupancy history yet.")

# ── Section 7: Report export ──────────────────────────────────────────────────

st.markdown("---")
st.subheader("7. Exported Report")

if REPORT_CSV.exists():
    try:
        df_report = pd.read_csv(REPORT_CSV, nrows=200)
        st.markdown(f"**Preview** (first 200 rows of `{REPORT_CSV.relative_to(ROOT)}`)")
        st.dataframe(df_report, hide_index=True, use_container_width=True)
        st.download_button(
            "⬇ Download report.csv",
            data=REPORT_CSV.read_bytes(),
            file_name="railway_analytics_report.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.warning(f"Could not read report: {e}")
else:
    st.info("No report CSV found yet.")
    if st.button("Generate Report CSV"):
        if not _db_ok():
            st.error("Run the analysis pipeline first to create the analytics database.")
        else:
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "export_report.py"),
                 "--db", str(DB_PATH), "--out", str(REPORT_CSV)],
                cwd=str(ROOT), capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0:
                st.success("Report generated.")
                st.rerun()
            else:
                st.error(f"Report generation failed:\n{result.stderr.strip()}")

# ── Section 8: Indian Railways context ───────────────────────────────────────

st.markdown("---")
st.subheader("8. What This Means for Indian Railways")
st.markdown(
    """
| Capability | Operational benefit |
|------------|---------------------|
| **Real-time zone occupancy** | Station managers see exactly which platform section is overloaded before an incident occurs |
| **Passenger flow counting** | Understand peak boarding/alighting patterns to schedule extra staff at the right gates |
| **Crowd alert levels** | Automated WARNING / CRITICAL alerts can trigger PA announcements or RPF deployment |
| **Historical analytics** | Shift reports, weekly trends, and peak-hour analysis feed into operational planning |
| **Line crossing counts** | Track how many passengers entered vs. left a zone — useful for evacuation accounting |

**Current status of this demo:**

- ✅ Detection, tracking, counting, and alerting pipeline works end-to-end
- ✅ Tested on public Indian railway platform footage
- ⚠️ YOLOv11n is a general-purpose person detector — not yet fine-tuned on Indian platform CCTV
- ⚠️ Zone thresholds (WARNING/CRITICAL counts) need calibration to each station's real density
- ⚠️ This demo runs on recorded video — live RTSP CCTV integration requires additional infrastructure
"""
)

# ── Section 9: Next steps ─────────────────────────────────────────────────────

st.markdown("---")
st.subheader("9. Next Steps Toward Production")
st.markdown(
    """
| Priority | Next step | What it enables |
|----------|-----------|-----------------|
| 🔴 High | **Connect live RTSP CCTV** | Real-time monitoring (the system already supports RTSP via `--source`) |
| 🔴 High | **Fine-tune YOLOv11 on Indian platform data** | Better accuracy in dense, cluttered platform environments |
| 🟡 Medium | **Multi-camera dashboard** | Monitor multiple platforms / stations from one screen |
| 🟡 Medium | **Alert notifications** | Push alerts to station control rooms via SMS / PA system |
| 🟡 Medium | **Calibrate crowding thresholds** | Set WARNING / CRITICAL limits based on each station's safe capacity |
| 🟢 Later | **TensorRT / DeepStream deployment** | 10–30× faster inference on NVIDIA hardware |
| 🟢 Later | **Annotate Indian platform dataset** | Required for fine-tuning — see `docs/ANNOTATION_WORKFLOW.md` |

> **To connect a live CCTV camera today:** enter its RTSP URL in the sidebar Advanced Options and click Run.
"""
)

st.markdown("---")
st.caption(
    "Indian Railway Passenger Counting Demo  ·  YOLOv11 + ByteTrack  ·  "
    "Requires fine-tuning on real Indian Railway CCTV footage before production deployment."
)
