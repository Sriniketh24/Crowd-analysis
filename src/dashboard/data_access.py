"""SQLite-backed analytics helpers for the Streamlit dashboard."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from src.dashboard.app_config import ALERT_ICON, ROOT


def db_ok(db_path: Path) -> bool:
    """Return whether an analytics DB exists and has content."""
    return db_path.exists() and db_path.stat().st_size > 0


def query(db_path: Path, sql: str, params: tuple = ()) -> pd.DataFrame:
    """Run a read-only dashboard query, returning an empty frame on failure."""
    if not db_ok(db_path):
        return pd.DataFrame()
    try:
        with sqlite3.connect(str(db_path), timeout=10) as conn:
            return pd.read_sql_query(sql, conn, params=params)
    except Exception:
        return pd.DataFrame()


def latest_session(db_path: Path) -> dict | None:
    """Return the latest completed session, falling back to the latest session."""
    df = query(
        db_path,
        "SELECT id, camera_id, source, started_at, total_frames, "
        "total_unique_passengers, model_weights, tracker_type "
        "FROM run_sessions "
        "WHERE ended_at IS NOT NULL AND total_frames IS NOT NULL "
        "ORDER BY started_at DESC, id DESC LIMIT 1",
    )
    if df.empty:
        df = query(
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


def zone_occupancy(db_path: Path, session_id: int) -> pd.DataFrame:
    """Return latest occupancy per zone for a run session."""
    df = query(
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


def line_crossings(db_path: Path, session_id: int) -> pd.DataFrame:
    """Return grouped line crossing counts for a run session."""
    return query(
        db_path,
        "SELECT line_id AS Line, direction AS Direction, COUNT(*) AS Count "
        "FROM line_crossing_events WHERE run_session_id=? GROUP BY line_id, direction",
        (session_id,),
    )


def summary_metrics(db_path: Path, session_id: int, session: dict) -> dict[str, int | float | str]:
    """Return compact metrics for one completed dashboard run."""
    frame_df = query(
        db_path,
        "SELECT total_detections, unique_passengers_seen FROM frames_processed "
        "WHERE run_session_id=? ORDER BY frame_index DESC, id DESC LIMIT 1",
        (session_id,),
    )
    avg_df = query(
        db_path,
        "SELECT AVG(total_detections) AS avg_detections FROM frames_processed "
        "WHERE run_session_id=?",
        (session_id,),
    )
    zone_df = zone_occupancy(db_path, session_id)
    line_df = line_crossings(db_path, session_id)

    current_detections: int | str = "—"
    unique_tracks: int | str = (
        session["total_unique_passengers"]
        if session["total_unique_passengers"] is not None
        else "—"
    )
    if not frame_df.empty:
        current_detections = int(frame_df.iloc[0].get("total_detections") or 0)
        unique_tracks = int(frame_df.iloc[0].get("unique_passengers_seen") or 0)
    avg_detections: float | str = "—"
    if not avg_df.empty and avg_df.iloc[0].get("avg_detections") is not None:
        avg_detections = round(float(avg_df.iloc[0]["avg_detections"]), 2)
    zone_occupancy_count: int | str = int(zone_df["Count"].sum()) if not zone_df.empty else "—"
    line_counts: int | str = int(line_df["Count"].sum()) if not line_df.empty else "—"
    return {
        "Current detections": current_detections,
        "Average detections/frame": avg_detections,
        "Unique tracks": unique_tracks,
        "Zone occupancy": zone_occupancy_count,
        "Line counts": line_counts,
    }


def alerts(db_path: Path, session_id: int) -> pd.DataFrame:
    """Return recent non-normal crowd alerts."""
    df = query(
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


def detector_mode_label(db_path: Path, session_id: int) -> str:
    """Return the detector mode recorded for a session."""
    df = query(db_path, "SELECT detector_mode FROM run_sessions WHERE id=?", (session_id,))
    if df.empty or "detector_mode" not in df.columns:
        return "—"
    return str(df.iloc[0]["detector_mode"] or "—")


def relative_or_display(path: Path | str) -> str:
    """Return a project-relative path when possible for dashboard captions."""
    candidate = Path(path)
    try:
        return str(candidate.relative_to(ROOT))
    except ValueError:
        return str(path)


def render_analytics(db_path: Path, label: str) -> None:
    """Render the dashboard analytics cards and tables for one DB."""
    if not db_ok(db_path):
        st.info(f"No {label} database. Run {label} detection first.")
        return
    session = latest_session(db_path)
    if session is None:
        st.info(f"{label} database has no sessions yet.")
        return
    sid = session["id"]
    st.caption(f"Mode: `{detector_mode_label(db_path, sid)}` · Session #{sid} · {session['started_at']}")
    metric_values = summary_metrics(db_path, sid, session)
    metric_cols = st.columns(5)
    for col, (name, value) in zip(metric_cols, metric_values.items()):
        col.metric(name, value)
    st.caption(f"Frames processed: {session['total_frames'] if session['total_frames'] is not None else '—'}")
    for title, df in [
        ("**Zone Occupancy**", zone_occupancy(db_path, sid)),
        ("**Line Crossings**", line_crossings(db_path, sid)),
    ]:
        st.markdown(title)
        if not df.empty:
            st.dataframe(df, hide_index=True, width="stretch")
        else:
            st.caption("No data yet.")
    alert_df = alerts(db_path, sid)
    st.markdown("**Crowd Alerts**")
    if not alert_df.empty:
        st.dataframe(alert_df, hide_index=True, width="stretch")
    else:
        st.success("No WARNING/CRITICAL alerts.")
