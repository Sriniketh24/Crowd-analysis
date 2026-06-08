"""Streamlit app — Indian Railway Platform Passenger Counting: Body, Head, and Hybrid Detection.

Run with:
    streamlit run src/dashboard/streamlit_app.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from subprocess import PIPE, STDOUT

import pandas as pd
import streamlit as st

from src.dashboard.app_config import (
    CCTV_VIDEO,
    HEAD_MODEL,
    INPUT_VIDEO,
    MODE_OPTIONS,
    PREVIOUS_VIDEO_BACKUP,
    ROOT,
    SCOPE_DOC,
    TUNING_CSV,
    ZONES_CONFIG,
)
from src.dashboard.content import (
    render_boss_guide,
    render_cctv_context,
    render_comparison_overview,
    render_data_collection,
    render_footer,
    render_limitations,
    render_model_flow,
    render_output_legend,
    render_railways_value,
    render_scope_brief,
)
from src.dashboard.data_access import (
    db_ok as _db_ok,
    relative_or_display as _relative_or_display,
    render_analytics as _render_analytics,
)

st.set_page_config(
    page_title="Indian Railway Passenger Counting: Body, Head, and Hybrid Detection",
    page_icon="🚉",
    layout="wide",
    initial_sidebar_state="collapsed",
)


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


def _add_hybrid_tuning_args(cmd: list[str]) -> list[str]:
    """Append tuning controls using hybrid-specific body/head CLI flags."""
    tuned = [
        *cmd,
        "--body-confidence", f"{tune_confidence:.2f}",
        "--head-confidence", f"{tune_confidence:.2f}",
        "--iou", f"{tune_iou:.2f}",
        "--body-imgsz", str(tune_imgsz),
        "--head-imgsz", str(tune_imgsz),
        "--tracker", tune_tracker,
        "--body-max-det", str(tune_max_det),
        "--head-max-det", str(tune_max_det),
    ]
    if tune_augment:
        tuned.append("--augment")
    return tuned


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

st.title("🚉 Indian Railway Platform Passenger Counting: Body, Head, and Hybrid Detection")
st.caption(
    "AI-powered crowd monitoring · Compares full-body, head-based, and hybrid body + head detection · "
    "YOLO + ByteTrack · CCTV-angle platform retest · Prototype — not production-ready"
)

render_comparison_overview()
render_scope_brief(SCOPE_DOC, ROOT)
render_cctv_context()
render_model_flow()

# ── Section 3: Input video ────────────────────────────────────────────────────

st.markdown("---")
st.subheader("3. Input Video — CCTV-Angle Platform Footage")
if display_video_path and display_video_path.exists():
    st.video(str(display_video_path))
    st.caption(
        f"`{_relative_or_display(display_video_path)}` — selected CCTV-angle platform footage used as input for body, head, and hybrid modes."
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
col_ms1, col_ms2, col_ms3 = st.columns(3)
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
with col_ms3:
    if HEAD_MODEL.exists() and ZONES_CONFIG.exists():
        st.success(
            "**Hybrid body + head**: Available.\n\n"
            "Runs body and head detectors together, then removes likely duplicates before tracking."
        )
    else:
        st.warning(
            "**Hybrid body + head** needs the head model and zone/ROI config before it can run."
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
    "Run Body, Head, and Hybrid Comparison With Selected Settings",
    width="stretch",
    disabled=not video_source or not HEAD_MODEL.exists(),
)
if run_tuned_compare:
    _run_subprocess(
        _add_tuning_args([
            sys.executable, "scripts/run_comparison_demo.py",
            "--source", video_source, "--zones-config", str(ZONES_CONFIG),
            "--head-confidence", f"{tune_confidence:.2f}",
        ]),
        "Tuned Body, Head, and Hybrid Comparison",
    )

st.markdown("**Latest tuning_results.csv**")
if TUNING_CSV.exists():
    try:
        tuning_df = pd.read_csv(TUNING_CSV)
        st.dataframe(tuning_df, hide_index=True, width="stretch")
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
    selected_mode_label = st.radio(
        "Detection mode",
        list(MODE_OPTIONS),
        horizontal=True,
        index=2,
        help="Choose which output to run/display: body output, head output, or hybrid body + head output.",
    )
    selected_mode = MODE_OPTIONS[selected_mode_label]
    needs_head_model = selected_mode["mode"] in {"head", "hybrid"}
    c1, c2, c3 = st.columns(3)
    run_selected = c1.button(
        f"▶ Run {selected_mode['label']}",
        type="primary",
        width="stretch",
        disabled=needs_head_model and not head_ready,
    )
    run_compare = c2.button(
        "▶ Run All Three Modes",
        type="secondary",
        width="stretch",
        disabled=not head_ready,
    )
    export_selected = c3.button(
        "📄 Export Selected Report CSV",
        width="stretch",
        disabled=not Path(selected_mode["db"]).exists(),
    )
    if not head_ready:
        st.caption("⚠️ Head and hybrid detection disabled — head model missing. See Section 4.")
    if run_selected:
        selected_command = [
            sys.executable, "scripts/run_video_demo.py",
            "--source", video_source, "--output", str(selected_mode["video"]),
            "--zones-config", str(ZONES_CONFIG), "--db", str(selected_mode["db"]),
            "--detector-mode", str(selected_mode["mode"]),
        ]
        if selected_mode["mode"] == "head":
            selected_command.extend(["--model", str(HEAD_MODEL)])
            selected_command = _add_tuning_args(selected_command)
        elif selected_mode["mode"] == "hybrid":
            selected_command.extend(["--head-model", str(HEAD_MODEL)])
            selected_command = _add_hybrid_tuning_args(selected_command)
        else:
            selected_command = _add_tuning_args(selected_command)
        _run_subprocess(selected_command, str(selected_mode["label"]))
    if run_compare:
        _run_subprocess(_add_tuning_args([
            sys.executable, "scripts/run_comparison_demo.py",
            "--source", video_source, "--zones-config", str(ZONES_CONFIG),
            "--head-confidence", f"{tune_confidence:.2f}",
        ]), "Body, Head, and Hybrid Comparison")
    if export_selected:
        _run_subprocess(
            [
                sys.executable, "scripts/export_report.py",
                "--db", str(selected_mode["db"]), "--out", str(selected_mode["csv"]),
            ],
            f"{selected_mode['label']} Report Export",
        )

# ── Section 7: Side-by-side output videos ────────────────────────────────────

st.markdown("---")
st.subheader("7. Detection Output — Body, Head, and Hybrid")
for col, (mode_name, mode_info) in zip(st.columns(3), MODE_OPTIONS.items()):
    with col:
        st.markdown(f"**{mode_info['label']} Output**")
        video_path = Path(mode_info["video"])
        if video_path.exists():
            st.video(str(video_path))
            st.caption(f"`{video_path.relative_to(ROOT)}`")
        else:
            st.info(f"No {mode_name.lower()} output yet. Select **{mode_name}** in Section 6 and run detection.")

render_output_legend()

# ── Section 9: Analytics comparison ──────────────────────────────────────────

st.markdown("---")
st.subheader("9. Analytics Comparison")
for col, (_, mode_info) in zip(st.columns(3), MODE_OPTIONS.items()):
    with col:
        st.markdown(f"### {mode_info['label']}")
        _render_analytics(Path(mode_info["db"]), str(mode_info["mode"]))

# ── Section 10: Report previews ───────────────────────────────────────────────

st.markdown("---")
st.subheader("10. Report Previews")
for col, (mode_name, mode_info) in zip(st.columns(3), MODE_OPTIONS.items()):
    with col:
        csv_path = Path(mode_info["csv"])
        st.markdown(f"**{mode_info['label']} Report**")
        if csv_path.exists():
            try:
                report_df = pd.read_csv(csv_path, nrows=100)
                st.dataframe(report_df, hide_index=True, width="stretch")
                st.download_button(
                    f"⬇ Download {csv_path.name}",
                    csv_path.read_bytes(),
                    csv_path.name,
                    "text/csv",
                )
            except Exception as exc:
                st.warning(f"Could not read {mode_name.lower()} report: {exc}")
        else:
            st.info(f"Select **{mode_name}** in Section 6 and export the report CSV.")

render_data_collection()
render_railways_value()
render_limitations()
render_boss_guide()
render_footer()
