"""Static explanatory sections for the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def render_comparison_overview() -> None:
    """Render the body/head/hybrid comparison overview."""
    st.markdown("---")
    with st.expander("**1. What is this demo comparing?**", expanded=True):
        st.markdown(
            """
This demo analyses railway platform CCTV-style video using three passenger detection approaches:

| | Full-Body Detection | Head Detection | Hybrid body + head |
|---|---|---|---|
| **What it detects** | The full person | The passenger's head only | Full persons and heads in one fused output |
| **Best when** | Body detection is better when the full person is visible. | Head detection is used for far/packed/occluded passengers where bodies are hidden. | Hybrid mode uses both, then removes duplicates. |
| **Challenge** | Misses passengers hidden by crowds, luggage, pillars, or low resolution | Smaller targets can cause more tracking ID switches | Fusion depends on camera-specific ROIs and still needs validation |
| **Model** | Pretrained YOLO person detector | Fine-tuned YOLO head detector trained in Google Colab | Body detector + head detector + duplicate removal |

**How to present the result honestly:**
Platform CCTV cameras often cover large areas. Passengers near the far end of the platform may be too
small or too crowded for full-body detection. The head is often still visible even when the rest of the
body is blocked, so head detection is useful as a second signal. Hybrid mode combines body and head
detections and removes likely duplicates, but accuracy still requires manual ground truth from counted
or labelled frames.
"""
        )


def render_scope_brief(scope_doc: Path, root: Path) -> None:
    """Render the project scope document preview/download section."""
    st.markdown("---")
    with st.expander("**Problem statement, assumptions, scope, and steps (boss brief)**", expanded=False):
        if scope_doc.exists():
            st.markdown(
                "Full written brief — problem statement, objective, assumptions, in/out of scope, "
                "steps, accuracy improvement plan, risks, and a one-page boss summary — is in "
                f"`{scope_doc.relative_to(root)}`."
            )
            try:
                scope_text = scope_doc.read_text(encoding="utf-8")
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
                f"`{scope_doc.relative_to(root)}`."
            )


def render_cctv_context() -> None:
    """Render the CCTV-angle context section."""
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


def render_model_flow() -> None:
    """Render the model pipeline explanation."""
    st.markdown("---")
    with st.expander("**2. What is the model actually doing?**", expanded=False):
        st.markdown(
            """
1. **Frame extraction** — video split into individual images.
2. **YOLO detection** — neural network draws boxes: *persons* (body mode), *heads* (head mode), or both (hybrid mode).
3. **Hybrid fusion** — in hybrid mode, likely duplicate body/head boxes are removed before tracking.
4. **ByteTrack tracking** — links the same box across frames with a persistent ID (e.g. `#42`) to avoid double-counting.
5. **Zone occupancy** — counts tracked IDs inside each platform zone per frame.
6. **Line crossings** — virtual line counts IDs crossing it (IN / OUT direction).
7. **Crowd alerts** — raises WARNING or CRITICAL when a zone exceeds its threshold.
8. **Database logging** — everything saved to SQLite for reporting.

Body mode uses the *bottom-centre* of the box (feet on ground). Head mode uses the *centre* of the head box. Hybrid mode keeps per-detection anchors so body and head detections can share one analytics flow.
"""
        )


def render_output_legend() -> None:
    """Render the output-video legend."""
    st.markdown("---")
    with st.expander("**8. How to read the output video**", expanded=False):
        st.markdown(
            "| Overlay | Meaning |\n"
            "|---------|--------|\n"
            "| **Bounding boxes** | Each box = one detected passenger or head |\n"
            "| **ID numbers** (e.g. `#12`) | Tracker ID — same person across frames, prevents double-counting |\n"
            "| **Mode label** | Full-Body, Head-Based, or Hybrid Body + Head detection |\n"
            "| **Zone polygons** | 🟢 Normal → 🟡 Warning → 🔴 Critical |\n"
            "| **Counting line** | Virtual boundary — IN / OUT crossings |\n"
            "| **Counter overlay** | Detections, unique IDs, FPS |\n"
            "| **ALERT banner** | Zone over threshold — staff action needed |\n\n"
            "Body boxes = *person/passenger*. Head boxes = *head/passenger*. Hybrid output keeps both signals after duplicate removal."
        )


def render_data_collection() -> None:
    """Render the privacy/data collection explanation."""
    st.markdown("---")
    with st.expander("**11. What data is collected?**", expanded=False):
        st.markdown(
            "Stores **operational analytics only** — no personal identity data.\n\n"
            "| Stored | Purpose |\n"
            "|--------|---------|\n"
            "| Detector mode (`body`/`head`/`hybrid`) | Know which pipeline produced the record |\n"
            "| Timestamp and frame number | Time-series analysis |\n"
            "| Anonymised tracking ID | Avoid double-counting — no name or identity |\n"
            "| Zone occupancy counts | Monitor crowding per area |\n"
            "| Line crossing events (IN / OUT) | Passenger flow |\n"
            "| Alert level (NORMAL / WARNING / CRITICAL) | Staff response trigger |\n"
            "| Camera / source ID | Multi-camera support |\n\n"
            "No face crops, names, biometric data, or personal identity information is stored."
        )


def render_railways_value() -> None:
    """Render the stakeholder value explanation."""
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

Head detection is especially relevant for platform views where distant or occluded passengers
may be missed by full-body detection. Hybrid mode is a practical way to review body and head
detections together, but its counts still need manual ground-truth validation.
"""
        )


def render_limitations() -> None:
    """Render dashboard limitations."""
    st.markdown("---")
    with st.expander("**13. Limitations (read before presenting to stakeholders)**", expanded=False):
        st.markdown(
            "- **Not production-ready.** Prototype only — no live Indian Railway CCTV validation.\n"
            "- **Head model trained on RPEE-Heads**, not Indian Railway CCTV. Domain shift expected.\n"
            "- **No production accuracy claims.** Training metrics are in `docs/HEAD_MODEL_TRAINING_REPORT.md`.\n"
            "- **Hybrid mode is not a guarantee of accuracy.** It uses body and head detections together, then removes likely duplicates.\n"
            "- **Missed detections** still occur in low resolution and heavy occlusion.\n"
            "- **Head tracking ID switches** are more frequent than body tracking in dense crowds.\n"
            "- **Thresholds need station-specific calibration** — defaults may not match real capacity.\n"
            "- **Pre-recorded video only** in this demo. Live RTSP needs additional infrastructure.\n"
            "- **Privacy:** obtain all legal/regulatory approvals before deploying on real CCTV."
        )


def render_boss_guide() -> None:
    """Render the stakeholder demo guide."""
    st.markdown("---")
    with st.expander("**14. Boss Demo Guide**", expanded=False):
        st.markdown(
            """
**~10 minute flow:**
1. **Section 1** — explain body, head, and hybrid comparison table.
2. **Section 3** — show the input video.
3. **Section 4** — confirm body, head, and hybrid mode are available.
4. **Section 5** — click **Run Body, Head, and Hybrid Comparison**.
5. **Section 7** — side-by-side output videos.
6. **Section 9** — analytics comparison (detections, average detections/frame, unique tracks, zone occupancy, line counts).
7. **Section 11** — confirm no personal data is stored.
8. **Section 12** — operational benefits.
9. **Section 13** — be transparent about limitations.

**On the head model:** "We fine-tuned this model in Google Colab on a railway-platform
head dataset. It is integrated alongside the body detector, and hybrid mode combines both
outputs after duplicate removal. Next step: validate the counts against manual ground truth
on approved Indian Railway platform footage."
"""
        )


def render_footer() -> None:
    """Render the dashboard footer."""
    st.markdown("---")
    st.caption(
        "Indian Railway Platform Passenger Counting · Body, Head, and Hybrid Detection Comparison · "
        "YOLO + ByteTrack · Prototype — requires validation on Indian Railway CCTV before production deployment."
    )
