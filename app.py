"""
Local test workbench — Stage A (automatic verification) + Stage B (full
cleanup) behind a browser UI styled to match the mockup we designed
earlier (card-based, color-coded status, rounded corners, plain-
language interface, advanced/technical detail tucked away).

This is NOT the final website — it's your own testing tool, built to
give a feel for how the real clinician-facing flow will work.

Run with:
    streamlit run app.py
"""

import os
import tempfile
import time

import streamlit as st

from pipeline import verify_scan
from stage_b import run_stage_b
from reporting import describe_stage_a, describe_stage_b

st.set_page_config(page_title="APEX Scan Verification", page_icon="🦿", layout="centered")

DEVICE_TYPES = ["AFO", "KAFO", "insole", "spinal", "arm"]

VERDICT_STYLE = {
    "pass": {"color": "#0F6E56", "bg": "#E1F5EE", "label": "Passed", "icon": "✓"},
    "needs_review": {"color": "#854F0B", "bg": "#FAEEDA", "label": "Needs review", "icon": "!"},
    "fail": {"color": "#8A1F11", "bg": "#FBE4E1", "label": "Failed! Rescan needed", "icon": "✕"},
}

REASON_LABELS = {
    "double_surface": "Double surface detected (scan defect)",
    "scale_deviation": "Scan scale looks wrong (possible unit error)",
    "hole_size": "A hole larger than expected was found",
    "size_range": "Proportions are outside the typical range (may be real anatomy)",
    "possible_disconnected_anatomy": "A separate section was detected may be real anatomy (e.g. swelling) rather than a scanning error",
}

# ---------------------------------------------------------------------------
# Styling — matches the earlier approved mockup: warm neutral background,
# rounded cards, plain-language status, minimal chrome.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    :root {
        --surface-0: #F4F3F0;
        --surface-1: #FFFFFF;
        --text-primary: #1E1E1C;
        --text-secondary: #6B6A66;
        --text-muted: #9C9B96;
        --border: #E4E2DC;
        --accent: #185FA5;
        --accent-bg: #E6F1FB;
        --radius: 14px;
    }
    .stApp { background: var(--surface-0); color: var(--text-primary); }
    .stApp, .stApp p, .stApp div, .stApp span, .stApp label { color: var(--text-primary); }
    #MainMenu, footer, header { visibility: hidden; }

    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
        color: var(--text-secondary) !important;
    }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p {
        color: var(--text-primary) !important;
    }

    .apex-topbar {
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: 4px;
    }
    .apex-title { font-size: 22px; font-weight: 700; color: var(--text-primary); margin: 0; }
    .apex-subtitle { font-size: 13px; color: var(--text-secondary); margin: 2px 0 20px; }

    .apex-card {
        background: var(--surface-1);
        border-radius: var(--radius);
        border: 1px solid var(--border);
        padding: 20px 22px;
        margin-bottom: 16px;
    }

    .apex-verdict-banner {
        border-radius: var(--radius);
        padding: 18px 22px;
        margin: 12px 0 8px;
        display: flex; align-items: center; gap: 12px;
    }
    .apex-verdict-icon {
        width: 34px; height: 34px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 18px; font-weight: 700; color: white; flex-shrink: 0;
    }
    .apex-verdict-text { font-size: 19px; font-weight: 600; }
    .apex-verdict-caption { font-size: 13px; color: var(--text-secondary); margin-top: 2px; }

    .apex-reason-row {
        display: flex; align-items: flex-start; gap: 10px;
        padding: 9px 0; border-bottom: 1px solid var(--border);
        font-size: 14px; color: var(--text-primary);
    }
    .apex-reason-row:last-child { border-bottom: none; }

    .apex-section-label {
        font-size: 12px; text-transform: uppercase; letter-spacing: 0.03em;
        color: var(--text-secondary); font-weight: 600; margin: 4px 0 10px;
    }

    .apex-stat-grid { display: flex; gap: 12px; margin: 10px 0 4px; }
    .apex-stat { flex: 1; background: var(--surface-0); border-radius: 10px; padding: 12px 14px; }
    .apex-stat-label { font-size: 12px; color: var(--text-secondary); margin: 0 0 2px; }
    .apex-stat-value { font-size: 20px; font-weight: 700; color: var(--text-primary); margin: 0; }

    div[data-testid="stFileUploader"] {
        background: var(--surface-1); border-radius: var(--radius);
        border: 1.5px dashed var(--border); padding: 8px;
    }
    div[data-testid="stFileUploader"] section {
        background: var(--surface-1) !important;
    }
    div[data-testid="stFileUploaderFile"],
    div[data-testid="stFileUploaderFile"] * {
        background: var(--surface-0) !important;
        color: var(--text-primary) !important;
    }
    div[data-testid="stFileUploaderFileName"] {
        color: var(--text-primary) !important;
    }
    div[data-testid="stFileUploader"] small {
        color: var(--text-secondary) !important;
    }
    div.stButton > button {
        border-radius: 10px; font-weight: 600; padding: 10px 18px;
    }
    div.stButton > button[kind="primary"],
    div.stButton > button[kind="primary"] p,
    div.stButton > button[kind="primary"] div {
        background: var(--text-primary) !important;
        color: #FFFFFF !important;
        border: none;
    }
    div.stButton > button[kind="secondary"],
    div.stButton > button[kind="secondary"] p,
    div.stButton > button[kind="secondary"] div {
        color: var(--text-primary) !important;
        background: var(--surface-1) !important;
        border: 1px solid var(--border) !important;
    }
    div[data-testid="stDownloadButton"] > button,
    div[data-testid="stDownloadButton"] > button p {
        color: var(--text-primary) !important;
        background: var(--surface-1) !important;
        border: 1px solid var(--border) !important;
    }
</style>
""", unsafe_allow_html=True)


def save_upload_to_temp(uploaded_file) -> str:
    suffix = "." + uploaded_file.name.rsplit(".", 1)[-1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


def reset_state():
    for key in ["stage_a_result", "stage_a_path", "stage_b_result", "stage_b_output_path", "confirmed_review"]:
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# Top bar
# ---------------------------------------------------------------------------
st.markdown("""
<div class="apex-topbar">
    <div>
        <p class="apex-title">Scan verification</p>
        <p class="apex-subtitle">APEX &middot; local test workbench</p>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<p class="apex-section-label">Scan details</p>', unsafe_allow_html=True)
    device_type = st.selectbox("What is this scan for?", DEVICE_TYPES, label_visibility="collapsed")
    st.caption(f"Checking against the **{device_type}** profile — thresholds live in `device_profiles.json`.")

st.markdown('<div class="apex-card">', unsafe_allow_html=True)
st.markdown('<p class="apex-section-label">Upload scan</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a scan", type=["obj", "stl", "ply"],
                                  on_change=reset_state, label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    scan_path = save_upload_to_temp(uploaded_file)

    # --- Stage A runs automatically the moment a file is uploaded ---
    if "stage_a_result" not in st.session_state:
        with st.spinner("Checking scan quality..."):
            t0 = time.perf_counter()
            try:
                result = verify_scan(scan_path, device_type)
                st.session_state["stage_a_result"] = result
                st.session_state["stage_a_path"] = scan_path
            except Exception as e:
                st.error(f"Something went wrong running Stage A: {e}")
                st.stop()

    result = st.session_state["stage_a_result"]
    verdict = result["final_verdict"]
    style = VERDICT_STYLE[verdict]

    st.markdown('<div class="apex-card">', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="apex-verdict-banner" style="background:{style['bg']};">
        <div class="apex-verdict-icon" style="background:{style['color']};">{style['icon']}</div>
        <div>
            <div class="apex-verdict-text" style="color:{style['color']};">{style['label']}</div>
            <div class="apex-verdict-caption">Verified against the {device_type} profile &middot; {result['total_pipeline_time_ms']}ms</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    reasons = result.get("reasons", [])
    check_lines = describe_stage_a(result)
    if check_lines:
        st.markdown('<p class="apex-section-label" style="margin-top:16px;">Checks performed</p>', unsafe_allow_html=True)
        rows = "".join(
            f'<div class="apex-reason-row">{line}</div>' for line in check_lines
        )
        st.markdown(rows, unsafe_allow_html=True)

    with st.expander("Advanced — full technical detail"):
        st.json(result)

    st.markdown('</div>', unsafe_allow_html=True)

    # --- Flow branches by verdict ---
    if verdict == "fail":
        st.markdown("""
        <div class="apex-card" style="border-left: 4px solid #8A1F11; color: var(--text-primary);">
            This scan cannot proceed to fabrication. Please rescan the patient while they're still present.
        </div>
        """, unsafe_allow_html=True)

    elif verdict == "needs_review":
        st.markdown('<div class="apex-card">', unsafe_allow_html=True)
        st.markdown('<p style="color: var(--text-primary);">This may reflect real anatomy (e.g. swelling) rather than a scanning error. A clinician should confirm before proceeding.</p>', unsafe_allow_html=True)
        if not st.session_state.get("confirmed_review"):
            if st.button("Confirm — reflects patient's actual condition, proceed anyway"):
                st.session_state["confirmed_review"] = True
                st.rerun()
        else:
            st.success("Confirmed by clinician. Ready for cleanup.")
        st.markdown('</div>', unsafe_allow_html=True)

    if verdict == "pass" or st.session_state.get("confirmed_review"):
        st.markdown('<div class="apex-card">', unsafe_allow_html=True)
        st.markdown('<p class="apex-section-label">Full cleanup</p>', unsafe_allow_html=True)
        st.caption("Reconstructs and repairs the mesh for fabrication. No time pressure the patient does not need to wait for this. Can take from under a minute to a few minutes depending on complexity.")

        if st.button("Run cleanup", type="primary"):
            with st.spinner("Running full cleanup this can take a few minutes..."):
                output_path = st.session_state["stage_a_path"].rsplit(".", 1)[0] + "_cleaned.ply"
                try:
                    b_result = run_stage_b(st.session_state["stage_a_path"], output_path)
                    st.session_state["stage_b_result"] = b_result
                    st.session_state["stage_b_output_path"] = output_path
                except Exception as e:
                    st.error(f"Something went wrong running Stage B: {e}")

        if "stage_b_result" in st.session_state:
            b_result = st.session_state["stage_b_result"]
            watertight = b_result["final_is_watertight"]

            st.markdown(f"""
            <div class="apex-stat-grid">
                <div class="apex-stat">
                    <p class="apex-stat-label">Triangles</p>
                    <p class="apex-stat-value">{b_result['final_triangle_count']:,}</p>
                </div>
                <div class="apex-stat">
                    <p class="apex-stat-label">Time taken</p>
                    <p class="apex-stat-value">{b_result['total_time_ms']/1000:.1f}s</p>
                </div>
                <div class="apex-stat">
                    <p class="apex-stat-label">Watertight</p>
                    <p class="apex-stat-value" style="color:{'#0F6E56' if watertight else '#854F0B'};">{'Yes' if watertight else 'Mostly'}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if not watertight:
                st.caption("Small remaining gaps may need a quick manual touch-up (e.g. Meshmixer) before fabrication.")

            st.markdown('<p class="apex-section-label" style="margin-top:16px;">What was done</p>', unsafe_allow_html=True)
            b_lines = describe_stage_b(b_result)
            b_rows = "".join(f'<div class="apex-reason-row">{line}</div>' for line in b_lines)
            st.markdown(b_rows, unsafe_allow_html=True)

            with st.expander("Advanced Stage B technical detail"):
                st.json(b_result)

            output_path = st.session_state["stage_b_output_path"]
            if os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    st.download_button("Download cleaned mesh", f, file_name=os.path.basename(output_path))

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="apex-card" style="text-align:center; color: var(--text-secondary); padding: 40px 20px;">
        Upload a scan above to begin. Verification runs automatically.
    </div>
    """, unsafe_allow_html=True)
