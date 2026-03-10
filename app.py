import streamlit as st
from PIL import Image
import numpy as np
import io

from modules.unit1 import show_image_info, color_space_viewer, sampling_demo
from modules.unit2 import enhancement_panel
from modules.unit3 import edge_detection_panel
from modules.unit4 import compression_panel

st.set_page_config(
    page_title="PixelLab",
    page_icon="⬛",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
}

.stApp {
    background-color: #0e0e0e;
    color: #e8e8e8;
}

section[data-testid="stSidebar"] {
    background-color: #141414;
    border-right: 1px solid #2a2a2a;
}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stFileUploader label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: #c8c8c8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.03em;
}

.stButton > button {
    background: #1e1e1e;
    color: #e8e8e8;
    border: 1px solid #333;
    border-radius: 2px;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    padding: 0.4rem 1rem;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background: #e8e8e8;
    color: #0e0e0e;
    border-color: #e8e8e8;
}

.stSelectbox > div > div {
    background-color: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    color: #e8e8e8 !important;
    border-radius: 2px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
}

.stSlider > div > div > div {
    background-color: #e8e8e8 !important;
}

div[data-testid="stMetric"] {
    background: #141414;
    border: 1px solid #222;
    border-radius: 2px;
    padding: 0.6rem 0.8rem;
}
div[data-testid="stMetric"] label {
    color: #888 !important;
    font-size: 0.65rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'DM Mono', monospace !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    color: #e8e8e8 !important;
    font-size: 0.95rem !important;
    font-family: 'DM Mono', monospace !important;
}

.stTabs [data-baseweb="tab-list"] {
    background-color: #141414;
    border-bottom: 1px solid #222;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #666;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.5rem 1.2rem;
    border-radius: 0;
}
.stTabs [aria-selected="true"] {
    background-color: #1e1e1e !important;
    color: #e8e8e8 !important;
    border-bottom: 1px solid #e8e8e8 !important;
}

hr {
    border-color: #1e1e1e;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

div[data-testid="stImage"] img {
    border: 1px solid #1e1e1e;
}

.pixel-header {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.6rem;
    letter-spacing: -0.04em;
    color: #e8e8e8;
    line-height: 1;
}
.pixel-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 0.2rem;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="pixel-header">PIXEL<br>LAB</div>', unsafe_allow_html=True)
    st.markdown('<div class="pixel-sub">Image Processing Tool</div>', unsafe_allow_html=True)
    st.markdown("---")

    uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "bmp", "tiff"])

    if uploaded:
        img = Image.open(uploaded)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        st.session_state["original"] = img
        st.markdown("---")
        st.markdown('<span style="font-size:0.68rem;color:#555;text-transform:uppercase;letter-spacing:0.1em">Navigate</span>', unsafe_allow_html=True)
        section = st.selectbox(
            "",
            ["Unit 1 — File & Color", "Unit 2 — Enhancement", "Unit 3 — Analysis", "Unit 4 — Compression"],
            label_visibility="collapsed"
        )
    else:
        section = None


# ── Main ───────────────────────────────────────────────────────────────────────
if "original" not in st.session_state or st.session_state["original"] is None:
    st.markdown("---")
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:55vh;gap:1rem;">
        <div style="font-family:'Syne',sans-serif;font-size:3rem;font-weight:800;color:#1e1e1e;letter-spacing:-0.05em;line-height:1;">PIXEL LAB</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#333;text-transform:uppercase;letter-spacing:0.15em;">Upload an image from the sidebar to begin</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

img = st.session_state["original"]

if section == "Unit 1 — File & Color":
    st.markdown('<h2 style="color:#e8e8e8;margin-bottom:0.2rem">File & Color</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#444;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;margin-top:0">Unit 1 — Introduction</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Info Panel", "Color Spaces", "Sampling"])
    with tab1:
        show_image_info(img, uploaded)
    with tab2:
        color_space_viewer(img)
    with tab3:
        sampling_demo(img)

elif section == "Unit 2 — Enhancement":
    st.markdown('<h2 style="color:#e8e8e8;margin-bottom:0.2rem">Enhancement</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#444;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;margin-top:0">Unit 2 — Spatial & Frequency Domain</p>', unsafe_allow_html=True)
    enhancement_panel(img)

elif section == "Unit 3 — Analysis":
    st.markdown('<h2 style="color:#e8e8e8;margin-bottom:0.2rem">Analysis</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#444;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;margin-top:0">Unit 3 — Edge Detection</p>', unsafe_allow_html=True)
    edge_detection_panel(img)

elif section == "Unit 4 — Compression":
    st.markdown('<h2 style="color:#e8e8e8;margin-bottom:0.2rem">Compression</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#444;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;margin-top:0">Unit 4 — Lossy & Lossless</p>', unsafe_allow_html=True)
    compression_panel(img)
