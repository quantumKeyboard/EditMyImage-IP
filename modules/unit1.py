import streamlit as st
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io


def show_image_info(img: Image.Image, uploaded_file):
    col_img, col_meta = st.columns([1, 1], gap="large")

    with col_img:
        st.image(img, use_container_width=True)

        buf = io.BytesIO()
        fmt = uploaded_file.name.split(".")[-1].upper()
        save_fmt = "JPEG" if fmt in ("JPG", "JPEG") else fmt
        try:
            img.save(buf, format=save_fmt)
        except Exception:
            img.save(buf, format="PNG")
        buf.seek(0)
        st.download_button("Download Image", buf, file_name=uploaded_file.name, mime=f"image/{fmt.lower()}")

    with col_meta:
        arr = np.array(img)
        file_bytes = uploaded_file.size

        st.markdown("##### Image Properties")
        c1, c2 = st.columns(2)
        c1.metric("Width", f"{img.width} px")
        c2.metric("Height", f"{img.height} px")
        c1.metric("Mode", img.mode)
        c2.metric("File Size", f"{file_bytes / 1024:.1f} KB")
        c1.metric("Channels", arr.ndim if arr.ndim == 2 else arr.shape[2])
        c2.metric("Format", uploaded_file.name.split(".")[-1].upper())

        if arr.ndim == 3:
            st.markdown("##### Channel Statistics")
            labels = ["R", "G", "B"]
            stat_cols = st.columns(3)
            for i, (label, col) in enumerate(zip(labels, stat_cols)):
                ch = arr[:, :, i]
                col.markdown(f"**{label}**")
                col.markdown(f"<span style='font-size:0.72rem;color:#888'>Mean: {ch.mean():.1f}<br>Std: {ch.std():.1f}<br>Min: {ch.min()} / Max: {ch.max()}</span>", unsafe_allow_html=True)


def color_space_viewer(img: Image.Image):
    mode = st.selectbox("View as", ["RGB Channels", "Grayscale", "HSV Channels"])
    arr = np.array(img)

    if mode == "RGB Channels":
        cols = st.columns(4)
        cols[0].image(img, caption="Original", use_container_width=True)
        for i, label in enumerate(["Red", "Green", "Blue"]):
            ch = np.zeros_like(arr)
            ch[:, :, i] = arr[:, :, i]
            cols[i + 1].image(ch.astype(np.uint8), caption=label, use_container_width=True)

    elif mode == "Grayscale":
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        col1, col2 = st.columns(2)
        col1.image(img, caption="Original (RGB)", use_container_width=True)
        col2.image(gray, caption="Grayscale", use_container_width=True, clamp=True)
        st.markdown(f"<span style='font-size:0.75rem;color:#666'>Conversion: Y = 0.299R + 0.587G + 0.114B</span>", unsafe_allow_html=True)

    elif mode == "HSV Channels":
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        cols = st.columns(4)
        cols[0].image(img, caption="Original", use_container_width=True)
        for i, label in enumerate(["Hue", "Saturation", "Value"]):
            cols[i + 1].image(hsv[:, :, i], caption=label, use_container_width=True, clamp=True)
        st.markdown("<span style='font-size:0.75rem;color:#666'>H: 0–179 (color angle) · S: 0–255 (intensity) · V: 0–255 (brightness)</span>", unsafe_allow_html=True)


def sampling_demo(img: Image.Image):
    factor = st.slider("Scale Factor", 0.1, 1.0, 0.5, 0.05)
    interp = st.selectbox("Interpolation", ["Nearest Neighbour", "Bilinear", "Bicubic"])

    interp_map = {
        "Nearest Neighbour": Image.NEAREST,
        "Bilinear": Image.BILINEAR,
        "Bicubic": Image.BICUBIC,
    }
    method = interp_map[interp]

    w, h = img.width, img.height
    small = img.resize((max(1, int(w * factor)), max(1, int(h * factor))), method)
    restored = small.resize((w, h), method)

    col1, col2, col3 = st.columns(3)
    col1.image(img, caption=f"Original — {w}×{h}", use_container_width=True)
    col2.image(small, caption=f"Downsampled — {small.width}×{small.height}", use_container_width=True)
    col3.image(restored, caption=f"Upsampled back — {w}×{h}", use_container_width=True)

    st.markdown(f"<span style='font-size:0.75rem;color:#555'>Pixels reduced from {w*h:,} → {small.width*small.height:,} ({factor*100:.0f}% of original)</span>", unsafe_allow_html=True)
