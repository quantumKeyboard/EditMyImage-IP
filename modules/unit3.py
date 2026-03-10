import streamlit as st
from PIL import Image
import numpy as np
import cv2


def edge_detection_panel(img: Image.Image):
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    method = st.radio("Detector", ["Sobel", "Canny"], horizontal=True)

    if method == "Sobel":
        st.markdown("**Sobel Edge Detection** — computes gradient magnitude using 3×3 derivative kernels in X and Y directions")

        col_ctrl, _ = st.columns([1, 2])
        with col_ctrl:
            ksize = st.select_slider("Kernel Size", options=[1, 3, 5, 7], value=3, key="sobel_k")

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = np.clip(magnitude / magnitude.max() * 255, 0, 255).astype(np.uint8)

        gx_vis = np.clip(np.abs(grad_x) / np.abs(grad_x).max() * 255, 0, 255).astype(np.uint8)
        gy_vis = np.clip(np.abs(grad_y) / np.abs(grad_y).max() * 255, 0, 255).astype(np.uint8)

        col1, col2, col3, col4 = st.columns(4)
        col1.image(gray, caption="Grayscale", use_container_width=True, clamp=True)
        col2.image(gx_vis, caption="Gradient X (∂/∂x)", use_container_width=True)
        col3.image(gy_vis, caption="Gradient Y (∂/∂y)", use_container_width=True)
        col4.image(magnitude, caption="Magnitude √(Gx²+Gy²)", use_container_width=True)

        st.markdown(f"<span style='font-size:0.73rem;color:#555'>Sobel kernels detect horizontal and vertical intensity transitions. Magnitude = combined edge strength.</span>", unsafe_allow_html=True)

    elif method == "Canny":
        st.markdown("**Canny Edge Detection** — multi-stage: Gaussian smoothing → gradient → non-max suppression → hysteresis thresholding")

        col_ctrl, _ = st.columns([1, 2])
        with col_ctrl:
            sigma_c = st.slider("Gaussian Sigma (pre-blur)", 0.5, 4.0, 1.0, 0.5, key="canny_sigma")
            t_low = st.slider("Low Threshold", 0, 200, 50, key="canny_low")
            t_high = st.slider("High Threshold", 0, 500, 150, key="canny_high")

        if t_low >= t_high:
            st.warning("Low threshold must be less than High threshold.")
        else:
            k = int(2 * round(3 * sigma_c) + 1)
            blurred = cv2.GaussianBlur(gray, (k, k), sigma_c)
            edges = cv2.Canny(blurred, t_low, t_high)

            col1, col2, col3 = st.columns(3)
            col1.image(gray, caption="Original (gray)", use_container_width=True, clamp=True)
            col2.image(blurred, caption=f"After Gaussian (σ={sigma_c})", use_container_width=True, clamp=True)
            col3.image(edges, caption=f"Canny Edges [{t_low}, {t_high}]", use_container_width=True)

            n_edge_px = np.count_nonzero(edges)
            total_px = edges.size
            st.markdown(f"<span style='font-size:0.73rem;color:#555'>Edge pixels: {n_edge_px:,} / {total_px:,} ({100*n_edge_px/total_px:.2f}%)</span>", unsafe_allow_html=True)
