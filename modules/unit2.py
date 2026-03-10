import streamlit as st
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io


def _to_gray(arr):
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def _show_pair(orig, result, label="Result"):
    col1, col2 = st.columns(2)
    col1.image(orig, caption="Original", use_container_width=True)
    col2.image(result, caption=label, use_container_width=True, clamp=True)


def _hist_fig(arr_gray, title=""):
    fig, ax = plt.subplots(figsize=(5, 2))
    fig.patch.set_facecolor("#141414")
    ax.set_facecolor("#0e0e0e")
    ax.hist(arr_gray.ravel(), bins=256, range=(0, 256), color="#e8e8e8", linewidth=0)
    ax.set_xlim(0, 255)
    ax.tick_params(colors="#555", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#222")
    if title:
        ax.set_title(title, color="#555", fontsize=7)
    fig.tight_layout(pad=0.3)
    return fig


def enhancement_panel(img: Image.Image):
    arr = np.array(img)
    gray = _to_gray(arr)

    tabs = st.tabs([
        "Negative", "Contrast Stretch", "Histogram EQ",
        "Bit Plane", "Box Filter", "Gaussian", "Median",
        "Laplacian", "High-pass"
    ])

    # ── Negative ──────────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("**Image Negative** — inverts pixel intensities: `I' = 255 - I`")
        neg = 255 - arr
        _show_pair(img, neg.astype(np.uint8))

    # ── Contrast Stretching ───────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("**Contrast Stretching** — linearly maps a selected intensity range to full [0, 255]")
        col_s, _ = st.columns([1, 2])
        with col_s:
            r_min = st.slider("Input min (r1)", 0, 127, 50, key="cs_min")
            r_max = st.slider("Input max (r2)", 128, 255, 200, key="cs_max")

        stretched = np.clip((gray.astype(np.float32) - r_min) / max(r_max - r_min, 1) * 255, 0, 255).astype(np.uint8)
        _show_pair(gray, stretched, "Stretched")

    # ── Histogram EQ ──────────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("**Histogram Equalization** — redistributes intensities to flatten the histogram")
        equalized = cv2.equalizeHist(gray)

        col1, col2 = st.columns(2)
        col1.image(gray, caption="Original", use_container_width=True, clamp=True)
        col2.image(equalized, caption="Equalized", use_container_width=True, clamp=True)

        hcol1, hcol2 = st.columns(2)
        hcol1.pyplot(_hist_fig(gray, "Before"))
        hcol2.pyplot(_hist_fig(equalized, "After"))
        plt.close("all")

    # ── Bit Plane Slicer ──────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("**Bit Plane Slicing** — extracts a single bit layer from each pixel's binary representation")
        plane = st.slider("Bit Plane (0 = LSB, 7 = MSB)", 0, 7, 7, key="bp")
        bit_plane = ((gray >> plane) & 1) * 255
        _show_pair(gray, bit_plane.astype(np.uint8), f"Plane {plane}")
        st.markdown(f"<span style='font-size:0.73rem;color:#555'>Plane {plane} contributes 2^{plane} = {2**plane} to each pixel value.</span>", unsafe_allow_html=True)

    # ── Box / Averaging Filter ─────────────────────────────────────────────────
    with tabs[4]:
        st.markdown("**Averaging (Box) Filter** — replaces each pixel with the mean of its neighbourhood")
        k = st.slider("Kernel Size", 3, 21, 5, step=2, key="box_k")
        blurred = cv2.blur(arr, (k, k))
        _show_pair(img, blurred.astype(np.uint8), f"Box Filter k={k}")

    # ── Gaussian Filter ───────────────────────────────────────────────────────
    with tabs[5]:
        st.markdown("**Gaussian Filter** — weighted smoothing; pixels closer to centre contribute more")
        sigma = st.slider("Sigma (σ)", 0.5, 10.0, 1.5, 0.5, key="gauss_s")
        k_size = int(2 * round(3 * sigma) + 1)
        gauss = cv2.GaussianBlur(arr, (k_size, k_size), sigma)
        _show_pair(img, gauss.astype(np.uint8), f"Gaussian σ={sigma}")
        st.markdown(f"<span style='font-size:0.73rem;color:#555'>Auto kernel size: {k_size}×{k_size} (= 6σ + 1)</span>", unsafe_allow_html=True)

    # ── Median Filter ─────────────────────────────────────────────────────────
    with tabs[6]:
        st.markdown("**Median Filter** — replaces each pixel with the median of its neighbourhood; excellent for salt-and-pepper noise")
        k_med = st.slider("Kernel Size", 3, 15, 3, step=2, key="med_k")
        median = cv2.medianBlur(arr, k_med)
        _show_pair(img, median.astype(np.uint8), f"Median k={k_med}")

    # ── Laplacian Sharpening ──────────────────────────────────────────────────
    with tabs[7]:
        st.markdown("**Laplacian Sharpening** — adds second-order derivative back to the image to enhance edges")
        strength = st.slider("Strength", 0.1, 3.0, 1.0, 0.1, key="lap_s")
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        sharpened = np.clip(gray.astype(np.float64) - strength * lap, 0, 255).astype(np.uint8)
        col1, col2, col3 = st.columns(3)
        col1.image(gray, caption="Original (gray)", use_container_width=True, clamp=True)
        col2.image(np.clip(np.abs(lap), 0, 255).astype(np.uint8), caption="Laplacian (edges)", use_container_width=True)
        col3.image(sharpened, caption="Sharpened", use_container_width=True)

    # ── High-pass Filter ──────────────────────────────────────────────────────
    with tabs[8]:
        st.markdown("**High-pass Filter** — `HPF = Original − Low-pass`. Isolates fine details and edges")
        sigma_hp = st.slider("Blur Sigma (for subtraction)", 1.0, 15.0, 3.0, 0.5, key="hp_s")
        strength_hp = st.slider("Add-back Strength", 0.5, 3.0, 1.5, 0.1, key="hp_st")
        k_hp = int(2 * round(3 * sigma_hp) + 1)
        blurred_hp = cv2.GaussianBlur(arr.astype(np.float32), (k_hp, k_hp), sigma_hp)
        high = arr.astype(np.float32) - blurred_hp
        result_hp = np.clip(arr.astype(np.float32) + strength_hp * high, 0, 255).astype(np.uint8)
        hpf_vis = np.clip(high + 128, 0, 255).astype(np.uint8)

        col1, col2, col3 = st.columns(3)
        col1.image(img, caption="Original", use_container_width=True)
        col2.image(hpf_vis, caption="HPF (centred at 128)", use_container_width=True)
        col3.image(result_hp, caption="Sharpened", use_container_width=True)
