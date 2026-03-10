import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import heapq
from collections import Counter


# ── Huffman ───────────────────────────────────────────────────────────────────
class _HuffNode:
    def __init__(self, symbol, freq):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def _build_huffman(data: bytes):
    freq = Counter(data)
    heap = [_HuffNode(sym, f) for sym, f in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        l, r = heapq.heappop(heap), heapq.heappop(heap)
        merged = _HuffNode(None, l.freq + r.freq)
        merged.left, merged.right = l, r
        heapq.heappush(heap, merged)

    codes = {}

    def _gen(node, code=""):
        if node is None:
            return
        if node.symbol is not None:
            codes[node.symbol] = code if code else "0"
            return
        _gen(node.left, code + "0")
        _gen(node.right, code + "1")

    _gen(heap[0])
    return codes


def _huffman_stats(data: bytes):
    codes = _build_huffman(data)
    freq = Counter(data)
    total_bits = sum(freq[s] * len(c) for s, c in codes.items())
    original_bits = len(data) * 8
    ratio = original_bits / total_bits if total_bits else 1
    avg_len = total_bits / len(data) if data else 8
    return {
        "original_bytes": len(data),
        "encoded_bits": total_bits,
        "encoded_bytes_approx": total_bits / 8,
        "ratio": ratio,
        "avg_code_len": avg_len,
        "unique_symbols": len(codes),
        "codes": codes,
    }


# ── Main Panel ────────────────────────────────────────────────────────────────
def compression_panel(img: Image.Image):
    tabs = st.tabs(["JPEG Quality", "Huffman Encoding"])

    # ── JPEG ──────────────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("**JPEG Compression** — lossy DCT-based compression; lower quality = higher compression ratio")

        col_ctrl, _ = st.columns([1, 2])
        with col_ctrl:
            quality = st.slider("JPEG Quality", 1, 95, 75, key="jpeg_q")

        buf_orig = io.BytesIO()
        img.save(buf_orig, format="PNG")
        orig_size = buf_orig.tell()

        buf_comp = io.BytesIO()
        img.save(buf_comp, format="JPEG", quality=quality)
        comp_size = buf_comp.tell()
        buf_comp.seek(0)
        compressed_img = Image.open(buf_comp)
        compressed_img.load()

        arr_orig = np.array(img).astype(np.float64)
        arr_comp = np.array(compressed_img.convert("RGB")).astype(np.float64)

        # Resize if needed (JPEG may slightly alter dims in edge cases)
        if arr_orig.shape != arr_comp.shape:
            arr_comp = arr_comp[:arr_orig.shape[0], :arr_orig.shape[1]]

        mse = np.mean((arr_orig - arr_comp) ** 2)
        psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float("inf")

        col1, col2 = st.columns(2)
        col1.image(img, caption=f"Original — {orig_size / 1024:.1f} KB (PNG)", use_container_width=True)
        col2.image(compressed_img, caption=f"JPEG q={quality} — {comp_size / 1024:.1f} KB", use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Original Size", f"{orig_size / 1024:.1f} KB")
        m2.metric("Compressed Size", f"{comp_size / 1024:.1f} KB")
        m3.metric("Ratio", f"{orig_size / comp_size:.2f}×")
        m4.metric("PSNR", f"{psnr:.1f} dB")

        st.markdown(f"<span style='font-size:0.73rem;color:#555'>PSNR > 40 dB: visually lossless · 30–40 dB: acceptable · < 30 dB: noticeable degradation</span>", unsafe_allow_html=True)

        # Difference map
        if st.checkbox("Show difference map", key="jpeg_diff"):
            diff = np.abs(arr_orig - arr_comp).mean(axis=2)
            diff_norm = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
            diff_colored = cv2.applyColorMap(diff_norm, cv2.COLORMAP_INFERNO)
            st.image(cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB), caption="Pixel difference (brighter = more loss)", use_container_width=True)

    # ── Huffman ───────────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("**Huffman Encoding** — lossless compression; assigns shorter codes to more frequent pixel values")

        channel = st.radio("Encode channel", ["Grayscale", "Red", "Green", "Blue"], horizontal=True)

        arr = np.array(img)
        if channel == "Grayscale":
            data = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).tobytes()
        elif channel == "Red":
            data = arr[:, :, 0].tobytes()
        elif channel == "Green":
            data = arr[:, :, 1].tobytes()
        else:
            data = arr[:, :, 2].tobytes()

        stats = _huffman_stats(data)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Original", f"{stats['original_bytes'] / 1024:.1f} KB")
        m2.metric("Encoded (approx)", f"{stats['encoded_bytes_approx'] / 1024:.1f} KB")
        m3.metric("Compression Ratio", f"{stats['ratio']:.3f}×")
        m4.metric("Avg Code Length", f"{stats['avg_code_len']:.2f} bits")

        st.markdown(f"<span style='font-size:0.73rem;color:#555'>Fixed 8-bit/pixel → avg {stats['avg_code_len']:.2f} bits/pixel · Unique symbols: {stats['unique_symbols']}</span>", unsafe_allow_html=True)

        if st.checkbox("Show code table (top 20 by frequency)", key="huff_table"):
            freq = Counter(data)
            top = sorted(freq.items(), key=lambda x: -x[1])[:20]
            rows = []
            for sym, f in top:
                code = stats["codes"].get(sym, "")
                rows.append(f"| `{sym:3d}` | {f:,} | {f/len(data)*100:.2f}% | `{code}` | {len(code)} |")

            table_md = "| Value | Frequency | Prob | Huffman Code | Bits |\n|---|---|---|---|---|\n" + "\n".join(rows)
            st.markdown(table_md)
