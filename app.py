"""
app.py
------
Main Streamlit application for the AI Audio Noise Reduction System.

Layout:
    ┌─────────────────────────────────┐
    │  🎙️ Header + intro              │
    ├─────────────────────────────────┤
    │  Sidebar: Settings              │
    ├─────────────────────────────────┤
    │  Upload panel                   │
    │  ↓ Original audio player       │
    │  ↓ Waveform preview            │
    │  [Clean Audio] button           │
    │  ↓ Progress bar                │
    │  ↓ Cleaned audio player        │
    │  ↓ Waveform comparison         │
    │  ↓ Spectrogram comparison      │
    │  ↓ AI Report card              │
    │  ↓ Download button             │
    └─────────────────────────────────┘

Run:
    streamlit run app.py
"""

import io
import logging
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — required for Streamlit

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ── Project imports ───────────────────────────────────────────────────────────
from pipeline.full_pipeline import run_pipeline, AudioReport
from utils.audio_utils import load_audio, save_upload
from utils.visualize import (
    plot_noise_profile,
    plot_spectrogram_comparison,
    plot_waveform_comparison,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s — %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")


# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Audio Noise Reducer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — minimal dark-style overrides
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Main background */
    .stApp { background-color: #1E1E2E; color: #CDD6F4; }

    /* Metric cards */
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; color: #89B4FA; }
    [data-testid="stMetricLabel"] { color: #A6ADC8 !important; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #89B4FA 0%, #74C7EC 100%);
        color: #1E1E2E;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        padding: 0.55rem 2rem;
        font-size: 1rem;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Section headers */
    h2, h3 { color: #CBA6F7 !important; }

    /* Info boxes */
    .st-alert { border-radius: 10px; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #181825;
    }

    /* Upload area */
    [data-testid="stFileUploader"] { background-color: #24243E; border-radius: 10px; padding: 1rem; }

    /* Divider */
    hr { border-color: #313244; }

    /* Report card */
    .report-card {
        background: #24243E;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        border: 1px solid #45475A;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style='text-align:center; padding: 1.5rem 0 0.5rem'>
        <h1 style='color:#CBA6F7; font-size:2.5rem; margin-bottom:0'>
            🎙️ Audio Studio Cleaner
        </h1>
        <p style='color:#A6ADC8; font-size:1.05rem; margin-top:0.4rem'>
            Professional Grade Noise Reduction & Audio Enhancement &nbsp;|&nbsp;
            Runs 100% locally
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — pipeline settings
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Studio Settings")
    st.markdown("---")

    st.markdown("### 🧹 Noise Reduction")
    use_noisereduce = st.toggle("Enable Noise Reduction", value=True,
                                help="Spectral-gating to remove background hiss/hum.")
    nr_strength = st.slider(
        "Reduction Strength",
        min_value=0.1, max_value=1.0, value=0.75, step=0.05,
        help="Higher = stronger (risk of warbling artefacts)."
    )

    st.markdown("### 🎚️ Equalizer (EQ)")
    low_gain = st.slider("Bass (Low)", -12.0, 12.0, 0.0, 0.5, help="Low frequency shelf (250Hz)")
    mid_gain = st.slider("Mids", -12.0, 12.0, 0.0, 0.5, help="Mid frequency peak (1000Hz)")
    high_gain = st.slider("Treble (High)", -12.0, 12.0, 0.0, 0.5, help="High frequency shelf (4000Hz)")

    st.markdown("### 📉 Compressor")
    comp_ratio = st.select_slider(
        "Ratio",
        options=[1.0, 2.0, 4.0, 8.0, 20.0],
        value=1.0,
        help="Compression strength. 1.0 = Off. 4.0 = Standard. 20.0 = Limiter."
    )
    comp_threshold = st.slider("Threshold (dB)", -60, 0, -20, 1, help="Level where compression starts.")

    st.markdown("### 🔊 Output")
    output_gain = st.slider("Output Gain (dB)", -20.0, 20.0, 0.0, 0.5)

    st.markdown("---")
    show_spectrogram = st.toggle("Show Spectrogram", value=True)
    show_noise_profile = st.toggle("Show Noise Profile", value=True)

    st.markdown("---")
    st.markdown(
        "<small style='color:#585B70'>All processing happens locally.<br>"
        "No audio is sent to any server.</small>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper — fig → bytes (for st.image if needed)
# ─────────────────────────────────────────────────────────────────────────────
def _fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# Helper — render the AI report
# ─────────────────────────────────────────────────────────────────────────────
def _render_report(report: AudioReport) -> None:
    st.markdown("## 📋 AI Analysis Report")

    # Top-level metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⏱ Duration",        f"{report.duration_s:.1f} s")
    col2.metric("📉 Noise Level",     report.noise_level_label)
    col3.metric("🔊 Noise Type",      report.noise_type_label.split("(")[0].strip())
    col4.metric("✂️ Noise Reduction", f"{report.noise_reduction_db:.1f} dB")

    st.markdown("---")

    # Detailed metrics expander
    with st.expander("📐 Detailed Signal Metrics", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Original RMS",   f"{report.original_rms_db:.1f} dB")
        c1.metric("Cleaned RMS",    f"{report.cleaned_rms_db:.1f} dB")
        c2.metric("Estimated SNR",  f"{report.original_snr_db:.1f} dB")
        c2.metric("Crest Factor",   f"{report.original_crest_factor_db:.1f} dB")
        c3.metric("Spectral Flatness", f"{report.original_spectral_flatness:.4f}")
        c3.metric("Processing Time",   f"{report.processing_time_s:.2f} s")

    # Suggestions
    st.markdown("### 💡 AI Suggestions")
    for suggestion in report.suggestions:
        st.info(suggestion)


# ─────────────────────────────────────────────────────────────────────────────
# Main panel — file upload
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 📁 Upload Audio or Video File")

uploaded_file = st.file_uploader(
    label="Drop your file here",
    type=["wav", "mp3", "flac", "ogg", "m4a", "aac", "aiff", "wma", "mpeg", "mp4", "mkv", "mov", "avi", "webm"],
    help="Supported: WAV, MP3, FLAC, OGG, M4A, AAC, AIFF, WMA, MPEG. "
         "Video support: MP4, MKV, MOV, AVI, WEBM (Audio will be extracted).",
)

if uploaded_file is None:
    st.markdown(
        """
        <div style='text-align:center; padding:2.5rem; color:#585B70;'>
            <p style='font-size:3rem'>🎬</p>
            <p>Upload an audio or video file to get started.<br>
            Supported: WAV · MP3 · MPEG · MP4 · MKV · MOV · AVI · FLAC</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Save the uploaded file and preview it
# ─────────────────────────────────────────────────────────────────────────────
try:
    input_path = save_upload(uploaded_file)
    logger.info("Uploaded file saved: %s", input_path)
except ValueError as e:
    st.error(f"❌ Upload failed: {e}")
    st.stop()

st.success(f"✅ File uploaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

# ── Original audio player ─────────────────────────────────────────────────────
st.markdown("### 🔊 Original Audio")
st.audio(str(input_path), format="audio/wav")

# ── Waveform preview of original ─────────────────────────────────────────────
with st.spinner("Loading waveform preview …"):
    try:
        orig_audio, orig_sr = load_audio(input_path)
        duration_orig = len(orig_audio) / orig_sr

        fig_orig, ax = plt.subplots(figsize=(10, 2))
        t = np.linspace(0, duration_orig, len(orig_audio))
        ax.plot(t, orig_audio, color="#4A90D9", linewidth=0.5, alpha=0.85)
        ax.set_facecolor("#1E1E2E")
        ax.tick_params(colors="#CDD6F4", labelsize=7)
        ax.set_xlabel("Time (s)", color="#A6ADC8", fontsize=8)
        ax.set_ylabel("Amplitude", color="#A6ADC8", fontsize=8)
        ax.set_title(
            f"Original Waveform — {duration_orig:.1f} s | {orig_sr} Hz",
            color="#CDD6F4", fontsize=9,
        )
        fig_orig.patch.set_facecolor("#1E1E2E")
        fig_orig.tight_layout()
        st.pyplot(fig_orig)
        plt.close(fig_orig)
    except Exception as e:
        st.warning(f"Could not render waveform preview: {e}")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Clean Audio button
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 🧹 Noise Reduction")

clean_btn = st.button(
    "🚀  Clean Audio",
    use_container_width=True,
    type="primary",
)

if not clean_btn:
    st.caption("👆  Click the button above to start the AI denoising pipeline.")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Run the pipeline with a live progress bar
# ─────────────────────────────────────────────────────────────────────────────
progress_bar  = st.progress(0)
status_text   = st.empty()

def _update_progress(fraction: float, message: str) -> None:
    progress_bar.progress(min(int(fraction * 100), 100))
    status_text.markdown(
        f"<small style='color:#A6ADC8'>{message}</small>",
        unsafe_allow_html=True,
    )

cleaned_audio: np.ndarray | None = None
cleaned_sr:    int                = 48_000
output_path:   Path | None        = None
report:        AudioReport | None = None

try:
    with st.spinner("Running audio enhancement pipeline …"):
        cleaned_audio, cleaned_sr, output_path, report = run_pipeline(
            input_path=input_path,
            progress_callback=_update_progress,
            noisereduce_strength=float(nr_strength),
            skip_noisereduce=not use_noisereduce,
            low_gain=low_gain,
            mid_gain=mid_gain,
            high_gain=high_gain,
            comp_threshold=float(comp_threshold),
            comp_ratio=float(comp_ratio),
            output_gain=output_gain,
        )

    progress_bar.progress(100)
    status_text.empty()
    st.balloons()
    st.success("🎉  Audio cleaned successfully!")

except FileNotFoundError as e:
    st.error(
        f"❌ **File not found**: {e}\n\n"
        "Make sure FFmpeg is installed and added to your PATH.\n"
        "See the README for installation instructions."
    )
    logger.exception("FileNotFoundError in pipeline")
    st.stop()
except RuntimeError as e:
    st.error(f"❌ **Pipeline error**: {e}")
    logger.exception("RuntimeError in pipeline")
    st.stop()
except Exception as e:
    st.error(
        f"❌ **Unexpected error**: {e}\n\n"
        "```\n" + traceback.format_exc() + "\n```"
    )
    logger.exception("Unexpected error in pipeline")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Results section
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("## ✅ Results")

# ── Cleaned audio player ──────────────────────────────────────────────────────
st.markdown("### 🎧 Cleaned Audio")
st.audio(str(output_path), format="audio/wav")

# ── Side-by-side players ──────────────────────────────────────────────────────
st.markdown("### 🆚 Side-by-Side Comparison")
col_orig, col_cln = st.columns(2)
with col_orig:
    st.markdown("**🔴 Original (Noisy)**")
    st.audio(str(input_path), format="audio/wav")
with col_cln:
    st.markdown("**🟢 Cleaned**")
    st.audio(str(output_path), format="audio/wav")

# ── Download button ───────────────────────────────────────────────────────────
st.markdown("### 💾 Download Cleaned Audio")
with open(output_path, "rb") as f:
    audio_bytes = f.read()
st.download_button(
    label="⬇️  Download Cleaned WAV",
    data=audio_bytes,
    file_name=output_path.name,
    mime="audio/wav",
    use_container_width=True,
)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 📊 Audio Visualisations")

# Waveform comparison
with st.spinner("Rendering waveform comparison …"):
    try:
        fig_wave = plot_waveform_comparison(orig_audio, cleaned_audio, cleaned_sr)
        st.pyplot(fig_wave)
        plt.close(fig_wave)
    except Exception as e:
        st.warning(f"Waveform plot failed: {e}")

# Spectrogram comparison
if show_spectrogram:
    with st.spinner("Rendering spectrogram comparison …"):
        try:
            fig_spec = plot_spectrogram_comparison(orig_audio, cleaned_audio, cleaned_sr)
            st.pyplot(fig_spec)
            plt.close(fig_spec)
        except Exception as e:
            st.warning(f"Spectrogram plot failed: {e}")

# Noise profile (band energy chart)
if show_noise_profile:
    with st.spinner("Rendering noise profile chart …"):
        try:
            fig_noise = plot_noise_profile(orig_audio, cleaned_audio, cleaned_sr)
            st.pyplot(fig_noise)
            plt.close(fig_noise)
        except Exception as e:
            st.warning(f"Noise profile chart failed: {e}")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# AI Report
# ─────────────────────────────────────────────────────────────────────────────
if report is not None:
    _render_report(report)

st.divider()
st.markdown(
    "<small style='color:#585B70; display:block; text-align:center'>"
    "Audio Studio Cleaner · Built with Noisereduce + Streamlit · "
    "Runs 100% locally</small>",
    unsafe_allow_html=True,
)