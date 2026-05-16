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

# ── Pipeline defaults ─────────────────────────────────────────────────────────
NR_STRENGTH_DEFAULT  = 0.75
LOW_GAIN_DEFAULT     = 0.0
MID_GAIN_DEFAULT     = 0.0
HIGH_GAIN_DEFAULT    = 0.0
COMP_RATIO_DEFAULT   = 1.0
COMP_THRESHOLD_DEFAULT = -20
OUTPUT_GAIN_DEFAULT  = 0.0


# ── Theme Configuration ───────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

PALETTES = {
    "dark": {
        "bg": "#0b0b14",
        "card_bg": "rgba(23, 23, 37, 0.7)",
        "text": "#cdd6f4",
        "text_dim": "#a6adc8",
        "accent": "#89b4fa",
        "accent2": "#cba6f7",
        "border": "rgba(255, 255, 255, 0.08)",
        "plot_bg": "#1e1e2e"
    },
    "light": {
        "bg": "#f4f5f7",
        "card_bg": "rgba(255, 255, 255, 0.85)",
        "text": "#4c4f69",
        "text_dim": "#5c5f77",
        "accent": "#1e66f5",
        "accent2": "#8839ef",
        "border": "rgba(0, 0, 0, 0.08)",
        "plot_bg": "#ffffff"
    }
}
cp = PALETTES[st.session_state.theme]

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Audio Studio Cleaner",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — minimal dark-style overrides
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@400;600&display=swap');

    :root {{
        --bg-color: {cp['bg']};
        --card-bg: {cp['card_bg']};
        --accent-color: {cp['accent']};
        --accent-color2: {cp['accent2']};
        --text-main: {cp['text']};
        --text-dim: {cp['text_dim']};
        --border-color: {cp['border']};
    }}

    .stApp {{
        background-color: var(--bg-color);
        font-family: 'Outfit', sans-serif;
    }}

    /* Ensure all text respects theme */
    .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp span, .stApp label, .stApp div {{
        color: var(--text-main);
    }}

    /* Glassmorphism Card */
    .glass-card {{
        background: var(--card-bg);
        backdrop-filter: blur(16px) saturate(180%);
        -webkit-backdrop-filter: blur(16px) saturate(180%);
        border: 1px solid var(--border-color);
        border-radius: 24px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
    }}

    /* Animated Gradient Title */
    .title-gradient {{
        background: linear-gradient(to right, var(--accent-color2), var(--accent-color), #94e2d5, var(--accent-color), var(--accent-color2));
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 5s linear infinite;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
    }}

    @keyframes shine {{
        to {{ background-position: 200% center; }}
    }}

    /* Premium Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, var(--accent-color) 0%, var(--accent-color2) 100%) !important;
        color: {(cp['bg'] if st.session_state.theme == 'dark' else '#ffffff')} !important;
        border: none !important;
        border-radius: 16px !important;
        font-weight: 700 !important;
        padding: 0.8rem 2.5rem !important;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1) !important;
        width: 100%;
    }}

    .stButton > button:hover {{
        transform: translateY(-4px) scale(1.02) !important;
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.2) !important;
    }}

    /* Sliders & Toggles */
    .stSlider [data-baseweb="slider"] {{
        margin-bottom: 10px;
    }}

    /* Metrics */
    [data-testid="stMetric"] {{
        background: rgba(128, 128, 128, 0.05);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 1rem !important;
    }}

    [data-testid="stMetricValue"] {{
        color: var(--accent-color) !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        background: rgba(128, 128, 128, 0.05);
        border-radius: 16px;
        padding: 6px;
        gap: 8px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 12px;
        color: var(--text-dim);
        transition: all 0.3s ease;
        padding: 10px 20px;
    }}

    .stTabs [aria-selected="true"] {{
        background: var(--accent-color) !important;
        color: {(cp['bg'] if st.session_state.theme == 'dark' else '#ffffff')} !important;
        font-weight: 600;
    }}

    /* Expander */
    .stExpander {{
        background: rgba(128, 128, 128, 0.03) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 16px !important;
    }}

    /* Progress Bar */
    .stProgress > div > div > div > div {{
        background-image: linear-gradient(to right, var(--accent-color), var(--accent-color2)) !important;
    }}

    /* File Uploader */
    [data-testid="stFileUploader"] {{
        background-color: rgba(128, 128, 128, 0.05);
        border: 2px dashed var(--border-color);
        border-radius: 16px;
        padding: 1rem;
    }}

    /* Hide default sidebar handle */
    [data-testid="collapsedControl"] {{
        display: none;
    }}

    /* Footer */
    .footer {{
        text-align: center;
        padding: 3rem 0;
        color: var(--text-dim);
        font-size: 0.9rem;
        border-top: 1px solid var(--border-color);
        margin-top: 4rem;
    }}

    .suggestion-card {{
        background: rgba(128, 128, 128, 0.05);
        border-left: 4px solid var(--accent-color);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
h_col1, h_col2 = st.columns([8, 2])
with h_col1:
    st.markdown(
        """
        <div style='text-align:left; padding: 1rem 0'>
            <h1 class='title-gradient'>🎙️ Audio Studio Cleaner</h1>
            <p style='color:var(--text-dim); font-size:1.1rem; margin-top:0.3rem; font-weight:300'>
                Professional Grade Noise Reduction &nbsp;|&nbsp; 
                <span style='color:var(--accent-color)'>100% Local AI</span>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with h_col2:
    st.markdown("<div style='padding-top: 2rem'></div>", unsafe_allow_html=True)
    theme_toggle = st.toggle(
        "🌞 Light Mode" if st.session_state.theme == "dark" else "🌙 Dark Mode",
        value=st.session_state.theme == "light",
        key="theme_switcher"
    )
    if theme_toggle != (st.session_state.theme == "light"):
        st.session_state.theme = "light" if theme_toggle else "dark"
        st.rerun()

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — pipeline settings
# ─────────────────────────────────────────────────────────────────────────────
# Sidebar removed - settings moved inline below


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
    with col1:
        st.metric("⏱ Duration", f"{report.duration_s:.1f} s")
    with col2:
        st.metric("📉 Noise Level", report.noise_level_label)
    with col3:
        st.metric("🔊 Noise Type", report.noise_type_label.split("(")[0].strip())
    with col4:
        st.metric("✂️ Reduction", f"{report.noise_reduction_db:.1f} dB")

    st.markdown("<br>", unsafe_allow_html=True)

    # Detailed metrics expander
    with st.expander("📐 Detailed Signal Metrics", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Original RMS", f"{report.original_rms_db:.1f} dB")
        c1.metric("Cleaned RMS", f"{report.cleaned_rms_db:.1f} dB")
        c2.metric("Estimated SNR", f"{report.original_snr_db:.1f} dB")
        c2.metric("Crest Factor", f"{report.original_crest_factor_db:.1f} dB")
        c3.metric("Spectral Flatness", f"{report.original_spectral_flatness:.4f}")
        c3.metric("Processing Time", f"{report.processing_time_s:.2f} s")

    # Suggestions
    st.markdown("### 💡 AI Suggestions")
    for suggestion in report.suggestions:
        st.markdown(f"<div class='suggestion-card'>{suggestion}</div>", unsafe_allow_html=True)


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
            <p style='font-size:3.5rem; margin-bottom: 1rem'>🎬</p>
            <p style='font-size:1.1rem; color: #a6adc8'>Upload an audio or video file to begin processing.<br>
            <span style='font-size:0.9rem; color: #6c7086'>Supported: WAV · MP3 · MPEG · MP4 · MKV · MOV · AVI · FLAC</span></p>
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
        ax.plot(t, orig_audio, color=cp["accent"], linewidth=0.5, alpha=0.85)
        ax.set_facecolor(cp["bg"])
        ax.tick_params(colors=cp["text"], labelsize=7)
        ax.set_xlabel("Time (s)", color=cp["text_dim"], fontsize=8)
        ax.set_ylabel("Amplitude", color=cp["text_dim"], fontsize=8)
        ax.set_title(
            f"Original Waveform — {duration_orig:.1f} s | {orig_sr} Hz",
            color=cp["text"], fontsize=9,
        )
        fig_orig.patch.set_facecolor(cp["bg"])
        fig_orig.tight_layout()
        st.pyplot(fig_orig)
        plt.close(fig_orig)
    except Exception as e:
        st.warning(f"Could not render waveform preview: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Clean Audio button
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 🚀 Enhance Audio")

# Primary Action Button
clean_btn = st.button(
    "🚀  Clean Audio",
    use_container_width=True,
    type="primary",
)

st.markdown(
    "<p style='text-align:center; color:#a6adc8; font-size:0.9rem; margin-top:-1rem; margin-bottom:1.5rem'>"
    "✨ Smart defaults active &nbsp;•&nbsp; One-click professional results</p>", 
    unsafe_allow_html=True
)

# Advanced Settings - Collapsed by default
with st.expander("⚙️ Advanced Studio Settings (Optional)", expanded=False):
    st.markdown("<p style='color:#585B70; font-size:0.85rem; margin-bottom:1rem'>Tweak these settings only if you need custom audio characteristics.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🧹 Noise Reduction")
        use_noisereduce = st.toggle("Enable Denoising", value=True, help="Remove background hiss/hum.")
        nr_strength = st.slider("Reduction Strength", 0.1, 1.0, NR_STRENGTH_DEFAULT, 0.05, help="Higher = stronger removal.")
        
    with col2:
        st.markdown("### 🎚️ EQ & Compressor")
        low_gain = st.slider("Bass", -12.0, 12.0, LOW_GAIN_DEFAULT, 0.5)
        mid_gain = st.slider("Mids", -12.0, 12.0, MID_GAIN_DEFAULT, 0.5)
        high_gain = st.slider("Treble", -12.0, 12.0, HIGH_GAIN_DEFAULT, 0.5)
        comp_ratio = st.select_slider("Comp Ratio", options=[1.0, 2.0, 4.0, 8.0, 20.0], value=COMP_RATIO_DEFAULT)
        
    with col3:
        st.markdown("### 🔊 Output & Display")
        comp_threshold = st.slider("Threshold (dB)", -60, 0, COMP_THRESHOLD_DEFAULT, 1)
        output_gain = st.slider("Output Gain (dB)", -20.0, 20.0, OUTPUT_GAIN_DEFAULT, 0.5)
        show_spectrogram = st.toggle("Show Spectrogram", value=True)
        show_noise_profile = st.toggle("Show Noise Profile", value=True)

    # Simple logic to show if settings are customized
    custom_count = 0
    if nr_strength != NR_STRENGTH_DEFAULT: custom_count += 1
    if low_gain != LOW_GAIN_DEFAULT: custom_count += 1
    if mid_gain != MID_GAIN_DEFAULT: custom_count += 1
    if high_gain != HIGH_GAIN_DEFAULT: custom_count += 1
    if comp_ratio != COMP_RATIO_DEFAULT: custom_count += 1
    if comp_threshold != COMP_THRESHOLD_DEFAULT: custom_count += 1
    if output_gain != OUTPUT_GAIN_DEFAULT: custom_count += 1
    
    if custom_count > 0:
        st.markdown(f"<div style='text-align:right; color:#fab387; font-size:0.8rem; font-weight:600'>⚡ {custom_count} custom settings active</div>", unsafe_allow_html=True)

if not clean_btn:
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Run the pipeline with a live progress bar
# ─────────────────────────────────────────────────────────────────────────────
progress_bar  = st.progress(0)
status_text   = st.empty()

def _update_progress(fraction: float, message: str) -> None:
    progress_bar.progress(min(int(fraction * 100), 100))
    status_text.markdown(
        f"<small style='color:var(--text-dim)'>{message}</small>",
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
st.markdown("<br>", unsafe_allow_html=True)
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

tab_wave, tab_spec, tab_noise = st.tabs(["📈 Waveform", "🌈 Spectrogram", "📉 Noise Profile"])

with tab_wave:
    with st.spinner("Rendering waveform comparison …"):
        try:
            fig_wave = plot_waveform_comparison(orig_audio, cleaned_audio, cleaned_sr, theme=st.session_state.theme)
            st.pyplot(fig_wave)
            plt.close(fig_wave)
        except Exception as e:
            st.warning(f"Waveform plot failed: {e}")

with tab_spec:
    if show_spectrogram:
        with st.spinner("Rendering spectrogram comparison …"):
            try:
                fig_spec = plot_spectrogram_comparison(orig_audio, cleaned_audio, cleaned_sr, theme=st.session_state.theme)
                st.pyplot(fig_spec)
                plt.close(fig_spec)
            except Exception as e:
                st.warning(f"Spectrogram plot failed: {e}")
    else:
        st.info("Spectrogram display is disabled in settings.")

with tab_noise:
    if show_noise_profile:
        with st.spinner("Rendering noise profile chart …"):
            try:
                fig_noise = plot_noise_profile(orig_audio, cleaned_audio, cleaned_sr, theme=st.session_state.theme)
                st.pyplot(fig_noise)
                plt.close(fig_noise)
            except Exception as e:
                st.warning(f"Noise profile chart failed: {e}")
    else:
        st.info("Noise profile display is disabled in settings.")

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# AI Report
# ─────────────────────────────────────────────────────────────────────────────
if report is not None:
    _render_report(report)

st.markdown(
    """
    <div class='footer'>
        <p>🎙️ Audio Studio Cleaner &nbsp;|&nbsp; Built with AI + Noisereduce + Streamlit</p>
        <p style='font-size:0.8rem; margin-top:0.5rem; opacity:0.6'>
            Developed for Final Year Project &nbsp;•&nbsp; 2026 &nbsp;•&nbsp; 100% Private & Secure
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
