"""
visualize.py
------------
All audio visualisation helpers used by the Streamlit UI.

Produces:
  - Side-by-side waveform comparison (original vs cleaned)
  - Mel spectrogram comparison
  - Noise profile bar chart

Why matplotlib + librosa.display?
    → Librosa ships a dedicated `display` module optimised for audio plots,
      including spectrogram tick formatting and frequency axis labelling.
      Matplotlib provides the rendering backend.
"""

import logging

import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

logger = logging.getLogger("visualize")

# ── Style config ─────────────────────────────────────────────────────────────
PALETTE = {
    "original": "#4A90D9",   # calm blue  – original (noisy)
    "cleaned":  "#27AE60",   # green      – cleaned output
    "noise":    "#E74C3C",   # red        – noise estimate
    "bg":       "#1E1E2E",   # dark background
    "text":     "#CDD6F4",   # light text
    "grid":     "#313244",   # subtle grid lines
}


def _apply_dark_style(fig: plt.Figure) -> None:
    """Apply a consistent dark theme to the figure."""
    fig.patch.set_facecolor(PALETTE["bg"])
    for ax in fig.get_axes():
        ax.set_facecolor(PALETTE["bg"])
        ax.tick_params(colors=PALETTE["text"], labelsize=8)
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        ax.title.set_color(PALETTE["text"])
        for spine in ax.spines.values():
            spine.set_color(PALETTE["grid"])
        ax.grid(color=PALETTE["grid"], linewidth=0.5, linestyle="--", alpha=0.6)


def plot_waveform_comparison(
    original: np.ndarray,
    cleaned: np.ndarray,
    sr: int,
) -> plt.Figure:
    """
    Render a 2-row waveform comparison plot.

    Args:
        original : raw (noisy) audio array
        cleaned  : denoised audio array
        sr       : sample rate (Hz)

    Returns:
        matplotlib Figure – pass directly to st.pyplot()
    """
    duration_orig = len(original) / sr
    duration_cln  = len(cleaned)  / sr

    t_orig = np.linspace(0, duration_orig, len(original))
    t_cln  = np.linspace(0, duration_cln,  len(cleaned))

    fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=False)

    # ── Top: original ────────────────────────────────────────────────────────
    axes[0].plot(t_orig, original, color=PALETTE["original"], linewidth=0.5, alpha=0.9)
    axes[0].set_title("🔴  Original (Noisy)", fontsize=10, pad=6)
    axes[0].set_ylabel("Amplitude", fontsize=8)
    axes[0].set_ylim(-1.05, 1.05)

    # ── Bottom: cleaned ──────────────────────────────────────────────────────
    axes[1].plot(t_cln, cleaned, color=PALETTE["cleaned"], linewidth=0.5, alpha=0.9)
    axes[1].set_title("🟢  Cleaned (Denoised)", fontsize=10, pad=6)
    axes[1].set_ylabel("Amplitude", fontsize=8)
    axes[1].set_xlabel("Time (s)", fontsize=8)
    axes[1].set_ylim(-1.05, 1.05)

    fig.suptitle("Waveform Comparison", fontsize=12, color=PALETTE["text"], y=1.01)
    fig.tight_layout()
    _apply_dark_style(fig)
    logger.info("Waveform comparison plot generated.")
    return fig


def plot_spectrogram_comparison(
    original: np.ndarray,
    cleaned: np.ndarray,
    sr: int,
) -> plt.Figure:
    """
    Mel spectrogram side-by-side comparison.

    Mel spectrograms map frequency content onto a perceptual scale,
    making it easy to see what noise bands were removed.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    def _draw_mel(ax, audio, title, cmap):
        # Compute mel spectrogram in dB
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
        S_db = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(
            S_db, sr=sr, x_axis="time", y_axis="mel",
            fmax=8000, ax=ax, cmap=cmap
        )
        ax.set_title(title, fontsize=10)
        fig.colorbar(img, ax=ax, format="%+2.0f dB", shrink=0.8)

    _draw_mel(axes[0], original, "🔴  Original Spectrogram", "magma")
    _draw_mel(axes[1], cleaned,  "🟢  Cleaned Spectrogram",  "viridis")

    fig.suptitle("Mel Spectrogram Comparison", fontsize=12, color=PALETTE["text"])
    fig.tight_layout()
    _apply_dark_style(fig)
    logger.info("Spectrogram comparison plot generated.")
    return fig


def plot_noise_profile(
    original: np.ndarray,
    cleaned: np.ndarray,
    sr: int,
) -> plt.Figure:
    """
    Bar chart showing estimated noise reduction across frequency bands.
    Computed as the power difference (original − cleaned) per band.
    """
    n_bands   = 8
    band_size = len(original) // n_bands

    band_labels = [f"B{i+1}" for i in range(n_bands)]
    orig_rms = []
    cln_rms  = []

    for i in range(n_bands):
        seg_orig = original[i * band_size : (i + 1) * band_size]
        seg_cln  = cleaned[i * band_size  : (i + 1) * band_size] if i * band_size < len(cleaned) else np.zeros(band_size)
        orig_rms.append(float(np.sqrt(np.mean(seg_orig ** 2))))
        cln_rms.append(float(np.sqrt(np.mean(seg_cln ** 2))))

    x = np.arange(n_bands)
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.bar(x - width / 2, orig_rms, width, label="Original",  color=PALETTE["original"], alpha=0.85)
    ax.bar(x + width / 2, cln_rms,  width, label="Cleaned",   color=PALETTE["cleaned"],  alpha=0.85)

    ax.set_xlabel("Audio Band", fontsize=8)
    ax.set_ylabel("RMS Energy", fontsize=8)
    ax.set_title("Energy Per Band — Before vs After Denoising", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels, fontsize=8)
    ax.legend(fontsize=8, facecolor=PALETTE["bg"], labelcolor=PALETTE["text"])

    fig.tight_layout()
    _apply_dark_style(fig)
    logger.info("Noise profile chart generated.")
    return fig