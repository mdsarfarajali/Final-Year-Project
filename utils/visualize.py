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
PALETTES = {
    "dark": {
        "original": "#89b4fa",   # blue
        "cleaned":  "#a6e3a1",   # green
        "noise":    "#f38ba8",   # red
        "bg":       "#0b0b14",   # deep dark
        "text":     "#cdd6f4",   # light text
        "grid":     "#313244",   # subtle grid
    },
    "light": {
        "original": "#1e66f5",   # deep blue
        "cleaned":  "#40a02b",   # rich green
        "noise":    "#d20f39",   # vivid red
        "bg":       "#f4f5f7",   # soft gray background
        "text":     "#4c4f69",   # dark text
        "grid":     "#ccd0da",   # soft grid
    }
}

def _apply_theme_style(fig: plt.Figure, theme: str = "dark") -> None:
    """Apply a consistent theme (light/dark) to the figure."""
    palette = PALETTES.get(theme, PALETTES["dark"])
    fig.patch.set_facecolor(palette["bg"])
    for ax in fig.get_axes():
        ax.set_facecolor(palette["bg"])
        ax.tick_params(colors=palette["text"], labelsize=8)
        ax.xaxis.label.set_color(palette["text"])
        ax.yaxis.label.set_color(palette["text"])
        if hasattr(ax, "title"):
            ax.title.set_color(palette["text"])
        for spine in ax.spines.values():
            spine.set_color(palette["grid"])
        ax.grid(color=palette["grid"], linewidth=0.5, linestyle="--", alpha=0.6)


def plot_waveform_comparison(
    original: np.ndarray,
    cleaned: np.ndarray,
    sr: int,
    theme: str = "dark"
) -> plt.Figure:
    """
    Render a 2-row waveform comparison plot.
    """
    palette = PALETTES.get(theme, PALETTES["dark"])
    duration_orig = len(original) / sr
    duration_cln  = len(cleaned)  / sr

    t_orig = np.linspace(0, duration_orig, len(original))
    t_cln  = np.linspace(0, duration_cln,  len(cleaned))

    fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=False)

    # ── Top: original ────────────────────────────────────────────────────────
    axes[0].plot(t_orig, original, color=palette["original"], linewidth=0.5, alpha=0.9)
    axes[0].set_title("🔴  Original (Noisy)", fontsize=10, pad=6)
    axes[0].set_ylabel("Amplitude", fontsize=8)
    axes[0].set_ylim(-1.05, 1.05)

    # ── Bottom: cleaned ──────────────────────────────────────────────────────
    axes[1].plot(t_cln, cleaned, color=palette["cleaned"], linewidth=0.5, alpha=0.9)
    axes[1].set_title("🟢  Cleaned (Denoised)", fontsize=10, pad=6)
    axes[1].set_ylabel("Amplitude", fontsize=8)
    axes[1].set_xlabel("Time (s)", fontsize=8)
    axes[1].set_ylim(-1.05, 1.05)

    fig.suptitle("Waveform Comparison", fontsize=12, color=palette["text"], y=1.01)
    fig.tight_layout()
    _apply_theme_style(fig, theme)
    logger.info("Waveform comparison plot generated (%s theme).", theme)
    return fig


def plot_spectrogram_comparison(
    original: np.ndarray,
    cleaned: np.ndarray,
    sr: int,
    theme: str = "dark"
) -> plt.Figure:
    """
    Mel spectrogram side-by-side comparison.
    """
    palette = PALETTES.get(theme, PALETTES["dark"])
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
        cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", shrink=0.8)
        cbar.ax.yaxis.set_tick_params(color=palette["text"])
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=palette["text"])

    _draw_mel(axes[0], original, "🔴  Original Spectrogram", "magma")
    _draw_mel(axes[1], cleaned,  "🟢  Cleaned Spectrogram",  "viridis")

    fig.suptitle("Mel Spectrogram Comparison", fontsize=12, color=palette["text"])
    fig.tight_layout()
    _apply_theme_style(fig, theme)
    logger.info("Spectrogram comparison plot generated (%s theme).", theme)
    return fig


def plot_noise_profile(
    original: np.ndarray,
    cleaned: np.ndarray,
    sr: int,
    theme: str = "dark"
) -> plt.Figure:
    """
    Bar chart showing estimated noise reduction across frequency bands.
    """
    palette = PALETTES.get(theme, PALETTES["dark"])
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
    ax.bar(x - width / 2, orig_rms, width, label="Original",  color=palette["original"], alpha=0.85)
    ax.bar(x + width / 2, cln_rms,  width, label="Cleaned",   color=palette["cleaned"],  alpha=0.85)

    ax.set_xlabel("Audio Band", fontsize=8)
    ax.set_ylabel("RMS Energy", fontsize=8)
    ax.set_title("Energy Per Band — Before vs After Denoising", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels, fontsize=8)
    ax.legend(fontsize=8, facecolor=palette["bg"], labelcolor=palette["text"])

    fig.tight_layout()
    _apply_theme_style(fig, theme)
    logger.info("Noise profile chart generated (%s theme).", theme)
    return fig
