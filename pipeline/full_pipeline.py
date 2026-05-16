"""
full_pipeline.py
----------------
Orchestrates the complete audio denoising pipeline and generates
an AI analysis report.

Pipeline stages:
    1. Load audio (librosa, 48 kHz mono)
    2. Apply DeepFilterNet  → primary broadband noise removal
    3. Apply Noisereduce    → secondary stationary-noise smoothing
    4. Normalize output
    5. Save cleaned WAV
    6. Generate AI report (noise metrics + suggestions)

The report is computed from signal statistics:
    - RMS energy    → volume level
    - SNR estimate  → signal-to-noise ratio
    - Spectral flatness → whether noise is tonal or broadband
    - Zero-crossing rate → roughness indicator
    - Crest factor  → dynamic range metric
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import librosa
import numpy as np

from pipeline.enhance import apply_equalizer, apply_compression, apply_gain
from pipeline.smooth import apply_noisereduce
from utils.audio_utils import (
    convert_to_wav,
    generate_filename,
    load_audio,
    normalize_audio,
    save_audio,
)

logger = logging.getLogger("full_pipeline")


# ── Report dataclass ──────────────────────────────────────────────────────────

@dataclass
class AudioReport:
    """Structured container for per-file audio analysis results."""

    # Basic info
    duration_s: float = 0.0
    sample_rate: int = 48_000
    channels: int = 1

    # Noise metrics (original signal)
    original_rms_db: float = 0.0
    original_snr_db: float = 0.0
    original_spectral_flatness: float = 0.0
    original_zcr: float = 0.0
    original_crest_factor_db: float = 0.0

    # Post-processing metrics
    cleaned_rms_db: float = 0.0
    noise_reduction_db: float = 0.0
    noise_level_label: str = "Unknown"     # Low / Medium / High / Very High
    noise_type_label: str = "Unknown"      # Broadband / Tonal / Mixed / Impulsive

    # Processing time
    processing_time_s: float = 0.0

    # suggestions
    suggestions: list[str] = field(default_factory=list)


# ── Metric helpers ────────────────────────────────────────────────────────────

def _rms_db(audio: np.ndarray) -> float:
    """RMS energy converted to dB. Silence guard prevents log(0)."""
    rms = np.sqrt(np.mean(audio ** 2))
    return float(20 * np.log10(max(rms, 1e-9)))


def _estimate_snr(audio: np.ndarray, sr: int) -> float:
    """
    Simple SNR estimate using the first 0.5 s as a noise reference.
    This works when audio starts with ambient noise before speech.
    Returns SNR in dB; clamped to [-10, 60].
    """
    ref_samples = min(int(0.5 * sr), len(audio) // 4)
    if ref_samples < 256:
        return 0.0
    noise_rms = np.sqrt(np.mean(audio[:ref_samples] ** 2))
    signal_rms = np.sqrt(np.mean(audio ** 2))
    if noise_rms < 1e-9:
        return 60.0
    snr = 20 * np.log10(signal_rms / noise_rms)
    return float(np.clip(snr, -10, 60))


def _spectral_flatness(audio: np.ndarray) -> float:
    """
    Spectral flatness (Wiener entropy) ∈ [0, 1].
    0 → tonal (pure tone / buzz)
    1 → white noise (completely flat spectrum)
    Computed from the mean across time frames.
    """
    flatness = librosa.feature.spectral_flatness(y=audio)
    return float(np.mean(flatness))


def _zero_crossing_rate(audio: np.ndarray) -> float:
    """ZCR is a roughness/noisiness indicator. Higher = more noise."""
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    return float(np.mean(zcr))


def _crest_factor_db(audio: np.ndarray) -> float:
    """
    Crest factor = peak / RMS.
    High crest factor → impulsive noise (clicks, pops).
    Low crest factor  → compressed / clipped audio.
    """
    peak = np.max(np.abs(audio))
    rms  = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-9:
        return 0.0
    return float(20 * np.log10(peak / rms))


# ── Noise classification ──────────────────────────────────────────────────────

def _classify_noise_level(snr_db: float) -> str:
    if snr_db > 30:
        return "Low"
    elif snr_db > 20:
        return "Medium"
    elif snr_db > 10:
        return "High"
    else:
        return "Very High"


def _classify_noise_type(
    spectral_flatness: float,
    zcr: float,
    crest_factor_db: float,
) -> str:
    """
    Heuristic noise type classification based on signal statistics.

    Rules:
      - flatness > 0.5 → Broadband (white/pink noise, hiss)
      - flatness < 0.1 → Tonal (hum, buzz — 50/60 Hz mains, fan drone)
      - crest_factor > 20 dB → Impulsive (clicks, pops, crackle)
      - else → Mixed
    """
    if crest_factor_db > 20:
        return "Impulsive (clicks/pops)"
    if spectral_flatness > 0.5:
        return "Broadband (hiss/white noise)"
    if spectral_flatness < 0.08:
        return "Tonal (hum/buzz)"
    return "Mixed"


def _generate_suggestions(report: AudioReport) -> list[str]:
    """
    Rule-based suggestion engine.
    Produces actionable advice based on measured audio characteristics.
    """
    suggestions = []

    if report.noise_level_label == "Very High":
        suggestions.append(
            "⚠️  Noise level is very high (SNR < 10 dB). Noise reduction is active, "
            "but re-recording in a quieter environment is recommended."
        )
    elif report.noise_level_label == "High":
        suggestions.append(
            "📢  Significant noise detected (SNR 10–20 dB). Spectral gating should "
            "help; consider a directional microphone for future recordings."
        )

    if "Tonal" in report.noise_type_label:
        suggestions.append(
            "🔌  Tonal/electrical hum detected. Check power supply, cable shielding, "
            "or use a ground-loop isolator."
        )

    if report.noise_reduction_db < 1.0:
        suggestions.append(
            "✅  Minimal noise was detected. Your recording is already quite clean."
        )
    
    if not suggestions:
        suggestions.append("✅  Audio quality looks good. Processing applied successfully.")

    return suggestions


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    input_path: Path,
    progress_callback: Callable[[float, str], None] | None = None,
    # Noise reduction settings
    noisereduce_strength: float = 0.75,
    skip_noisereduce: bool = False,
    # Enhancement settings
    low_gain: float = 0.0,
    mid_gain: float = 0.0,
    high_gain: float = 0.0,
    comp_threshold: float = -20.0,
    comp_ratio: float = 1.0,
    output_gain: float = 0.0,
) -> tuple[np.ndarray, int, Path, AudioReport]:
    """
    Execute the complete audio studio cleaning pipeline.

    Args:
        input_path           : path to the input audio file
        progress_callback    : optional callable(fraction, message) for UI updates
        noisereduce_strength : noisereduce prop_decrease [0.0–1.0]
        skip_noisereduce     : bypass noisereduce stage
        low_gain/mid/high    : EQ band adjustments in dB
        comp_threshold       : compressor threshold in dB
        comp_ratio           : compressor ratio (1.0 = no compression)
        output_gain          : final gain adjustment in dB

    Returns:
        cleaned_audio : float32 numpy array
        sr            : sample rate (48000)
        output_path   : path to saved WAV
        report        : AudioReport with metrics and suggestions
    """
    def _progress(fraction: float, msg: str) -> None:
        if progress_callback:
            progress_callback(fraction, msg)
        logger.info("[%.0f%%] %s", fraction * 100, msg)

    start_time = time.time()
    report = AudioReport()

    # ── Step 1: Format conversion ─────────────────────────────────────────────
    _progress(0.05, "Converting audio format (if needed) …")
    wav_path = convert_to_wav(input_path)

    # ── Step 2: Load audio ────────────────────────────────────────────────────
    _progress(0.10, "Loading audio …")
    audio, sr = load_audio(wav_path)
    report.duration_s  = len(audio) / sr
    report.sample_rate = sr

    # ── Step 3: Compute original metrics ──────────────────────────────────────
    _progress(0.18, "Analysing audio characteristics …")
    report.original_rms_db           = _rms_db(audio)
    report.original_snr_db           = _estimate_snr(audio, sr)
    report.original_spectral_flatness = _spectral_flatness(audio)
    report.original_zcr               = _zero_crossing_rate(audio)
    report.original_crest_factor_db   = _crest_factor_db(audio)

    report.noise_level_label = _classify_noise_level(report.original_snr_db)
    report.noise_type_label  = _classify_noise_type(
        report.original_spectral_flatness,
        report.original_zcr,
        report.original_crest_factor_db,
    )

    # ── Step 4: Noisereduce ───────────────────────────────────────────────────
    if not skip_noisereduce:
        _progress(0.30, "Applying spectral noise reduction …")
        audio = apply_noisereduce(audio, sr, prop_decrease=noisereduce_strength)
    else:
        _progress(0.30, "Skipping noise reduction.")

    # ── Step 5: Equalizer ─────────────────────────────────────────────────────
    _progress(0.50, "Applying 3-band EQ …")
    audio = apply_equalizer(audio, sr, low_gain, mid_gain, high_gain)

    # ── Step 6: Compression ───────────────────────────────────────────────────
    if comp_ratio > 1.0:
        _progress(0.70, "Applying dynamic range compression …")
        audio = apply_compression(audio, threshold_db=comp_threshold, ratio=comp_ratio, sr=sr)
    else:
        _progress(0.70, "Skipping compression (ratio = 1.0).")

    # ── Step 7: Output Gain ───────────────────────────────────────────────────
    _progress(0.80, "Applying final gain …")
    audio = apply_gain(audio, output_gain)

    # ── Step 8: Normalise ─────────────────────────────────────────────────────
    _progress(0.88, "Normalising output …")
    audio = normalize_audio(audio)

    # ── Step 9: Save ──────────────────────────────────────────────────────────
    _progress(0.92, "Saving processed audio …")
    output_path = generate_filename("processed")
    save_audio(audio, sr, output_path)

    # ── Step 10: Post-processing metrics ──────────────────────────────────────
    _progress(0.96, "Generating report …")
    report.cleaned_rms_db     = _rms_db(audio)
    report.noise_reduction_db = max(
        0.0,
        report.original_rms_db - report.cleaned_rms_db,
    )
    report.processing_time_s = time.time() - start_time
    report.suggestions       = _generate_suggestions(report)

    _progress(1.0, "Done!")
    return audio, sr, output_path, report