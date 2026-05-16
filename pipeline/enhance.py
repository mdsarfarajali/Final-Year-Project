"""
enhance.py
----------
Classical signal processing stages for audio enhancement:
  - 3-Band Equalizer (Bass, Mid, Treble)
  - Dynamic Range Compression
  - Manual Gain Control
"""

import logging
import numpy as np
from scipy.signal import iirfilter, lfilter

logger = logging.getLogger("enhance")

def apply_equalizer(
    audio: np.ndarray,
    sr: int,
    low_gain_db: float = 0.0,
    mid_gain_db: float = 0.0,
    high_gain_db: float = 0.0,
) -> np.ndarray:
    """
    Apply a simple 3-band equalizer using IIR filters.
    - Low: Shelving filter at 250 Hz
    - Mid: Peaking filter at 1000 Hz
    - High: Shelving filter at 4000 Hz
    """
    if low_gain_db == 0 and mid_gain_db == 0 and high_gain_db == 0:
        return audio

    logger.info("Applying EQ | Low: %.1f dB | Mid: %.1f dB | High: %.1f dB", 
                low_gain_db, mid_gain_db, high_gain_db)

    processed = audio.copy()

    # Low Shelf (approx 250 Hz)
    if low_gain_db != 0:
        b, a = iirfilter(2, 250 / (sr / 2), btype='lowpass', ftype='butter')
        low_component = lfilter(b, a, processed)
        processed = (processed - low_component) + low_component * (10 ** (low_gain_db / 20))

    # High Shelf (approx 4000 Hz)
    if high_gain_db != 0:
        b, a = iirfilter(2, 4000 / (sr / 2), btype='highpass', ftype='butter')
        high_component = lfilter(b, a, processed)
        processed = (processed - high_component) + high_component * (10 ** (high_gain_db / 20))

    # Mid Band (approx 1000 Hz)
    if mid_gain_db != 0:
        # Simple bandpass approach for mid
        b, a = iirfilter(2, [500 / (sr / 2), 2000 / (sr / 2)], btype='bandpass', ftype='butter')
        mid_component = lfilter(b, a, processed)
        processed = (processed - mid_component) + mid_component * (10 ** (mid_gain_db / 20))

    return processed.astype(np.float32)

def apply_compression(
    audio: np.ndarray,
    threshold_db: float = -20.0,
    ratio: float = 4.0,
    attack_ms: float = 5.0,
    release_ms: float = 50.0,
    sr: int = 48000
) -> np.ndarray:
    """
    Apply dynamic range compression.
    Reduces the volume of signals above the threshold.
    """
    if ratio <= 1.0:
        return audio

    logger.info("Applying Compression | Threshold: %.1f dB | Ratio: %.1f:1", threshold_db, ratio)

    # Convert threshold to linear
    threshold = 10 ** (threshold_db / 20)
    
    # Calculate envelope (RMS-ish)
    # Using a simple moving average for the envelope
    window_size = int(0.01 * sr) # 10ms window
    envelope = np.sqrt(np.convolve(audio**2, np.ones(window_size)/window_size, mode='same'))
    
    # Calculate gain reduction
    gain_reduction = np.ones_like(envelope)
    mask = envelope > threshold
    
    # For samples above threshold: 
    # gain = threshold + (envelope - threshold) / ratio
    # reduction = gain / envelope
    gain_reduction[mask] = (threshold + (envelope[mask] - threshold) / ratio) / envelope[mask]
    
    # Apply smoothing to gain reduction (attack/release)
    # Simple one-pole filter for smoothing
    alpha_attack = np.exp(-1.0 / (attack_ms * sr / 1000.0))
    alpha_release = np.exp(-1.0 / (release_ms * sr / 1000.0))
    
    smoothed_gain = np.ones_like(gain_reduction)
    current_gain = 1.0
    for i in range(len(gain_reduction)):
        target = gain_reduction[i]
        if target < current_gain:
            current_gain = alpha_attack * current_gain + (1 - alpha_attack) * target
        else:
            current_gain = alpha_release * current_gain + (1 - alpha_release) * target
        smoothed_gain[i] = current_gain

    return (audio * smoothed_gain).astype(np.float32)

def apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply manual gain in dB."""
    if gain_db == 0:
        return audio
    logger.info("Applying Manual Gain: %.1f dB", gain_db)
    return (audio * (10 ** (gain_db / 20))).astype(np.float32)
