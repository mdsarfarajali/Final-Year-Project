"""
smooth.py
---------
Secondary noise reduction stage using the `noisereduce` library.

What is noisereduce?
    A signal-processing library that applies spectral gating / statistical
    noise profiling to suppress residual noise that DeepFilterNet may leave
    behind — especially low-frequency rumble, stationary hiss, and
    electrical buzz.

How it works:
    1. Estimates a noise profile from a "quiet" portion of the signal
       (or uses the entire signal statistically).
    2. Builds a Wiener-filter-like mask in the STFT domain.
    3. Applies the mask, attenuating frames below the noise threshold.

Why use it after DeepFilterNet?
    DeepFilterNet is designed for speech and handles dynamic noise well,
    but may leave faint stationary noise. Noisereduce handles exactly that
    — stationary hiss, air-conditioning hum, etc. The two-stage pipeline
    gives best-of-both-worlds results.
"""

import logging

import numpy as np
import noisereduce as nr

logger = logging.getLogger("smooth")


def apply_noisereduce(
    audio: np.ndarray,
    sr: int,
    prop_decrease: float = 0.75,
    stationary: bool = False,
    time_mask_smooth_ms: int = 50,
    freq_mask_smooth_hz: int = 300,
    n_std_thresh_stationary: float = 1.5,
) -> np.ndarray:
    """
    Apply spectral gating noise reduction as a post-processing step.

    Args:
        audio                     : float32 mono audio array
        sr                        : sample rate in Hz
        prop_decrease             : proportion of noise to remove [0.0–1.0].
                                    0.75 = reduce noise by 75% of estimated level.
                                    Lower = gentler (less artefacts);
                                    Higher = more aggressive (risk of warbling).
        stationary                : if True, treats noise as constant (good for
                                    steady hiss/hum). If False, uses time-varying
                                    estimation (better for mixed environments).
        time_mask_smooth_ms       : time smoothing of the noise mask in ms.
                                    Higher = smoother but slower response.
        freq_mask_smooth_hz       : frequency smoothing of the mask.
                                    Higher = broader frequency buckets.
        n_std_thresh_stationary   : standard deviations above noise floor to
                                    keep signal. Higher = more conservative
                                    (keeps more, removes less).

    Returns:
        reduced : float32 numpy array of same shape
    """
    if len(audio) == 0:
        logger.warning("Empty audio passed to smooth stage — skipping.")
        return audio

    logger.info(
        "Applying noisereduce | prop_decrease=%.2f | stationary=%s | SR=%d",
        prop_decrease, stationary, sr,
    )

    reduced = nr.reduce_noise(
        y=audio,
        sr=sr,
        prop_decrease=prop_decrease,
        stationary=stationary,
        time_mask_smooth_ms=time_mask_smooth_ms,
        freq_mask_smooth_hz=freq_mask_smooth_hz,
        n_std_thresh_stationary=n_std_thresh_stationary,
        use_torch=False,           # CPU-only; set True if CUDA available
        n_jobs=1,                  # single-threaded for reproducibility
    )

    logger.info("Noisereduce complete. Output shape: %s", reduced.shape)
    return reduced.astype(np.float32)


def apply_gentle_smooth(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    A lighter variant of noise reduction with very gentle settings.
    Use this when the audio is already fairly clean and you want
    minimal post-processing to avoid introducing artefacts.
    """
    return apply_noisereduce(
        audio,
        sr,
        prop_decrease=0.40,
        stationary=True,
        time_mask_smooth_ms=100,
        freq_mask_smooth_hz=500,
        n_std_thresh_stationary=2.0,
    )


def apply_aggressive_smooth(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    A more aggressive variant for heavily noisy recordings.
    May introduce slight musical-noise artefacts on some signals.
    """
    return apply_noisereduce(
        audio,
        sr,
        prop_decrease=0.95,
        stationary=False,
        time_mask_smooth_ms=25,
        freq_mask_smooth_hz=150,
        n_std_thresh_stationary=1.0,
    )