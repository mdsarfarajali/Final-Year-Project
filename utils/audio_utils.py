"""
audio_utils.py
--------------
Handles all audio I/O operations:
  - Loading audio files (WAV, MP3, FLAC, OGG, M4A, etc.)
  - Converting non-WAV formats to WAV via FFmpeg
  - Saving processed audio to disk
  - Generating unique timestamped filenames

Why librosa?  → Industry-standard, handles resampling & multichannel gracefully.
Why soundfile? → Fast, supports 32-bit float WAV needed by DeepFilterNet.
Why FFmpeg?    → Universal format conversion; handles codecs Python alone cannot.
"""

import os
import logging
import subprocess
import tempfile
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s — %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("audio_utils")

# ── Constants ─────────────────────────────────────────────────────────────────
SUPPORTED_FORMATS = {
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".aiff", ".mpeg",
    ".mp4", ".mkv", ".mov", ".avi", ".webm"
}
TARGET_SR = 48_000          # DeepFilterNet's native sample rate
UPLOADS_DIR = Path("uploads")
OUTPUTS_DIR = Path("outputs")


def ensure_dirs() -> None:
    """Create upload/output directories if they don't exist."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_filename(prefix: str, suffix: str = ".wav") -> Path:
    """
    Generate a unique timestamped filename.
    Example: outputs/cleaned_1713800123.wav
    """
    timestamp = int(time.time())
    return OUTPUTS_DIR / f"{prefix}_{timestamp}{suffix}"


def save_upload(uploaded_file) -> Path:
    """
    Persist a Streamlit UploadedFile object to disk.
    Returns the saved file path.
    """
    ensure_dirs()
    ext = Path(uploaded_file.name).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )
    dest = UPLOADS_DIR / f"input_{int(time.time())}{ext}"
    dest.write_bytes(uploaded_file.read())
    logger.info("Saved upload → %s (%s bytes)", dest, dest.stat().st_size)
    return dest


def convert_to_wav(src: Path) -> Path:
    """
    Convert any audio file to 48 kHz mono WAV using FFmpeg.
    This is required because DeepFilterNet only accepts WAV at 48 kHz.

    FFmpeg handles codecs (MP3, AAC, FLAC, OGG, etc.) that Python
    audio libraries cannot decode natively.
    """
    if src.suffix.lower() == ".wav":
        logger.info("File is already WAV, skipping conversion.")
        return src

    dest = UPLOADS_DIR / f"{src.stem}_converted.wav"
    cmd = [
        "ffmpeg",
        "-y",                  # overwrite without asking
        "-i", str(src),        # input file
        "-ar", str(TARGET_SR), # resample to 48 kHz
        "-ac", "1",            # force mono
        "-f", "wav",           # output format
        "-acodec", "pcm_f32le",# 32-bit float PCM (DeepFilterNet compatible)
        str(dest),
    ]
    logger.info("Running FFmpeg: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg conversion failed:\n{result.stderr}"
        )
    logger.info("Converted → %s", dest)
    return dest


def load_audio(path: Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """
    Load audio from disk, resample to target_sr, convert to mono.

    Returns:
        audio  : float32 numpy array, shape (samples,), range [-1, 1]
        sr     : actual sample rate after resampling

    Why librosa.load?
        - Handles automatic resampling with high-quality algorithms.
        - Converts stereo → mono by averaging channels.
        - Returns float32 arrays ready for neural network inference.
    """
    logger.info("Loading audio: %s (target SR=%d)", path, target_sr)
    audio, sr = librosa.load(str(path), sr=target_sr, mono=True, dtype=np.float32)

    # Sanity checks
    if audio is None or len(audio) == 0:
        raise ValueError("Loaded audio is empty.")
    if np.all(audio == 0):
        raise ValueError("Audio contains only silence — nothing to process.")

    logger.info("Loaded: %.2f s | SR=%d | samples=%d", len(audio) / sr, sr, len(audio))
    return audio, sr


def save_audio(audio: np.ndarray, sr: int, path: Path | None = None) -> Path:
    """
    Save a float32 audio array to a WAV file.

    Why soundfile?
        - Supports 32-bit float WAV natively.
        - Much faster than scipy.io.wavfile for large files.
        - No clipping: handles values outside [-1, 1] gracefully.
    """
    if path is None:
        path = generate_filename("cleaned")

    ensure_dirs()

    # Clip to prevent distortion in players that expect [-1, 1]
    audio_clipped = np.clip(audio, -1.0, 1.0)
    sf.write(str(path), audio_clipped, sr, subtype="PCM_16")
    logger.info("Saved cleaned audio → %s (%.2f s)", path, len(audio) / sr)
    return path


def get_duration(audio: np.ndarray, sr: int) -> float:
    """Return duration of audio in seconds."""
    return len(audio) / sr


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Peak-normalize audio to [-1, 1].
    Applied before saving to ensure consistent loudness.
    """
    peak = np.max(np.abs(audio))
    if peak > 0:
        return audio / peak
    return audio