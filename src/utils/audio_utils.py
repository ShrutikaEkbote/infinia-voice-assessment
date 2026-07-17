"""
audio_utils.py
===============
Audio I/O and preprocessing shared by every TTS wrapper and evaluator.
Deliberately built on `soundfile` + `librosa` rather than `torchaudio`
directly, so this module has no hard dependency on a particular torch
build — it works even before a model-specific environment is set up
(useful for the reference-audio validation step in notebook 00).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioInfo:
    """Lightweight metadata about a wav file, computed once and reused
    instead of re-reading the file with every caller."""

    path: Path
    duration_s: float
    sample_rate: int
    channels: int
    peak_amplitude: float
    rms: float

    @property
    def is_clipping(self) -> bool:
        return self.peak_amplitude >= 0.999

    @property
    def is_silent(self) -> bool:
        return self.rms < 1e-4


def load_audio(path: str | Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load a wav file as float32 mono. If `target_sr` is given and differs
    from the file's native rate, resample with librosa (high-quality,
    deterministic — avoids the subtly different resampling behaviour of
    torchaudio's default resampler, which would otherwise make cross-model
    comparisons slightly unfair)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, sr = sf.read(str(path), always_2d=False, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # downmix to mono

    if target_sr is not None and sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio, sr


def save_audio(audio: np.ndarray, sample_rate: int, path: str | Path) -> Path:
    """Write float32 audio to disk as 16-bit PCM wav, creating parent dirs
    as needed. Returns the resolved path for convenient chaining."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")
    return path


def get_audio_info(path: str | Path) -> AudioInfo:
    """Compute duration, peak, and RMS without loading via librosa twice —
    used by the reference-audio validator and the audio-quality checks in
    the evaluation pipeline (see diagram section 5.1: 'Audio Quality Checks')."""
    audio, sr = load_audio(path)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    return AudioInfo(
        path=Path(path),
        duration_s=len(audio) / sr if sr else 0.0,
        sample_rate=sr,
        channels=1,
        peak_amplitude=peak,
        rms=rms,
    )


def trim_silence(audio: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    """Trim leading/trailing silence. Used before computing RTF so a model
    that pads its output with silence isn't penalized (or flattered) on
    real-time-factor relative to one that doesn't."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def validate_reference_audio(
    path: str | Path,
    min_duration_s: float,
    max_duration_s: float,
) -> list[str]:
    """Sanity-check a reference/cloning sample before it's fed to any model.
    Returns a list of human-readable problems; an empty list means the file
    passed. Used in notebook 00_environment_check so a bad reference clip
    is caught before burning GPU time on Colab."""
    problems: list[str] = []
    info = get_audio_info(path)

    if info.duration_s < min_duration_s:
        problems.append(
            f"Reference too short: {info.duration_s:.2f}s (min {min_duration_s}s)"
        )
    if info.duration_s > max_duration_s:
        problems.append(
            f"Reference too long: {info.duration_s:.2f}s (max {max_duration_s}s)"
        )
    if info.is_clipping:
        problems.append(f"Reference clips at peak amplitude {info.peak_amplitude:.3f}")
    if info.is_silent:
        problems.append("Reference is effectively silent (RMS below threshold)")

    return problems