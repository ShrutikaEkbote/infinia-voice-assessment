"""
xtts.py
=======
Wrapper around XTTS-v2 for zero-shot voice cloning.

API verified (2026-07) against the maintained fork, published on PyPI as
`coqui-tts` (idiap/coqui-ai-TTS). The original `coqui-ai/TTS` package is
unmaintained and raises RuntimeError on Python >= 3.12, so this wrapper
requires the fork specifically:

    pip install coqui-tts

Both packages expose the same import path (`from TTS.api import TTS`),
so no other code in this repo needs to know which one is installed.

License note: XTTS-v2 weights are distributed under the Coqui Public
Model License (non-commercial). Fine for this benchmark; flag it in the
report if Infinia intends to ship XTTS-v2 in a paid product.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.tts.base import BaseTTS, GenerationResult
from src.utils.config import SAMPLE_RATE

logger = logging.getLogger(__name__)

# XTTS-v2 supports 17 languages; this is the subset relevant to this project.
# Passing an unsupported code raises inside the TTS library with a clear
# message, so we don't duplicate validation here.
_LANGUAGE_MAP = {
    "english": "en",
    "arabic": "ar",
    "hindi": "hi",
}


class XTTS(BaseTTS):
    """XTTS-v2 wrapper. See module docstring for install requirements."""

    model_key = "xtts_v2"
    _HF_MODEL_ID = "tts_models/multilingual/multi-dataset/xtts_v2"

    def __init__(self, device: str | None = None, agree_to_tos: bool = True) -> None:
        super().__init__(device=device)
        # The fork's non-interactive checkpoint download requires explicit
        # ToS acceptance (COQUI_TOS_AGREED=1); we set it in-process rather
        # than requiring the user to export an env var, since a Colab cell
        # restart would otherwise silently drop it.
        if agree_to_tos:
            import os
            os.environ.setdefault("COQUI_TOS_AGREED", "1")

    def load_model(self) -> None:
        if self._is_loaded:
            return
        try:
            from TTS.api import TTS
        except ImportError as exc:
            raise ImportError(
                "XTTS-v2 requires the maintained fork. Install with:\n"
                "    pip install coqui-tts\n"
                "(NOT `pip install TTS` — that package is unmaintained and "
                "breaks on Python >= 3.12.)"
            ) from exc

        logger.info("Loading XTTS-v2 onto device=%s (first run downloads ~2GB of weights)", self._device)
        use_gpu = self._device == "cuda"
        self._model = TTS(self._HF_MODEL_ID, progress_bar=False, gpu=use_gpu)
        self._is_loaded = True
        logger.info("XTTS-v2 loaded.")

    def generate(
        self,
        text: str,
        reference_audio_path: str | Path,
        language: str,
        prompt_id: int = 0,
    ) -> GenerationResult:
        if not self._is_loaded:
            self.load_model()

        if language not in _LANGUAGE_MAP:
            raise ValueError(
                f"XTTS-v2 wrapper has no language mapping for '{language}'. "
                f"Known: {list(_LANGUAGE_MAP)}"
            )
        lang_code = _LANGUAGE_MAP[language]
        reference_audio_path = str(reference_audio_path)

        def _run():
            # tts.tts() returns a python list[float] at XTTS's native sample
            # rate (24kHz), not the project-wide SAMPLE_RATE — resampling to
            # a common rate happens once, centrally, in save_audio's caller
            # in base.benchmark(), so every model's output ends up directly
            # comparable regardless of native rate.
            return self._model.tts(
                text=text,
                speaker_wav=reference_audio_path,
                language=lang_code,
            )

        raw_audio, elapsed_s, peak_mb = self._time_and_measure_memory(_run)

        import numpy as np
        audio = np.asarray(raw_audio, dtype="float32")
        native_sr = self._model.synthesizer.output_sample_rate

        # Resample to the project's canonical rate here (rather than
        # deferring it) so GenerationResult.sample_rate is always accurate
        # for whatever the caller does next (save, score, play).
        if native_sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=native_sr, target_sr=SAMPLE_RATE)

        return GenerationResult(
            audio=audio,
            sample_rate=SAMPLE_RATE,
            latency_s=elapsed_s,
            time_to_first_chunk_s=None,  # this wrapper uses batch (non-streaming) inference;
            # XTTS does support a streaming API (tts.tts_stream) with <200ms
            # first-chunk latency claims — implement as a second wrapper
            # (xtts_streaming.py) if the report needs the streaming metric
            # specifically, since batch and streaming latency aren't
            # comparable numbers and shouldn't share one code path.
            peak_gpu_memory_mb=peak_mb,
            model_key=self.model_key,
            prompt_id=prompt_id,
            language=language,
            text=text,
        )