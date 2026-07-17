"""
base.py
=======
Abstract interface every TTS wrapper implements. Keeping this thin and
strict is what lets the benchmark loop, the notebooks, and the leaderboard
treat every model identically — a notebook cell that benchmarks XTTS
should be line-for-line swappable to benchmark Chatterbox by changing
one import.

Design notes
------------
- `load_model()` / `unload_model()` are separate from `__init__` so a
  benchmark loop can construct all model objects up front (cheap) and
  then load/generate/unload one at a time (expensive, GPU-memory bound).
  This matters a lot on a single T4 in Colab, where you cannot hold
  four large TTS models in VRAM simultaneously.
- `generate()` returns a `GenerationResult`, not a bare numpy array,
  because latency and peak GPU memory have to be measured *inside* the
  call (measuring them outside would include unrelated overhead like
  file I/O) but the caller still needs the raw audio to save and score.
- `benchmark()` has a default implementation built on top of
  `generate()` so a subclass only has to implement `load_model()` and
  `generate()` to be fully benchmarkable. Override `benchmark()` only
  if a model needs a genuinely different measurement strategy (e.g. a
  streaming model measuring time-to-first-chunk).
"""

from __future__ import annotations

import gc
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is a hard dependency at runtime,
    _TORCH_AVAILABLE = False  # but importing base.py alone (e.g. for type checking) should not fail.


@dataclass
class GenerationResult:
    """Everything the evaluation pipeline needs from a single generation call."""

    audio: np.ndarray
    sample_rate: int
    latency_s: float                 # wall-clock time for generate() itself
    time_to_first_chunk_s: float | None  # None for non-streaming models
    peak_gpu_memory_mb: float | None     # None if run on CPU
    model_key: str
    prompt_id: int
    language: str
    text: str


@dataclass
class BenchmarkRecord:
    """One row of `results/raw_benchmark_results.csv`. Deliberately flat
    (no nested objects) so it serializes to CSV with zero custom logic —
    `pandas.DataFrame([asdict(r) for r in records])` is the entire export."""

    model_key: str
    language: str
    prompt_id: int
    text: str
    audio_path: str
    duration_s: float
    latency_s: float
    rtf: float                       # latency_s / duration_s
    time_to_first_chunk_ms: float | None
    peak_gpu_memory_mb: float | None
    success: bool
    error_message: str | None = None


class BaseTTS(ABC):
    """Abstract base every model wrapper (XTTS, Chatterbox, FishSpeech, ...)
    must subclass. See module docstring for the design rationale."""

    #: Overridden by subclasses with the registry key from config.MODEL_REGISTRY.
    model_key: str = "base"

    def __init__(self, device: str | None = None) -> None:
        self._device = device or self._auto_select_device()
        self._model = None  # populated by load_model()
        self._is_loaded = False

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------
    @abstractmethod
    def load_model(self) -> None:
        """Load weights into memory/VRAM. Must set `self._model` and
        `self._is_loaded = True`. Should be idempotent — calling it twice
        in a row must not re-download or double-allocate."""
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        text: str,
        reference_audio_path: str | Path,
        language: str,
        prompt_id: int = 0,
    ) -> GenerationResult:
        """Synthesize `text` cloning the speaker in `reference_audio_path`.
        Must internally time the actual model.forward()/inference call —
        not file I/O — for `latency_s` to be comparable across models."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared functionality — override only if a model needs it
    # ------------------------------------------------------------------
    def unload_model(self) -> None:
        """Release the model and free GPU memory. Called between models in
        a benchmark loop so peak-memory measurements aren't contaminated by
        a previous model still resident in VRAM."""
        self._model = None
        self._is_loaded = False
        gc.collect()
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def benchmark(
        self,
        prompts: list[dict],
        reference_audio_path: str | Path,
        language: str,
        output_dir: str | Path,
    ) -> list[BenchmarkRecord]:
        """Run every prompt through `generate()`, save the audio, and return
        one BenchmarkRecord per prompt. A failure on one prompt is recorded
        (success=False) rather than raising, so one bad prompt can't kill
        an otherwise-multi-hour Colab run.
        """
        from src.utils.audio_utils import save_audio  # local import avoids a
        # circular import between src.tts and src.utils at module load time.

        if not self._is_loaded:
            self.load_model()

        output_dir = Path(output_dir)
        records: list[BenchmarkRecord] = []

        for prompt in prompts:
            prompt_id = prompt["id"]
            text = prompt["text"]
            try:
                result = self.generate(
                    text=text,
                    reference_audio_path=reference_audio_path,
                    language=language,
                    prompt_id=prompt_id,
                )
                audio_path = output_dir / f"{prompt_id:03d}.wav"
                save_audio(result.audio, result.sample_rate, audio_path)

                duration_s = len(result.audio) / result.sample_rate if result.sample_rate else 0.0
                rtf = result.latency_s / duration_s if duration_s > 0 else float("inf")

                records.append(
                    BenchmarkRecord(
                        model_key=self.model_key,
                        language=language,
                        prompt_id=prompt_id,
                        text=text,
                        audio_path=str(audio_path),
                        duration_s=duration_s,
                        latency_s=result.latency_s,
                        rtf=rtf,
                        time_to_first_chunk_ms=(
                            result.time_to_first_chunk_s * 1000
                            if result.time_to_first_chunk_s is not None else None
                        ),
                        peak_gpu_memory_mb=result.peak_gpu_memory_mb,
                        success=True,
                    )
                )
                logger.info(
                    "[%s/%s] prompt %s ok — latency=%.2fs rtf=%.2f",
                    self.model_key, language, prompt_id, result.latency_s, rtf,
                )
            except Exception as exc:  # noqa: BLE001 - intentionally broad; see docstring
                logger.exception("[%s/%s] prompt %s FAILED", self.model_key, language, prompt_id)
                records.append(
                    BenchmarkRecord(
                        model_key=self.model_key,
                        language=language,
                        prompt_id=prompt_id,
                        text=text,
                        audio_path="",
                        duration_s=0.0,
                        latency_s=0.0,
                        rtf=float("nan"),
                        time_to_first_chunk_ms=None,
                        peak_gpu_memory_mb=None,
                        success=False,
                        error_message=str(exc),
                    )
                )
        return records

    # ------------------------------------------------------------------
    # Helpers available to every subclass
    # ------------------------------------------------------------------
    @staticmethod
    def _auto_select_device() -> str:
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def _time_and_measure_memory(fn, *args, **kwargs) -> tuple[np.ndarray, float, float | None]:
        """Run `fn`, returning (result, elapsed_seconds, peak_gpu_mb).
        Shared by every subclass's `generate()` so the timing methodology
        (perf_counter around the call, CUDA memory stats reset beforehand)
        is identical across all models — required for the latency/memory
        columns in the leaderboard to be an apples-to-apples comparison."""
        peak_mb: float | None = None
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start = time.perf_counter()
        result = fn(*args, **kwargs)
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if _TORCH_AVAILABLE and torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        return result, elapsed, peak_mb

    def __repr__(self) -> str:
        status = "loaded" if self._is_loaded else "not loaded"
        return f"<{self.__class__.__name__} device={self._device} status={status}>"