"""
config.py
=========
Single source of truth for every constant used across the pipeline:
languages, model registry, evaluation targets (from the Infinia brief,
section 3), sample rates, and ASR configuration.

Nothing in this file touches disk or imports torch — it is safe to
import from any context (local Windows, Colab, Kaggle, a notebook,
or a unit test) with zero side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Language(str, Enum):
    """Supported languages. String-valued so they serialize cleanly to
    JSON/CSV without a custom encoder."""

    ENGLISH = "english"
    ARABIC = "arabic"
    HINDI = "hindi"

    @property
    def iso_code(self) -> str:
        """ISO 639-1 code, used when a model's API expects e.g. 'en'/'ar'/'hi'."""
        return {
            Language.ENGLISH: "en",
            Language.ARABIC: "ar",
            Language.HINDI: "hi",
        }[self]

    @property
    def whisper_code(self) -> str:
        """Whisper's language argument is the same ISO code for these three."""
        return self.iso_code


@dataclass(frozen=True)
class EvalTargets:
    """Section 3 targets from the Infinia brief. Kept as one object so a
    report can print 'target vs. actual' without magic numbers scattered
    through the codebase."""

    mos_min: float = 4.0                 # Naturalness, 1-5 scale
    speaker_similarity_min: float = 0.75  # Cosine similarity, ECAPA-TDNN embeddings
    latency_streaming_ms_max: float = 500.0   # Time to first audio chunk
    latency_batch_s_max: float = 2.0          # Full clip, ~10-word sentence
    rtf_max: float = 0.5                 # Real-time factor (gen_time / audio_len)
    wer_max_pct: float = 10.0            # Round-trip word error rate


@dataclass(frozen=True)
class ModelSpec:
    """Metadata for one TTS model under benchmark. `supported_languages`
    drives the language router — a model is only invoked for a language
    it actually claims to support, so the leaderboard never silently
    penalizes a model for a language it was never meant to speak."""

    key: str                       # short id, matches folder names under audio/generated/<lang>/
    display_name: str
    supports_cloning: bool
    supported_languages: tuple[Language, ...]
    module_path: str               # "src.tts.xtts" style, used for dynamic import
    class_name: str                # "XTTS"
    license_note: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# This is the ONE place a new model gets added. Everything else (notebooks,
# language router, leaderboard) reads from here.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "xtts_v2": ModelSpec(
        key="xtts_v2",
        display_name="XTTS-v2 (Coqui, idiap fork)",
        supports_cloning=True,
        supported_languages=(Language.ENGLISH, Language.ARABIC, Language.HINDI),
        module_path="src.tts.xtts",
        class_name="XTTS",
        license_note="Coqui Public Model License — non-commercial",
        notes="Zero-shot cloning from ~6s reference. 17 languages incl. ar, hi.",
    ),
    "chatterbox": ModelSpec(
        key="chatterbox",
        display_name="Chatterbox (Resemble AI)",
        supports_cloning=True,
        supported_languages=(Language.ENGLISH,),
        module_path="src.tts.chatterbox",
        class_name="Chatterbox",
        notes="English-only in the base release; strongest latency/quality in English.",
    ),
    "fish_speech": ModelSpec(
        key="fish_speech",
        display_name="Fish-Speech",
        supports_cloning=True,
        supported_languages=(Language.ENGLISH, Language.ARABIC, Language.HINDI),
        module_path="src.tts.fish_speech",
        class_name="FishSpeech",
        notes="Multilingual; Arabic/Hindi coverage should be spot-checked, not assumed.",
    ),
    "indextts2": ModelSpec(
        key="indextts2",
        display_name="IndexTTS-2",
        supports_cloning=True,
        supported_languages=(Language.ENGLISH,),
        module_path="src.tts.indextts2",
        class_name="IndexTTS2",
        notes="Emotion control; English/Chinese focus, no confirmed ar/hi support.",
    ),
    "cosyvoice2": ModelSpec(
        key="cosyvoice2",
        display_name="CosyVoice2",
        supports_cloning=True,
        supported_languages=(Language.ENGLISH,),
        module_path="src.tts.cosyvoice2",
        class_name="CosyVoice2",
        notes="Strong streaming latency; zh/en primary, no confirmed ar/hi support.",
    ),
}


# ---------------------------------------------------------------------------
# Audio / ASR constants
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 16000            # target rate for all saved generations + eval
REFERENCE_MIN_DURATION_S: float = 5.0
REFERENCE_MAX_DURATION_S: float = 30.0
WHISPER_MODEL_NAME: str = "large-v3"
WHISPER_MODEL_NAME_CPU_FALLBACK: str = "medium"  # used when no CUDA device is present

SPEAKER_EMBEDDING_MODEL: str = "speechbrain/spkrec-ecapa-voxceleb"

EVAL_TARGETS = EvalTargets()

RANDOM_SEED: int = 42