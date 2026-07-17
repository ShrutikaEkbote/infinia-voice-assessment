"""
Global project configuration.
"""

from pathlib import Path

# Change this only if your project folder name changes.
PROJECT_NAME = "infinia-voice-assessment"

# Default sample rate for generated audio.
DEFAULT_SAMPLE_RATE = 16000

# Default audio format.
AUDIO_FORMAT = "wav"

# Random seed for reproducibility.
RANDOM_SEED = 42

# Default device preference.
DEVICE = "cuda"

# Benchmark settings.
BENCHMARK_LANGUAGE = "english"

# Speaker similarity model.
SPEAKER_EMBEDDING_MODEL = "speechbrain_ecapa"

# Whisper model for ASR.
ASR_MODEL = "large-v3"