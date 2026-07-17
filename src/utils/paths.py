"""
Centralized project paths.

Works in both VS Code and Google Colab.
"""

from pathlib import Path

PROJECT_ROOT = Path.cwd()

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

PROMPTS_DIR = PROJECT_ROOT / "prompts"

AUDIO_DIR = PROJECT_ROOT / "audio"

REFERENCE_DIR = AUDIO_DIR / "references"

GENERATED_DIR = AUDIO_DIR / "generated"

RESULTS_DIR = PROJECT_ROOT / "results"

SRC_DIR = PROJECT_ROOT / "src"