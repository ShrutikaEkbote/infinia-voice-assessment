"""
paths.py
========
Resolves the project root and every derived directory, transparently,
whether we're running:

  - locally on Windows in VS Code (D:\\INFINIA-VOICE-ASSESMENT)
  - in Google Colab (repo cloned under /content)
  - in Kaggle (repo cloned under /kaggle/working)

Every path returned is a `pathlib.Path`, never a raw string, so the
same code is correct on both Windows and POSIX separators.

Directories are created on first access (idempotent) so notebooks never
crash with FileNotFoundError on a fresh checkout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _detect_project_root() -> Path:
    """Walk upward from this file until we find the repo root, identified
    by the presence of a `requirements.txt` alongside a `src/` directory.

    This avoids hardcoding an absolute path, which is the #1 reason
    "works on my machine" notebooks break when someone else clones them.
    """
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "requirements.txt").is_file() and (parent / "src").is_dir():
            return parent

    # Fallback for interactive environments (e.g. a Colab cell that pip-installs
    # this file directly rather than importing it as part of the package) where
    # __file__ resolution can be unreliable. Common Colab/Kaggle mount points
    # are checked explicitly; otherwise we fall back to the current working dir.
    for candidate in (Path("/content/INFINIA-VOICE-ASSESMENT"),
                      Path("/kaggle/working/INFINIA-VOICE-ASSESMENT")):
        if candidate.is_dir():
            return candidate

    return Path.cwd()


PROJECT_ROOT: Path = Path(os.environ.get("INFINIA_PROJECT_ROOT", "")) if os.environ.get(
    "INFINIA_PROJECT_ROOT"
) else _detect_project_root()


@dataclass(frozen=True)
class ProjectPaths:
    """All project directories, resolved once and reused everywhere.
    Import `PATHS` (the module-level singleton below) rather than
    constructing this class directly."""

    root: Path

    @property
    def audio(self) -> Path:
        return self.root / "audio"

    @property
    def references(self) -> Path:
        return self.audio / "references"

    @property
    def generated(self) -> Path:
        return self.audio / "generated"

    @property
    def prompts(self) -> Path:
        return self.root / "prompts"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def plots(self) -> Path:
        return self.results / "plots"

    @property
    def notebooks(self) -> Path:
        return self.root / "notebooks"

    @property
    def src(self) -> Path:
        return self.root / "src"

    @property
    def model_cache(self) -> Path:
        """Local cache dir for downloaded checkpoints. Kept inside the repo
        by default but overridable via HF_HOME / TORCH_HOME so large weight
        files never accidentally get committed to git."""
        return self.root / ".model_cache"

    def reference_dir(self, language: str) -> Path:
        return self.references / language

    def generated_dir(self, language: str, model_key: str) -> Path:
        return self.generated / language / model_key

    def prompts_file(self, language: str) -> Path:
        return self.prompts / f"{language}.json"

    def ensure_all(self) -> None:
        """Create every directory that should exist at the start of a run.
        Safe to call repeatedly."""
        dirs = [
            self.audio, self.references, self.generated, self.prompts,
            self.results, self.plots, self.notebooks, self.model_cache,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        for lang in ("english", "arabic", "hindi"):
            self.reference_dir(lang).mkdir(parents=True, exist_ok=True)


PATHS = ProjectPaths(root=PROJECT_ROOT)


def running_in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def running_in_kaggle() -> bool:
    return Path("/kaggle/working").is_dir()