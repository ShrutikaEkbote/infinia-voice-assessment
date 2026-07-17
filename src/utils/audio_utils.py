"""
Audio helper functions.
"""

from pathlib import Path

import soundfile as sf

from IPython.display import Audio


def load_audio(path: Path):

    audio, sr = sf.read(path)

    return audio, sr


def save_audio(path: Path, audio, sample_rate):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sf.write(
        path,
        audio,
        sample_rate
    )


def play_audio(audio, sample_rate):

    return Audio(
        audio,
        rate=sample_rate
    )