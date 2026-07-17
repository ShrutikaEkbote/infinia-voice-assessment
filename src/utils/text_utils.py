"""
Text preprocessing utilities.
"""

import re


def normalize_text(text: str):

    text = text.strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


def remove_extra_spaces(text: str):

    return " ".join(text.split())