"""Text normalization for voice transcripts."""

from __future__ import annotations

import re

_FILLER_WORDS = {
    "um",
    "uh",
    "er",
    "ah",
    "like",
    "you know",
    "i mean",
    "basically",
    "actually",
    "literally",
    "so",
    "well",
}

_FILLER_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(_FILLER_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    text = _FILLER_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = text.strip(" ,.;:!?\"'")
    return text
