from __future__ import annotations

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class MoodMode(Enum):
    CASUAL = "casual"
    WORK = "work"
    SUPPORTIVE = "supportive"


WORK_APPS = {
    "IntelliJ IDEA", "IntelliJ IDEA CE", "WebStorm", "PyCharm",
    "Visual Studio Code", "Cursor", "Terminal", "iTerm2",
    "Xcode", "Android Studio",
}

FRUSTRATION_WORDS = [
    "frustrated", "annoyed", "angry", "stuck", "broken",
    "doesn't work", "not working", "hate", "ugh", "damn",
    "what the", "wtf", "failing", "sucks", "tired of",
    "can't figure", "giving up", "impossible", "stupid",
]

_MOOD_PROMPTS = {
    MoodMode.CASUAL: (
        "Speak like a close friend. Playful, warm, uses humor. "
        "Match Tony's energy — if he's chill, be chill. If he's excited, match it."
    ),
    MoodMode.WORK: (
        "Act as a senior SDE-3 colleague. Be precise, call out issues, suggest improvements. "
        "Professional but not stiff. No fluff — Tony is in the zone."
    ),
    MoodMode.SUPPORTIVE: (
        "Be empathetic and supportive. Listen first. Don't try to fix everything immediately. "
        "Acknowledge Tony's frustration before offering solutions."
    ),
}


def detect_mood(active_app: str, transcript: str, hour: int) -> MoodMode:
    """Detect mood from context. Simple rule-based for now."""
    transcript_lower = transcript.lower()
    
    # Frustration signals take highest priority
    if any(word in transcript_lower for word in FRUSTRATION_WORDS):
        return MoodMode.SUPPORTIVE
    
    # Work context during work hours
    if active_app in WORK_APPS and 9 <= hour <= 18:
        return MoodMode.WORK
    
    return MoodMode.CASUAL


def get_mood_prompt(mood: MoodMode) -> str:
    """Get the personality injection prompt for a given mood."""
    return _MOOD_PROMPTS.get(mood, _MOOD_PROMPTS[MoodMode.CASUAL])
