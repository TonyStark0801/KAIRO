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

_MUSIC_NOTE = re.compile(r"[♪♫♬🎵🎶]")


def normalize(text: str) -> str:
    text = _FILLER_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = text.strip(" ,.;:!?\"'")
    return text


def looks_like_music(text: str) -> bool:
    """Heuristic: returns True if the text looks like transcribed music/lyrics, not a command."""
    if _MUSIC_NOTE.search(text):
        return True
    words = text.split()
    if len(words) < 2:
        return False
    unique = set(words)
    if len(unique) <= len(words) * 0.4 and len(words) >= 5:
        return True
    return False


_NOISE_PHRASES = re.compile(
    r"^(thank you for watching|thanks for watching|please subscribe"
    r"|like and subscribe|don't forget to subscribe|hit the bell"
    r"|see you next time|see you in the next|bye bye|goodbye"
    r"|check out my|link in the description|comment below"
    r"|you|bye|thanks|thank you|okay|oh)$",
    re.IGNORECASE,
)


def looks_like_noise(text: str) -> bool:
    """Returns True for common YouTube/background noise phrases that aren't user commands."""
    return bool(_NOISE_PHRASES.match(text.strip()))


# Single-word or very-short utterances that are pure ambient/conversational filler.
# Whisper commonly hallucinates these from background speech, especially with Indian English
# where retroflex consonants create short confident-sounding transcriptions.
_AMBIENT_SINGLE = frozenset({
    # English fillers
    "yes", "no", "yeah", "nah", "nope", "yep", "yup", "sure", "right",
    "hmm", "hm", "mm", "mmm", "oh", "ah", "uh", "uhh",
    "ok", "okay", "fine", "cool", "nice", "wow", "great",
    "what", "why", "when", "how", "where", "who",
    "really", "seriously", "exactly", "absolutely", "definitely",
    "sorry", "excuse me", "pardon",
    "hello", "hi", "hey", "bye", "goodbye",
    "please", "thanks", "thank",
    # Common Hinglish / Indian English fillers that Whisper picks up
    "haan", "han", "hah", "arre", "arrey", "yaar", "yar",
    "achha", "acha", "accha", "theek", "theekh",
    "nahin", "nahi", "bas", "kya",
    # Whisper hallucinations from silence / breath sounds
    "the", "a", "an", "and", "or", "but", "so",
    "i", "you", "he", "she", "we", "they", "it",
})

# Short ambient phrases (2-4 words) that are never commands.
# "i don't know", "that's fine", "yeah okay", etc.
_AMBIENT_PHRASES = re.compile(
    r"^("
    r"i don'?t know|i don'?t think so|i don'?t understand"
    r"|that'?s fine|that'?s okay|that'?s right|that'?s good|that'?s great"
    r"|yeah okay|yeah sure|yeah fine|okay fine|okay sure"
    r"|no no|yes yes|oh okay|oh right|oh yeah|oh wow|oh no"
    r"|i see|i know|i think|i mean|i was|i am"
    r"|let me see|let me think|hold on|wait wait"
    r"|what do you|why don'?t you|how are you"
    r"|you know what|you know|i know right"
    r"|it'?s fine|it'?s okay|it'?s good|it'?s alright"
    r"|are you sure|is that right|is that so"
    r"|haan haan|arre yaar|bas bas|theek hai"
    r")$",
    re.IGNORECASE,
)


def looks_like_ambient(text: str) -> bool:
    """Returns True for utterances that are clearly conversational/ambient — not directed at KAIRO.

    Catches:
    - Single filler words (yeah, haan, yaar, etc.)
    - Common 2-4 word conversational phrases
    - Single pronouns / articles that Whisper hallucinates from breath sounds
    """
    t = text.strip().lower().strip(" .,!?\"'-")
    if not t:
        return False
    # Single word check
    if " " not in t and t in _AMBIENT_SINGLE:
        return True
    # Short phrase check
    words = t.split()
    if len(words) <= 4 and _AMBIENT_PHRASES.match(t):
        return True
    return False
