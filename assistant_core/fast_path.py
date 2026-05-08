"""Tier 1: Keyword fast-path — instant command matching, no LLM needed.

Handles ~60% of daily commands with zero latency.
Returns None if no pattern matched (falls through to LLM tiers).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class FastPathResult:
    tool_name: str
    params: dict[str, Any]
    message: str = ""


_PATTERNS: list[tuple[re.Pattern, str, dict[str, Any], str]] = [
    # YouTube playback controls
    (re.compile(r"^(next|skip|skip this|next song|skip song)$"), "youtube_control", {"action": "next"}, ""),
    (re.compile(r"^(pause|stop|stop this|stop music|pause music)$"), "youtube_control", {"action": "pause"}, ""),
    (re.compile(r"^(resume|unpause|continue|continue playing)$"), "youtube_control", {"action": "play"}, ""),
    (re.compile(r"^(what'?s playing|now playing|current song|what song is this)$"), "youtube_now_playing", {}, ""),

    # System volume
    (re.compile(r"(volume up|louder|turn.*up|raise.*volume|increase.*volume)"), "system_volume", {"action": "up"}, ""),
    (re.compile(r"(volume down|quieter|turn.*down|lower.*volume|decrease.*volume|reduce.*volume)"), "system_volume", {"action": "down"}, ""),
    (re.compile(r"^mute$"), "system_volume", {"action": "mute"}, ""),
    (re.compile(r"^unmute$"), "system_volume", {"action": "unmute"}, ""),

    # Open common URLs
    (re.compile(r"^open (youtube|yt)$"), "open_url", {"url": "https://www.youtube.com", "browser": "Brave Browser"}, ""),
    (re.compile(r"^open (gmail|mail|email)$"), "open_url", {"url": "https://mail.google.com", "browser": "Brave Browser"}, ""),
    (re.compile(r"^open (calendar|cal)$"), "open_url", {"url": "https://calendar.google.com", "browser": "Brave Browser"}, ""),
    (re.compile(r"^open (github|gh)$"), "open_url", {"url": "https://github.com", "browser": "Brave Browser"}, ""),

    # Open apps — direct AppleScript
    (re.compile(r"open (intellij|idea)"), "open_app", {"app_name": "IntelliJ IDEA"}, ""),
    (re.compile(r"open (brave|browser)"), "open_app", {"app_name": "Brave Browser"}, ""),
    (re.compile(r"open (chrome|google chrome)"), "open_app", {"app_name": "Google Chrome"}, ""),
    (re.compile(r"open (terminal|term)"), "open_app", {"app_name": "Terminal"}, ""),
    (re.compile(r"open (finder|files)"), "open_app", {"app_name": "Finder"}, ""),
    (re.compile(r"open (slack)"), "open_app", {"app_name": "Slack"}, ""),
    (re.compile(r"open (spotify)"), "open_app", {"app_name": "Spotify"}, ""),
    (re.compile(r"open (notes)"), "open_app", {"app_name": "Notes"}, ""),
    (re.compile(r"open (cursor)"), "open_app", {"app_name": "Cursor"}, ""),
    (re.compile(r"open (vscode|vs code|visual studio)"), "open_app", {"app_name": "Visual Studio Code"}, ""),
]


def try_fast_path(transcript: str, media_playing: bool = False) -> FastPathResult | None:
    """Try to match transcript against keyword patterns.

    Returns FastPathResult if matched, None if no match (fall through to LLM).
    """
    text = transcript.strip().lower()
    # Strip trailing filler: "for me", "please", "will you"
    text = re.sub(r"\s+(for me|please|will you|can you|could you)$", "", text)

    # "play" with no query while media is playing → resume
    if media_playing and text in ("play", "play music"):
        return FastPathResult("youtube_control", {"action": "play"})

    for pattern, tool, params, msg in _PATTERNS:
        if pattern.search(text):
            return FastPathResult(tool, dict(params), msg)

    # Volume set with explicit number: "set volume to 60", "volume 50"
    vol_match = re.search(r"(?:set\s+)?volume\s+(?:to\s+)?(\d+)", text)
    if vol_match:
        level = int(vol_match.group(1))
        return FastPathResult("system_volume", {"action": "set", "level": level})

    return None
