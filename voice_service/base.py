"""Abstract voice engine interface — swap TTS backends without touching callers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable


class VoiceEngine(ABC):
    @abstractmethod
    async def speak(self, text: str) -> None:
        """Synthesize and play the given text. Must block until playback finishes."""

    @abstractmethod
    async def initialize(self) -> bool:
        """Set up the engine. Return True if ready."""

    def set_mic_mute_callback(self, mute: Callable[[], None], unmute: Callable[[], None]) -> None:
        """Optional: register callbacks to mute/unmute mic during speech."""
        self._mute_mic = mute
        self._unmute_mic = unmute

    _mute_mic: Callable[[], None] | None = None
    _unmute_mic: Callable[[], None] | None = None
