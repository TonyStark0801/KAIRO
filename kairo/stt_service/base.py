"""Abstract STT engine interface — swap speech-to-text backends without touching callers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np


class TranscriptMeta(NamedTuple):
    text: str
    speech_prob: float  # 1.0 = definitely speech, 0.0 = definitely not speech


class STTEngine(ABC):
    @abstractmethod
    def initialize(self) -> bool:
        """Load model. Return True if ready."""

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe float32 audio array at 16kHz. Returns cleaned text."""

    def transcribe_with_meta(self, audio: np.ndarray) -> TranscriptMeta:
        """Transcribe and return text + speech probability.

        Default: calls transcribe() with speech_prob=1.0.
        Override in engines that expose no_speech_prob (faster-whisper, mlx-whisper).
        speech_prob = 1.0 - no_speech_prob averaged across segments.
        """
        return TranscriptMeta(text=self.transcribe(audio), speech_prob=1.0)
