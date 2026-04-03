"""Speech-to-text via pywhispercpp."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000


class Transcriber:
    def __init__(self, model_name: str = "base.en") -> None:
        self._model_name = model_name
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
        self._healthy = False

    @property
    def healthy(self) -> bool:
        return self._healthy

    def initialize(self) -> bool:
        try:
            from pywhispercpp.model import Model

            self._model = Model(self._model_name)
            self._healthy = True
            return True
        except Exception:
            logger.exception("Failed to initialize Whisper model %s", self._model_name)
            return False

    def _transcribe_sync(self, audio_bytes: bytes) -> str:
        if self._model is None:
            return ""
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments = self._model.transcribe(samples)
        return " ".join(seg.text for seg in segments).strip()

    async def transcribe(self, audio_bytes: bytes) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._transcribe_sync, audio_bytes)
