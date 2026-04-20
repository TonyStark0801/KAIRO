"""Streaming wake detection via openWakeWord (optional dependency)."""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

# openWakeWord expects 16 kHz int16 chunks; 1280 samples is the default frame size.
_OWW_CHUNK_SAMPLES = 1280
_RING_MAX_BYTES = 16000 * 2 * 2  # ~2s mono int16 for voice verify / debug


class OpenWakeWordStreamDetector:
    """Feed raw paInt16 mono frames; returns True when score exceeds threshold."""

    def __init__(
        self,
        wakeword_models: list[str] | None = None,
        threshold: float = 0.5,
        inference_framework: str = "tflite",
    ) -> None:
        from openwakeword.model import Model

        kwargs: dict = {"inference_framework": inference_framework}
        if wakeword_models:
            kwargs["wakeword_models"] = wakeword_models
        self._model = Model(**kwargs)
        self._threshold = float(threshold)
        self._pending = bytearray()
        self._ring: deque[bytes] = deque()
        self._ring_bytes = 0

    def _push_ring(self, chunk: bytes) -> None:
        self._ring.append(chunk)
        self._ring_bytes += len(chunk)
        while self._ring_bytes > _RING_MAX_BYTES and self._ring:
            old = self._ring.popleft()
            self._ring_bytes -= len(old)

    def recent_audio(self) -> bytes:
        return b"".join(self._ring)

    def feed_pcm_frame(self, frame: bytes) -> bool:
        """Append one pyaudio int16 mono frame; run inference when buffer is full."""
        self._push_ring(frame)
        self._pending.extend(frame)
        chunk_b = _OWW_CHUNK_SAMPLES * 2
        triggered = False
        while len(self._pending) >= chunk_b:
            block = bytes(self._pending[:chunk_b])
            del self._pending[:chunk_b]
            audio = np.frombuffer(block, dtype=np.int16)
            self._model.predict(audio)
            for _name, scores in self._model.prediction_buffer.items():
                if not scores:
                    continue
                try:
                    last = scores[-1]
                except (TypeError, IndexError):
                    continue
                if float(last) >= self._threshold:
                    triggered = True
                    logger.info("openWakeWord hit model=%s score=%s", _name, last)
                    break
            if triggered:
                self._pending.clear()
                break
        return triggered
