"""Energy-based Voice Activity Detection."""

from __future__ import annotations

import logging
import struct
import time

import numpy as np

logger = logging.getLogger(__name__)

_FRAME_DURATION_MS = 30
_SAMPLE_RATE = 16000
_FRAME_SIZE = int(_SAMPLE_RATE * _FRAME_DURATION_MS / 1000)
_ONSET_THRESHOLD = 500
_OFFSET_THRESHOLD = 300
_ONSET_DURATION = 0.3
_OFFSET_DURATION = 0.7


class VoiceActivityDetector:
    def __init__(
        self,
        sample_rate: int = _SAMPLE_RATE,
        onset_threshold: float = _ONSET_THRESHOLD,
        offset_threshold: float = _OFFSET_THRESHOLD,
    ) -> None:
        self._sample_rate = sample_rate
        self._onset_threshold = onset_threshold
        self._offset_threshold = offset_threshold
        self._is_speaking = False
        self._onset_start: float | None = None
        self._offset_start: float | None = None
        self._audio_buffer: list[bytes] = []

    def reset(self) -> None:
        self._is_speaking = False
        self._onset_start = None
        self._offset_start = None
        self._audio_buffer.clear()

    def process_frame(self, frame: bytes) -> bytes | None:
        energy = self._compute_energy(frame)
        now = time.monotonic()

        if not self._is_speaking:
            if energy > self._onset_threshold:
                if self._onset_start is None:
                    self._onset_start = now
                elif now - self._onset_start >= _ONSET_DURATION:
                    self._is_speaking = True
                    self._onset_start = None
                    self._offset_start = None
                    self._audio_buffer.append(frame)
            else:
                self._onset_start = None
            return None

        self._audio_buffer.append(frame)

        if energy < self._offset_threshold:
            if self._offset_start is None:
                self._offset_start = now
            elif now - self._offset_start >= _OFFSET_DURATION:
                self._is_speaking = False
                self._offset_start = None
                result = b"".join(self._audio_buffer)
                self._audio_buffer.clear()
                return result
        else:
            self._offset_start = None

        return None

    @staticmethod
    def _compute_energy(frame: bytes) -> float:
        count = len(frame) // 2
        if count == 0:
            return 0.0
        samples = struct.unpack(f"<{count}h", frame)
        arr = np.array(samples, dtype=np.float32)
        return float(np.sqrt(np.mean(arr**2)))
