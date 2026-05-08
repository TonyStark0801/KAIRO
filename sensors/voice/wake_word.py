"""Unified mic listener — handles wake word detection AND command processing.

Single mic thread avoids the macOS issue of two threads fighting for the mic.
Routes audio based on FSM state:
  - WAKE_PENDING: check for wake phrases
  - ACTIVE_SESSION: transcribe commands and publish VoiceTranscriptEvent
"""

from __future__ import annotations

import asyncio
import logging
import struct
import threading
import time
from enum import Enum, auto

import numpy as np

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_FRAME_SIZE = 480
_ENERGY_NORMAL = 250
_ENERGY_MEDIA = 1500
_SILENCE_AFTER_SPEECH = 0.6
_MAX_PHRASE_SECONDS = 5.0

_WAKE_KEYWORDS = [
    "kairo", "hey kairo", "cairo", "kyro", "caro", "kai ro", "ky ro",
]


class MicMode(Enum):
    IDLE = auto()
    WAKE_WORD = auto()
    COMMAND = auto()


class UnifiedMicListener:
    """Single always-on mic thread. Mode is switched externally by the daemon
    based on FSM state changes."""

    def __init__(
        self,
        event_bus,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._bus = event_bus
        self._loop = loop
        self._transcriber = None
        self._healthy = False
        self._mode = MicMode.IDLE
        self._mode_lock = threading.Lock()
        self._session_id = ""
        self._media_playing = False

    @property
    def healthy(self) -> bool:
        return self._healthy

    @property
    def mode(self) -> MicMode:
        with self._mode_lock:
            return self._mode

    def set_mode(self, mode: MicMode, session_id: str = "") -> None:
        with self._mode_lock:
            old = self._mode
            self._mode = mode
            self._session_id = session_id
        if old != mode:
            logger.info("Mic mode: %s → %s", old.name, mode.name)

    def set_media_playing(self, playing: bool) -> None:
        self._media_playing = playing
        threshold = _ENERGY_MEDIA if playing else _ENERGY_NORMAL
        logger.info("Mic energy threshold: %d (media=%s)", threshold, playing)

    @property
    def _energy_threshold(self) -> float:
        return _ENERGY_MEDIA if self._media_playing else _ENERGY_NORMAL

    def initialize(self) -> bool:
        try:
            from sensors.voice.transcriber import Transcriber
            self._transcriber = Transcriber()
            if not self._transcriber.initialize():
                logger.error("Unified mic: Whisper init failed")
                return False
            self._healthy = True
            return True
        except Exception:
            logger.exception("Unified mic init failed")
            return False

    def run(self, stop_event: threading.Event) -> None:
        if not self._healthy:
            return
        try:
            import pyaudio
        except ImportError:
            logger.error("pyaudio not installed — mic listener disabled")
            return

        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(
                format=pyaudio.paInt16, channels=1,
                rate=_SAMPLE_RATE, input=True,
                frames_per_buffer=_FRAME_SIZE,
            )
        except Exception:
            logger.exception("Unified mic: cannot open mic")
            pa.terminate()
            return

        logger.info("Unified mic listener started")
        try:
            while not stop_event.is_set():
                current_mode = self.mode
                if current_mode == MicMode.IDLE:
                    time.sleep(0.1)
                    continue

                try:
                    frame = stream.read(_FRAME_SIZE, exception_on_overflow=False)
                except Exception:
                    time.sleep(0.05)
                    continue

                energy = _rms_energy(frame)
                threshold = self._energy_threshold
                if energy < threshold:
                    continue

                audio_chunks = [frame]
                silence_start = None
                capture_start = time.monotonic()
                silence_threshold = threshold * 0.4

                while not stop_event.is_set():
                    if time.monotonic() - capture_start > _MAX_PHRASE_SECONDS:
                        break
                    try:
                        frame = stream.read(_FRAME_SIZE, exception_on_overflow=False)
                    except Exception:
                        break
                    audio_chunks.append(frame)
                    e = _rms_energy(frame)
                    if e < silence_threshold:
                        if silence_start is None:
                            silence_start = time.monotonic()
                        elif time.monotonic() - silence_start >= _SILENCE_AFTER_SPEECH:
                            break
                    else:
                        silence_start = None

                audio_bytes = b"".join(audio_chunks)
                current_mode = self.mode

                if current_mode == MicMode.WAKE_WORD:
                    self._handle_wake_word(audio_bytes)
                elif current_mode == MicMode.COMMAND:
                    self._handle_command(audio_bytes)
        finally:
            stream.close()
            pa.terminate()
            logger.info("Unified mic listener stopped")

    def _transcribe(self, audio_bytes: bytes) -> str:
        try:
            samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            segments = self._transcriber._model.transcribe(samples)
            text = " ".join(seg.text for seg in segments).strip().lower()
            return text.strip(" .,!?\"'-")
        except Exception:
            logger.exception("Transcription error")
            return ""

    def _handle_wake_word(self, audio_bytes: bytes) -> None:
        text = self._transcribe(audio_bytes)
        if not text or text.startswith("[") or text.startswith("("):
            return

        for keyword in _WAKE_KEYWORDS:
            if keyword in text:
                logger.info("Wake keyword matched: '%s' in '%s'", keyword, text)
                from runtime.event_bus import GestureEvent, GestureType
                event = GestureEvent(
                    type=GestureType.WAKE_WORD_DETECTED,
                    timestamp=time.time(),
                )
                self._loop.call_soon_threadsafe(
                    asyncio.ensure_future, self._bus.publish(event)
                )
                time.sleep(2.0)
                return

    def _handle_command(self, audio_bytes: bytes) -> None:
        text = self._transcribe(audio_bytes)
        if not text or text.startswith("[") or text.startswith("("):
            return

        from sensors.voice.normalizer import normalize
        text = normalize(text)
        if not text:
            return

        logger.info("Voice command: '%s'", text)

        with self._mode_lock:
            session_id = self._session_id

        from runtime.event_bus import VoiceTranscriptEvent
        self._loop.call_soon_threadsafe(
            asyncio.ensure_future,
            self._bus.publish(
                VoiceTranscriptEvent(text=text, confidence=1.0, session_id=session_id)
            ),
        )


def _rms_energy(frame: bytes) -> float:
    count = len(frame) // 2
    if count == 0:
        return 0.0
    samples = struct.unpack(f"<{count}h", frame)
    arr = np.array(samples, dtype=np.float32)
    return float(np.sqrt(np.mean(arr ** 2)))
