"""MicListener variant: sounddevice + high-pass + peak/silence segmentation (newThing/test.py style).

Publishes the same GestureEvent / VoiceTranscriptEvent as MicListener so the daemon agent + LLM path is unchanged.
Does not support openWakeWord frame streaming; wake uses STT on each completed utterance in WAKE_WORD mode.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np

from stt_service.mic_listener import MicListener, MicMode
from stt_service.sounddevice_segmenter import run_sounddevice_mic_loop, try_import_deps

if TYPE_CHECKING:
    from runtime.event_bus import EventBus
    from sensors.voice.voice_verifier import VoiceVerifier
    from stt_service.base import STTEngine

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000


class SounddeviceMicListener(MicListener):
    def __init__(
        self,
        stt_engine: STTEngine,
        event_bus: EventBus,
        loop,
        *,
        sd_threshold: float = 0.04,
        sd_threshold_media: float | None = None,
        sd_silence_chunks: int = 10,
        sd_chunk_seconds: float = 0.1,
        sd_min_duration_seconds: float = 0.25,
        sd_device: int | None = None,
        sd_show_meter: bool = False,
        wake_words: list[str] | None = None,
        voice_verifier: VoiceVerifier | None = None,
        openwakeword_detector=None,
        wake_stt_engine: STTEngine | None = None,
    ) -> None:
        if openwakeword_detector is not None:
            logger.warning("SounddeviceMicListener ignores openWakeWord (use STT wake on segments)")
        super().__init__(
            stt_engine,
            event_bus,
            loop,
            wake_words=wake_words,
            voice_verifier=voice_verifier,
            openwakeword_detector=None,
            wake_stt_engine=wake_stt_engine,
        )
        self._sd_threshold = sd_threshold
        self._sd_threshold_media = (
            sd_threshold_media if sd_threshold_media is not None else sd_threshold * 2.2
        )
        # During TTS playback, raise threshold aggressively to block earphone bleed.
        # 0.04 (normal) → 0.15 (speaking): requires ~4x louder audio for capture.
        # A user's deliberate barge-in shout clears this; KAIRO's TTS bleed doesn't.
        self._sd_threshold_speaking = sd_threshold * 3.75
        self._sd_silence_chunks = sd_silence_chunks
        self._sd_chunk_seconds = sd_chunk_seconds
        self._sd_min_duration_seconds = sd_min_duration_seconds
        self._sd_device = sd_device
        self._sd_show_meter = sd_show_meter

    def run(self, stop_event: threading.Event) -> None:
        err = try_import_deps()
        if err:
            logger.error("sounddevice mic: %s", err)
            return

        def threshold() -> float:
            # Priority: speaking > media playing > normal
            if self._is_speaking:
                return self._sd_threshold_speaking
            return self._sd_threshold_media if self._media_playing else self._sd_threshold

        def should_process() -> bool:
            return self.mode in (MicMode.WAKE_WORD, MicMode.COMMAND)

        def on_utterance(audio_f32: np.ndarray) -> None:
            pcm = np.clip(audio_f32, -1.0, 1.0)
            samples_i16 = (pcm * 32767.0).astype(np.int16)
            audio_bytes = samples_i16.tobytes()
            mode = self.mode
            if mode == MicMode.WAKE_WORD:
                self._handle_wake_word(audio_bytes)
            elif mode == MicMode.COMMAND:
                self._handle_command(audio_bytes)

        logger.info(
            "Sounddevice mic loop (threshold=%.3f media=%.3f silence_chunks=%d)",
            self._sd_threshold,
            self._sd_threshold_media,
            self._sd_silence_chunks,
        )
        run_sounddevice_mic_loop(
            stop_event,
            sample_rate=_SAMPLE_RATE,
            chunk_seconds=self._sd_chunk_seconds,
            silence_chunks=self._sd_silence_chunks,
            get_threshold=threshold,
            show_meter=self._sd_show_meter,
            device=self._sd_device,
            min_duration_s=self._sd_min_duration_seconds,
            should_process=should_process,
            on_utterance=on_utterance,
        )
        logger.info("Sounddevice mic loop stopped")
