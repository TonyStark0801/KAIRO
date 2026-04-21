"""Unified mic listener — single mic thread with mode-based routing.

Modes:
  IDLE       — mic open but not processing (during speech or shutdown)
  WAKE_WORD  — listening for wake keywords, publishes GestureEvent
  COMMAND    — transcribes full speech, publishes VoiceTranscriptEvent
"""

from __future__ import annotations

import asyncio
import logging
import struct
import threading
import time
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from runtime.event_bus import EventBus
    from sensors.voice.voice_verifier import VoiceVerifier
    from sensors.wake.openwakeword_stream import OpenWakeWordStreamDetector
    from stt_service.base import STTEngine

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_FRAME_SIZE = 480
_ENERGY_NORMAL = 700    # filters ceiling fan / AC hum (~300-600 RMS)
_ENERGY_MEDIA = 1500    # raised threshold when media is playing
_ENERGY_BARGE_IN = 3932 # 0.12 * 32768 — requires intentional loud command to cut through earphone playback
_SILENCE_AFTER_SPEECH = 0.5  # longer silence gate — reduces spurious captures
_MAX_PHRASE_SECONDS = 4.0
_POST_COMMAND_COOLDOWN = 0.8


class MicMode(Enum):
    IDLE = auto()
    WAKE_WORD = auto()
    COMMAND = auto()


class MicListener:
    """Single always-on mic thread. Mode switched externally by the daemon."""

    def __init__(
        self,
        stt_engine: STTEngine,
        event_bus: EventBus,
        loop: asyncio.AbstractEventLoop,
        wake_words: list[str] | None = None,
        voice_verifier: VoiceVerifier | None = None,
        openwakeword_detector: OpenWakeWordStreamDetector | None = None,
        wake_stt_engine: STTEngine | None = None,
    ) -> None:
        # Two engines: tiny/fast for wake-word matching, full for commands.
        # If wake_stt_engine is None we fall back to stt_engine (single-model mode).
        self._stt = stt_engine
        self._wake_stt = wake_stt_engine or stt_engine
        self._bus = event_bus
        self._loop = loop
        self._oww_detector = openwakeword_detector
        self._wake_words = wake_words or [
            # Canonical
            "kairo", "hey kairo",
            # Indian English phonetic variants — retroflex r, different stress
            "cairo", "kyro", "kiro", "keiro", "kaero",
            "kai ro", "ky ro", "ki ro",
            # Whisper transcription variants (NOT common English words)
            "karo", "kyrow", "cayro", "care o",
            # Deliberately removed: "hero", "micro", "tyro", "high row", "caro", "k row"
            # These are too common in song lyrics / ambient speech → high false-positive rate
        ]
        self._voice_verifier = voice_verifier
        self._mode = MicMode.IDLE
        self._mode_lock = threading.Lock()
        self._session_id = ""
        self._media_playing = False
        self._is_speaking = False  # True while KAIRO's TTS is playing through earphones
        self._interrupt_callback = None
        self._interrupt_words = {"stop", "shut up", "quiet", "enough", "never mind", "nevermind", "ok stop", "stop it"}

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

    def set_interrupt_callback(self, callback) -> None:
        """Set a callback to be called when user says an interrupt word."""
        self._interrupt_callback = callback

    def set_media_playing(self, playing: bool) -> None:
        self._media_playing = playing
        logger.info("Mic energy threshold: %d (media=%s, speaking=%s)",
                    self._energy_threshold, playing, self._is_speaking)

    def set_speaking(self, speaking: bool) -> None:
        """Raise mic threshold while KAIRO is talking through earphones.

        Earphone feedback means the mic hears KAIRO's voice directly at low volume.
        At normal threshold (700 RMS) that bleeds through as ghost commands.
        Barge-in requires the user to speak at ~12% of int16 max — loud enough
        to be intentional, quiet enough to not need shouting.
        """
        self._is_speaking = speaking
        logger.info("Mic energy threshold: %d (speaking=%s)", self._energy_threshold, speaking)

    @property
    def _energy_threshold(self) -> float:
        if self._is_speaking:
            return _ENERGY_BARGE_IN
        return _ENERGY_MEDIA if self._media_playing else _ENERGY_NORMAL

    def run(self, stop_event: threading.Event) -> None:
        try:
            import pyaudio
        except ImportError:
            logger.error("pyaudio not installed — mic listener disabled")
            return

        from sensors.voice.vad import VoiceActivityDetector

        # VAD for COMMAND mode — proper onset/offset, no false triggers on transient noise.
        # Thresholds are absolute RMS values of int16 PCM (range 0–32768).
        _vad = VoiceActivityDetector(
            sample_rate=_SAMPLE_RATE,
            onset_threshold=400,   # sustained speech onset
            offset_threshold=180,  # back-to-silence offset
        )

        pa = pyaudio.PyAudio()
        builtin_idx = self._find_builtin_mic(pa)
        try:
            kwargs = dict(
                format=pyaudio.paInt16, channels=1,
                rate=_SAMPLE_RATE, input=True,
                frames_per_buffer=_FRAME_SIZE,
            )
            if builtin_idx is not None:
                kwargs["input_device_index"] = builtin_idx
            stream = pa.open(**kwargs)
        except Exception:
            logger.exception("Cannot open mic stream")
            pa.terminate()
            return

        logger.info("Mic listener started (vad=enabled, wake_words=%d)", len(self._wake_words))
        _min_bytes = int(_SAMPLE_RATE * 2 * 0.3)  # 0.3 s minimum utterance

        try:
            while not stop_event.is_set():
                current_mode = self.mode

                if current_mode == MicMode.IDLE:
                    _vad.reset()
                    time.sleep(0.1)
                    continue

                # openWakeWord streaming — frame-by-frame, no energy gate
                if current_mode == MicMode.WAKE_WORD and self._oww_detector is not None:
                    try:
                        oww_frame = stream.read(_FRAME_SIZE, exception_on_overflow=False)
                    except Exception:
                        time.sleep(0.02)
                        continue
                    if self._oww_detector.feed_pcm_frame(oww_frame):
                        self._publish_openwakeword_wake()
                    continue

                try:
                    frame = stream.read(_FRAME_SIZE, exception_on_overflow=False)
                except Exception:
                    time.sleep(0.05)
                    continue

                current_mode = self.mode  # re-read after blocking mic read

                if current_mode == MicMode.COMMAND:
                    # ── VAD-gated command capture ──────────────────────────────
                    # Feed every frame; VAD handles onset+offset with hysteresis.
                    # process_frame() returns None until utterance ends, then
                    # returns the full accumulated audio bytes.
                    audio = _vad.process_frame(frame)
                    if audio is not None:
                        if len(audio) >= _min_bytes:
                            self._handle_command(audio)
                        _vad.reset()

                elif current_mode == MicMode.WAKE_WORD:
                    # ── STT-keyword wake: energy gate + manual silence loop ────
                    # Keep existing approach — wake phrase is a short burst and
                    # we pass the whole chunk to Whisper for keyword matching.
                    _vad.reset()
                    energy = _rms_energy(frame)
                    if energy < self._energy_threshold:
                        continue

                    audio_chunks: list[bytes] = [frame]
                    silence_start: float | None = None
                    capture_start = time.monotonic()
                    silence_threshold = self._energy_threshold * 0.4

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
                    if len(audio_bytes) >= _min_bytes:
                        self._handle_wake_word(audio_bytes)

        finally:
            stream.close()
            pa.terminate()
            logger.info("Mic listener stopped")

    @staticmethod
    def _find_builtin_mic(pa) -> int | None:
        """Find the built-in Mac microphone, ignoring AirPods/Bluetooth mics."""
        builtin_keywords = ("macbook", "built-in", "internal")
        best = None
        for i in range(pa.get_device_count()):
            try:
                info = pa.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) < 1:
                    continue
                name = info.get("name", "").lower()
                logger.info("Mic device %d: %s (channels=%d)", i, info.get("name"), info["maxInputChannels"])
                if any(kw in name for kw in builtin_keywords):
                    best = i
            except Exception:
                continue
        if best is not None:
            logger.info("Using built-in mic (device %d)", best)
        else:
            logger.info("Built-in mic not found — using system default")
        return best

    def _transcribe(self, audio_bytes: bytes, *, wake: bool = False) -> str:
        engine = self._wake_stt if wake else self._stt
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return engine.transcribe(samples)

    def _transcribe_meta(self, audio_bytes: bytes, *, wake: bool = False):
        """Returns TranscriptMeta(text, speech_prob). Use this over _transcribe() when filtering matters."""
        engine = self._wake_stt if wake else self._stt
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return engine.transcribe_with_meta(samples)

    def _publish_openwakeword_wake(self) -> None:
        from runtime.event_bus import GestureEvent, GestureType

        now = time.time()
        wake_event = GestureEvent(type=GestureType.WAKE_WORD_DETECTED, timestamp=now)
        self._loop.call_soon_threadsafe(asyncio.ensure_future, self._bus.publish(wake_event))

        if self._voice_verifier and self._voice_verifier.healthy and self._oww_detector:
            audio = self._oww_detector.recent_audio()
            if audio:
                is_owner, score = self._voice_verifier.verify(audio)
                if is_owner:
                    voice_event = GestureEvent(type=GestureType.VOICE_VERIFIED, timestamp=now)
                    self._loop.call_soon_threadsafe(asyncio.ensure_future, self._bus.publish(voice_event))

    def _handle_wake_word(self, audio_bytes: bytes) -> None:
        meta = self._transcribe_meta(audio_bytes, wake=True)
        text = meta.text
        # Wake detection: low bar on speech_prob (0.25) — missing a wake is worse than a false positive.
        # But if Whisper is very confident it's NOT speech, skip.
        if meta.speech_prob < 0.25:
            logger.debug("Wake-word audio rejected (speech_prob=%.2f)", meta.speech_prob)
            return
        logger.info("Wake-word heard: '%s' (speech_prob=%.2f)", text, meta.speech_prob)
        if not text or text.startswith("[") or text.startswith("("):
            return

        # Repetition hallucination filter — "er, er, er er er..." from fan/hum.
        # Strip punctuation before uniqueness check — "er," and "er" are the same token.
        _raw_words = text.split()
        _words = [w.strip(".,;:!?-") for w in _raw_words if w.strip(".,;:!?-")]
        if len(_words) >= 4:
            _unique = set(_words)
            if len(_unique) <= max(1, len(_words) * 0.25):
                logger.debug("Wake-word audio: repetition hallucination (%d unique / %d words), dropping", len(_unique), len(_words))
                return

        text_lower = text.lower().strip()
        if text_lower in self._interrupt_words or any(w in text_lower for w in self._interrupt_words):
            logger.info("Interrupt detected: '%s'", text)
            if self._interrupt_callback:
                self._interrupt_callback()
            return

        matched_keyword = self._find_wake_keyword(text_lower)
        if matched_keyword:
            logger.info("Wake word matched: '%s' in '%s'", matched_keyword, text)
            from runtime.event_bus import GestureEvent, GestureType

            now = time.time()

            wake_event = GestureEvent(type=GestureType.WAKE_WORD_DETECTED, timestamp=now)
            self._loop.call_soon_threadsafe(
                asyncio.ensure_future, self._bus.publish(wake_event)
            )

            if self._voice_verifier and self._voice_verifier.healthy:
                is_owner, score = self._voice_verifier.verify(audio_bytes)
                if is_owner:
                    voice_event = GestureEvent(type=GestureType.VOICE_VERIFIED, timestamp=now)
                    self._loop.call_soon_threadsafe(
                        asyncio.ensure_future, self._bus.publish(voice_event)
                    )

            # Only extract inline command if the keyword appears literally in the text.
            # Fuzzy matches mean the whole utterance WAS the wake phrase — no remainder.
            if matched_keyword in text:
                remainder = text.split(matched_keyword, 1)[-1].strip()
            else:
                remainder = ""
            remainder = remainder.strip(" .,!?\"'-")
            if remainder and len(remainder) > 2:
                logger.info("Inline command after wake word: '%s'", remainder)
                from sensors.voice.normalizer import normalize
                remainder = normalize(remainder)
                if remainder:
                    with self._mode_lock:
                        session_id = self._session_id
                    from runtime.event_bus import VoiceTranscriptEvent
                    time.sleep(0.3)
                    self._loop.call_soon_threadsafe(
                        asyncio.ensure_future,
                        self._bus.publish(
                            VoiceTranscriptEvent(text=remainder, confidence=1.0, session_id=session_id)
                        ),
                    )
            return

    def _find_wake_keyword(self, text: str) -> str:
        """Return the matching wake keyword or '' if none match.

        Strategy (in priority order):
        1. Exact substring — fast, zero false positives.
        2. Fuzzy per-ngram — catches accent variants without an exhaustive list.
           Uses difflib.SequenceMatcher (stdlib, no extra deps).
           Threshold 0.82: "kairu" ↔ "kairo" = 0.89 ✓, "cairo" ↔ "kairo" = 0.80 ✓
                           "hello" ↔ "kairo" = 0.20 ✗, "fire" ↔ "kairo" = 0.22 ✗
        """
        import difflib

        text_words = text.split()

        for keyword in self._wake_words:
            # ── Pass 1: exact substring ──────────────────────────────────────
            if keyword in text:
                return keyword

            # ── Pass 2: fuzzy n-gram window ─────────────────────────────────
            kw_words = keyword.split()
            n = len(kw_words)
            for i in range(max(1, len(text_words) - n + 1)):
                ngram = " ".join(text_words[i : i + n])
                ratio = difflib.SequenceMatcher(None, ngram, keyword).ratio()
                if ratio >= 0.82:
                    logger.debug("Fuzzy wake match: '%s' ~ '%s' (%.2f)", ngram, keyword, ratio)
                    return keyword

        return ""

    def _handle_command(self, audio_bytes: bytes) -> None:
        meta = self._transcribe_meta(audio_bytes, wake=False)
        text = meta.text

        # During TTS playback: raise the bar significantly.
        # KAIRO's own voice bleeds through earphones at low speech_prob (~0.45-0.55).
        # A real barge-in command from the user will have higher speech_prob AND volume.
        # This filters TTS echoes without blocking intentional interruptions.
        if self._is_speaking and meta.speech_prob < 0.70:
            logger.debug("Command rejected during TTS (speech_prob=%.2f): '%s'", meta.speech_prob, text)
            return

        # Hard speech probability filter — if Whisper is > 55% sure it's NOT speech,
        # drop it. This kills ambient conversation, background TV, half-heard words.
        if meta.speech_prob < 0.45:
            logger.debug("Command audio rejected (speech_prob=%.2f): '%s'", meta.speech_prob, text)
            return

        if not text or text.startswith("[") or text.startswith("("):
            return

        # Interrupt check in COMMAND mode: if KAIRO is speaking and user says a stop word,
        # fire the interrupt callback instead of routing to the LLM.
        # (In WAKE_WORD mode this is already handled in _handle_wake_word.)
        if self._is_speaking:
            text_lower_check = text.lower().strip().rstrip(".,!?")
            if text_lower_check in self._interrupt_words or any(w in text_lower_check for w in self._interrupt_words):
                logger.info("Interrupt word during command mode: '%s'", text)
                if self._interrupt_callback:
                    self._interrupt_callback()
                return

        from sensors.voice.normalizer import normalize, looks_like_music, looks_like_noise, looks_like_ambient
        raw_text = text
        text = normalize(text)
        if not text:
            return

        # Always filter music — ♪ symbols in raw text are never a real command
        if looks_like_music(raw_text) or looks_like_music(text):
            logger.debug("Filtered music/lyrics: '%s'", text)
            return

        if looks_like_noise(text):
            logger.debug("Filtered noise: '%s'", text)
            return

        # Filter ambient/conversational utterances that weren't directed at KAIRO.
        # Catches "yeah", "okay that's fine", "I don't know", "haan", etc.
        if looks_like_ambient(text):
            logger.debug("Filtered ambient speech: '%s' (speech_prob=%.2f)", text, meta.speech_prob)
            return

        logger.info("Voice command: '%s' (speech_prob=%.2f)", text, meta.speech_prob)

        with self._mode_lock:
            session_id = self._session_id

        from runtime.event_bus import VoiceTranscriptEvent
        self._loop.call_soon_threadsafe(
            asyncio.ensure_future,
            self._bus.publish(
                VoiceTranscriptEvent(text=text, confidence=meta.speech_prob, session_id=session_id)
            ),
        )

        time.sleep(_POST_COMMAND_COOLDOWN)


def _rms_energy(frame: bytes) -> float:
    count = len(frame) // 2
    if count == 0:
        return 0.0
    samples = struct.unpack(f"<{count}h", frame)
    arr = np.array(samples, dtype=np.float32)
    return float(np.sqrt(np.mean(arr ** 2)))
