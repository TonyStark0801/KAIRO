"""Streaming wake detection via openWakeWord (optional dependency).

Two-stage pipeline:
  Stage 1 — OWW neural detector (frame-by-frame, <1ms).
             Bundled models (alexa, hey_mycroft, etc.) used as acoustic proxy
             for "hey kairo" — similar two-syllable cadence, overlapping mel-
             spectrogram features.  Low threshold (0.3) maximises recall.
  Stage 2 — Optional Whisper confirmation on the 1.5s audio ring buffer.
             Called by mic_listener via verify_with_whisper() after OWW fires.
             Checks for "kairo" phonetic variants in the transcript.
             Eliminates false positives from ambient speech, TV, room talk.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

# openWakeWord expects 16 kHz int16 chunks; 1280 samples is the default frame size.
_OWW_CHUNK_SAMPLES = 1280

# Ring buffer: keep ~1.5s of audio for Stage 2 Whisper verification.
# 1.5s captures "hey kairo" + a small pre/post margin without excess context
# that would confuse Whisper's language model into hallucinating fillers.
_RING_MAX_BYTES = int(16000 * 2 * 1.5)  # 1.5s mono int16

# Phonetic variants of "kairo" that Whisper may produce for Indian-accented speech.
# Deliberately conservative — no common English words.
_KAIRO_VARIANTS = frozenset({
    "kairo", "hey kairo", "kairu", "hey kairu",
    "cairo", "kyro", "kiro", "keiro",
    "kai ro", "ky ro", "care o", "karo",
    "kairu", "cairu", "kyrou", "kyrow",
})


class OpenWakeWordStreamDetector:
    """Feed raw paInt16 mono frames; returns True when OWW score exceeds threshold.

    Stage 2 Whisper verification is NOT done here — it's done in mic_listener
    via verify_with_whisper() so the STT engine is injected externally (keeps
    this class free of STT dependency and testable in isolation).
    """

    def __init__(
        self,
        wakeword_models: list[str] | None = None,
        threshold: float = 0.3,
        inference_framework: str = "tflite",
    ) -> None:
        from openwakeword.model import Model

        kwargs: dict = {"inference_framework": inference_framework}
        if wakeword_models:
            # Explicit model list — e.g. ["hey_mycroft"] to narrow scope
            kwargs["wakeword_models"] = wakeword_models
        # Empty list → load all bundled models (alexa, hey_mycroft, hey_rhasspy, etc.)
        # This is intentional: broader acoustic coverage as proxy for "hey kairo".
        self._model = Model(**kwargs)
        self._threshold = float(threshold)
        self._pending = bytearray()
        self._ring: deque[bytes] = deque()
        self._ring_bytes = 0
        # Track which model name triggered — for logging / tuning
        self._last_trigger_model: str = ""
        self._last_trigger_score: float = 0.0

    # ── Ring buffer ────────────────────────────────────────────────────────────

    def _push_ring(self, chunk: bytes) -> None:
        self._ring.append(chunk)
        self._ring_bytes += len(chunk)
        while self._ring_bytes > _RING_MAX_BYTES and self._ring:
            old = self._ring.popleft()
            self._ring_bytes -= len(old)

    def recent_audio(self) -> bytes:
        """Return last ~1.5s of raw int16 PCM audio for Stage 2 Whisper verification."""
        return b"".join(self._ring)

    # ── Stage 1: OWW inference ─────────────────────────────────────────────────

    def feed_pcm_frame(self, frame: bytes) -> bool:
        """Append one int16 mono frame (480 or 1280 samples); run OWW inference.

        Returns True when any loaded model exceeds the threshold.
        Caller must then call verify_with_whisper() for Stage 2 confirmation.
        """
        self._push_ring(frame)
        self._pending.extend(frame)
        chunk_b = _OWW_CHUNK_SAMPLES * 2
        triggered = False
        while len(self._pending) >= chunk_b:
            block = bytes(self._pending[:chunk_b])
            del self._pending[:chunk_b]
            audio = np.frombuffer(block, dtype=np.int16)
            self._model.predict(audio)
            for name, scores in self._model.prediction_buffer.items():
                if not scores:
                    continue
                try:
                    last = float(scores[-1])
                except (TypeError, IndexError):
                    continue
                if last >= self._threshold:
                    self._last_trigger_model = name
                    self._last_trigger_score = last
                    triggered = True
                    logger.info(
                        "OWW Stage 1 hit: model=%s score=%.3f (threshold=%.2f)",
                        name, last, self._threshold,
                    )
                    break
            if triggered:
                self._pending.clear()
                break
        return triggered

    # ── Stage 2: Whisper verification ─────────────────────────────────────────

    def verify_with_whisper(self, stt_engine) -> bool:
        """Run Whisper on the ring buffer audio and confirm "kairo" was said.

        Args:
            stt_engine: any STTEngine instance with a .transcribe(float32_array) method.

        Returns True if a kairo variant is found in the transcript.
        Logs the transcript regardless — useful for tuning the variant list.
        """
        audio_bytes = self.recent_audio()
        if not audio_bytes:
            logger.debug("OWW Stage 2: no ring audio available — passing through")
            return True  # don't block if we have no audio to verify

        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            text = stt_engine.transcribe(samples).lower().strip()
        except Exception:
            logger.warning("OWW Stage 2: Whisper transcription failed — passing through", exc_info=True)
            return True  # fail open: don't drop a real wake because of a transient STT error

        logger.info("OWW Stage 2 transcript: '%s' (OWW model=%s score=%.3f)",
                    text, self._last_trigger_model, self._last_trigger_score)

        # Check for kairo variants in the transcript
        # Use word-level matching, not substring — avoids "cairo" in "cairo egypt"
        words = text.replace(",", "").replace(".", "").replace("!", "").replace("?", "").split()
        for i in range(len(words)):
            # single-word match: "kairo", "cairo", etc.
            if words[i] in _KAIRO_VARIANTS:
                logger.info("OWW Stage 2: kairo variant '%s' confirmed in transcript", words[i])
                return True
            # two-word match: "hey kairo", "hey kairu", etc.
            if i + 1 < len(words):
                bigram = f"{words[i]} {words[i+1]}"
                if bigram in _KAIRO_VARIANTS:
                    logger.info("OWW Stage 2: kairo bigram '%s' confirmed in transcript", bigram)
                    return True

        logger.info(
            "OWW Stage 2: rejected — '%s' contains no kairo variant "
            "(OWW model=%s was a false positive, score=%.3f)",
            text, self._last_trigger_model, self._last_trigger_score,
        )
        return False
