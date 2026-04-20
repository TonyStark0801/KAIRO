"""Speaker verification — compares voice against enrolled voiceprint.

Uses resemblyzer to create/compare speaker embeddings.
Designed to be called from the mic listener on wake word audio.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_SIMILARITY_THRESHOLD = 0.75
_SAMPLE_RATE = 16000


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


class VoiceVerifier:
    def __init__(
        self,
        embedding_path: str = "~/.kairo/voice_embedding.npy",
        threshold: float = _SIMILARITY_THRESHOLD,
    ) -> None:
        self._embedding_path = Path(embedding_path).expanduser()
        self._threshold = threshold
        self._enrolled_embedding: np.ndarray | None = None
        self._encoder = None
        self._healthy = False

    @property
    def healthy(self) -> bool:
        return self._healthy

    def initialize(self) -> bool:
        if not self._embedding_path.exists():
            logger.warning(
                "No enrolled voice at %s — run 'kairo-enroll-voice'",
                self._embedding_path,
            )
            return False

        try:
            self._enrolled_embedding = np.load(str(self._embedding_path))
        except Exception:
            logger.exception("Failed to load voice embedding")
            return False

        try:
            from resemblyzer import VoiceEncoder
            self._encoder = VoiceEncoder()
        except ImportError:
            logger.error("resemblyzer not installed — run: pip install resemblyzer")
            return False
        except Exception:
            logger.exception("Failed to initialize VoiceEncoder")
            return False

        self._healthy = True
        logger.info("Voice verifier ready (threshold=%.2f)", self._threshold)
        return True

    def verify(self, audio_bytes: bytes) -> tuple[bool, float]:
        """Verify speaker from raw int16 audio bytes.

        Returns (is_owner, similarity_score).
        """
        if not self._healthy or self._encoder is None or self._enrolled_embedding is None:
            return False, 0.0

        try:
            samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            if len(samples) < _SAMPLE_RATE * 0.5:
                logger.debug("Audio too short for voice verification (%.1fs)", len(samples) / _SAMPLE_RATE)
                return False, 0.0

            from resemblyzer import preprocess_wav
            processed = preprocess_wav(samples, source_sr=_SAMPLE_RATE)
            embedding = self._encoder.embed_utterance(processed)

            similarity = _cosine_similarity(embedding, self._enrolled_embedding)
            is_owner = similarity >= self._threshold
            logger.info("Voice verification: similarity=%.3f owner=%s", similarity, is_owner)
            return is_owner, similarity

        except Exception:
            logger.exception("Voice verification failed")
            return False, 0.0


def enroll_voice_cli() -> None:
    """One-time voice enrollment — run as 'kairo-enroll-voice'."""
    import time

    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError:
        print("resemblyzer not installed. Run: pip install resemblyzer")
        return

    try:
        import pyaudio
    except ImportError:
        print("pyaudio not installed. Run: pip install pyaudio")
        return

    dest = Path("~/.kairo/voice_embedding.npy").expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)

    encoder = VoiceEncoder()
    pa = pyaudio.PyAudio()

    sample_rate = 16000
    chunk = 1024
    record_seconds = 8
    captures_needed = 3

    input_device = None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) > 0:
            input_device = i
            print(f"   Using mic: {info['name']}")
            break

    if input_device is None:
        print("No microphone found!")
        pa.terminate()
        return

    try:
        stream = pa.open(
            format=pyaudio.paInt16, channels=1,
            rate=sample_rate, input=True,
            frames_per_buffer=chunk,
            input_device_index=input_device,
        )
    except Exception as e:
        print(f"Cannot open microphone: {e}")
        pa.terminate()
        return

    embeddings: list[np.ndarray] = []

    prompts = [
        "Hey Kairo, can you check what I have on my calendar today and play some focus music while I work on this project.",
        "Kairo, open IntelliJ for the CodeJam project, lower the volume a bit, and tell me what file I was working on last time.",
        "Good morning Kairo, I need you to search for some lofi beats on YouTube and set the volume to about forty percent.",
    ]

    print(f"\n🎤 Voice Enrollment for Kairo")
    print(f"   You'll record {captures_needed} samples of your voice.")
    print(f"   Read the text shown on screen naturally — like you're talking to Kairo.")
    print(f"   Each recording is {record_seconds} seconds.\n")

    for i in range(captures_needed):
        print(f"   📖 Read this aloud:")
        print(f"   \"{prompts[i]}\"\n")
        input(f"   Press ENTER when ready to record {i + 1}/{captures_needed}...")
        print(f"   🔴 Recording... speak now!")

        frames = []
        for _ in range(0, int(sample_rate / chunk * record_seconds)):
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)

        audio_bytes = b"".join(frames)
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        processed = preprocess_wav(samples, source_sr=sample_rate)
        embedding = encoder.embed_utterance(processed)
        embeddings.append(embedding)
        print(f"   ✓ Sample {i + 1} captured\n")

    stream.close()
    pa.terminate()

    if len(embeddings) == captures_needed:
        avg = np.mean(embeddings, axis=0)
        np.save(str(dest), avg)
        print(f"   ✅ Voice enrollment saved to {dest}")
        print(f"   Kairo will now recognize your voice.\n")
    else:
        print("   ❌ Enrollment cancelled — not enough samples")
