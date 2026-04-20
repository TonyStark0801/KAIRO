#!/usr/bin/env python3
"""Live mic test — records audio and shows what Whisper hears.

Run: .venv/bin/python -u tests/test_mic_live.py
"""

import struct
import sys
import time

def p(msg):
    print(msg, flush=True)

import numpy as np

SAMPLE_RATE = 16000
FRAME_SIZE = 480
ENERGY_THRESHOLD = 250
SILENCE_AFTER_SPEECH = 0.6
MAX_PHRASE_SECONDS = 5.0

WAKE_WORDS = [
    "kairo", "hey kairo", "cairo", "kyro", "caro", "kai ro", "ky ro",
]


def rms_energy(frame: bytes) -> float:
    count = len(frame) // 2
    if count == 0:
        return 0.0
    samples = struct.unpack(f"<{count}h", frame)
    arr = np.array(samples, dtype=np.float32)
    return float(np.sqrt(np.mean(arr ** 2)))


def main():
    try:
        import pyaudio
    except ImportError:
        print("pyaudio not installed")
        return

    from pywhispercpp.model import Model
    p("Loading Whisper small.en (this takes a few seconds)...")
    model = Model("small.en", n_threads=4)
    p("Whisper loaded!\n")

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16, channels=1,
        rate=SAMPLE_RATE, input=True,
        frames_per_buffer=FRAME_SIZE,
    )

    p("=" * 60)
    p("  LIVE MIC TEST — Say 'hey kairo' or anything")
    p("  Ctrl+C to exit")
    p("=" * 60)
    p("")

    try:
        round_num = 0
        while True:
            frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
            energy = rms_energy(frame)

            if energy < ENERGY_THRESHOLD:
                continue

            round_num += 1
            p(f"[{round_num}] Speech detected (energy={energy:.0f}) — recording...")

            audio_chunks = [frame]
            silence_start = None
            capture_start = time.monotonic()
            silence_threshold = ENERGY_THRESHOLD * 0.4

            while True:
                if time.monotonic() - capture_start > MAX_PHRASE_SECONDS:
                    break
                frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
                audio_chunks.append(frame)
                e = rms_energy(frame)
                if e < silence_threshold:
                    if silence_start is None:
                        silence_start = time.monotonic()
                    elif time.monotonic() - silence_start >= SILENCE_AFTER_SPEECH:
                        break
                else:
                    silence_start = None

            audio_bytes = b"".join(audio_chunks)
            duration = len(audio_bytes) / (SAMPLE_RATE * 2)
            p(f"    Captured {duration:.1f}s of audio")

            if duration < 0.3:
                p(f"    Too short, skipping")
                continue

            samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            t0 = time.monotonic()
            segments = model.transcribe(samples)
            elapsed = time.monotonic() - t0
            text = " ".join(seg.text.strip() for seg in segments).strip().lower()

            p(f"    Whisper heard: '{text}' ({elapsed:.2f}s)")

            matched = False
            for kw in WAKE_WORDS:
                if kw in text:
                    p(f"    >>> WAKE WORD MATCH: '{kw}'")
                    matched = True
                    break
            if not matched:
                p(f"    --- No wake word match")
            p("")

    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        stream.close()
        pa.terminate()


if __name__ == "__main__":
    main()
