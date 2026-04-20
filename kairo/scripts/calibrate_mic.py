#!/usr/bin/env python3
"""Measure ambient noise RMS to calibrate the mic energy threshold.

Run with the ceiling fan on, from your normal working position (not close to mic).
This tells you what _ENERGY_NORMAL should be set to in mic_listener.py.

Usage:
    cd ~/Jarvis/kairo
    python scripts/calibrate_mic.py
"""

import struct
import time
import numpy as np

SAMPLE_RATE = 16000
FRAME_SIZE = 480
MEASURE_SECONDS = 5


def rms(frame: bytes) -> float:
    count = len(frame) // 2
    if count == 0:
        return 0.0
    samples = struct.unpack(f"<{count}h", frame)
    arr = np.array(samples, dtype=np.float32)
    return float(np.sqrt(np.mean(arr ** 2)))


def main() -> None:
    try:
        import pyaudio
    except ImportError:
        print("pyaudio not installed")
        return

    pa = pyaudio.PyAudio()

    # Find built-in mic
    builtin_idx = None
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) >= 1:
            name = info.get("name", "").lower()
            if any(kw in name for kw in ("macbook", "built-in", "internal")):
                builtin_idx = i
                print(f"Using: {info.get('name')} (device {i})")
                break

    stream = pa.open(
        format=pyaudio.paInt16, channels=1,
        rate=SAMPLE_RATE, input=True,
        frames_per_buffer=FRAME_SIZE,
        input_device_index=builtin_idx,
    )

    print(f"\nMeasuring ambient noise for {MEASURE_SECONDS}s — DON'T speak, let the fan run...")
    samples = []
    n_frames = int(SAMPLE_RATE / FRAME_SIZE * MEASURE_SECONDS)
    for i in range(n_frames):
        frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
        samples.append(rms(frame))
        if i % 30 == 0:
            print(f"  {i * FRAME_SIZE / SAMPLE_RATE:.1f}s / {MEASURE_SECONDS}s  current RMS: {samples[-1]:.0f}", end="\r")

    stream.close()
    pa.terminate()

    ambient = np.mean(samples)
    peak = np.max(samples)
    p95 = np.percentile(samples, 95)

    print(f"\n\nResults (fan on, no speech):")
    print(f"  Ambient mean RMS : {ambient:.0f}")
    print(f"  95th percentile  : {p95:.0f}")
    print(f"  Peak RMS         : {peak:.0f}")
    print()

    recommended = int(p95 * 2.5)
    print(f"Recommended _ENERGY_NORMAL: {recommended}")
    print(f"(2.5x your ambient 95th percentile — speech should be 3-10x this)")
    print()

    # Reuse same pa instance — macOS PyAudio segfaults on terminate+reopen
    input("Now say 'hey kairo' from your normal position. Press Enter then speak immediately: ")
    print("Listening for 3s...")
    stream2 = pa.open(
        format=pyaudio.paInt16, channels=1,
        rate=SAMPLE_RATE, input=True,
        frames_per_buffer=FRAME_SIZE,
        input_device_index=builtin_idx,
    )
    speech_samples = []
    for _ in range(int(SAMPLE_RATE / FRAME_SIZE * 3)):
        frame = stream2.read(FRAME_SIZE, exception_on_overflow=False)
        speech_samples.append(rms(frame))
    stream2.close()
    pa.terminate()

    speech_peak = np.max(speech_samples)
    active = [s for s in speech_samples if s > ambient * 1.5]
    speech_mean = np.mean(active) if active else ambient
    print(f"\nYour speech peak RMS : {speech_peak:.0f}")
    print(f"Your speech mean RMS : {speech_mean:.0f}")
    print(f"SNR (speech/ambient) : {speech_mean / max(ambient, 1):.1f}x")
    print()

    if speech_mean < recommended * 1.2:
        print("⚠️  Your voice barely clears the recommended threshold from this distance.")
        print("   openWakeWord is the real fix — run: python scripts/train_wake_word.py")
        print(f"   Compromise threshold for now: {int(p95 * 1.5)}")
    else:
        print(f"✓  Your voice clears the threshold with {speech_mean / max(recommended, 1):.1f}x headroom.")
        print(f"   _ENERGY_NORMAL = {recommended} in stt_service/mic_listener.py is correct")


if __name__ == "__main__":
    main()
