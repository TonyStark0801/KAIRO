"""Utterance segmentation matching newThing/test.py: sounddevice + high-pass + peak gate + silence tail.

Requires optional deps: pip install 'kairo-runtime[sd-mic]' (sounddevice, scipy).
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

_FS = 16000


def highpass_filter(data: np.ndarray, cutoff: float = 300.0, fs: float = _FS, order: int = 5) -> np.ndarray:
    from scipy.signal import butter, lfilter

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return lfilter(b, a, data)


def iter_utterances(
    *,
    sample_rate: int = _FS,
    chunk_seconds: float = 0.1,
    threshold: float = 0.04,
    silence_chunks: int = 10,
    show_meter: bool = True,
    device: int | str | None = None,
    min_duration_s: float = 0.25,
) -> Iterator[np.ndarray]:
    """Yield peak-normalized float32 mono utterances (same semantics as test.py process_audio input)."""
    import sounddevice as sd

    chunk_frames = int(chunk_seconds * sample_rate)
    audio_buffer: list[np.ndarray] = []
    silent_count = 0
    recording = False

    stream_kw: dict[str, Any] = dict(samplerate=sample_rate, channels=1, dtype="float32")
    if device is not None:
        stream_kw["device"] = device

    with sd.InputStream(**stream_kw) as stream:
        while True:
            raw_chunk, _ = stream.read(chunk_frames)
            filtered_chunk = highpass_filter(raw_chunk.flatten())
            volume = float(np.max(np.abs(filtered_chunk)))

            if show_meter:
                meter_val = min(20, int(volume * 100))
                meter = "█" * meter_val
                status = "REC" if recording else "WAIT"
                print(f"\rVol: [{meter.ljust(20)}] {status} (Val: {volume:.4f})", end="", flush=True)

            if volume > threshold:
                if not recording:
                    recording = True
                    audio_buffer = []
                audio_buffer.append(filtered_chunk)
                silent_count = 0
            elif recording:
                audio_buffer.append(filtered_chunk)
                silent_count += 1
                if silent_count > silence_chunks:
                    if show_meter:
                        print(flush=True)
                    combined = np.concatenate(audio_buffer)
                    recording = False
                    audio_buffer = []
                    silent_count = 0
                    dur = combined.size / sample_rate
                    if dur < min_duration_s:
                        if show_meter:
                            print("Listening again...", flush=True)
                        continue
                    peak = float(np.max(np.abs(combined)))
                    if peak > 0:
                        combined = (combined / peak).astype(np.float32)
                    yield combined
                    if show_meter:
                        print("Listening again...", flush=True)


def try_import_deps() -> str | None:
    """Return an error message if sounddevice/scipy are missing, else None."""
    try:
        import sounddevice  # noqa: F401
        import scipy  # noqa: F401
    except ImportError as e:
        return f"{e} — install from the kairo dir: pip install -e '.[sd-mic]'"
    return None


def run_sounddevice_mic_loop(
    stop_event: threading.Event,
    *,
    sample_rate: int = _FS,
    chunk_seconds: float = 0.1,
    silence_chunks: int = 10,
    get_threshold: Callable[[], float],
    show_meter: bool = False,
    device: int | None = None,
    min_duration_s: float = 0.25,
    should_process: Callable[[], bool],
    on_utterance: Callable[[np.ndarray], None],
) -> None:
    """Blocking loop for daemon thread: segment utterances, optionally discard when should_process is False."""
    import sounddevice as sd

    chunk_frames = int(chunk_seconds * sample_rate)
    audio_buffer: list[np.ndarray] = []
    silent_count = 0
    recording = False

    stream_kw: dict[str, Any] = dict(samplerate=sample_rate, channels=1, dtype="float32")
    if device is not None:
        stream_kw["device"] = device

    with sd.InputStream(**stream_kw) as stream:
        while not stop_event.is_set():
            raw_chunk, _ = stream.read(chunk_frames)
            filtered_chunk = highpass_filter(raw_chunk.flatten())
            threshold = float(get_threshold())
            volume = float(np.max(np.abs(filtered_chunk)))

            if show_meter:
                meter_val = min(20, int(volume * 100))
                meter = "█" * meter_val
                status = "REC" if recording else "WAIT"
                print(f"\rVol: [{meter.ljust(20)}] {status} (Val: {volume:.4f})", end="", flush=True)

            if volume > threshold:
                if not recording:
                    recording = True
                    audio_buffer = []
                audio_buffer.append(filtered_chunk)
                silent_count = 0
            elif recording:
                audio_buffer.append(filtered_chunk)
                silent_count += 1
                if silent_count > silence_chunks:
                    if show_meter:
                        print(flush=True)
                    combined = np.concatenate(audio_buffer)
                    recording = False
                    audio_buffer = []
                    silent_count = 0
                    dur = combined.size / sample_rate
                    if dur < min_duration_s:
                        continue
                    peak = float(np.max(np.abs(combined)))
                    if peak > 0:
                        combined = (combined / peak).astype(np.float32)
                    if should_process():
                        on_utterance(combined)
