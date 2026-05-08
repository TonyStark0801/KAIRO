"""CLI echo: microphone → STT → print line; optional Piper / say parrot.

Mic backends:
  sounddevice — same segmentation as newThing/test.py (high-pass + level gate + silence tail).
               pip install '.[sd-mic]' from the kairo directory.
  pyaudio     — legacy VAD + PyAudio.

Run: kairo-echo --mic-backend sounddevice
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time

import numpy as np

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_FRAME_SIZE = 480
_MIN_UTTERANCE_BYTES = int(_SAMPLE_RATE * 2 * 0.3)


def _find_builtin_mic(pa) -> int | None:
    builtin_keywords = ("macbook", "built-in", "internal")
    best = None
    for i in range(pa.get_device_count()):
        try:
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) < 1:
                continue
            name = info.get("name", "").lower()
            if any(kw in name for kw in builtin_keywords):
                best = i
        except Exception:
            continue
    return best


def _resolve_mic_backend(choice: str) -> str:
    if choice != "auto":
        return choice
    from stt_service.sounddevice_segmenter import try_import_deps

    if try_import_deps() is None:
        return "sounddevice"
    return "pyaudio"


async def _parrot(text: str, *, engine: str, require_piper: bool, piper_model: str) -> None:
    if not text.strip():
        return
    if engine == "say":
        proc = await asyncio.create_subprocess_exec(
            "say",
            "-v",
            "Samantha",
            "-r",
            "185",
            text,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.communicate()
        return

    from voice_service.piper_engine import PiperVoiceEngine

    ve = PiperVoiceEngine(model=piper_model, local_only=True, require_piper=require_piper)
    if not await ve.initialize():
        sys.exit(
            "Parrot: Piper not available. Install piper and a voice under ~/.local/share/piper-voices/ "
            "or use --parrot-engine auto or say."
        )
    await ve.speak(text)


def _run_pyaudio_echo(args, whisper) -> None:
    try:
        import pyaudio
    except ImportError:
        sys.exit("pyaudio is required for --mic-backend pyaudio. Install pyaudio or use --mic-backend sounddevice.")

    from sensors.voice.vad import VoiceActivityDetector

    onset, offset = (280, 140) if args.sensitive_vad else (400, 180)
    vad = VoiceActivityDetector(
        sample_rate=_SAMPLE_RATE,
        onset_threshold=onset,
        offset_threshold=offset,
    )

    pa = pyaudio.PyAudio()
    stream_kwargs: dict = dict(
        format=pyaudio.paInt16,
        channels=1,
        rate=_SAMPLE_RATE,
        input=True,
        frames_per_buffer=_FRAME_SIZE,
    )
    idx: int | None = None
    if not args.system_default_mic:
        idx = _find_builtin_mic(pa)
        if idx is not None:
            stream_kwargs["input_device_index"] = idx

    try:
        stream = pa.open(**stream_kwargs)
    except Exception:
        pa.terminate()
        logger.exception("Cannot open microphone")
        sys.exit(1)

    if idx is not None:
        try:
            mic_name = pa.get_device_info_by_index(idx).get("name", "?")
        except Exception:
            mic_name = "?"
    else:
        try:
            mic_name = pa.get_default_input_device_info().get("name", "?")
        except Exception:
            mic_name = "system default"

    print("Microphone:", mic_name, flush=True)
    _print_stt_line(whisper)
    print("", flush=True)

    round_num = 0
    try:
        while True:
            try:
                frame = stream.read(_FRAME_SIZE, exception_on_overflow=False)
            except Exception:
                time.sleep(0.05)
                continue

            audio = vad.process_frame(frame)
            if audio is None:
                continue
            if len(audio) < _MIN_UTTERANCE_BYTES:
                vad.reset()
                continue

            samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            meta = whisper.transcribe_with_meta(samples)
            text = (meta.text or "").strip()
            vad.reset()

            if not text:
                continue

            round_num += 1
            print(f"[{round_num}] {text}", flush=True)
            _maybe_parrot(args, text)
    except KeyboardInterrupt:
        print("\nDone.", flush=True)
    finally:
        stream.close()
        pa.terminate()


def _print_stt_line(whisper) -> None:
    _m = whisper._model_name
    if whisper._backend == "pywhispercpp":
        _m = whisper._cpp_fallback_model
    print("STT backend:", whisper._backend, "| model:", _m, flush=True)


def _maybe_parrot(args, text: str) -> None:
    if not args.parrot:
        return
    if args.parrot_engine == "say":
        pe, req = "say", False
    elif args.parrot_engine == "piper":
        pe, req = "piper", True
    else:
        pe, req = "piper", False
    try:
        asyncio.run(_parrot(text, engine=pe, require_piper=req, piper_model=args.piper_model))
    except KeyboardInterrupt:
        raise
    except SystemExit:
        raise
    except Exception:
        logger.exception("Parrot playback failed")


def _run_sounddevice_echo(args, whisper) -> None:
    from stt_service.sounddevice_segmenter import iter_utterances, try_import_deps

    err = try_import_deps()
    if err:
        sys.exit(err)

    print("Mic backend: sounddevice (high-pass + peak gate, same family as newThing/test.py)", flush=True)
    try:
        import sounddevice as sd

        mic_name = "default"
        di, _ = sd.default.device
        if isinstance(di, int) and di >= 0:
            mic_name = str(sd.query_devices(di).get("name", mic_name))
    except Exception:
        mic_name = "default"
    print("Microphone:", mic_name, flush=True)
    _print_stt_line(whisper)
    print("", flush=True)

    round_num = 0
    try:
        for utterance in iter_utterances(
            sample_rate=_SAMPLE_RATE,
            chunk_seconds=args.sd_chunk_s,
            threshold=args.sd_threshold,
            silence_chunks=args.sd_silence_chunks,
            show_meter=not args.no_meter,
            device=args.sd_device,
            min_duration_s=args.sd_min_duration,
        ):
            print("[Processing Sentence...]", flush=True)
            meta = whisper.transcribe_with_meta(utterance)
            text = (meta.text or "").strip()
            if not text:
                continue
            round_num += 1
            print(f">>> {text}", flush=True)
            _maybe_parrot(args, text)
    except KeyboardInterrupt:
        print("\nOffline.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Echo mic transcript to stdout; optional TTS parrot.")
    parser.add_argument("--config", default=None, help="Path to kairo.yaml (default: package config/kairo.yaml)")
    parser.add_argument("--parrot", action="store_true", help="Speak recognized text after each line")
    parser.add_argument(
        "--parrot-engine",
        choices=("auto", "piper", "say"),
        default="auto",
        help="auto: Piper if installed, else macOS say; piper: require Piper; say: macOS say only",
    )
    parser.add_argument(
        "--piper-model",
        default="en_US-amy-medium",
        help="Piper voice id (looks for ~/.local/share/piper-voices/<id>.onnx)",
    )
    parser.add_argument(
        "--mic-backend",
        choices=("auto", "pyaudio", "sounddevice"),
        default="auto",
        help="auto: sounddevice if scipy+sounddevice installed, else pyaudio",
    )
    parser.add_argument(
        "--system-default-mic",
        action="store_true",
        help="(pyaudio only) Use OS default input instead of built-in laptop mic",
    )
    parser.add_argument(
        "--sensitive-vad",
        action="store_true",
        help="(pyaudio only) Lower VAD energy gates",
    )
    parser.add_argument(
        "--sd-threshold",
        type=float,
        default=0.04,
        help="(sounddevice) Peak amplitude gate after high-pass (default: 0.04)",
    )
    parser.add_argument(
        "--sd-silence-chunks",
        type=int,
        default=10,
        help="(sounddevice) Chunks of silence to end utterance (100ms each; default 10 = 1.0s)",
    )
    parser.add_argument(
        "--sd-chunk-s",
        type=float,
        default=0.1,
        help="(sounddevice) Read chunk size in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--sd-min-duration",
        type=float,
        default=0.25,
        help="(sounddevice) Minimum utterance duration in seconds",
    )
    parser.add_argument(
        "--sd-device",
        type=int,
        default=None,
        help="(sounddevice) sounddevice input device index (default: system default)",
    )
    parser.add_argument("--no-meter", action="store_true", help="(sounddevice) Hide the level meter line")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from core.config.loader import load_config
    from stt_service.whisper_engine import WhisperEngine

    cfg = load_config(args.config)
    stt = cfg.stt
    whisper = WhisperEngine(
        model_name=stt.model,
        cpp_fallback_model=stt.cpp_fallback_model,
        engine=stt.engine,
        initial_prompt=stt.initial_prompt,
    )
    if not whisper.initialize():
        sys.exit("Whisper STT failed to initialize. Check stt.engine / model in kairo.yaml.")

    backend = _resolve_mic_backend(args.mic_backend)
    print("Kairo echo — Ctrl+C to stop.", flush=True)

    if backend == "sounddevice":
        _run_sounddevice_echo(args, whisper)
    else:
        _run_pyaudio_echo(args, whisper)


if __name__ == "__main__":
    main()
