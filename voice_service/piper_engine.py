"""Voice engine — Kokoro (primary), Edge TTS streaming (fallback), macOS say (last resort).

Kokoro: local ONNX-free neural TTS, British-female 'bf_emma' voice, ~300MB model,
runs on CPU at ~2x realtime. The 'real' Kairo voice.
Edge TTS: cloud fallback if Kokoro import/init fails (unofficial Microsoft API).
macOS say: robotic last resort — only if both above fail (e.g. offline + Kokoro broken).
Supports mid-speech interruption via stop_speaking().
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import tempfile
from pathlib import Path

from voice_service.base import VoiceEngine

logger = logging.getLogger(__name__)

_POST_SPEECH_BUFFER = 0.2
_SAY_VOICE = "Samantha"
_SAY_RATE = "185"
_EDGE_VOICE = "en-US-AriaNeural"
_KOKORO_VOICE = "bf_emma"        # British female — picked in voice audition
_KOKORO_LANG = "en-gb"           # kokoro-onnx language code
_KOKORO_SPEED = 1.1

# TTS hates these. Emoji get spelled out ("smiley face"); short interjections
# get read letter-by-letter ("M M M"). Strip before synthesis.
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF\U0001F600-\U0001F64F]+"
)
_INTERJECTION_RE = re.compile(r"\b(m+m+|h+m+|u+h+|a+h+|e+h+|o+h+)\b[,.!?]?\s*", re.IGNORECASE)


def _sanitize_for_tts(text: str) -> str:
    text = _EMOJI_RE.sub("", text)
    text = _INTERJECTION_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()

# kokoro-onnx model files — downloaded once, cached.
_KOKORO_CACHE_DIR = Path("~/.cache/kokoro-onnx").expanduser()
_KOKORO_MODEL_FILE = _KOKORO_CACHE_DIR / "kokoro-v1.0.onnx"
_KOKORO_VOICES_FILE = _KOKORO_CACHE_DIR / "voices-v1.0.bin"
_KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
_KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"


class PiperVoiceEngine(VoiceEngine):
    def __init__(
        self,
        model: str = "en_US-amy-medium",
        *,
        local_only: bool = False,
        require_piper: bool = False,
    ) -> None:
        self._model = model
        self._local_only = local_only
        self._require_piper = require_piper
        self._piper_path: str | None = None
        self._model_path: str | None = None
        self._engine: str = "say"
        self._kokoro = None  # lazy-loaded Kokoro (onnx) instance
        self._current_proc: asyncio.subprocess.Process | None = None
        self._interrupted = False

    async def initialize(self) -> bool:
        # ── Tier 1: Kokoro ONNX (local, expressive, British bf_emma) ────────
        try:
            from kokoro_onnx import Kokoro  # noqa: F401
            await self._ensure_kokoro_models()
            loop = asyncio.get_running_loop()
            self._kokoro = await loop.run_in_executor(
                None,
                lambda: Kokoro(str(_KOKORO_MODEL_FILE), str(_KOKORO_VOICES_FILE)),
            )
            self._engine = "kokoro"
            logger.info("Voice engine: Kokoro ONNX (voice=%s, lang=%s)", _KOKORO_VOICE, _KOKORO_LANG)
            return True
        except ImportError:
            logger.info("kokoro-onnx not installed, falling back to Edge TTS")
        except Exception:
            logger.exception("Kokoro init failed, falling back to Edge TTS")
            self._kokoro = None

        # ── Tier 2: Edge TTS streaming (cloud fallback) ──────────────────────
        if not self._local_only:
            if shutil.which("mpv"):
                try:
                    import edge_tts  # noqa: F401
                    self._engine = "edge_stream"
                    logger.info("Voice engine: Edge TTS streaming (voice=%s)", _EDGE_VOICE)
                    return True
                except ImportError:
                    logger.info("edge-tts not installed, trying macOS say")

        # ── Tier 3: macOS say (always available) ─────────────────────────────
        self._engine = "say"
        logger.info("Voice engine: macOS say (voice=%s, rate=%s)", _SAY_VOICE, _SAY_RATE)
        return True

    def stop_speaking(self) -> None:
        """Kill current speech immediately. Safe to call from any thread."""
        self._interrupted = True
        proc = self._current_proc
        if proc and proc.returncode is None:
            try:
                proc.kill()
                logger.info("Speech interrupted")
            except Exception:
                pass

    @property
    def is_speaking(self) -> bool:
        proc = self._current_proc
        return proc is not None and proc.returncode is None

    async def speak(self, text: str) -> None:
        if not text:
            return

        text = _sanitize_for_tts(text)
        if not text:
            return

        self._interrupted = False

        try:
            if self._engine == "kokoro":
                await self._speak_kokoro(text)
            elif self._engine == "edge_stream":
                await self._speak_edge_stream(text)
            elif self._engine == "piper":
                await self._speak_piper(text)
            else:
                await self._speak_macos_say(text)

            if not self._interrupted:
                await asyncio.sleep(_POST_SPEECH_BUFFER)
        except Exception:
            if not self._interrupted:
                logger.exception("Speech failed, falling back to say")
                try:
                    await self._speak_macos_say(text)
                except Exception:
                    logger.exception("Even macOS say failed")
        finally:
            self._current_proc = None

    async def _ensure_kokoro_models(self) -> None:
        """Download the Kokoro model + voices files on first run."""
        _KOKORO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        targets = [
            (_KOKORO_MODEL_FILE, _KOKORO_MODEL_URL, "model (~325MB)"),
            (_KOKORO_VOICES_FILE, _KOKORO_VOICES_URL, "voices (~26MB)"),
        ]
        missing = [(p, u, label) for p, u, label in targets if not p.exists()]
        if not missing:
            return

        import httpx

        for path, url, label in missing:
            logger.info("Downloading Kokoro %s from %s", label, url)
            try:
                async with httpx.AsyncClient(follow_redirects=True, timeout=600) as client:
                    async with client.stream("GET", url) as resp:
                        resp.raise_for_status()
                        tmp = path.with_suffix(path.suffix + ".part")
                        with open(tmp, "wb") as f:
                            async for chunk in resp.aiter_bytes(1 << 16):
                                f.write(chunk)
                        tmp.rename(path)
                logger.info("Kokoro %s saved to %s", label, path)
            except Exception:
                logger.exception("Failed to download Kokoro %s", label)
                raise

    async def _speak_kokoro(self, text: str) -> None:
        import soundfile as sf

        loop = asyncio.get_running_loop()

        def _synthesize():
            # Kokoro.create returns (samples: np.ndarray, sample_rate: int)
            return self._kokoro.create(
                text, voice=_KOKORO_VOICE, speed=_KOKORO_SPEED, lang=_KOKORO_LANG
            )

        try:
            samples, sample_rate = await loop.run_in_executor(None, _synthesize)
        except Exception:
            logger.exception("Kokoro synth failed")
            return
        if self._interrupted or samples is None or len(samples) == 0:
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            sf.write(wav_path, samples, sample_rate)
            proc = await asyncio.create_subprocess_exec(
                "afplay", wav_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            self._current_proc = proc
            await proc.wait()
        finally:
            Path(wav_path).unlink(missing_ok=True)

    async def _speak_edge_stream(self, text: str) -> None:
        import edge_tts

        communicate = edge_tts.Communicate(text, _EDGE_VOICE)

        proc = await asyncio.create_subprocess_exec(
            "mpv", "--no-video", "--no-terminal", "--", "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._current_proc = proc

        try:
            async for chunk in communicate.stream():
                if self._interrupted:
                    break
                if chunk["type"] == "audio" and proc.stdin:
                    proc.stdin.write(chunk["data"])
                    await proc.stdin.drain()
        except Exception:
            if not self._interrupted:
                logger.exception("Edge TTS stream error")
        finally:
            if proc.stdin:
                proc.stdin.close()
            await proc.wait()

    async def _speak_piper(self, text: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        try:
            proc = await asyncio.create_subprocess_exec(
                self._piper_path,
                "--model", self._model_path,
                "--output_file", wav_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate(input=text.encode())

            if proc.returncode != 0:
                logger.warning("Piper failed, falling back to say")
                await self._speak_macos_say(text)
                return

            play_proc = await asyncio.create_subprocess_exec(
                "afplay", wav_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            self._current_proc = play_proc
            await play_proc.communicate()
        finally:
            Path(wav_path).unlink(missing_ok=True)

    async def _speak_macos_say(self, text: str) -> None:
        escaped = text.replace('"', '\\"')
        proc = await asyncio.create_subprocess_exec(
            "say", "-v", _SAY_VOICE, "-r", _SAY_RATE, escaped,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._current_proc = proc
        await proc.communicate()
