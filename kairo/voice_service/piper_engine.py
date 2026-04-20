"""Voice engine — Edge TTS streaming (primary), Piper (secondary), macOS say (fallback).

Edge TTS streams audio chunks to mpv for instant playback (~300ms to first word).
Supports mid-speech interruption via stop_speaking().
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path

from voice_service.base import VoiceEngine

logger = logging.getLogger(__name__)

_POST_SPEECH_BUFFER = 0.2
_SAY_VOICE = "Samantha"
_SAY_RATE = "185"
_EDGE_VOICE = "en-US-AriaNeural"


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
        self._current_proc: asyncio.subprocess.Process | None = None
        self._interrupted = False

    async def initialize(self) -> bool:
        if not self._local_only:
            if shutil.which("mpv"):
                try:
                    import edge_tts  # noqa: F401
                    self._engine = "edge_stream"
                    logger.info("Voice engine: Edge TTS streaming (voice=%s)", _EDGE_VOICE)
                    return True
                except ImportError:
                    logger.info("edge-tts not installed, trying Piper")

        self._piper_path = shutil.which("piper")
        if self._piper_path:
            model_dir = Path("~/.local/share/piper-voices").expanduser()
            model_file = model_dir / f"{self._model}.onnx"
            if model_file.exists():
                self._model_path = str(model_file)
                self._engine = "piper"
                logger.info("Voice engine: Piper (model=%s)", self._model)
                return True
            if self._require_piper:
                logger.error("Piper model missing: %s", model_file)
                return False

        if self._require_piper:
            logger.error("piper binary not found on PATH")
            return False

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

        self._interrupted = False

        try:
            if self._engine == "edge_stream":
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
