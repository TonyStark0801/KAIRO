"""Clipboard monitor — watches system clipboard and emits ClipboardChangedEvent.

Runs as a background thread (not async) because clipboard reads are
blocking subprocess calls. Publishes events back to the asyncio event
loop via loop.call_soon_threadsafe().

Platform support:
  macOS  → pbpaste
  Linux  → xclip -o  (fallback: xsel --output --clipboard)
  Windows→ powershell.exe Get-Clipboard
"""

from __future__ import annotations

import asyncio
import logging
import platform
import subprocess
import threading
import time

from runtime.event_bus import ClipboardChangedEvent

logger = logging.getLogger(__name__)

_MAX_CONTENT_LEN = 500   # chars stored in event; long pastes truncated
_POLL_INTERVAL = 1.0     # seconds between clipboard reads
_MIN_CONTENT_LEN = 3     # ignore single chars / whitespace


class ClipboardMonitor:
    """Background thread that polls the system clipboard."""

    def __init__(
        self,
        event_bus,
        loop: asyncio.AbstractEventLoop,
        poll_interval: float = _POLL_INTERVAL,
    ) -> None:
        self._bus = event_bus
        self._loop = loop
        self._interval = max(0.5, poll_interval)
        self._last_content: str = ""
        self._os = platform.system()  # "Darwin" | "Linux" | "Windows"

    # ------------------------------------------------------------------
    # Public — called from ThreadPoolExecutor
    # ------------------------------------------------------------------

    def run(self, stop_event: threading.Event) -> None:
        logger.info("Clipboard monitor started (os=%s, interval=%.1fs)", self._os, self._interval)
        while not stop_event.is_set():
            try:
                content = self._read_clipboard()
                if content and content != self._last_content and len(content) >= _MIN_CONTENT_LEN:
                    self._last_content = content
                    content_type = self._classify(content)
                    event = ClipboardChangedEvent(
                        content=content[:_MAX_CONTENT_LEN],
                        content_type=content_type,
                        timestamp=time.time(),
                    )
                    self._loop.call_soon_threadsafe(
                        lambda e=event: asyncio.ensure_future(self._bus.publish(e))
                    )
                    logger.debug(
                        "Clipboard changed: type=%s len=%d",
                        content_type,
                        len(content),
                    )
            except Exception:
                logger.debug("Clipboard read error", exc_info=True)
            time.sleep(self._interval)
        logger.info("Clipboard monitor stopped")

    # ------------------------------------------------------------------
    # Platform clipboard read
    # ------------------------------------------------------------------

    def _read_clipboard(self) -> str:
        try:
            if self._os == "Darwin":
                return self._run_cmd(["pbpaste"])
            if self._os == "Linux":
                text = self._run_cmd(["xclip", "-o", "-selection", "clipboard"])
                if text is None:
                    text = self._run_cmd(["xsel", "--output", "--clipboard"])
                return text or ""
            if self._os == "Windows":
                return self._run_cmd(
                    ["powershell.exe", "-Command", "Get-Clipboard"],
                    timeout=2.0,
                )
        except FileNotFoundError:
            pass  # clipboard tool not installed — monitor silently no-ops
        except Exception:
            logger.debug("Clipboard read failed", exc_info=True)
        return ""

    @staticmethod
    def _run_cmd(cmd: list[str], timeout: float = 1.0) -> str:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout  # may be empty string; caller strips

    # ------------------------------------------------------------------
    # Content classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify(content: str) -> str:
        stripped = content.strip()

        if stripped.startswith(("http://", "https://", "ftp://")):
            return "url"

        # Code heuristics: common tokens that appear in source code
        _code_tokens = (
            "def ", "class ", "import ", "from ", "return ",
            "function ", "const ", "let ", "var ", "=>",
            "public ", "private ", "static ", "void ", "int ",
            "async ", "await ", "#include", "fn ", "struct ",
            "SELECT ", "INSERT ", "UPDATE ", "DELETE ",
        )
        has_newline = "\n" in stripped
        has_token = any(tok in stripped for tok in _code_tokens)
        if has_newline and has_token:
            return "code"

        return "text"
