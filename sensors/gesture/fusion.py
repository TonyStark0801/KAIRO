"""Gesture fusion — combines identity signals (face/voice) with wake word.

Verification modes:
  face  — FACE_VERIFIED + WAKE_WORD_DETECTED
  voice — VOICE_VERIFIED + WAKE_WORD_DETECTED
  any   — (FACE_VERIFIED OR VOICE_VERIFIED) + WAKE_WORD_DETECTED
  both  — FACE_VERIFIED + VOICE_VERIFIED + WAKE_WORD_DETECTED
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING

from runtime.event_bus import (
    GestureEvent,
    GestureType,
    SessionState,
    SessionStateChangedEvent,
    ToolCancelEvent,
)

if TYPE_CHECKING:
    from runtime.event_bus import EventBus

logger = logging.getLogger(__name__)

_DEFAULT_SIGNAL_TTL = 10.0
_VALID_MODES = {"none", "face", "voice", "any", "both"}


class GestureFusion:
    def __init__(
        self,
        event_bus: EventBus,
        loop: asyncio.AbstractEventLoop,
        wake_window: float = _DEFAULT_SIGNAL_TTL,
        verification_mode: str = "any",
    ) -> None:
        self._bus = event_bus
        self._loop = loop
        self._signal_ttl = wake_window
        self._mode = verification_mode if verification_mode in _VALID_MODES else "any"
        self._lock = threading.Lock()
        self._last_face_time: float = 0
        self._last_voice_time: float = 0
        self._last_wake_time: float = 0
        self._confirmed = False
        self._current_state = SessionState.SLEEP
        self._session_id = ""
        logger.info("GestureFusion verification_mode=%s", self._mode)

    async def on_state_changed(self, event: SessionStateChangedEvent) -> None:
        self._current_state = event.new_state
        self._session_id = event.session_id
        if event.new_state in (SessionState.ACTIVE_SESSION, SessionState.SLEEP):
            with self._lock:
                self._confirmed = False

    async def on_gesture(self, event: GestureEvent) -> None:
        if self._current_state == SessionState.EXECUTING:
            if event.type == GestureType.DUAL_SNAP:
                await self._bus.publish(ToolCancelEvent(
                    session_id=self._session_id,
                    reason="User dual-snap cancel",
                ))
            return

        if self._current_state == SessionState.ACTIVE_SESSION:
            return

        now = time.time()

        with self._lock:
            if event.type == GestureType.FACE_VERIFIED:
                self._last_face_time = now
            elif event.type == GestureType.VOICE_VERIFIED:
                self._last_voice_time = now
            elif event.type == GestureType.WAKE_WORD_DETECTED:
                self._last_wake_time = now
            else:
                return

            if self._confirmed:
                return

            wake_fresh = (now - self._last_wake_time) < self._signal_ttl and self._last_wake_time > 0
            if not wake_fresh:
                return

            if self._mode == "none":
                confirmed = True
            else:
                face_fresh = (now - self._last_face_time) < self._signal_ttl and self._last_face_time > 0
                voice_fresh = (now - self._last_voice_time) < self._signal_ttl and self._last_voice_time > 0

                confirmed = False
                if self._mode == "face":
                    confirmed = face_fresh
                elif self._mode == "voice":
                    confirmed = voice_fresh
                elif self._mode == "any":
                    confirmed = face_fresh or voice_fresh
                elif self._mode == "both":
                    confirmed = face_fresh and voice_fresh

            if confirmed:
                self._confirmed = True
                self._last_face_time = 0
                self._last_voice_time = 0
                self._last_wake_time = 0

                if self._mode == "none":
                    logger.info("Wake word confirmed (no identity check)")
                else:
                    id_source = []
                    if face_fresh:
                        id_source.append("face")
                    if voice_fresh:
                        id_source.append("voice")
                    logger.info("Identity confirmed (%s + wake word)", " + ".join(id_source))

                confirmed_event = GestureEvent(
                    type=GestureType.ALL_SIGNALS_CONFIRMED,
                    timestamp=now,
                )
                self._loop.call_soon_threadsafe(
                    asyncio.ensure_future, self._bus.publish(confirmed_event)
                )
