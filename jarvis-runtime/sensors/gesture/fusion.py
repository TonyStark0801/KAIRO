"""Gesture fusion — self-managing 3s window for multi-signal wake confirmation."""

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

_WAKE_WINDOW_SECONDS = 3.0
_REQUIRED_SIGNALS = {GestureType.FACE_VERIFIED, GestureType.DOUBLE_CLAP, GestureType.DUAL_SNAP}


class GestureFusion:
    def __init__(
        self,
        event_bus: EventBus,
        loop: asyncio.AbstractEventLoop,
        wake_window: float = _WAKE_WINDOW_SECONDS,
    ) -> None:
        self._bus = event_bus
        self._loop = loop
        self._wake_window = wake_window
        self._lock = threading.Lock()
        self._signals_received: set[GestureType] = set()
        self._window_start: float | None = None
        self._window_timer: threading.Timer | None = None
        self._current_state = SessionState.SLEEP
        self._session_id = ""

    async def on_state_changed(self, event: SessionStateChangedEvent) -> None:
        self._current_state = event.new_state
        self._session_id = event.session_id
        if event.new_state == SessionState.WAKE_PENDING:
            self._start_window()
        elif event.new_state in (SessionState.SLEEP, SessionState.ACTIVE_SESSION):
            self._reset()

    async def on_gesture(self, event: GestureEvent) -> None:
        if self._current_state == SessionState.EXECUTING:
            if event.type == GestureType.DUAL_SNAP:
                cancel = ToolCancelEvent(
                    session_id=self._session_id,
                    reason="User dual-snap cancel",
                )
                await self._bus.publish(cancel)
            return

        if self._current_state != SessionState.WAKE_PENDING:
            return

        if event.type not in _REQUIRED_SIGNALS:
            return

        with self._lock:
            self._signals_received.add(event.type)
            if self._signals_received >= _REQUIRED_SIGNALS:
                self._cancel_timer()
                confirmed = GestureEvent(
                    type=GestureType.ALL_SIGNALS_CONFIRMED,
                    timestamp=time.time(),
                )
                self._loop.call_soon_threadsafe(
                    asyncio.ensure_future, self._bus.publish(confirmed)
                )

    def _start_window(self) -> None:
        with self._lock:
            self._signals_received = {GestureType.FACE_VERIFIED}
            self._window_start = time.time()
            self._cancel_timer()
            self._window_timer = threading.Timer(
                self._wake_window, self._on_window_expired
            )
            self._window_timer.daemon = True
            self._window_timer.start()

    def _on_window_expired(self) -> None:
        with self._lock:
            if self._signals_received >= _REQUIRED_SIGNALS:
                return
            logger.info(
                "Wake window expired with signals: %s",
                {s.name for s in self._signals_received},
            )
            timeout = GestureEvent(
                type=GestureType.WAKE_TIMEOUT,
                timestamp=time.time(),
            )
            self._loop.call_soon_threadsafe(
                asyncio.ensure_future, self._bus.publish(timeout)
            )

    def _cancel_timer(self) -> None:
        if self._window_timer is not None:
            self._window_timer.cancel()
            self._window_timer = None

    def _reset(self) -> None:
        with self._lock:
            self._cancel_timer()
            self._signals_received.clear()
            self._window_start = None
