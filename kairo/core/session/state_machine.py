"""Finite state machine for session lifecycle."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from runtime.event_bus import (
    GestureEvent,
    GestureType,
    IntentRoutedEvent,
    SessionState,
    SessionStateChangedEvent,
    ToolExecutionEvent,
)

if TYPE_CHECKING:
    from runtime.event_bus import EventBus

logger = logging.getLogger(__name__)


class InvalidTransitionError(Exception):
    pass


_VALID_TRANSITIONS: dict[
    SessionState, dict[type, dict[str | None, SessionState]]
] = {
    SessionState.SLEEP: {
        GestureEvent: {
            GestureType.FACE_VERIFIED.name: SessionState.WAKE_PENDING,
            GestureType.ALL_SIGNALS_CONFIRMED.name: SessionState.ACTIVE_SESSION,
        },
    },
    SessionState.WAKE_PENDING: {
        GestureEvent: {
            GestureType.ALL_SIGNALS_CONFIRMED.name: SessionState.ACTIVE_SESSION,
            GestureType.WAKE_TIMEOUT.name: SessionState.SLEEP,
        },
    },
    SessionState.ACTIVE_SESSION: {
        IntentRoutedEvent: {None: SessionState.EXECUTING},
    },
    SessionState.EXECUTING: {
        ToolExecutionEvent: {None: SessionState.ACTIVE_SESSION},
    },
}

_NOOP_EVENTS: dict[SessionState, set[type]] = {
    SessionState.SLEEP: {GestureEvent, IntentRoutedEvent, ToolExecutionEvent},
    SessionState.WAKE_PENDING: {GestureEvent, IntentRoutedEvent, ToolExecutionEvent},
    SessionState.ACTIVE_SESSION: {GestureEvent, ToolExecutionEvent},
    SessionState.EXECUTING: {GestureEvent, IntentRoutedEvent},
    SessionState.IDLE_TIMEOUT: {IntentRoutedEvent, ToolExecutionEvent},
}


class SessionStateMachine:
    def __init__(self, event_bus: EventBus, session_id: str = "") -> None:
        self._bus = event_bus
        self._state = SessionState.SLEEP
        self._session_id = session_id
        self._idle_handle: asyncio.TimerHandle | None = None

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def session_id(self) -> str:
        return self._session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        self._session_id = value

    async def handle_event(self, event: object) -> None:
        event_type = type(event)
        state_transitions = _VALID_TRANSITIONS.get(self._state, {})
        event_transitions = state_transitions.get(event_type)

        if event_transitions is None:
            noop_set = _NOOP_EVENTS.get(self._state, set())
            if event_type in noop_set:
                logger.debug(
                    "Ignoring %s in state %s", event_type.__name__, self._state.name
                )
                return
            raise InvalidTransitionError(
                f"No transition for {event_type.__name__} in state {self._state.name}"
            )

        if isinstance(event, GestureEvent):
            key = event.type.name
        else:
            key = None

        new_state = event_transitions.get(key)
        if new_state is None:
            logger.debug(
                "Ignoring %s (subtype %s) in state %s",
                event_type.__name__,
                key,
                self._state.name,
            )
            return

        await self._transition(new_state)

    async def trigger_idle_timeout(self) -> None:
        if self._state != SessionState.ACTIVE_SESSION:
            return
        await self._transition(SessionState.IDLE_TIMEOUT)
        await self._transition(SessionState.SLEEP)

    async def _transition(self, new_state: SessionState) -> None:
        old_state = self._state
        self._state = new_state
        logger.info("State: %s → %s", old_state.name, new_state.name)
        await self._bus.publish(
            SessionStateChangedEvent(
                old_state=old_state,
                new_state=new_state,
                session_id=self._session_id,
            )
        )
