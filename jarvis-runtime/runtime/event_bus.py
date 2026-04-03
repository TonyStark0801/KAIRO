"""Async event bus — pure asyncio pub/sub with per-subscriber queues."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class GestureType(Enum):
    FACE_VERIFIED = auto()
    DOUBLE_CLAP = auto()
    DUAL_SNAP = auto()
    ALL_SIGNALS_CONFIRMED = auto()
    WAKE_TIMEOUT = auto()


class SessionState(Enum):
    SLEEP = auto()
    WAKE_PENDING = auto()
    ACTIVE_SESSION = auto()
    EXECUTING = auto()
    IDLE_TIMEOUT = auto()


@dataclass(frozen=True)
class GestureEvent:
    type: GestureType
    timestamp: float


@dataclass(frozen=True)
class VoiceTranscriptEvent:
    text: str
    confidence: float
    session_id: str


@dataclass(frozen=True)
class SessionStateChangedEvent:
    old_state: SessionState
    new_state: SessionState
    session_id: str


@dataclass(frozen=True)
class IntentRoutedEvent:
    tool_name: str
    params: dict[str, Any]
    confidence: float
    session_id: str


@dataclass(frozen=True)
class ToolResult:
    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolExecutionEvent:
    tool_name: str
    success: bool
    result: ToolResult
    session_id: str


@dataclass(frozen=True)
class ToolCancelEvent:
    session_id: str
    reason: str


@dataclass(frozen=True)
class MemoryWriteEvent:
    tool_name: str
    command_text: str
    params: dict[str, Any]
    session_id: str
    timestamp: float


EventHandler = Callable[..., Coroutine[Any, Any, None]]

_QUEUE_MAX = 100


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[type, list[tuple[asyncio.Queue, EventHandler]]] = {}
        self._running = False
        self._dispatch_tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False
        for task in self._dispatch_tasks:
            task.cancel()
        self._dispatch_tasks.clear()

    def subscribe(self, event_type: type, handler: EventHandler) -> None:
        queue: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAX)
        self._subscribers.setdefault(event_type, []).append((queue, handler))
        task = asyncio.ensure_future(self._dispatch_loop(queue, handler))
        self._dispatch_tasks.append(task)

    async def publish(self, event: object) -> None:
        event_type = type(event)
        for queue, _ in self._subscribers.get(event_type, []):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(
                    "Event queue full for %s subscriber, dropping event %s",
                    event_type.__name__,
                    event,
                )

    async def _dispatch_loop(
        self, queue: asyncio.Queue, handler: EventHandler
    ) -> None:
        while True:
            event = await queue.get()
            try:
                await handler(event)
            except Exception:
                logger.exception(
                    "Error in event handler %s for %s",
                    handler.__name__,
                    type(event).__name__,
                )
