# Jarvis Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the complete Jarvis macOS desktop assistant runtime — a background daemon activated by multi-modal gestures, accepting voice commands, routing via local LLM, and executing desktop actions.

**Architecture:** Event-driven asyncio runtime with sensor threads bridging via `call_soon_threadsafe()`. Pure FSM for session lifecycle, dynamic tool registry, platform adapters isolating OS calls, and async memory stores. All inter-module communication via internal event bus.

**Tech Stack:** Python 3.11+, asyncio, Ollama (llama3.1), pywhispercpp, MediaPipe, InsightFace, ChromaDB, aiosqlite, aioredis, Pydantic v2, Jinja2, PyYAML

**Spec:** `docs/superpowers/specs/2026-04-03-jarvis-runtime-design.md`

---

## Task 1: Config Models + Loader

**Files:**
- Modify: `jarvis-runtime/core/config/models.py` (add `tool_timeout_seconds`)
- Create: `jarvis-runtime/core/config/loader.py`
- Create: `jarvis-runtime/config/jarvis.yaml`

- [ ] **Step 1: Add `tool_timeout_seconds` to `SessionConfig`**

In `jarvis-runtime/core/config/models.py`, add to `SessionConfig`:

```python
tool_timeout_seconds: int = Field(default=30, alias="tool_timeout")
```

- [ ] **Step 2: Create `core/config/loader.py`**

```python
"""Configuration loader — reads jarvis.yaml once at startup."""

from __future__ import annotations

from pathlib import Path

import yaml

from core.config.models import JarvisConfig


def load_config(path: str | Path | None = None) -> JarvisConfig:
    if path is None:
        path = Path(__file__).resolve().parent.parent.parent / "config" / "jarvis.yaml"
    path = Path(path)
    if not path.exists():
        return JarvisConfig()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return JarvisConfig(**raw)
```

- [ ] **Step 3: Create `config/jarvis.yaml`**

```yaml
ollama:
  host: localhost
  port: 11434
  model: llama3.1
  embed_model: nomic-embed-text

session:
  idle_timeout: 30
  wake_window: 3
  tool_timeout: 30

memory:
  chroma_path: ~/.jarvis/chroma
  behavioral_db: ~/.jarvis/behavior.db

redis:
  redis_enabled: false
  url: redis://localhost:6379/0

paths:
  face_embedding_path: ~/.jarvis/face_embedding.npy

projects:
  office:
    name: Office Dashboard
    path: ~/Projects/office-dashboard
    intellij_module: office-dashboard
  codejam:
    name: CodeJam Solutions
    path: ~/Projects/codejam
    intellij_module: codejam
  personal:
    name: Personal Site
    path: ~/Projects/personal-site
    intellij_module: null

workspace_modes:
  office:
    description: Opens office tools and communication apps
    steps:
      - tool: open_project
        params:
          project: office
      - tool: open_url
        params:
          url: https://mail.google.com
          browser: Chrome
      - tool: open_url
        params:
          url: https://calendar.google.com
          browser: Chrome

  focus:
    description: Deep work mode with IDE and music
    steps:
      - tool: open_project
        params:
          project: codejam
      - tool: play_music
        params:
          query: focus playlist
      - tool: open_notes
        params: {}

  evening:
    description: Wind-down mode with personal projects
    steps:
      - tool: open_project
        params:
          project: personal
      - tool: open_url
        params:
          url: https://news.ycombinator.com
          browser: Chrome
      - tool: play_music
        params:
          query: chill playlist
```

- [ ] **Step 4: Verify config loads correctly**

Run: `cd jarvis-runtime && python -c "from core.config.loader import load_config; c = load_config(); print(c.model_dump_json(indent=2))"`

Expected: Full JSON output with all sections populated, paths expanded.

- [ ] **Step 5: Commit**

```bash
git add jarvis-runtime/core/config/ jarvis-runtime/config/
git commit -m "feat: add config loader and sample jarvis.yaml"
```

---

## Task 2: Event Bus

**Files:**
- Create: `jarvis-runtime/runtime/__init__.py`
- Create: `jarvis-runtime/runtime/event_bus.py`

- [ ] **Step 1: Create `runtime/__init__.py`**

Empty file.

- [ ] **Step 2: Create `runtime/event_bus.py`**

```python
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
```

- [ ] **Step 3: Verify event bus imports cleanly**

Run: `cd jarvis-runtime && python -c "from runtime.event_bus import EventBus, GestureEvent, GestureType; print('OK')"`

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add jarvis-runtime/runtime/
git commit -m "feat: add async event bus with typed events"
```

---

## Task 3: Platform Adapter ABC

**Files:**
- Create: `jarvis-runtime/adapters/__init__.py`
- Create: `jarvis-runtime/adapters/base/__init__.py`
- Create: `jarvis-runtime/adapters/base/platform_adapter.py`

- [ ] **Step 1: Create package init files**

Both `__init__.py` files are empty.

- [ ] **Step 2: Create `adapters/base/platform_adapter.py`**

```python
"""Abstract base class for platform adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AppWindow:
    app_name: str
    title: str
    pid: int


class PlatformAdapter(ABC):

    @abstractmethod
    async def open_application(self, app_name: str) -> bool: ...

    @abstractmethod
    async def switch_window(
        self, app_name: str, title_pattern: str | None = None
    ) -> bool: ...

    @abstractmethod
    async def run_script(self, script: str) -> str: ...

    @abstractmethod
    async def get_running_apps(self) -> list[AppWindow]: ...

    @abstractmethod
    async def open_url_in_browser(self, url: str, browser: str = "Safari") -> bool: ...

    @abstractmethod
    async def send_notification(self, title: str, body: str) -> None: ...

    @abstractmethod
    async def play_audio_file(self, path: str) -> None: ...

    @abstractmethod
    async def get_active_workspace(self) -> str | None: ...
```

- [ ] **Step 3: Verify import**

Run: `cd jarvis-runtime && python -c "from adapters.base.platform_adapter import PlatformAdapter, AppWindow; print('OK')"`

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add jarvis-runtime/adapters/
git commit -m "feat: add PlatformAdapter ABC"
```

---

## Task 4: BaseTool ABC

**Files:**
- Create: `jarvis-runtime/tools/__init__.py`
- Create: `jarvis-runtime/tools/_base.py`

- [ ] **Step 1: Create `tools/__init__.py`**

Empty file.

- [ ] **Step 2: Create `tools/_base.py`**

```python
"""Base class for all tool plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from adapters.base.platform_adapter import PlatformAdapter


@dataclass
class ToolMeta:
    name: str
    description: str
    parameters_schema: dict[str, Any]


@dataclass
class ToolResult:
    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]: ...

    def get_meta(self) -> ToolMeta:
        return ToolMeta(
            name=self.name,
            description=self.description,
            parameters_schema=self.parameters_schema,
        )

    @abstractmethod
    async def execute(
        self, params: dict[str, Any], adapter: PlatformAdapter
    ) -> ToolResult: ...
```

- [ ] **Step 3: Verify import**

Run: `cd jarvis-runtime && python -c "from tools._base import BaseTool, ToolResult, ToolMeta; print('OK')"`

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add jarvis-runtime/tools/
git commit -m "feat: add BaseTool ABC with ToolResult and ToolMeta"
```

---

## Task 5: State Machine

**Files:**
- Create: `jarvis-runtime/core/session/__init__.py`
- Create: `jarvis-runtime/core/session/state_machine.py`
- Create: `jarvis-runtime/core/session/session_context.py`
- Test: `jarvis-runtime/tests/core/test_state_machine.py`

- [ ] **Step 1: Write the failing tests for state machine**

Create `jarvis-runtime/tests/__init__.py`, `jarvis-runtime/tests/core/__init__.py`, and `jarvis-runtime/tests/core/test_state_machine.py`:

```python
"""Tests for the session state machine."""

import asyncio

import pytest
import pytest_asyncio

from runtime.event_bus import (
    EventBus,
    GestureEvent,
    GestureType,
    IntentRoutedEvent,
    SessionState,
    SessionStateChangedEvent,
    ToolExecutionEvent,
    ToolResult,
)
from core.session.state_machine import InvalidTransitionError, SessionStateMachine


@pytest_asyncio.fixture
async def bus():
    b = EventBus()
    await b.start()
    yield b
    await b.stop()


@pytest_asyncio.fixture
async def fsm(bus):
    sm = SessionStateMachine(bus)
    return sm


@pytest.mark.asyncio
async def test_initial_state_is_sleep(fsm):
    assert fsm.state == SessionState.SLEEP


@pytest.mark.asyncio
async def test_sleep_to_wake_pending_on_face_verified(fsm, bus):
    events = []
    bus.subscribe(SessionStateChangedEvent, lambda e: events.append(e))
    await asyncio.sleep(0.05)

    await fsm.handle_event(
        GestureEvent(type=GestureType.FACE_VERIFIED, timestamp=1.0)
    )
    await asyncio.sleep(0.05)

    assert fsm.state == SessionState.WAKE_PENDING
    assert len(events) == 1
    assert events[0].old_state == SessionState.SLEEP
    assert events[0].new_state == SessionState.WAKE_PENDING


@pytest.mark.asyncio
async def test_wake_pending_to_active_on_all_signals(fsm, bus):
    await fsm.handle_event(
        GestureEvent(type=GestureType.FACE_VERIFIED, timestamp=1.0)
    )
    await fsm.handle_event(
        GestureEvent(type=GestureType.ALL_SIGNALS_CONFIRMED, timestamp=2.0)
    )
    assert fsm.state == SessionState.ACTIVE_SESSION


@pytest.mark.asyncio
async def test_wake_pending_to_sleep_on_timeout(fsm, bus):
    await fsm.handle_event(
        GestureEvent(type=GestureType.FACE_VERIFIED, timestamp=1.0)
    )
    await fsm.handle_event(
        GestureEvent(type=GestureType.WAKE_TIMEOUT, timestamp=4.0)
    )
    assert fsm.state == SessionState.SLEEP


@pytest.mark.asyncio
async def test_active_to_executing_on_intent(fsm, bus):
    await fsm.handle_event(
        GestureEvent(type=GestureType.FACE_VERIFIED, timestamp=1.0)
    )
    await fsm.handle_event(
        GestureEvent(type=GestureType.ALL_SIGNALS_CONFIRMED, timestamp=2.0)
    )
    await fsm.handle_event(
        IntentRoutedEvent(tool_name="test", params={}, confidence=0.9, session_id="s1")
    )
    assert fsm.state == SessionState.EXECUTING


@pytest.mark.asyncio
async def test_executing_to_active_on_success(fsm, bus):
    await fsm.handle_event(GestureEvent(type=GestureType.FACE_VERIFIED, timestamp=1.0))
    await fsm.handle_event(GestureEvent(type=GestureType.ALL_SIGNALS_CONFIRMED, timestamp=2.0))
    await fsm.handle_event(IntentRoutedEvent(tool_name="t", params={}, confidence=0.9, session_id="s1"))
    await fsm.handle_event(
        ToolExecutionEvent(
            tool_name="t", success=True,
            result=ToolResult(success=True, message="ok"),
            session_id="s1",
        )
    )
    assert fsm.state == SessionState.ACTIVE_SESSION


@pytest.mark.asyncio
async def test_executing_to_active_on_failure(fsm, bus):
    await fsm.handle_event(GestureEvent(type=GestureType.FACE_VERIFIED, timestamp=1.0))
    await fsm.handle_event(GestureEvent(type=GestureType.ALL_SIGNALS_CONFIRMED, timestamp=2.0))
    await fsm.handle_event(IntentRoutedEvent(tool_name="t", params={}, confidence=0.9, session_id="s1"))
    await fsm.handle_event(
        ToolExecutionEvent(
            tool_name="t", success=False,
            result=ToolResult(success=False, message="error"),
            session_id="s1",
        )
    )
    assert fsm.state == SessionState.ACTIVE_SESSION


@pytest.mark.asyncio
async def test_idle_timeout_transition(fsm, bus):
    await fsm.handle_event(GestureEvent(type=GestureType.FACE_VERIFIED, timestamp=1.0))
    await fsm.handle_event(GestureEvent(type=GestureType.ALL_SIGNALS_CONFIRMED, timestamp=2.0))
    await fsm.trigger_idle_timeout()
    assert fsm.state == SessionState.SLEEP


@pytest.mark.asyncio
async def test_irrelevant_event_is_noop(fsm, bus):
    await fsm.handle_event(
        GestureEvent(type=GestureType.DOUBLE_CLAP, timestamp=1.0)
    )
    assert fsm.state == SessionState.SLEEP


@pytest.mark.asyncio
async def test_invalid_transition_raises(fsm, bus):
    with pytest.raises(InvalidTransitionError):
        await fsm.handle_event(
            IntentRoutedEvent(tool_name="t", params={}, confidence=0.9, session_id="s1")
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd jarvis-runtime && python -m pytest tests/core/test_state_machine.py -v`

Expected: FAIL with import errors (state_machine module doesn't exist yet).

- [ ] **Step 3: Create `core/session/__init__.py`**

Empty file.

- [ ] **Step 4: Create `core/session/session_context.py`**

```python
"""Session context — holds per-session state."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class SessionContext:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    start_time: float = field(default_factory=time.time)
    workspace_mode: str | None = None
    command_history: list[str] = field(default_factory=list)
```

- [ ] **Step 5: Create `core/session/state_machine.py`**

```python
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
    SessionState.SLEEP: {GestureEvent},
    SessionState.WAKE_PENDING: {GestureEvent},
    SessionState.ACTIVE_SESSION: {GestureEvent, ToolExecutionEvent},
    SessionState.EXECUTING: {GestureEvent, IntentRoutedEvent},
    SessionState.IDLE_TIMEOUT: set(),
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd jarvis-runtime && python -m pytest tests/core/test_state_machine.py -v`

Expected: All 10 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add jarvis-runtime/core/session/ jarvis-runtime/tests/
git commit -m "feat: add session state machine with full transition logic"
```

---

## Task 6: Tool Registry

**Files:**
- Create: `jarvis-runtime/core/registry/__init__.py`
- Create: `jarvis-runtime/core/registry/tool_registry.py`

- [ ] **Step 1: Create `core/registry/__init__.py`**

Empty file.

- [ ] **Step 2: Create `core/registry/tool_registry.py`**

```python
"""Dynamic tool registry — discovers and registers BaseTool subclasses."""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools._base import BaseTool, ToolMeta

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            logger.warning("Duplicate tool name %r — overwriting", tool.name)
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_all(self) -> list[ToolMeta]:
        return [t.get_meta() for t in self._tools.values()]

    def discover(self, tools_package_path: str | Path | None = None) -> None:
        from tools._base import BaseTool as BaseToolCls

        if tools_package_path is None:
            tools_package_path = (
                Path(__file__).resolve().parent.parent.parent / "tools"
            )
        else:
            tools_package_path = Path(tools_package_path)

        for importer, modname, ispkg in pkgutil.walk_packages(
            [str(tools_package_path)], prefix="tools."
        ):
            if modname == "tools._base":
                continue
            try:
                module = importlib.import_module(modname)
            except Exception:
                logger.exception("Failed to import tool module %s", modname)
                continue

            for _name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, BaseToolCls)
                    and obj is not BaseToolCls
                    and not inspect.isabstract(obj)
                ):
                    try:
                        instance = obj()
                        self.register(instance)
                    except Exception:
                        logger.exception("Failed to instantiate tool %s", _name)
```

- [ ] **Step 3: Verify import**

Run: `cd jarvis-runtime && python -c "from core.registry.tool_registry import ToolRegistry; print('OK')"`

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add jarvis-runtime/core/registry/
git commit -m "feat: add dynamic tool registry with auto-discovery"
```

---

## Task 7: Sensors — Camera + Gesture

**Files:**
- Create: `jarvis-runtime/sensors/__init__.py`
- Create: `jarvis-runtime/sensors/camera.py`
- Create: `jarvis-runtime/sensors/gesture/__init__.py`
- Create: `jarvis-runtime/sensors/gesture/face_verifier.py`
- Create: `jarvis-runtime/sensors/gesture/gesture_detector.py`
- Create: `jarvis-runtime/sensors/gesture/fusion.py`

- [ ] **Step 1: Create `sensors/__init__.py` and `sensors/gesture/__init__.py`**

Both empty.

- [ ] **Step 2: Create `sensors/camera.py`**

```python
"""Shared camera thread — single cv2.VideoCapture, fans out frames."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FrameCallback = Callable[[np.ndarray], None]

_MAX_RETRIES = 3
_TARGET_FPS = 15


class CameraThread:
    def __init__(self, camera_index: int = 0) -> None:
        self._camera_index = camera_index
        self._subscribers: list[deque] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._healthy = False

    @property
    def healthy(self) -> bool:
        return self._healthy

    def add_subscriber(self) -> deque:
        d: deque = deque(maxlen=2)
        with self._lock:
            self._subscribers.append(d)
        return d

    def start(self) -> bool:
        cap = None
        for attempt in range(1, _MAX_RETRIES + 1):
            cap = cv2.VideoCapture(self._camera_index)
            if cap.isOpened():
                break
            logger.warning("Camera open attempt %d/%d failed", attempt, _MAX_RETRIES)
            cap.release()
            cap = None
            time.sleep(2 ** (attempt - 1))

        if cap is None:
            logger.error("Camera unavailable after %d retries", _MAX_RETRIES)
            return False

        self._healthy = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, args=(cap,), daemon=True, name="camera"
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._healthy = False

    def _capture_loop(self, cap: cv2.VideoCapture) -> None:
        frame_interval = 1.0 / _TARGET_FPS
        try:
            while not self._stop_event.is_set():
                start = time.monotonic()
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Camera read failed, retrying")
                    time.sleep(0.1)
                    continue
                with self._lock:
                    for d in self._subscribers:
                        d.append(frame)
                elapsed = time.monotonic() - start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            cap.release()
            self._healthy = False
            logger.info("Camera thread stopped")
```

- [ ] **Step 3: Create `sensors/gesture/face_verifier.py`**

```python
"""Face verification via InsightFace — compares against enrolled embedding."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from runtime.event_bus import EventBus

_SIMILARITY_THRESHOLD = 0.5


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


class FaceVerifier:
    def __init__(
        self,
        frame_deque: deque,
        event_bus: EventBus,
        loop: asyncio.AbstractEventLoop,
        embedding_path: str = "~/.jarvis/face_embedding.npy",
        check_interval: float = 1.0,
    ) -> None:
        self._frame_deque = frame_deque
        self._bus = event_bus
        self._loop = loop
        self._embedding_path = Path(embedding_path).expanduser()
        self._check_interval = check_interval
        self._enrolled_embedding: np.ndarray | None = None
        self._app: object | None = None
        self._healthy = False

    @property
    def healthy(self) -> bool:
        return self._healthy

    def initialize(self) -> bool:
        if not self._embedding_path.exists():
            logger.error("No enrolled face at %s — run 'jarvis-enroll'", self._embedding_path)
            return False
        try:
            self._enrolled_embedding = np.load(str(self._embedding_path))
        except Exception:
            logger.exception("Failed to load face embedding")
            return False

        try:
            from insightface.app import FaceAnalysis
            self._app = FaceAnalysis(allowed_modules=["detection", "recognition"])
            self._app.prepare(ctx_id=-1, det_size=(640, 640))
        except Exception:
            logger.exception("Failed to initialize InsightFace")
            return False

        self._healthy = True
        return True

    def run(self, stop_event) -> None:
        if not self._healthy:
            return
        from runtime.event_bus import GestureEvent, GestureType
        import time as _time

        while not stop_event.is_set():
            if not self._frame_deque:
                _time.sleep(0.1)
                continue
            frame = self._frame_deque[-1]
            try:
                faces = self._app.get(frame)
                if not faces:
                    _time.sleep(self._check_interval)
                    continue
                embedding = faces[0].embedding
                similarity = _cosine_similarity(embedding, self._enrolled_embedding)
                if similarity >= _SIMILARITY_THRESHOLD:
                    event = GestureEvent(
                        type=GestureType.FACE_VERIFIED,
                        timestamp=time.time(),
                    )
                    self._loop.call_soon_threadsafe(
                        asyncio.ensure_future, self._bus.publish(event)
                    )
                    _time.sleep(3.0)
                else:
                    _time.sleep(self._check_interval)
            except Exception:
                logger.exception("Face verification error")
                _time.sleep(self._check_interval)


def enroll_cli() -> None:
    """One-time face enrollment — run as 'jarvis-enroll'."""
    import cv2

    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        print("InsightFace not installed. Run: pip install insightface")
        return

    dest = Path("~/.jarvis/face_embedding.npy").expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)

    app = FaceAnalysis(allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    embeddings: list[np.ndarray] = []
    captures_needed = 3
    print(f"Position your face in the frame. Press SPACE to capture ({captures_needed} captures needed). Press Q to quit.")

    while len(embeddings) < captures_needed:
        ret, frame = cap.read()
        if not ret:
            continue
        faces = app.get(frame)
        display = frame.copy()
        for face in faces:
            box = face.bbox.astype(int)
            cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imshow("Jarvis Enrollment", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and faces:
            embeddings.append(faces[0].embedding)
            print(f"Captured {len(embeddings)}/{captures_needed}")

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) == captures_needed:
        avg = np.mean(embeddings, axis=0)
        np.save(str(dest), avg)
        print(f"Enrollment saved to {dest}")
    else:
        print("Enrollment cancelled — not enough captures")
```

- [ ] **Step 4: Create `sensors/gesture/gesture_detector.py`**

```python
"""Gesture detection via MediaPipe — detects double clap and dual snap."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from runtime.event_bus import EventBus

_CLAP_DISTANCE_THRESHOLD = 0.05
_SNAP_WRIST_THRESHOLD = 0.08


class GestureDetector:
    def __init__(
        self,
        frame_deque: deque,
        event_bus: EventBus,
        loop: asyncio.AbstractEventLoop,
        check_interval: float = 0.1,
    ) -> None:
        self._frame_deque = frame_deque
        self._bus = event_bus
        self._loop = loop
        self._check_interval = check_interval
        self._hands = None
        self._healthy = False
        self._clap_timestamps: list[float] = []
        self._snap_count = 0

    @property
    def healthy(self) -> bool:
        return self._healthy

    def initialize(self) -> bool:
        try:
            import mediapipe as mp
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
            )
            self._healthy = True
            return True
        except Exception:
            logger.exception("Failed to initialize MediaPipe Hands")
            return False

    def run(self, stop_event) -> None:
        if not self._healthy:
            return
        import cv2
        from runtime.event_bus import GestureEvent, GestureType

        while not stop_event.is_set():
            if not self._frame_deque:
                time.sleep(0.05)
                continue
            frame = self._frame_deque[-1]
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._hands.process(rgb)
                if not results.multi_hand_landmarks:
                    time.sleep(self._check_interval)
                    continue

                hands = results.multi_hand_landmarks
                now = time.time()

                if len(hands) == 2:
                    h1, h2 = hands[0].landmark, hands[1].landmark
                    palm_dist = abs(h1[9].x - h2[9].x) + abs(h1[9].y - h2[9].y)
                    if palm_dist < _CLAP_DISTANCE_THRESHOLD:
                        self._clap_timestamps = [
                            t for t in self._clap_timestamps if now - t < 1.5
                        ]
                        self._clap_timestamps.append(now)
                        if len(self._clap_timestamps) >= 2:
                            event = GestureEvent(
                                type=GestureType.DOUBLE_CLAP, timestamp=now
                            )
                            self._loop.call_soon_threadsafe(
                                asyncio.ensure_future, self._bus.publish(event)
                            )
                            self._clap_timestamps.clear()
                            time.sleep(0.5)
                            continue

                for hand_landmarks in hands:
                    lm = hand_landmarks.landmark
                    thumb_tip = lm[4]
                    middle_tip = lm[12]
                    wrist = lm[0]
                    dist = abs(thumb_tip.x - middle_tip.x) + abs(thumb_tip.y - middle_tip.y)
                    if dist < _SNAP_WRIST_THRESHOLD:
                        self._snap_count += 1
                        if self._snap_count >= 2:
                            event = GestureEvent(
                                type=GestureType.DUAL_SNAP, timestamp=now
                            )
                            self._loop.call_soon_threadsafe(
                                asyncio.ensure_future, self._bus.publish(event)
                            )
                            self._snap_count = 0
                            time.sleep(0.5)
                            continue

                time.sleep(self._check_interval)
            except Exception:
                logger.exception("Gesture detection error")
                time.sleep(self._check_interval)
```

- [ ] **Step 5: Create `sensors/gesture/fusion.py`**

```python
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
```

- [ ] **Step 6: Verify all sensor modules import**

Run: `cd jarvis-runtime && python -c "from sensors.camera import CameraThread; from sensors.gesture.fusion import GestureFusion; print('OK')"`

Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add jarvis-runtime/sensors/
git commit -m "feat: add camera thread, face verifier, gesture detector, and fusion"
```

---

## Task 8: Voice Sensors

**Files:**
- Create: `jarvis-runtime/sensors/voice/__init__.py`
- Create: `jarvis-runtime/sensors/voice/vad.py`
- Create: `jarvis-runtime/sensors/voice/transcriber.py`
- Create: `jarvis-runtime/sensors/voice/normalizer.py`

- [ ] **Step 1: Create `sensors/voice/__init__.py`**

Empty file.

- [ ] **Step 2: Create `sensors/voice/vad.py`**

```python
"""Energy-based Voice Activity Detection."""

from __future__ import annotations

import logging
import struct
import time

import numpy as np

logger = logging.getLogger(__name__)

_FRAME_DURATION_MS = 30
_SAMPLE_RATE = 16000
_FRAME_SIZE = int(_SAMPLE_RATE * _FRAME_DURATION_MS / 1000)
_ONSET_THRESHOLD = 500
_OFFSET_THRESHOLD = 300
_ONSET_DURATION = 0.3
_OFFSET_DURATION = 0.7


class VoiceActivityDetector:
    def __init__(
        self,
        sample_rate: int = _SAMPLE_RATE,
        onset_threshold: float = _ONSET_THRESHOLD,
        offset_threshold: float = _OFFSET_THRESHOLD,
    ) -> None:
        self._sample_rate = sample_rate
        self._onset_threshold = onset_threshold
        self._offset_threshold = offset_threshold
        self._is_speaking = False
        self._onset_start: float | None = None
        self._offset_start: float | None = None
        self._audio_buffer: list[bytes] = []

    def reset(self) -> None:
        self._is_speaking = False
        self._onset_start = None
        self._offset_start = None
        self._audio_buffer.clear()

    def process_frame(self, frame: bytes) -> bytes | None:
        energy = self._compute_energy(frame)
        now = time.monotonic()

        if not self._is_speaking:
            if energy > self._onset_threshold:
                if self._onset_start is None:
                    self._onset_start = now
                elif now - self._onset_start >= _ONSET_DURATION:
                    self._is_speaking = True
                    self._onset_start = None
                    self._offset_start = None
                    self._audio_buffer.append(frame)
            else:
                self._onset_start = None
            return None

        self._audio_buffer.append(frame)

        if energy < self._offset_threshold:
            if self._offset_start is None:
                self._offset_start = now
            elif now - self._offset_start >= _OFFSET_DURATION:
                self._is_speaking = False
                self._offset_start = None
                result = b"".join(self._audio_buffer)
                self._audio_buffer.clear()
                return result
        else:
            self._offset_start = None

        return None

    @staticmethod
    def _compute_energy(frame: bytes) -> float:
        count = len(frame) // 2
        if count == 0:
            return 0.0
        samples = struct.unpack(f"<{count}h", frame)
        arr = np.array(samples, dtype=np.float32)
        return float(np.sqrt(np.mean(arr ** 2)))
```

- [ ] **Step 3: Create `sensors/voice/transcriber.py`**

```python
"""Speech-to-text via pywhispercpp."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000


class Transcriber:
    def __init__(self, model_name: str = "base.en") -> None:
        self._model_name = model_name
        self._model = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
        self._healthy = False

    @property
    def healthy(self) -> bool:
        return self._healthy

    def initialize(self) -> bool:
        try:
            from pywhispercpp.model import Model
            self._model = Model(self._model_name)
            self._healthy = True
            return True
        except Exception:
            logger.exception("Failed to initialize Whisper model %s", self._model_name)
            return False

    def _transcribe_sync(self, audio_bytes: bytes) -> str:
        if self._model is None:
            return ""
        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        segments = self._model.transcribe(samples)
        return " ".join(seg.text for seg in segments).strip()

    async def transcribe(self, audio_bytes: bytes) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._transcribe_sync, audio_bytes)
```

- [ ] **Step 4: Create `sensors/voice/normalizer.py`**

```python
"""Text normalization for voice transcripts."""

from __future__ import annotations

import re

_FILLER_WORDS = {
    "um", "uh", "er", "ah", "like", "you know", "i mean",
    "basically", "actually", "literally", "so", "well",
}

_FILLER_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(_FILLER_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    text = _FILLER_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text
```

- [ ] **Step 5: Verify imports**

Run: `cd jarvis-runtime && python -c "from sensors.voice.normalizer import normalize; print(normalize('Um, open the uh project'))"`

Expected: `open the project`

- [ ] **Step 6: Commit**

```bash
git add jarvis-runtime/sensors/voice/
git commit -m "feat: add VAD, transcriber, and text normalizer"
```

---

## Task 9: macOS Adapter

**Files:**
- Create: `jarvis-runtime/adapters/macos/__init__.py`
- Create: `jarvis-runtime/adapters/macos/applescript.py`
- Create: `jarvis-runtime/adapters/macos/adapter.py`
- Create: `jarvis-runtime/adapters/macos/process_manager.py`
- Test: `jarvis-runtime/tests/adapters/__init__.py`
- Test: `jarvis-runtime/tests/adapters/test_macos_adapter.py`

- [ ] **Step 1: Write failing tests**

Create `jarvis-runtime/tests/adapters/__init__.py` (empty) and `jarvis-runtime/tests/adapters/test_macos_adapter.py`:

```python
"""Tests for macOS adapter and AppleScript builders."""

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from adapters.macos.applescript import (
    build_open_app_script,
    build_switch_window_script,
    build_notification_script,
    build_open_url_script,
    build_say_script,
)
from adapters.macos.adapter import MacOSAdapter


def test_build_open_app_script():
    script = build_open_app_script("IntelliJ IDEA")
    assert 'tell application "IntelliJ IDEA"' in script
    assert "activate" in script


def test_build_switch_window_script():
    script = build_switch_window_script("Safari", "GitHub")
    assert 'tell application "Safari"' in script
    assert "GitHub" in script


def test_build_notification_script():
    script = build_notification_script("Hello", "World")
    assert "display notification" in script
    assert "Hello" in script or "World" in script


def test_build_open_url_script():
    script = build_open_url_script("https://example.com", "Chrome")
    assert "https://example.com" in script
    assert "Chrome" in script


def test_build_say_script():
    script = build_say_script("Hello Shubham")
    assert "say" in script.lower() or "Hello Shubham" in script


@pytest.mark.asyncio
async def test_adapter_run_script():
    adapter = MacOSAdapter()
    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"result", b"")
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc

        result = await adapter.run_script('tell application "Finder" to activate')
        assert result == "result"
        mock_exec.assert_called_once()


@pytest.mark.asyncio
async def test_adapter_open_application():
    adapter = MacOSAdapter()
    with patch.object(adapter, "run_script", new_callable=AsyncMock, return_value=""):
        result = await adapter.open_application("Safari")
        assert result is True


@pytest.mark.asyncio
async def test_adapter_send_notification():
    adapter = MacOSAdapter()
    with patch.object(adapter, "run_script", new_callable=AsyncMock, return_value=""):
        await adapter.send_notification("Test", "Body")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd jarvis-runtime && python -m pytest tests/adapters/test_macos_adapter.py -v`

Expected: FAIL with import errors.

- [ ] **Step 3: Create `adapters/macos/__init__.py`**

Empty.

- [ ] **Step 4: Create `adapters/macos/applescript.py`**

```python
"""AppleScript builder helpers — return script strings, never execute."""

from __future__ import annotations


def build_open_app_script(app_name: str) -> str:
    return f'tell application "{app_name}" to activate'


def build_switch_window_script(app_name: str, title_pattern: str | None = None) -> str:
    if title_pattern:
        return (
            f'tell application "System Events"\n'
            f'  tell process "{app_name}"\n'
            f'    set frontmost to true\n'
            f'    set targetWindow to first window whose name contains "{title_pattern}"\n'
            f'    perform action "AXRaise" of targetWindow\n'
            f'  end tell\n'
            f'end tell'
        )
    return build_open_app_script(app_name)


def build_notification_script(title: str, body: str) -> str:
    return (
        f'display notification "{body}" with title "{title}"'
    )


def build_open_url_script(url: str, browser: str = "Safari") -> str:
    return (
        f'tell application "{browser}"\n'
        f'  activate\n'
        f'  open location "{url}"\n'
        f'end tell'
    )


def build_say_script(text: str) -> str:
    escaped = text.replace('"', '\\"')
    return f'do shell script "say \\"{escaped}\\""'


def build_play_audio_script(path: str) -> str:
    return f'do shell script "afplay \\"{path}\\"" '


def build_list_running_apps_script() -> str:
    return (
        'tell application "System Events"\n'
        '  set appList to name of every application process whose background only is false\n'
        'end tell\n'
        'return appList'
    )


def build_get_active_workspace_script() -> str:
    return (
        'tell application "System Events"\n'
        '  set frontApp to name of first application process whose frontmost is true\n'
        'end tell\n'
        'return frontApp'
    )
```

- [ ] **Step 5: Create `adapters/macos/adapter.py`**

```python
"""macOS platform adapter — all OS interaction goes through here."""

from __future__ import annotations

import asyncio
import logging

from adapters.base.platform_adapter import AppWindow, PlatformAdapter
from adapters.macos.applescript import (
    build_get_active_workspace_script,
    build_list_running_apps_script,
    build_notification_script,
    build_open_app_script,
    build_open_url_script,
    build_play_audio_script,
    build_say_script,
    build_switch_window_script,
)

logger = logging.getLogger(__name__)


class MacOSAdapter(PlatformAdapter):

    async def run_script(self, script: str) -> str:
        proc = await asyncio.create_subprocess_exec(
            "osascript", "-e", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.warning(
                "osascript failed (rc=%d): %s", proc.returncode, stderr.decode().strip()
            )
        return stdout.decode().strip()

    async def open_application(self, app_name: str) -> bool:
        try:
            await self.run_script(build_open_app_script(app_name))
            return True
        except Exception:
            logger.exception("Failed to open %s", app_name)
            return False

    async def switch_window(
        self, app_name: str, title_pattern: str | None = None
    ) -> bool:
        try:
            await self.run_script(build_switch_window_script(app_name, title_pattern))
            return True
        except Exception:
            logger.exception("Failed to switch window for %s", app_name)
            return False

    async def get_running_apps(self) -> list[AppWindow]:
        raw = await self.run_script(build_list_running_apps_script())
        if not raw:
            return []
        names = [n.strip() for n in raw.split(",")]
        return [AppWindow(app_name=n, title="", pid=0) for n in names if n]

    async def open_url_in_browser(
        self, url: str, browser: str = "Safari"
    ) -> bool:
        try:
            await self.run_script(build_open_url_script(url, browser))
            return True
        except Exception:
            logger.exception("Failed to open URL %s", url)
            return False

    async def send_notification(self, title: str, body: str) -> None:
        await self.run_script(build_notification_script(title, body))

    async def play_audio_file(self, path: str) -> None:
        await self.run_script(build_play_audio_script(path))

    async def get_active_workspace(self) -> str | None:
        result = await self.run_script(build_get_active_workspace_script())
        return result or None
```

- [ ] **Step 6: Create `adapters/macos/process_manager.py`**

```python
"""Process manager — queries running applications via AppleScript."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adapters.macos.adapter import MacOSAdapter

logger = logging.getLogger(__name__)


class ProcessManager:
    def __init__(self, adapter: MacOSAdapter) -> None:
        self._adapter = adapter

    async def get_open_intellij_projects(self) -> list[str]:
        script = (
            'tell application "System Events"\n'
            '  if exists (process "IntelliJ IDEA") then\n'
            '    tell process "IntelliJ IDEA"\n'
            '      set windowNames to name of every window\n'
            '    end tell\n'
            '    return windowNames\n'
            '  else\n'
            '    return ""\n'
            '  end if\n'
            'end tell'
        )
        raw = await self._adapter.run_script(script)
        if not raw:
            return []
        return [w.strip() for w in raw.split(",") if w.strip()]

    async def is_app_running(self, app_name: str) -> bool:
        script = (
            f'tell application "System Events"\n'
            f'  return exists (process "{app_name}")\n'
            f'end tell'
        )
        result = await self._adapter.run_script(script)
        return result.lower() == "true"
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd jarvis-runtime && python -m pytest tests/adapters/test_macos_adapter.py -v`

Expected: All tests PASS.

- [ ] **Step 8: Commit**

```bash
git add jarvis-runtime/adapters/macos/ jarvis-runtime/tests/adapters/
git commit -m "feat: add macOS adapter with AppleScript builders"
```

---

## Task 10: Windows Stub Adapter

**Files:**
- Create: `jarvis-runtime/adapters/windows/__init__.py`
- Create: `jarvis-runtime/adapters/windows/adapter.py`

- [ ] **Step 1: Create both files**

`__init__.py` is empty. `adapter.py`:

```python
"""Windows adapter stub — all methods raise NotImplementedError."""

from __future__ import annotations

from adapters.base.platform_adapter import AppWindow, PlatformAdapter


class WindowsAdapter(PlatformAdapter):

    async def open_application(self, app_name: str) -> bool:
        raise NotImplementedError("Windows adapter not yet implemented")

    async def switch_window(
        self, app_name: str, title_pattern: str | None = None
    ) -> bool:
        raise NotImplementedError("Windows adapter: switch_window not implemented")

    async def run_script(self, script: str) -> str:
        raise NotImplementedError("Windows adapter: run_script not implemented")

    async def get_running_apps(self) -> list[AppWindow]:
        raise NotImplementedError("Windows adapter: get_running_apps not implemented")

    async def open_url_in_browser(self, url: str, browser: str = "Safari") -> bool:
        raise NotImplementedError("Windows adapter: open_url_in_browser not implemented")

    async def send_notification(self, title: str, body: str) -> None:
        raise NotImplementedError("Windows adapter: send_notification not implemented")

    async def play_audio_file(self, path: str) -> None:
        raise NotImplementedError("Windows adapter: play_audio_file not implemented")

    async def get_active_workspace(self) -> str | None:
        raise NotImplementedError("Windows adapter: get_active_workspace not implemented")
```

- [ ] **Step 2: Verify import on macOS**

Run: `cd jarvis-runtime && python -c "from adapters.windows.adapter import WindowsAdapter; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add jarvis-runtime/adapters/windows/
git commit -m "feat: add Windows adapter stub"
```

---

## Task 11: Tool Plugins

**Files:**
- Create: `jarvis-runtime/tools/intellij/__init__.py`
- Create: `jarvis-runtime/tools/intellij/open_project.py`
- Create: `jarvis-runtime/tools/intellij/switch_window.py`
- Create: `jarvis-runtime/tools/chrome/__init__.py`
- Create: `jarvis-runtime/tools/chrome/open_url.py`
- Create: `jarvis-runtime/tools/notes/__init__.py`
- Create: `jarvis-runtime/tools/notes/open_notes.py`
- Create: `jarvis-runtime/tools/music/__init__.py`
- Create: `jarvis-runtime/tools/music/play_music.py`
- Create: `jarvis-runtime/tools/workspace_modes/__init__.py`
- Create: `jarvis-runtime/tools/workspace_modes/open_workspace_mode.py`
- Test: `jarvis-runtime/tests/tools/__init__.py`
- Test: `jarvis-runtime/tests/tools/test_workspace_mode.py`

- [ ] **Step 1: Write failing test for workspace mode**

Create `jarvis-runtime/tests/tools/__init__.py` (empty) and `jarvis-runtime/tests/tools/test_workspace_mode.py`:

```python
"""Tests for workspace mode tool."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools._base import ToolResult
from tools.workspace_modes.open_workspace_mode import OpenWorkspaceModeTool


@pytest.mark.asyncio
async def test_workspace_mode_executes_steps_in_order():
    tool = OpenWorkspaceModeTool()
    mock_executor = AsyncMock()
    mock_executor.execute_tool = AsyncMock(
        return_value=ToolResult(success=True, message="ok")
    )

    config = {
        "workspace_modes": {
            "office": {
                "description": "Office mode",
                "steps": [
                    {"tool": "open_project", "params": {"project": "office"}},
                    {"tool": "open_url", "params": {"url": "https://mail.google.com", "browser": "Chrome"}},
                ],
            }
        }
    }
    params = {"mode": "office", "_config": config, "_executor": mock_executor}
    adapter = AsyncMock()

    result = await tool.execute(params, adapter)

    assert result.success is True
    assert mock_executor.execute_tool.call_count == 2
    calls = mock_executor.execute_tool.call_args_list
    assert calls[0][1]["tool_name"] == "open_project"
    assert calls[1][1]["tool_name"] == "open_url"


@pytest.mark.asyncio
async def test_workspace_mode_unknown_mode():
    tool = OpenWorkspaceModeTool()
    params = {"mode": "nonexistent", "_config": {"workspace_modes": {}}}
    adapter = AsyncMock()

    result = await tool.execute(params, adapter)
    assert result.success is False


@pytest.mark.asyncio
async def test_workspace_mode_partial_failure():
    tool = OpenWorkspaceModeTool()
    mock_executor = AsyncMock()
    mock_executor.execute_tool = AsyncMock(
        side_effect=[
            ToolResult(success=True, message="ok"),
            ToolResult(success=False, message="failed"),
        ]
    )
    config = {
        "workspace_modes": {
            "test": {
                "description": "Test",
                "steps": [
                    {"tool": "a", "params": {}},
                    {"tool": "b", "params": {}},
                ],
            }
        }
    }
    params = {"mode": "test", "_config": config, "_executor": mock_executor}
    adapter = AsyncMock()

    result = await tool.execute(params, adapter)
    assert result.success is True
    assert "1 failed" in result.message
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd jarvis-runtime && python -m pytest tests/tools/test_workspace_mode.py -v`

Expected: FAIL with import errors.

- [ ] **Step 3: Create all tool package `__init__.py` files**

Create empty `__init__.py` in: `tools/intellij/`, `tools/chrome/`, `tools/notes/`, `tools/music/`, `tools/workspace_modes/`.

- [ ] **Step 4: Create `tools/intellij/open_project.py`**

```python
"""Tool: Open an IntelliJ project."""

from __future__ import annotations

from typing import Any

from tools._base import BaseTool, ToolResult


class OpenProjectTool(BaseTool):

    @property
    def name(self) -> str:
        return "open_project"

    @property
    def description(self) -> str:
        return "Opens a project in IntelliJ IDEA by name"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Project name from config"},
            },
            "required": ["project"],
        }

    async def execute(self, params: dict[str, Any], adapter) -> ToolResult:
        try:
            project = params.get("project", "")
            await adapter.open_application("IntelliJ IDEA")
            return ToolResult(success=True, message=f"Opened project: {project}")
        except Exception as e:
            return ToolResult(success=False, message=str(e))
```

- [ ] **Step 5: Create `tools/intellij/switch_window.py`**

```python
"""Tool: Switch to a specific IntelliJ window."""

from __future__ import annotations

from typing import Any

from tools._base import BaseTool, ToolResult


class SwitchWindowTool(BaseTool):

    @property
    def name(self) -> str:
        return "switch_intellij_window"

    @property
    def description(self) -> str:
        return "Switches to a specific IntelliJ IDEA window by title pattern"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title_pattern": {"type": "string", "description": "Window title to match"},
            },
            "required": ["title_pattern"],
        }

    async def execute(self, params: dict[str, Any], adapter) -> ToolResult:
        try:
            pattern = params.get("title_pattern", "")
            success = await adapter.switch_window("IntelliJ IDEA", pattern)
            if success:
                return ToolResult(success=True, message=f"Switched to window: {pattern}")
            return ToolResult(success=False, message=f"Window not found: {pattern}")
        except Exception as e:
            return ToolResult(success=False, message=str(e))
```

- [ ] **Step 6: Create `tools/chrome/open_url.py`**

```python
"""Tool: Open a URL in a browser."""

from __future__ import annotations

from typing import Any

from tools._base import BaseTool, ToolResult


class OpenUrlTool(BaseTool):

    @property
    def name(self) -> str:
        return "open_url"

    @property
    def description(self) -> str:
        return "Opens a URL in the specified browser"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to open"},
                "browser": {"type": "string", "description": "Browser name", "default": "Chrome"},
            },
            "required": ["url"],
        }

    async def execute(self, params: dict[str, Any], adapter) -> ToolResult:
        try:
            url = params["url"]
            browser = params.get("browser", "Chrome")
            success = await adapter.open_url_in_browser(url, browser)
            if success:
                return ToolResult(success=True, message=f"Opened {url} in {browser}")
            return ToolResult(success=False, message=f"Failed to open {url}")
        except Exception as e:
            return ToolResult(success=False, message=str(e))
```

- [ ] **Step 7: Create `tools/notes/open_notes.py`**

```python
"""Tool: Open Apple Notes."""

from __future__ import annotations

from typing import Any

from tools._base import BaseTool, ToolResult


class OpenNotesTool(BaseTool):

    @property
    def name(self) -> str:
        return "open_notes"

    @property
    def description(self) -> str:
        return "Opens Apple Notes application"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, params: dict[str, Any], adapter) -> ToolResult:
        try:
            await adapter.open_application("Notes")
            return ToolResult(success=True, message="Opened Notes")
        except Exception as e:
            return ToolResult(success=False, message=str(e))
```

- [ ] **Step 8: Create `tools/music/play_music.py`**

```python
"""Tool: Play music via Apple Music."""

from __future__ import annotations

from typing import Any

from tools._base import BaseTool, ToolResult


class PlayMusicTool(BaseTool):

    @property
    def name(self) -> str:
        return "play_music"

    @property
    def description(self) -> str:
        return "Plays music or a playlist in Apple Music"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Playlist or song name to play"},
            },
            "required": [],
        }

    async def execute(self, params: dict[str, Any], adapter) -> ToolResult:
        try:
            query = params.get("query", "")
            script = (
                'tell application "Music"\n'
                '  activate\n'
                '  play\n'
                'end tell'
            )
            if query:
                script = (
                    f'tell application "Music"\n'
                    f'  activate\n'
                    f'  play (first playlist whose name contains "{query}")\n'
                    f'end tell'
                )
            await adapter.run_script(script)
            return ToolResult(success=True, message=f"Playing music: {query or 'resumed'}")
        except Exception as e:
            return ToolResult(success=False, message=str(e))
```

- [ ] **Step 9: Create `tools/workspace_modes/open_workspace_mode.py`**

```python
"""Tool: Execute a workspace mode — runs a sequence of tool steps."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from tools._base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_STEP_DELAY = 0.5


class OpenWorkspaceModeTool(BaseTool):

    @property
    def name(self) -> str:
        return "open_workspace_mode"

    @property
    def description(self) -> str:
        return "Activates a named workspace mode, executing each configured step sequentially"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "description": "Workspace mode name from config"},
            },
            "required": ["mode"],
        }

    async def execute(self, params: dict[str, Any], adapter) -> ToolResult:
        try:
            mode_name = params.get("mode", "")
            config = params.get("_config", {})
            executor = params.get("_executor")

            modes = config.get("workspace_modes", {})
            mode = modes.get(mode_name)
            if mode is None:
                return ToolResult(
                    success=False,
                    message=f"Unknown workspace mode: {mode_name}",
                )

            steps = mode.get("steps", [])
            if isinstance(mode, dict) and "steps" not in mode:
                steps = mode.get("steps", [])

            succeeded = 0
            failed = 0
            for step in steps:
                tool_name = step.get("tool", "")
                tool_params = step.get("params", {})
                if executor is not None:
                    result = await executor.execute_tool(
                        tool_name=tool_name, params=tool_params, adapter=adapter
                    )
                    if result.success:
                        succeeded += 1
                    else:
                        failed += 1
                        logger.warning(
                            "Workspace step %s failed: %s", tool_name, result.message
                        )
                await asyncio.sleep(_STEP_DELAY)

            summary = f"Workspace '{mode_name}': {succeeded} succeeded"
            if failed:
                summary += f", {failed} failed"
            return ToolResult(success=True, message=summary)
        except Exception as e:
            return ToolResult(success=False, message=str(e))
```

- [ ] **Step 10: Run workspace mode tests**

Run: `cd jarvis-runtime && python -m pytest tests/tools/test_workspace_mode.py -v`

Expected: All 3 tests PASS.

- [ ] **Step 11: Commit**

```bash
git add jarvis-runtime/tools/ jarvis-runtime/tests/tools/
git commit -m "feat: add all tool plugins — intellij, chrome, notes, music, workspace_modes"
```

---

## Task 12: Memory Subsystems

**Files:**
- Create: `jarvis-runtime/memory/__init__.py`
- Create: `jarvis-runtime/memory/vector/__init__.py`
- Create: `jarvis-runtime/memory/vector/embedder.py`
- Create: `jarvis-runtime/memory/vector/client.py`
- Create: `jarvis-runtime/memory/behavioral/__init__.py`
- Create: `jarvis-runtime/memory/behavioral/tracker.py`
- Create: `jarvis-runtime/memory/behavioral/query.py`
- Create: `jarvis-runtime/memory/session_cache/__init__.py`
- Create: `jarvis-runtime/memory/session_cache/redis_client.py`

- [ ] **Step 1: Create all package `__init__.py` files**

Empty files in: `memory/`, `memory/vector/`, `memory/behavioral/`, `memory/session_cache/`.

- [ ] **Step 2: Create `memory/vector/embedder.py`**

```python
"""Text embedder via Ollama embeddings API (nomic-embed-text)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config.models import OllamaConfig

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, config: OllamaConfig) -> None:
        self._config = config
        self._healthy = False

    @property
    def healthy(self) -> bool:
        return self._healthy

    async def initialize(self) -> bool:
        try:
            import ollama
            response = await ollama.AsyncClient(
                host=f"http://{self._config.host}:{self._config.port}"
            ).embeddings(model=self._config.embed_model, prompt="test")
            if response and "embedding" in response:
                self._healthy = True
                return True
            logger.warning("Embedding model %s returned empty response", self._config.embed_model)
            return False
        except Exception:
            logger.exception(
                "Failed to initialize embedding model %s — run 'ollama pull %s'",
                self._config.embed_model,
                self._config.embed_model,
            )
            return False

    async def embed(self, text: str) -> list[float] | None:
        if not self._healthy:
            return None
        try:
            import ollama
            response = await ollama.AsyncClient(
                host=f"http://{self._config.host}:{self._config.port}"
            ).embeddings(model=self._config.embed_model, prompt=text)
            return response.get("embedding")
        except Exception:
            logger.exception("Embedding failed for text: %s...", text[:50])
            return None
```

- [ ] **Step 3: Create `memory/vector/client.py`**

```python
"""ChromaDB vector memory client — subscribes to MemoryWriteEvent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config.models import MemoryConfig
    from memory.vector.embedder import Embedder
    from runtime.event_bus import EventBus

logger = logging.getLogger(__name__)


class VectorMemoryClient:
    def __init__(
        self, config: MemoryConfig, embedder: Embedder, event_bus: EventBus
    ) -> None:
        self._config = config
        self._embedder = embedder
        self._bus = event_bus
        self._collection = None
        self._healthy = False

    @property
    def healthy(self) -> bool:
        return self._healthy

    async def initialize(self) -> bool:
        try:
            import chromadb
            from pathlib import Path

            path = Path(self._config.chroma_path)
            path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(path))
            self._collection = client.get_or_create_collection("jarvis_commands")
            self._healthy = True
            return True
        except Exception:
            logger.exception("Failed to initialize ChromaDB")
            return False

    async def on_memory_write(self, event) -> None:
        if not self._healthy or self._collection is None:
            return
        try:
            embedding = await self._embedder.embed(event.command_text)
            if embedding is None:
                return
            doc_id = f"{event.session_id}_{event.timestamp}"
            self._collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[event.command_text],
                metadatas=[{
                    "tool_name": event.tool_name,
                    "timestamp": str(event.timestamp),
                    "session_id": event.session_id,
                }],
            )
        except Exception:
            logger.exception("Vector memory write failed")

    async def search(self, query: str, n_results: int = 5) -> list[dict]:
        if not self._healthy or self._collection is None:
            return []
        try:
            embedding = await self._embedder.embed(query)
            if embedding is None:
                return []
            results = self._collection.query(
                query_embeddings=[embedding], n_results=n_results
            )
            return [
                {"document": doc, "metadata": meta}
                for doc, meta in zip(
                    results.get("documents", [[]])[0],
                    results.get("metadatas", [[]])[0],
                )
            ]
        except Exception:
            logger.exception("Vector search failed")
            return []
```

- [ ] **Step 4: Create `memory/behavioral/tracker.py`**

```python
"""Behavioral tracker — records tool executions in SQLite."""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config.models import MemoryConfig
    from runtime.event_bus import EventBus, ToolExecutionEvent

logger = logging.getLogger(__name__)


class BehavioralTracker:
    def __init__(self, config: MemoryConfig, event_bus: EventBus) -> None:
        self._config = config
        self._bus = event_bus
        self._db = None
        self._healthy = False

    @property
    def healthy(self) -> bool:
        return self._healthy

    async def initialize(self) -> bool:
        try:
            import aiosqlite
            from pathlib import Path

            db_path = Path(self._config.behavioral_db)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = await aiosqlite.connect(str(db_path))
            await self._db.execute(
                """CREATE TABLE IF NOT EXISTS commands (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    tool_name TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    success INTEGER NOT NULL
                )"""
            )
            await self._db.commit()
            self._healthy = True
            return True
        except Exception:
            logger.exception("Failed to initialize behavioral store")
            return False

    async def record(self, event: ToolExecutionEvent) -> None:
        if not self._healthy or self._db is None:
            return
        try:
            params_json = json.dumps(event.result.data) if event.result else "{}"
            await self._db.execute(
                "INSERT INTO commands (timestamp, tool_name, params_json, session_id, success) VALUES (?, ?, ?, ?, ?)",
                (time.time(), event.tool_name, params_json, event.session_id, int(event.success)),
            )
            await self._db.commit()
        except Exception:
            logger.exception("Failed to record command")

    async def on_tool_execution(self, event: ToolExecutionEvent) -> None:
        await self.record(event)

    async def get_frequent_tools(self, limit: int = 5) -> list[dict]:
        if not self._healthy or self._db is None:
            return []
        try:
            cursor = await self._db.execute(
                "SELECT tool_name, COUNT(*) as cnt FROM commands WHERE success = 1 "
                "GROUP BY tool_name ORDER BY cnt DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
            return [{"tool_name": r[0], "count": r[1]} for r in rows]
        except Exception:
            logger.exception("Failed to query frequent tools")
            return []

    async def get_time_of_day_pattern(self) -> dict[str, list[str]]:
        if not self._healthy or self._db is None:
            return {}
        try:
            cursor = await self._db.execute(
                "SELECT tool_name, timestamp FROM commands WHERE success = 1 "
                "ORDER BY timestamp DESC LIMIT 100"
            )
            rows = await cursor.fetchall()
            import datetime
            patterns: dict[str, list[str]] = {"morning": [], "afternoon": [], "evening": [], "night": []}
            for tool_name, ts in rows:
                hour = datetime.datetime.fromtimestamp(ts).hour
                if 5 <= hour < 12:
                    period = "morning"
                elif 12 <= hour < 17:
                    period = "afternoon"
                elif 17 <= hour < 21:
                    period = "evening"
                else:
                    period = "night"
                if tool_name not in patterns[period]:
                    patterns[period].append(tool_name)
            return patterns
        except Exception:
            logger.exception("Failed to query time patterns")
            return {}

    async def close(self) -> None:
        if self._db:
            await self._db.close()
```

- [ ] **Step 5: Create `memory/behavioral/query.py`**

```python
"""Read-only query interface for behavioral data."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memory.behavioral.tracker import BehavioralTracker


class BehavioralQuery:
    def __init__(self, tracker: BehavioralTracker) -> None:
        self._tracker = tracker

    async def get_recent_tools(self, limit: int = 3) -> list[str]:
        frequent = await self._tracker.get_frequent_tools(limit)
        return [entry["tool_name"] for entry in frequent]

    async def get_time_of_day_pattern(self) -> dict[str, list[str]]:
        return await self._tracker.get_time_of_day_pattern()
```

- [ ] **Step 6: Create `memory/session_cache/redis_client.py`**

```python
"""Session cache — Redis with in-memory dict fallback."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class SessionCache:
    def __init__(self, redis_enabled: bool = False, redis_url: str = "redis://localhost:6379/0") -> None:
        self._redis_enabled = redis_enabled
        self._redis_url = redis_url
        self._redis = None
        self._fallback: dict[str, str] = {}
        self._using_fallback = True

    @property
    def healthy(self) -> bool:
        return True

    async def initialize(self) -> bool:
        if not self._redis_enabled:
            logger.info("Redis disabled — using in-memory session cache")
            self._using_fallback = True
            return True
        try:
            import aioredis
            self._redis = await aioredis.from_url(self._redis_url)
            await self._redis.ping()
            self._using_fallback = False
            logger.info("Connected to Redis at %s", self._redis_url)
            return True
        except Exception:
            logger.warning("Redis unavailable — falling back to in-memory cache")
            self._using_fallback = True
            return True

    async def set(self, key: str, value: Any) -> None:
        encoded = json.dumps(value)
        if self._using_fallback:
            self._fallback[key] = encoded
        else:
            try:
                await self._redis.set(key, encoded)
            except Exception:
                logger.warning("Redis set failed, using fallback")
                self._fallback[key] = encoded

    async def get(self, key: str) -> Any | None:
        if self._using_fallback:
            raw = self._fallback.get(key)
        else:
            try:
                raw = await self._redis.get(key)
                if isinstance(raw, bytes):
                    raw = raw.decode()
            except Exception:
                raw = self._fallback.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    async def append_command(self, session_id: str, command: dict) -> None:
        key = f"session:{session_id}:commands"
        commands = await self.get(key) or []
        commands.append(command)
        commands = commands[-10:]
        await self.set(key, commands)

    async def get_recent_commands(self, session_id: str, limit: int = 5) -> list[dict]:
        key = f"session:{session_id}:commands"
        commands = await self.get(key) or []
        return commands[-limit:]

    async def set_session_state(self, session_id: str, state: str) -> None:
        await self.set(f"session:{session_id}:state", state)

    async def set_session_start(self, session_id: str) -> None:
        await self.set(f"session:{session_id}:start", time.time())

    async def close(self) -> None:
        if self._redis and not self._using_fallback:
            await self._redis.close()
```

- [ ] **Step 7: Verify imports**

Run: `cd jarvis-runtime && python -c "from memory.session_cache.redis_client import SessionCache; from memory.behavioral.tracker import BehavioralTracker; print('OK')"`

Expected: `OK`

- [ ] **Step 8: Commit**

```bash
git add jarvis-runtime/memory/
git commit -m "feat: add vector memory, behavioral tracker, and session cache"
```

---

## Task 13: Intent System

**Files:**
- Create: `jarvis-runtime/core/intent/__init__.py`
- Create: `jarvis-runtime/core/intent/router.py`
- Create: `jarvis-runtime/core/intent/slot_filler.py`
- Create: `jarvis-runtime/core/intent/prompts/system.j2`
- Create: `jarvis-runtime/core/intent/prompts/user.j2`
- Test: `jarvis-runtime/tests/core/test_intent_router.py`

- [ ] **Step 1: Write failing tests**

Create `jarvis-runtime/tests/core/test_intent_router.py`:

```python
"""Tests for intent router."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.intent.router import IntentResult, IntentRouter
from tools._base import ToolMeta


@pytest.fixture
def tool_metas():
    return [
        ToolMeta(
            name="open_project",
            description="Opens a project in IntelliJ",
            parameters_schema={"type": "object", "properties": {"project": {"type": "string"}}, "required": ["project"]},
        ),
        ToolMeta(
            name="open_url",
            description="Opens a URL in a browser",
            parameters_schema={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
        ),
    ]


@pytest.mark.asyncio
async def test_router_parses_valid_response(tool_metas):
    mock_response = {
        "message": {
            "content": json.dumps({
                "tool": "open_project",
                "params": {"project": "office"},
                "confidence": 0.95,
            })
        }
    }
    with patch("ollama.AsyncClient") as MockClient:
        instance = MockClient.return_value
        instance.chat = AsyncMock(return_value=mock_response)

        router = IntentRouter(host="localhost", port=11434, model="llama3.1")
        result = await router.route("open the office project", tool_metas, [])

    assert isinstance(result, IntentResult)
    assert result.tool_name == "open_project"
    assert result.params == {"project": "office"}
    assert result.confidence == 0.95


@pytest.mark.asyncio
async def test_router_returns_none_on_low_confidence(tool_metas):
    mock_response = {
        "message": {
            "content": json.dumps({
                "tool": "open_project",
                "params": {"project": "office"},
                "confidence": 0.3,
            })
        }
    }
    with patch("ollama.AsyncClient") as MockClient:
        instance = MockClient.return_value
        instance.chat = AsyncMock(return_value=mock_response)

        router = IntentRouter(host="localhost", port=11434, model="llama3.1")
        result = await router.route("something vague", tool_metas, [])

    assert result.tool_name is None


@pytest.mark.asyncio
async def test_router_handles_invalid_json(tool_metas):
    mock_response = {
        "message": {"content": "I don't understand that request."}
    }
    with patch("ollama.AsyncClient") as MockClient:
        instance = MockClient.return_value
        instance.chat = AsyncMock(return_value=mock_response)

        router = IntentRouter(host="localhost", port=11434, model="llama3.1")
        result = await router.route("gibberish", tool_metas, [])

    assert result.tool_name is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd jarvis-runtime && python -m pytest tests/core/test_intent_router.py -v`

Expected: FAIL with import errors.

- [ ] **Step 3: Create `core/intent/__init__.py`**

Empty.

- [ ] **Step 4: Create `core/intent/prompts/system.j2`**

```jinja2
You are Jarvis, a macOS desktop assistant. Your job is to interpret the user's voice command and match it to one of the available tools.

Available tools:
```json
{{ tools_json }}
```

{% if recent_commands %}
Recent commands (for context):
{% for cmd in recent_commands %}
- {{ cmd }}
{% endfor %}
{% endif %}

Respond with a JSON object containing:
- "tool": the tool name to execute (or null if no tool matches)
- "params": a dictionary of parameters for the tool
- "confidence": a float from 0.0 to 1.0 indicating your confidence in the match

Respond ONLY with valid JSON, no explanation.
```

- [ ] **Step 5: Create `core/intent/prompts/user.j2`**

```jinja2
User command: "{{ transcript }}"
```

- [ ] **Step 6: Create `core/intent/router.py`**

```python
"""Intent router — sends transcripts to Ollama and parses tool matches."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

_CONFIDENCE_THRESHOLD = 0.6
_PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class IntentResult:
    tool_name: str | None
    params: dict[str, Any]
    confidence: float
    raw_response: str


class IntentRouter:
    def __init__(self, host: str, port: int, model: str) -> None:
        self._host = host
        self._port = port
        self._model = model
        self._jinja = Environment(
            loader=FileSystemLoader(str(_PROMPTS_DIR)),
            autoescape=False,
        )
        self._healthy = False

    @property
    def healthy(self) -> bool:
        return self._healthy

    async def initialize(self) -> bool:
        try:
            import ollama
            client = ollama.AsyncClient(host=f"http://{self._host}:{self._port}")
            await client.list()
            self._healthy = True
            return True
        except Exception:
            logger.exception("Failed to connect to Ollama")
            return False

    async def route(
        self,
        transcript: str,
        tool_metas: list,
        recent_commands: list[str],
    ) -> IntentResult:
        if not self._healthy:
            return IntentResult(
                tool_name=None, params={}, confidence=0.0,
                raw_response="Intent routing unavailable",
            )

        tools_json = json.dumps(
            [{"name": t.name, "description": t.description, "parameters": t.parameters_schema} for t in tool_metas],
            indent=2,
        )

        system_template = self._jinja.get_template("system.j2")
        user_template = self._jinja.get_template("user.j2")

        system_prompt = system_template.render(
            tools_json=tools_json, recent_commands=recent_commands
        )
        user_prompt = user_template.render(transcript=transcript)

        try:
            import ollama
            client = ollama.AsyncClient(host=f"http://{self._host}:{self._port}")
            response = await client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = response["message"]["content"]
            return self._parse_response(raw)
        except Exception:
            logger.exception("Ollama chat failed")
            return IntentResult(
                tool_name=None, params={}, confidence=0.0,
                raw_response="LLM call failed",
            )

    def _parse_response(self, raw: str) -> IntentResult:
        try:
            data = json.loads(raw)
            tool_name = data.get("tool")
            params = data.get("params", {})
            confidence = float(data.get("confidence", 0.0))

            if confidence < _CONFIDENCE_THRESHOLD:
                return IntentResult(
                    tool_name=None, params=params, confidence=confidence,
                    raw_response=raw,
                )
            return IntentResult(
                tool_name=tool_name, params=params, confidence=confidence,
                raw_response=raw,
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("Failed to parse LLM response: %s", raw[:200])
            return IntentResult(
                tool_name=None, params={}, confidence=0.0, raw_response=raw,
            )
```

- [ ] **Step 7: Create `core/intent/slot_filler.py`**

```python
"""Slot filler — fills missing tool parameters from context."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SlotFillingResult:
    params: dict[str, Any]
    unfilled: list[str] = field(default_factory=list)


class SlotFiller:

    async def fill(
        self,
        params: dict[str, Any],
        parameters_schema: dict[str, Any],
        recent_commands: list[dict],
        time_patterns: dict[str, list[str]],
    ) -> SlotFillingResult:
        required = parameters_schema.get("required", [])
        properties = parameters_schema.get("properties", {})
        filled = dict(params)
        unfilled = []

        for param_name in required:
            if param_name in filled and filled[param_name]:
                continue

            value = self._fill_from_recent(param_name, recent_commands)
            if value is not None:
                filled[param_name] = value
                continue

            value = self._fill_from_patterns(param_name, time_patterns)
            if value is not None:
                filled[param_name] = value
                continue

            unfilled.append(param_name)

        return SlotFillingResult(params=filled, unfilled=unfilled)

    @staticmethod
    def _fill_from_recent(param_name: str, recent_commands: list[dict]) -> Any | None:
        for cmd in reversed(recent_commands):
            cmd_params = cmd.get("params", {})
            if param_name in cmd_params:
                return cmd_params[param_name]
        return None

    @staticmethod
    def _fill_from_patterns(param_name: str, patterns: dict[str, list[str]]) -> Any | None:
        return None
```

- [ ] **Step 8: Run tests**

Run: `cd jarvis-runtime && python -m pytest tests/core/test_intent_router.py -v`

Expected: All 3 tests PASS.

- [ ] **Step 9: Commit**

```bash
git add jarvis-runtime/core/intent/ jarvis-runtime/tests/core/test_intent_router.py
git commit -m "feat: add intent router, slot filler, and Jinja2 prompt templates"
```

---

## Task 14: Pipelines

**Files:**
- Create: `jarvis-runtime/core/pipeline/__init__.py`
- Create: `jarvis-runtime/core/pipeline/wake_pipeline.py`
- Create: `jarvis-runtime/core/pipeline/greeting_pipeline.py`

- [ ] **Step 1: Create `core/pipeline/__init__.py`**

Empty.

- [ ] **Step 2: Create `core/pipeline/wake_pipeline.py`**

```python
"""Wake pipeline — initializes session context on WAKE_PENDING."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.session.session_context import SessionContext
from runtime.event_bus import SessionState, SessionStateChangedEvent

if TYPE_CHECKING:
    from memory.session_cache.redis_client import SessionCache

logger = logging.getLogger(__name__)


class WakePipeline:
    def __init__(self, session_cache: SessionCache) -> None:
        self._cache = session_cache
        self._current_context: SessionContext | None = None

    @property
    def current_context(self) -> SessionContext | None:
        return self._current_context

    async def on_state_changed(self, event: SessionStateChangedEvent) -> None:
        if event.new_state == SessionState.WAKE_PENDING:
            ctx = SessionContext()
            self._current_context = ctx
            await self._cache.set_session_start(ctx.session_id)
            await self._cache.set_session_state(ctx.session_id, "WAKE_PENDING")
            logger.info("Session context created: %s", ctx.session_id)
        elif event.new_state == SessionState.SLEEP:
            self._current_context = None
```

- [ ] **Step 3: Create `core/pipeline/greeting_pipeline.py`**

```python
"""Greeting pipeline — speaks a short greeting when ACTIVE_SESSION starts."""

from __future__ import annotations

import datetime
import logging
from typing import TYPE_CHECKING

from runtime.event_bus import SessionState, SessionStateChangedEvent

if TYPE_CHECKING:
    from adapters.base.platform_adapter import PlatformAdapter
    from adapters.macos.process_manager import ProcessManager
    from memory.behavioral.query import BehavioralQuery
    from memory.session_cache.redis_client import SessionCache

logger = logging.getLogger(__name__)


class GreetingPipeline:
    def __init__(
        self,
        adapter: PlatformAdapter,
        behavioral_query: BehavioralQuery,
        session_cache: SessionCache,
        process_manager: ProcessManager | None = None,
    ) -> None:
        self._adapter = adapter
        self._query = behavioral_query
        self._cache = session_cache
        self._process_manager = process_manager

    async def on_state_changed(self, event: SessionStateChangedEvent) -> None:
        if event.new_state != SessionState.ACTIVE_SESSION:
            return
        try:
            greeting = await self._build_greeting(event.session_id)
            from adapters.macos.applescript import build_say_script
            await self._adapter.run_script(build_say_script(greeting))
        except Exception:
            logger.exception("Greeting pipeline failed")

    async def _build_greeting(self, session_id: str) -> str:
        now = datetime.datetime.now()
        hour = now.hour
        if 5 <= hour < 12:
            time_greeting = "Good morning"
        elif 12 <= hour < 17:
            time_greeting = "Good afternoon"
        elif 17 <= hour < 21:
            time_greeting = "Good evening"
        else:
            time_greeting = "Hey"

        day_name = now.strftime("%A")

        recent_tools = await self._query.get_recent_tools(limit=3)
        tool_hint = ""
        if recent_tools:
            tool_hint = f" Your recent tools: {', '.join(recent_tools)}."

        projects_hint = ""
        if self._process_manager:
            try:
                projects = await self._process_manager.get_open_intellij_projects()
                if projects:
                    projects_hint = f" IntelliJ has {', '.join(projects[:2])} open."
            except Exception:
                pass

        greeting = f"{time_greeting}! Happy {day_name}.{tool_hint}{projects_hint} Ready."
        if len(greeting) > 200:
            greeting = f"{time_greeting}! Ready."
        return greeting
```

- [ ] **Step 4: Verify imports**

Run: `cd jarvis-runtime && python -c "from core.pipeline.wake_pipeline import WakePipeline; from core.pipeline.greeting_pipeline import GreetingPipeline; print('OK')"`

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add jarvis-runtime/core/pipeline/
git commit -m "feat: add wake and greeting pipelines"
```

---

## Task 15: Executor

**Files:**
- Create: `jarvis-runtime/core/registry/executor.py`
- Test: `jarvis-runtime/tests/core/test_executor.py`

- [ ] **Step 1: Write failing tests**

Create `jarvis-runtime/tests/core/test_executor.py`:

```python
"""Tests for tool executor."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from runtime.event_bus import (
    EventBus,
    IntentRoutedEvent,
    MemoryWriteEvent,
    ToolCancelEvent,
    ToolExecutionEvent,
    ToolResult,
)
from core.registry.executor import ToolExecutor


@pytest_asyncio.fixture
async def bus():
    b = EventBus()
    await b.start()
    yield b
    await b.stop()


@pytest.fixture
def mock_registry():
    registry = MagicMock()
    tool = AsyncMock()
    tool.name = "test_tool"
    tool.execute = AsyncMock(return_value=ToolResult(success=True, message="ok"))
    registry.get.return_value = tool
    return registry


@pytest.fixture
def mock_adapter():
    return AsyncMock()


@pytest.mark.asyncio
async def test_executor_publishes_tool_execution_event(bus, mock_registry, mock_adapter):
    events = []
    bus.subscribe(ToolExecutionEvent, lambda e: events.append(e))
    await asyncio.sleep(0.05)

    executor = ToolExecutor(bus, mock_registry, mock_adapter, timeout=30)
    event = IntentRoutedEvent(tool_name="test_tool", params={"a": 1}, confidence=0.9, session_id="s1")
    await executor.on_intent_routed(event)
    await asyncio.sleep(0.1)

    assert len(events) == 1
    assert events[0].success is True
    assert events[0].tool_name == "test_tool"


@pytest.mark.asyncio
async def test_executor_publishes_memory_write_event(bus, mock_registry, mock_adapter):
    events = []
    bus.subscribe(MemoryWriteEvent, lambda e: events.append(e))
    await asyncio.sleep(0.05)

    executor = ToolExecutor(bus, mock_registry, mock_adapter, timeout=30)
    event = IntentRoutedEvent(tool_name="test_tool", params={}, confidence=0.9, session_id="s1")
    await executor.on_intent_routed(event)
    await asyncio.sleep(0.1)

    assert len(events) == 1
    assert events[0].tool_name == "test_tool"


@pytest.mark.asyncio
async def test_executor_handles_timeout(bus, mock_adapter):
    registry = MagicMock()
    tool = AsyncMock()
    tool.name = "slow_tool"

    async def slow_execute(params, adapter):
        await asyncio.sleep(10)
        return ToolResult(success=True, message="done")

    tool.execute = slow_execute
    registry.get.return_value = tool

    events = []
    bus.subscribe(ToolExecutionEvent, lambda e: events.append(e))
    await asyncio.sleep(0.05)

    executor = ToolExecutor(bus, registry, mock_adapter, timeout=0.1)
    event = IntentRoutedEvent(tool_name="slow_tool", params={}, confidence=0.9, session_id="s1")
    await executor.on_intent_routed(event)
    await asyncio.sleep(0.5)

    assert len(events) == 1
    assert events[0].success is False
    assert "timed out" in events[0].result.message


@pytest.mark.asyncio
async def test_executor_handles_tool_not_found(bus, mock_adapter):
    registry = MagicMock()
    registry.get.return_value = None

    events = []
    bus.subscribe(ToolExecutionEvent, lambda e: events.append(e))
    await asyncio.sleep(0.05)

    executor = ToolExecutor(bus, registry, mock_adapter, timeout=30)
    event = IntentRoutedEvent(tool_name="missing", params={}, confidence=0.9, session_id="s1")
    await executor.on_intent_routed(event)
    await asyncio.sleep(0.1)

    assert len(events) == 1
    assert events[0].success is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd jarvis-runtime && python -m pytest tests/core/test_executor.py -v`

Expected: FAIL with import errors.

- [ ] **Step 3: Create `core/registry/executor.py`**

```python
"""Tool executor — runs tools with timeout, publishes results."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from runtime.event_bus import (
    IntentRoutedEvent,
    MemoryWriteEvent,
    ToolCancelEvent,
    ToolExecutionEvent,
    ToolResult,
)

if TYPE_CHECKING:
    from adapters.base.platform_adapter import PlatformAdapter
    from core.registry.tool_registry import ToolRegistry
    from runtime.event_bus import EventBus

logger = logging.getLogger(__name__)


class ToolExecutor:
    def __init__(
        self,
        event_bus: EventBus,
        registry: ToolRegistry,
        adapter: PlatformAdapter,
        timeout: float = 30.0,
    ) -> None:
        self._bus = event_bus
        self._registry = registry
        self._adapter = adapter
        self._timeout = timeout
        self._current_task: asyncio.Task | None = None

    async def on_intent_routed(self, event: IntentRoutedEvent) -> None:
        tool = self._registry.get(event.tool_name)
        if tool is None:
            await self._publish_result(
                event.tool_name,
                ToolResult(success=False, message=f"Tool not found: {event.tool_name}"),
                event.session_id,
                event.params,
            )
            return

        try:
            self._current_task = asyncio.current_task()
            result = await asyncio.wait_for(
                tool.execute(event.params, self._adapter),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            result = ToolResult(success=False, message=f"Tool '{event.tool_name}' timed out")
        except asyncio.CancelledError:
            result = ToolResult(success=False, message="cancelled by user")
        except Exception as e:
            logger.exception("Unexpected executor error for %s", event.tool_name)
            result = ToolResult(success=False, message=str(e))
        finally:
            self._current_task = None

        await self._publish_result(event.tool_name, result, event.session_id, event.params)

    async def on_cancel(self, event: ToolCancelEvent) -> None:
        if self._current_task is not None and not self._current_task.done():
            self._current_task.cancel()
            logger.info("Tool execution cancelled: %s", event.reason)

    async def execute_tool(
        self, tool_name: str, params: dict, adapter=None
    ) -> ToolResult:
        tool = self._registry.get(tool_name)
        if tool is None:
            return ToolResult(success=False, message=f"Tool not found: {tool_name}")
        try:
            return await tool.execute(params, adapter or self._adapter)
        except Exception as e:
            return ToolResult(success=False, message=str(e))

    async def _publish_result(
        self, tool_name: str, result: ToolResult, session_id: str, params: dict
    ) -> None:
        await self._bus.publish(
            ToolExecutionEvent(
                tool_name=tool_name,
                success=result.success,
                result=result,
                session_id=session_id,
            )
        )
        await self._bus.publish(
            MemoryWriteEvent(
                tool_name=tool_name,
                command_text=f"{tool_name} {params}",
                params=params,
                session_id=session_id,
                timestamp=time.time(),
            )
        )
```

- [ ] **Step 4: Run tests**

Run: `cd jarvis-runtime && python -m pytest tests/core/test_executor.py -v`

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add jarvis-runtime/core/registry/executor.py jarvis-runtime/tests/core/test_executor.py
git commit -m "feat: add tool executor with timeout and cancel support"
```

---

## Task 16: Health Tracker

**Files:**
- Create: `jarvis-runtime/runtime/health.py`

- [ ] **Step 1: Create `runtime/health.py`**

```python
"""Subsystem health tracker — tracks init status with retry logic."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = auto()
    DEGRADED = auto()
    DOWN = auto()


@dataclass
class SubsystemHealth:
    name: str
    status: HealthStatus
    message: str = ""


class HealthTracker:
    def __init__(self) -> None:
        self._subsystems: dict[str, SubsystemHealth] = {}

    def get_status(self) -> dict[str, SubsystemHealth]:
        return dict(self._subsystems)

    def mark(self, name: str, status: HealthStatus, message: str = "") -> None:
        self._subsystems[name] = SubsystemHealth(name=name, status=status, message=message)
        level = logging.INFO if status == HealthStatus.HEALTHY else logging.WARNING
        logger.log(level, "Subsystem %s: %s %s", name, status.name, message)

    async def init_with_retry(
        self,
        name: str,
        init_fn: Callable[[], Awaitable[bool]],
        max_retries: int = 3,
    ) -> bool:
        for attempt in range(1, max_retries + 1):
            try:
                success = await init_fn()
                if success:
                    self.mark(name, HealthStatus.HEALTHY)
                    return True
            except Exception as e:
                logger.warning(
                    "Subsystem %s init attempt %d/%d failed: %s",
                    name, attempt, max_retries, e,
                )
            if attempt < max_retries:
                backoff = 2 ** (attempt - 1)
                await asyncio.sleep(backoff)

        self.mark(name, HealthStatus.DOWN, f"Failed after {max_retries} attempts")
        return False
```

- [ ] **Step 2: Verify import**

Run: `cd jarvis-runtime && python -c "from runtime.health import HealthTracker, HealthStatus; print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add jarvis-runtime/runtime/health.py
git commit -m "feat: add subsystem health tracker with retry logic"
```

---

## Task 17: Daemon Entry Point

**Files:**
- Create: `jarvis-runtime/runtime/daemon.py`

- [ ] **Step 1: Create `runtime/daemon.py`**

```python
"""Jarvis runtime daemon — entry point that wires everything together."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler
from pathlib import Path

from core.config.loader import load_config
from core.config.models import JarvisConfig
from core.intent.router import IntentRouter
from core.intent.slot_filler import SlotFiller
from core.pipeline.greeting_pipeline import GreetingPipeline
from core.pipeline.wake_pipeline import WakePipeline
from core.registry.executor import ToolExecutor
from core.registry.tool_registry import ToolRegistry
from core.session.state_machine import SessionStateMachine
from memory.behavioral.query import BehavioralQuery
from memory.behavioral.tracker import BehavioralTracker
from memory.session_cache.redis_client import SessionCache
from memory.vector.client import VectorMemoryClient
from memory.vector.embedder import Embedder
from runtime.event_bus import (
    EventBus,
    GestureEvent,
    IntentRoutedEvent,
    MemoryWriteEvent,
    SessionState,
    SessionStateChangedEvent,
    ToolCancelEvent,
    ToolExecutionEvent,
    VoiceTranscriptEvent,
)
from runtime.health import HealthStatus, HealthTracker

logger = logging.getLogger("jarvis")


def _setup_logging() -> None:
    log_dir = Path("~/.jarvis").expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        str(log_dir / "jarvis.log"), maxBytes=10 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


class JarvisDaemon:
    def __init__(self, config: JarvisConfig) -> None:
        self._config = config
        self._bus = EventBus()
        self._health = HealthTracker()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._executor_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sensor")
        self._stop_event = threading.Event()

        self._fsm: SessionStateMachine | None = None
        self._tool_registry: ToolRegistry | None = None
        self._tool_executor: ToolExecutor | None = None
        self._adapter = None
        self._router: IntentRouter | None = None
        self._slot_filler = SlotFiller()
        self._session_cache: SessionCache | None = None
        self._behavioral_tracker: BehavioralTracker | None = None
        self._behavioral_query: BehavioralQuery | None = None
        self._vector_client: VectorMemoryClient | None = None
        self._wake_pipeline: WakePipeline | None = None
        self._greeting_pipeline: GreetingPipeline | None = None
        self._fusion = None
        self._camera = None

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        await self._bus.start()

        await self._init_adapter()
        await self._init_memory()
        await self._init_tools()
        await self._init_intent()
        await self._init_sensors()
        await self._init_pipelines()
        await self._init_fsm()
        await self._wire_subscriptions()

        logger.info("Jarvis daemon started — health: %s", {
            k: v.status.name for k, v in self._health.get_status().items()
        })

    async def _init_adapter(self) -> None:
        from adapters.macos.adapter import MacOSAdapter
        self._adapter = MacOSAdapter()
        self._health.mark("adapter", HealthStatus.HEALTHY)

    async def _init_memory(self) -> None:
        self._session_cache = SessionCache(
            redis_enabled=self._config.redis.enabled,
            redis_url=self._config.redis.url,
        )
        await self._session_cache.initialize()
        self._health.mark("redis", HealthStatus.HEALTHY)

        self._behavioral_tracker = BehavioralTracker(self._config.memory, self._bus)
        ok = await self._health.init_with_retry("sqlite", self._behavioral_tracker.initialize)
        self._behavioral_query = BehavioralQuery(self._behavioral_tracker)

        embedder = Embedder(self._config.ollama)
        embed_ok = await self._health.init_with_retry("ollama_embed", embedder.initialize)

        self._vector_client = VectorMemoryClient(self._config.memory, embedder, self._bus)
        await self._health.init_with_retry("chromadb", self._vector_client.initialize)

    async def _init_tools(self) -> None:
        self._tool_registry = ToolRegistry()
        self._tool_registry.discover()
        logger.info("Discovered %d tools", len(self._tool_registry.list_all()))

        self._tool_executor = ToolExecutor(
            self._bus,
            self._tool_registry,
            self._adapter,
            timeout=self._config.session.tool_timeout_seconds,
        )

    async def _init_intent(self) -> None:
        self._router = IntentRouter(
            host=self._config.ollama.host,
            port=self._config.ollama.port,
            model=self._config.ollama.model,
        )
        await self._health.init_with_retry("ollama", self._router.initialize)

    async def _init_sensors(self) -> None:
        try:
            from sensors.camera import CameraThread
            self._camera = CameraThread()
            started = self._camera.start()
            if started:
                self._health.mark("camera", HealthStatus.HEALTHY)
            else:
                self._health.mark("camera", HealthStatus.DOWN, "Camera unavailable")
        except Exception:
            logger.exception("Camera init failed")
            self._health.mark("camera", HealthStatus.DOWN, "Import or init error")

        if self._camera and self._camera.healthy:
            try:
                from sensors.gesture.face_verifier import FaceVerifier
                from sensors.gesture.gesture_detector import GestureDetector
                from sensors.gesture.fusion import GestureFusion

                face_deque = self._camera.add_subscriber()
                gesture_deque = self._camera.add_subscriber()

                face_verifier = FaceVerifier(
                    face_deque, self._bus, self._loop,
                    embedding_path=self._config.paths.face_embedding_path,
                )
                face_ok = face_verifier.initialize()
                if face_ok:
                    self._executor_pool.submit(face_verifier.run, self._stop_event)

                gesture_detector = GestureDetector(gesture_deque, self._bus, self._loop)
                gesture_ok = gesture_detector.initialize()
                if gesture_ok:
                    self._executor_pool.submit(gesture_detector.run, self._stop_event)

                self._fusion = GestureFusion(
                    self._bus, self._loop,
                    wake_window=self._config.session.wake_window_seconds,
                )
            except Exception:
                logger.exception("Gesture sensor init failed")

    async def _init_pipelines(self) -> None:
        self._wake_pipeline = WakePipeline(self._session_cache)

        process_manager = None
        try:
            from adapters.macos.process_manager import ProcessManager
            process_manager = ProcessManager(self._adapter)
        except Exception:
            pass

        self._greeting_pipeline = GreetingPipeline(
            self._adapter,
            self._behavioral_query,
            self._session_cache,
            process_manager,
        )

    async def _init_fsm(self) -> None:
        self._fsm = SessionStateMachine(self._bus)

    async def _wire_subscriptions(self) -> None:
        self._bus.subscribe(GestureEvent, self._fsm.handle_event)
        self._bus.subscribe(IntentRoutedEvent, self._fsm.handle_event)
        self._bus.subscribe(ToolExecutionEvent, self._fsm.handle_event)

        self._bus.subscribe(IntentRoutedEvent, self._tool_executor.on_intent_routed)
        self._bus.subscribe(ToolCancelEvent, self._tool_executor.on_cancel)

        if self._fusion:
            self._bus.subscribe(SessionStateChangedEvent, self._fusion.on_state_changed)
            self._bus.subscribe(GestureEvent, self._fusion.on_gesture)

        self._bus.subscribe(SessionStateChangedEvent, self._wake_pipeline.on_state_changed)
        self._bus.subscribe(SessionStateChangedEvent, self._greeting_pipeline.on_state_changed)

        if self._behavioral_tracker and self._behavioral_tracker.healthy:
            self._bus.subscribe(ToolExecutionEvent, self._behavioral_tracker.on_tool_execution)

        if self._vector_client and self._vector_client.healthy:
            self._bus.subscribe(MemoryWriteEvent, self._vector_client.on_memory_write)

        self._bus.subscribe(VoiceTranscriptEvent, self._on_voice_transcript)
        self._bus.subscribe(SessionStateChangedEvent, self._on_session_state_changed)

    async def _on_voice_transcript(self, event: VoiceTranscriptEvent) -> None:
        if self._router is None or not self._router.healthy:
            await self._adapter.send_notification("Jarvis", "Intent routing unavailable")
            return

        tool_metas = self._tool_registry.list_all()
        recent = await self._session_cache.get_recent_commands(event.session_id)
        recent_texts = [str(c) for c in recent]

        result = await self._router.route(event.text, tool_metas, recent_texts)

        if result.tool_name is None:
            await self._adapter.send_notification("Jarvis", "I didn't understand that")
            return

        tool = self._tool_registry.get(result.tool_name)
        if tool is None:
            await self._adapter.send_notification("Jarvis", f"Unknown tool: {result.tool_name}")
            return

        slot_result = await self._slot_filler.fill(
            result.params,
            tool.parameters_schema,
            recent,
            await self._behavioral_query.get_time_of_day_pattern() if self._behavioral_query else {},
        )

        if slot_result.unfilled:
            await self._adapter.send_notification(
                "Jarvis", f"Missing info: {', '.join(slot_result.unfilled)}"
            )
            return

        config_dict = self._config.model_dump()
        slot_result.params["_config"] = config_dict
        slot_result.params["_executor"] = self._tool_executor

        await self._bus.publish(
            IntentRoutedEvent(
                tool_name=result.tool_name,
                params=slot_result.params,
                confidence=result.confidence,
                session_id=event.session_id,
            )
        )
        await self._session_cache.append_command(
            event.session_id,
            {"tool": result.tool_name, "params": result.params, "text": event.text},
        )

    async def _on_session_state_changed(self, event: SessionStateChangedEvent) -> None:
        if event.new_state == SessionState.ACTIVE_SESSION:
            self._start_voice_pipeline()
            self._start_idle_timer()
        elif event.new_state in (SessionState.EXECUTING, SessionState.SLEEP, SessionState.IDLE_TIMEOUT):
            self._stop_voice_pipeline()
            self._cancel_idle_timer()

        if self._fsm:
            if self._wake_pipeline and self._wake_pipeline.current_context:
                self._fsm.session_id = self._wake_pipeline.current_context.session_id

    def _start_voice_pipeline(self) -> None:
        pass

    def _stop_voice_pipeline(self) -> None:
        pass

    _idle_task: asyncio.Task | None = None

    def _start_idle_timer(self) -> None:
        self._cancel_idle_timer()
        self._idle_task = asyncio.ensure_future(self._idle_timeout())

    def _cancel_idle_timer(self) -> None:
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
            self._idle_task = None

    async def _idle_timeout(self) -> None:
        try:
            await asyncio.sleep(self._config.session.idle_timeout_seconds)
            if self._fsm:
                await self._fsm.trigger_idle_timeout()
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        logger.info("Shutting down Jarvis daemon")
        self._stop_event.set()
        if self._camera:
            self._camera.stop()
        self._executor_pool.shutdown(wait=False)
        await self._bus.stop()
        if self._behavioral_tracker:
            await self._behavioral_tracker.close()
        if self._session_cache:
            await self._session_cache.close()


async def _run() -> None:
    _setup_logging()
    config = load_config()
    daemon = JarvisDaemon(config)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(daemon.stop()))

    await daemon.start()

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await daemon.stop()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax (don't actually run the daemon)**

Run: `cd jarvis-runtime && python -c "import py_compile; py_compile.compile('runtime/daemon.py', doraise=True); print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add jarvis-runtime/runtime/daemon.py
git commit -m "feat: add daemon entry point — wires all subsystems together"
```

---

## Task 18: Tests

**Files:**
- All test files already created in Tasks 5, 9, 11, 13, 15

- [ ] **Step 1: Run all tests**

Run: `cd jarvis-runtime && python -m pytest tests/ -v`

Expected: All tests PASS (state machine: 10, intent router: 3, executor: 4, macos adapter: 6, workspace mode: 3).

- [ ] **Step 2: Commit any test fixes if needed**

```bash
git add jarvis-runtime/tests/
git commit -m "test: verify all tests pass"
```

---

## Task 19: launchd Integration

**Files:**
- Create: `jarvis-runtime/adapters/macos/launchd/com.jarvis.runtime.plist`
- Create: `jarvis-runtime/adapters/macos/launchd/install_launchd.sh`

- [ ] **Step 1: Create `com.jarvis.runtime.plist`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.jarvis.runtime</string>
    <key>ProgramArguments</key>
    <array>
        <string>__PYTHON_PATH__</string>
        <string>-m</string>
        <string>runtime.daemon</string>
    </array>
    <key>WorkingDirectory</key>
    <string>__WORKING_DIR__</string>
    <key>KeepAlive</key>
    <true/>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>__HOME__/.jarvis/launchd-stdout.log</string>
    <key>StandardErrorPath</key>
    <string>__HOME__/.jarvis/launchd-stderr.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
```

- [ ] **Step 2: Create `install_launchd.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../../" && pwd)"
PLIST_TEMPLATE="$SCRIPT_DIR/com.jarvis.runtime.plist"
LABEL="com.jarvis.runtime"

CURRENT_USER="$(whoami)"
HOME_DIR="$(eval echo ~$CURRENT_USER)"
PLIST_DEST="$HOME_DIR/Library/LaunchAgents/$LABEL.plist"

if [ -n "${VIRTUAL_ENV:-}" ]; then
    PYTHON_PATH="$VIRTUAL_ENV/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_PATH="$(which python3)"
else
    echo "Error: No Python found. Activate a virtualenv or install Python 3.11+." >&2
    exit 1
fi

echo "Installing Jarvis launchd agent..."
echo "  User:       $CURRENT_USER"
echo "  Python:     $PYTHON_PATH"
echo "  Project:    $PROJECT_DIR"
echo "  Plist:      $PLIST_DEST"

mkdir -p "$HOME_DIR/.jarvis"
mkdir -p "$(dirname "$PLIST_DEST")"

sed -e "s|__PYTHON_PATH__|$PYTHON_PATH|g" \
    -e "s|__WORKING_DIR__|$PROJECT_DIR|g" \
    -e "s|__HOME__|$HOME_DIR|g" \
    "$PLIST_TEMPLATE" > "$PLIST_DEST"

launchctl unload "$PLIST_DEST" 2>/dev/null || true
launchctl load "$PLIST_DEST"

echo "Jarvis daemon installed and started."
echo "  Stop:   launchctl unload $PLIST_DEST"
echo "  Start:  launchctl load $PLIST_DEST"
echo "  Logs:   $HOME_DIR/.jarvis/"
```

- [ ] **Step 3: Make install script executable and commit**

```bash
chmod +x jarvis-runtime/adapters/macos/launchd/install_launchd.sh
git add jarvis-runtime/adapters/macos/launchd/
git commit -m "feat: add launchd plist and install script"
```

---

## Task 20: pyproject.toml

**Files:**
- Create: `jarvis-runtime/pyproject.toml`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "jarvis-runtime"
version = "0.1.0"
requires-python = ">=3.11"
description = "Production-grade macOS desktop assistant runtime"

dependencies = [
    "pydantic>=2.5",
    "pyyaml>=6.0",
    "jinja2>=3.1",
    "ollama>=0.4",
    "pywhispercpp>=1.2",
    "mediapipe>=0.10",
    "insightface>=0.7",
    "chromadb>=0.4",
    "aiosqlite>=0.19",
    "aioredis>=2.0",
    "opencv-python>=4.8",
    "numpy>=1.24",
    "pyaudio>=0.2",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-asyncio>=0.23",
    "pytest-mock>=3.12",
]

[project.scripts]
jarvis = "runtime.daemon:main"
jarvis-enroll = "sensors.gesture.face_verifier:enroll_cli"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.setuptools.packages.find]
include = ["runtime*", "core*", "sensors*", "adapters*", "tools*", "memory*"]
```

- [ ] **Step 2: Commit**

```bash
git add jarvis-runtime/pyproject.toml
git commit -m "feat: add pyproject.toml with all dependencies and entry points"
```

---

## Task 21: README

**Files:**
- Create: `jarvis-runtime/README.md`

- [ ] **Step 1: Create `README.md`**

```markdown
# Jarvis Runtime

A production-grade macOS desktop assistant daemon. Activated by multi-modal gestures (face + clap + snap), accepts voice commands, routes them via local LLM (Ollama), and executes desktop actions.

## Prerequisites

- macOS 13+
- Python 3.11+
- Ollama installed and running (`brew install ollama && ollama serve`)
- Camera access for gesture/face detection

## Setup

```bash
# Clone and install
cd jarvis-runtime
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Pull required Ollama models
ollama pull llama3.1
ollama pull nomic-embed-text

# Enroll your face (one-time)
jarvis-enroll
```

## Running

```bash
# Run directly
jarvis

# Or install as launchd service (auto-start on login, auto-restart on crash)
bash adapters/macos/launchd/install_launchd.sh
```

## Configuration

Edit `config/jarvis.yaml` to configure:
- Ollama model and host
- Session timeouts
- Projects and workspace modes
- Memory store paths

## Workspace Modes

Voice command: "Open office mode" / "Switch to focus mode"

Workspace modes execute a sequence of tool actions. Define them in `jarvis.yaml`:

```yaml
workspace_modes:
  office:
    description: Opens office tools
    steps:
      - tool: open_project
        params: { project: office }
      - tool: open_url
        params: { url: "https://mail.google.com", browser: Chrome }
```

## Architecture

- **Event-driven**: All inter-module communication via async event bus
- **Sensor threads**: Camera, face, gesture, voice run in ThreadPoolExecutor
- **Pure FSM**: Session states (SLEEP → WAKE_PENDING → ACTIVE_SESSION → EXECUTING)
- **Dynamic tools**: Plugins auto-discovered from `tools/` directory
- **Platform adapters**: macOS adapter uses AppleScript; Windows stub ready for future

## Tests

```bash
pytest tests/ -v
```

## Logs

Logs are written to `~/.jarvis/jarvis.log` (rotating, 10MB max).
```

- [ ] **Step 2: Commit**

```bash
git add jarvis-runtime/README.md
git commit -m "docs: add README with setup and usage instructions"
```

---

## Task 22: Final Verification

- [ ] **Step 1: Run full test suite**

Run: `cd jarvis-runtime && python -m pytest tests/ -v --tb=short`

Expected: All tests pass, no failures.

- [ ] **Step 2: Verify all imports work end-to-end**

Run: `cd jarvis-runtime && python -c "from runtime.daemon import JarvisDaemon; from core.config.loader import load_config; print('All imports OK')"`

Expected: `All imports OK`

- [ ] **Step 3: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "chore: final cleanup and verification"
```
