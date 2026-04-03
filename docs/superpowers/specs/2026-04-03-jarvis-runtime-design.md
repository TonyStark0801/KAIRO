# Jarvis Runtime — Design Specification

**Date:** 2026-04-03
**Status:** Approved
**Scope:** Full implementation of a production-grade macOS desktop assistant runtime

---

## 1. Overview

Jarvis is a background agent platform — a daemon that runs 24/7 on macOS, activated by multi-modal gestures (face verification + clap + snap), accepting voice commands, routing them to tools via a local LLM, and executing actions on the desktop. It is not a chatbot or single script.

### Core Principles

- Async-first (single asyncio event loop, sensor threads bridge via `call_soon_threadsafe`)
- All inter-module communication via event bus (no direct cross-layer calls)
- Platform-specific code isolated in adapters
- Tools are self-contained plugins discovered at runtime
- Config injected via constructors, no global singletons
- Single user, local only

---

## 2. Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11+ with asyncio |
| LLM | Ollama (llama3.1 model, local) |
| Speech-to-text | Whisper.cpp via pywhispercpp |
| Gesture detection | MediaPipe (clap + dual snap) |
| Face verification | InsightFace |
| Vector memory | ChromaDB |
| Behavioral store | SQLite via aiosqlite |
| Session cache | Redis via aioredis (in-memory dict fallback) |
| Config models | Pydantic v2 |
| Prompt templates | Jinja2 |
| Config format | PyYAML |

---

## 3. Folder Structure

```
jarvis-runtime/
├── runtime/
│   ├── daemon.py               # Entry point, wires everything
│   ├── event_bus.py            # asyncio internal pub/sub
│   └── health.py               # Subsystem health tracking
├── core/
│   ├── session/
│   │   ├── state_machine.py    # FSM implementation
│   │   └── session_context.py
│   ├── intent/
│   │   ├── router.py           # Ollama-based intent routing
│   │   ├── slot_filler.py
│   │   └── prompts/
│   │       ├── system.j2
│   │       └── user.j2
│   ├── pipeline/
│   │   ├── wake_pipeline.py
│   │   └── greeting_pipeline.py
│   ├── registry/
│   │   ├── tool_registry.py
│   │   └── executor.py
│   └── config/
│       ├── loader.py
│       └── models.py
├── sensors/
│   ├── camera.py               # Shared camera thread (NEW)
│   ├── gesture/
│   │   ├── face_verifier.py
│   │   ├── gesture_detector.py
│   │   └── fusion.py
│   └── voice/
│       ├── vad.py
│       ├── transcriber.py
│       └── normalizer.py
├── adapters/
│   ├── base/
│   │   └── platform_adapter.py # ABC
│   ├── macos/
│   │   ├── adapter.py
│   │   ├── applescript.py
│   │   ├── process_manager.py
│   │   └── launchd/
│   │       ├── com.jarvis.runtime.plist
│   │       └── install_launchd.sh
│   └── windows/
│       └── adapter.py          # Stub with NotImplementedError
├── tools/
│   ├── _base.py                # BaseTool ABC
│   ├── intellij/
│   │   ├── __init__.py
│   │   ├── open_project.py
│   │   └── switch_window.py
│   ├── chrome/
│   │   └── open_url.py
│   ├── notes/
│   │   └── open_notes.py
│   ├── music/
│   │   └── play_music.py
│   └── workspace_modes/
│       ├── __init__.py
│       └── open_workspace_mode.py
├── memory/
│   ├── vector/
│   │   ├── client.py
│   │   └── embedder.py
│   ├── behavioral/
│   │   ├── tracker.py
│   │   └── query.py
│   └── session_cache/
│       └── redis_client.py
├── config/
│   └── jarvis.yaml
├── tests/
│   ├── core/
│   │   ├── test_state_machine.py
│   │   ├── test_intent_router.py
│   │   └── test_executor.py
│   ├── adapters/
│   │   └── test_macos_adapter.py
│   └── tools/
│       └── test_workspace_mode.py
├── pyproject.toml
└── README.md
```

---

## 4. Event System

### 4.1 Event Bus (`runtime/event_bus.py`)

Pure asyncio pub/sub. No third-party libraries.

- Each subscriber gets its own `asyncio.Queue(maxsize=100)`
- `subscribe(event_type, handler)` — registers an async callable
- `publish(event)` — fans out to all subscribers of that event type
- Backpressure: if a subscriber's queue is full, the event is dropped for that subscriber with a warning log

### 4.2 Event Types (dataclasses)

| Event | Fields |
|---|---|
| `GestureEvent` | `type: GestureType, timestamp: float` |
| `VoiceTranscriptEvent` | `text: str, confidence: float, session_id: str` |
| `SessionStateChangedEvent` | `old_state: State, new_state: State, session_id: str` |
| `IntentRoutedEvent` | `tool_name: str, params: dict, confidence: float, session_id: str` |
| `ToolExecutionEvent` | `tool_name: str, success: bool, result: ToolResult, session_id: str` |
| `ToolCancelEvent` | `session_id: str, reason: str` |
| `MemoryWriteEvent` | `tool_name: str, command_text: str, params: dict, session_id: str, timestamp: float` |

`GestureType` enum: `FACE_VERIFIED`, `DOUBLE_CLAP`, `DUAL_SNAP`, `ALL_SIGNALS_CONFIRMED`

---

## 5. State Machine (`core/session/state_machine.py`)

### 5.1 States

`SLEEP`, `WAKE_PENDING`, `ACTIVE_SESSION`, `EXECUTING`, `IDLE_TIMEOUT`

### 5.2 Transitions

| From | To | Trigger |
|---|---|---|
| SLEEP | WAKE_PENDING | `GestureEvent(type=FACE_VERIFIED)` |
| WAKE_PENDING | ACTIVE_SESSION | `GestureEvent(type=ALL_SIGNALS_CONFIRMED)` — fusion.py only emits this if all 3 signals arrived within its 3s window |
| WAKE_PENDING | SLEEP | Timeout (3s window expired without `ALL_SIGNALS_CONFIRMED`) |
| ACTIVE_SESSION | EXECUTING | `IntentRoutedEvent` |
| EXECUTING | ACTIVE_SESSION | `ToolExecutionEvent(success=True)` |
| EXECUTING | ACTIVE_SESSION | `ToolExecutionEvent(success=False)` (with error notification) |
| EXECUTING | ACTIVE_SESSION | `ToolCancelEvent` (user-initiated or timeout cancel) |
| ACTIVE_SESSION | IDLE_TIMEOUT | 30s idle timer |
| IDLE_TIMEOUT | SLEEP | Immediately after firing timeout notification |

Invalid transitions raise `InvalidTransitionError`.

Publishes `SessionStateChangedEvent` on every transition.

Pure module — no adapter imports.

---

## 6. Subsystem Health & Degraded Startup (`runtime/health.py`)

### 6.1 Health States

`HEALTHY`, `DEGRADED`, `DOWN`

### 6.2 Startup Behavior

Each subsystem (Ollama, Camera, ChromaDB, SQLite, Redis) attempts initialization with **3 retries, exponential backoff (1s, 2s, 4s)**. If all attempts fail:

- Subsystem marked `DOWN`
- Warning logged with failure reason
- Daemon continues with remaining healthy subsystems
- **No periodic retry.** Stays DOWN until daemon restart.

### 6.3 Impact Table

| Subsystem DOWN | Impact |
|---|---|
| Ollama | Gestures wake, greeting works, voice commands return "intent routing unavailable" |
| Camera | No gesture wake — daemon idle until restart with camera available |
| ChromaDB | Tool execution works, vector memory writes silently skipped |
| SQLite | Tool execution works, behavioral tracking silently skipped |
| Redis | Falls back to in-memory dict (already in spec) |

### 6.4 Interface

`get_status() -> dict[str, SubsystemHealth]` for debugging and future health endpoint expansion.

---

## 7. Shared Camera Architecture (`sensors/camera.py`)

Single camera thread owns `cv2.VideoCapture(0)`:

- Reads frames at ~15 FPS
- Distributes to subscribers via **bounded deque per subscriber** (maxlen=2, always latest frame)
- Subscribers: `face_verifier.py`, `gesture_detector.py`
- Camera thread starts on daemon boot, runs continuously
- If camera can't open (3 retries), subsystem marked DOWN, thread exits

### Thread Safety Model

```
Camera Thread (cv2.VideoCapture)
    ├── frame → FaceVerifier deque → ThreadPool task → loop.call_soon_threadsafe(publish GestureEvent)
    └── frame → GestureDetector deque → ThreadPool task → loop.call_soon_threadsafe(publish GestureEvent)
```

Frame format: raw BGR numpy arrays, passed by reference.

---

## 8. Gesture Fusion (`sensors/gesture/fusion.py`)

Fusion activates **only after the FSM enters `WAKE_PENDING`** (i.e., after `FACE_VERIFIED` is received). Clap and snap signals during `SLEEP` are ignored — face verification is always the first required signal.

**Sequence:**
1. Face verifier detects enrolled face → publishes `GestureEvent(type=FACE_VERIFIED)`
2. State machine transitions `SLEEP → WAKE_PENDING`
3. Fusion starts its 3-second window, pre-populating `FACE_VERIFIED` as already received
4. Gesture detector picks up `DOUBLE_CLAP` and `DUAL_SNAP` within the window
5. When all 3 confirmed → fusion emits `GestureEvent(type=ALL_SIGNALS_CONFIRMED)`
6. State machine transitions `WAKE_PENDING → ACTIVE_SESSION`

If window expires with incomplete set → fusion resets, state machine transitions `WAKE_PENDING → SLEEP`.

Thread-safe: sensors run in threads, event bus is asyncio. Uses `threading.Lock` for signal tracking, `loop.call_soon_threadsafe()` for event publishing.

---

## 9. Voice Pipeline

### 9.1 Lifecycle

- **Active only during `ACTIVE_SESSION` state**
- `daemon.py` subscribes to `SessionStateChangedEvent` to start/stop voice pipeline
- Mic closes during EXECUTING (gestures handle cancel — see Section 5.2)

### 9.2 Pipeline Flow

```
Mic (pyaudio, in thread) → VAD → Transcriber (whisper.cpp) → Normalizer → VoiceTranscriptEvent
```

- **vad.py:** Energy-based. Speech onset: energy above threshold for 300ms. Speech end: energy below threshold for 700ms. Collects frames between onset/end.
- **transcriber.py:** Feeds audio chunk to pywhispercpp in ThreadPoolExecutor. Returns raw text.
- **normalizer.py:** Strips filler words, normalizes whitespace, lowercases. Pure function.

---

## 10. Intent Router (`core/intent/router.py`)

- Sends transcript + session context + ToolMeta list to Ollama via Python SDK
- System prompt from `prompts/system.j2` (includes tool registry as JSON + last 5 commands from session cache)
- User prompt from `prompts/user.j2`
- Returns `IntentResult(tool_name, params, confidence, raw_response)`
- Confidence < 0.6 or no match → `IntentResult(tool_name=None, ...)`

### 10.2 Slot Filler (`core/intent/slot_filler.py`)

Called by the router after LLM returns an `IntentResult` with a matched tool. Compares extracted `params` against the tool's `parameters_schema`. For any required parameter that is missing:

1. Checks session cache for recent context that could fill the slot (e.g., last-used project name)
2. Checks behavioral patterns (e.g., time-of-day defaults)
3. If still missing: marks the slot as unfilled

Returns `SlotFillingResult(params: dict, unfilled: list[str])`. If `unfilled` is non-empty, the executor sends a notification asking the user to clarify rather than executing with incomplete params.

### 10.3 Transcript-to-Intent Wiring

The **intent routing subscriber** is wired in `daemon.py`:

1. Subscribes to `VoiceTranscriptEvent` on the event bus
2. Calls `router.route(transcript, session_context, tool_metas)` → `IntentResult`
3. Calls `slot_filler.fill(intent_result, tool_schema, session_cache)` → `SlotFillingResult`
4. If confidence >= 0.6 and no unfilled slots: publishes `IntentRoutedEvent`
5. If confidence < 0.6: sends notification "I didn't understand that" via adapter
6. If unfilled slots: sends notification asking for missing info

This is an async function defined in `daemon.py` (not a separate module) since it's pure wiring logic.

---

## 11. Tool System

### 11.1 BaseTool ABC (`tools/_base.py`)

Abstract base with `name`, `description`, `parameters_schema` properties and `execute(params, adapter)` method.

Every tool catches its own exceptions and returns `ToolResult(success=False, message=...)`. No exceptions propagate to executor.

### 11.2 Tool Registry (`core/registry/tool_registry.py`)

At startup: walks `tools/` directory, imports modules, finds `BaseTool` subclasses via `inspect`, instantiates, registers by `tool.name`.

- `registry.get(name) -> BaseTool`
- `registry.list_all() -> list[ToolMeta]`

### 11.3 Executor (`core/registry/executor.py`)

- Receives `IntentRoutedEvent` from event bus
- Looks up tool from registry, calls `tool.execute(params, adapter)`
- **Timeout:** Wraps execution in `asyncio.wait_for(timeout=tool_timeout_seconds)` (default 30s)
- On completion: publishes `ToolExecutionEvent` and `MemoryWriteEvent`
- On timeout: cancels task, publishes `ToolExecutionEvent(success=False)`
- On `ToolCancelEvent`: cancels running task, same recovery path

### 11.4 Tool Cancel Mechanism

- **Dual snap during EXECUTING** → `ToolCancelEvent` published
- **Timeout (30s default)** → executor self-cancels
- Both paths: `EXECUTING → ACTIVE_SESSION`, mic reopens

---

## 12. Platform Adapters

### 12.1 ABC (`adapters/base/platform_adapter.py`)

Abstract methods: `open_application`, `switch_window`, `run_script`, `get_running_apps`, `open_url_in_browser`, `send_notification`, `play_audio_file`, `get_active_workspace`.

### 12.2 macOS Adapter

- `applescript.py` provides script builder functions (returns strings)
- `adapter.py` calls `run_script()` which executes via `asyncio.create_subprocess_exec("osascript", ...)`
- `process_manager.py` queries running apps via AppleScript

### 12.3 Windows Stub

Every method raises `NotImplementedError`. No Windows imports — importable on macOS.

---

## 13. Memory System

### 13.1 Vector Memory (`memory/vector/`)

- Subscribes to `MemoryWriteEvent` on event bus
- `embedder.py` calls Ollama embeddings API (nomic-embed-text)
- If embedding model unavailable: logs warning, skips write (subsystem DOWN)
- `client.py` upserts to ChromaDB with metadata `{tool_name, timestamp, session_id}`

### 13.2 Behavioral Store (`memory/behavioral/`)

- SQLite schema: `CREATE TABLE commands (id, timestamp, tool_name, params_json, session_id, success)`
- Subscribes to `ToolExecutionEvent` on event bus
- `tracker.py`: `record(event)`, `get_frequent_tools(limit=5)`, `get_time_of_day_pattern()`
- `query.py`: read-only query interface for greeting pipeline and other consumers

### 13.3 Session Cache (`memory/session_cache/`)

- Redis with in-memory dict fallback
- Stores: last 10 commands per session, current state, session start timestamp, active workspace mode
- Intent router reads last 5 commands for recency context

---

## 14. Pipelines

### 14.1 Greeting Pipeline (`core/pipeline/greeting_pipeline.py`)

Runs on `ACTIVE_SESSION` entry. Collects: current time, day of week, top 3 recent tools from behavioral store, open IntelliJ projects, last session timestamp. Formats greeting < 40 words. Speaks via `adapter.run_script()` (macOS `say` command). Fire-and-forget.

### 14.2 Wake Pipeline (`core/pipeline/wake_pipeline.py`)

Subscribes to `SessionStateChangedEvent` where `new_state == WAKE_PENDING`. Responsibilities:

1. Starts/resets the fusion timer (ensures the 3s window is running)
2. Initializes a new `SessionContext` (generates session_id, records start time)
3. Writes session start to session cache
4. On `ALL_SIGNALS_CONFIRMED`: passes the initialized context to the greeting pipeline
5. On timeout (window expired): cleans up the session context

This is the bridge between the FSM state transition and the sensor fusion layer. It does NOT control the FSM — the FSM transitions on events independently. The wake pipeline prepares the session context so it's ready when `ACTIVE_SESSION` is entered.

---

## 15. Face Enrollment CLI

Entry point: `jarvis-enroll = "sensors.gesture.face_verifier:enroll_cli"` in pyproject.toml.

Flow:
1. Opens camera, shows live preview with face bounding box
2. Prompts: "Position your face, press SPACE to capture"
3. Takes 3 captures, averages InsightFace 512-d embeddings
4. Saves to `~/.jarvis/face_embedding.npy`

Runtime: `face_verifier.py` loads `.npy` at startup, compares via cosine similarity (threshold 0.5). No enrollment file → subsystem DOWN.

Single user only.

---

## 16. Configuration

### 16.1 Pydantic Models (`core/config/models.py`)

Partially implemented — existing code in `core/config/models.py` is the source of truth and must be preserved. Will be extended to add `tool_timeout_seconds: int = 30` to `SessionConfig`. Contains: `JarvisConfig` with nested `OllamaConfig`, `SessionConfig`, `MemoryConfig`, `RedisConfig`, `ProjectEntry`, `WorkspaceModeEntry`, `PathsConfig`.

### 16.2 Loader (`core/config/loader.py`)

Loads `config/jarvis.yaml`, validates via Pydantic, returns `JarvisConfig`. Loaded once at startup, passed to all modules via constructor.

### 16.3 Sample Config (`config/jarvis.yaml`)

- 3 projects: office, codejam, personal
- 3 workspace modes: office, focus, evening
- Ollama: localhost:11434, llama3.1
- Session: idle_timeout=30, wake_window=3, tool_timeout=30
- Memory: chroma_path=~/.jarvis/chroma, behavioral_db=~/.jarvis/behavior.db
- redis_enabled: false

---

## 17. Daemon Entry Point (`runtime/daemon.py`)

Single asyncio event loop. Wiring order:
1. Load config
2. Initialize event bus
3. Initialize health tracker
4. Initialize subsystems with retry (camera, Ollama, ChromaDB, SQLite, Redis)
5. Initialize adapters, tool registry, executor
6. Initialize sensor threads (camera → face verifier + gesture detector)
7. Wire event subscriptions
8. Start state machine in SLEEP state
9. Run forever

Sensor threads use `ThreadPoolExecutor` and `loop.call_soon_threadsafe()` to bridge into asyncio.

---

## 18. Logging

Standard `logging` module. Root logger configured in `daemon.py`:
- Format: `[%(asctime)s] %(name)s %(levelname)s: %(message)s`
- Outputs: stderr + `~/.jarvis/jarvis.log` (rotating, 10MB, 3 backups)
- Each module: `logging.getLogger(__name__)`

---

## 19. launchd Integration

- `com.jarvis.runtime.plist`: `KeepAlive=true` for auto-restart on crash
- `install_launchd.sh`: detects current username and Python venv path dynamically. No hardcoded usernames.

---

## 20. Implementation Rules

1. ZERO platform imports in `core/` or `tools/`
2. ALL inter-module communication via event bus
3. Tool plugins are self-contained (import only `_base.py`, pydantic, stdlib)
4. Async-first (single event loop, threads bridge via `call_soon_threadsafe`)
5. Config loaded once, injected via constructors
6. Tool registry is dynamic (no hardcoded tool names)
7. AppleScript calls go through `adapter.run_script()` exclusively
8. Every `tool.execute()` catches its own exceptions
9. launchd plist uses KeepAlive=true
10. Windows adapter stub is importable on macOS without errors

---

## 21. Tests

pytest + pytest-asyncio + pytest-mock:

- **test_state_machine.py:** All valid transitions, invalid transitions raise `InvalidTransitionError`, cancel transition
- **test_intent_router.py:** Mock Ollama response, IntentResult parsing, confidence threshold
- **test_executor.py:** Mock tool execution, verify `MemoryWriteEvent` published, timeout behavior
- **test_macos_adapter.py:** Mock subprocess, verify AppleScript builder output
- **test_workspace_mode.py:** Mock executor, verify steps fire in order with delays

---

## 22. Build Order

1. `core/config/models.py` + `loader.py`
2. `runtime/event_bus.py`
3. `adapters/base/platform_adapter.py`
4. `tools/_base.py`
5. `core/session/state_machine.py`
6. `core/registry/tool_registry.py`
7. `sensors/camera.py` + `sensors/gesture/` (all files)
8. `sensors/voice/` (all files)
9. `adapters/macos/` (all files)
10. `adapters/windows/adapter.py`
11. `tools/` (all plugins)
12. `memory/` (all subsystems)
13. `core/intent/` (router, slot_filler, prompts)
14. `core/pipeline/` (wake + greeting)
15. `core/registry/executor.py`
16. `runtime/health.py`
17. `runtime/daemon.py`
18. `tests/`
19. `config/jarvis.yaml`
20. `adapters/macos/launchd/`
21. `pyproject.toml`
22. `README.md`
