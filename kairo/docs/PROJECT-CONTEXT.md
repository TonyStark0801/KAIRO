# KAIRO — project context and architecture

This document captures **product intent**, **as-built behavior**, **configuration**, **extension interfaces**, and **changes made during the agent-runtime upgrade** so future work (human or AI) can stay aligned without re-deriving context from chat history.

---

## 1. What KAIRO is

**KAIRO** = *Knowledge-Aware Interactive Runtime Operator*.

It is a **local-first, voice-driven developer companion**: not a stateless chatbot, but a **runtime layer** that can observe workflow context, remember patterns, speak and listen on-device, and execute tools (apps, URLs, workspace modes, etc.).

The installable Python distribution is **`kairo-runtime`** (`pyproject.toml`). The on-disk source tree is **`kairo/`**. User data defaults to **`~/.kairo/`** (logs, SQLite, Chroma, enrollments).

---

## 2. Target pipeline vs what runs today

### 2.1 Target (product / README roadmap)

End state is often described as:

`Mic → Wake Word → VAD → STT → Dialogue Planner → LLM Brain → Tool Router → Memory Layer → TTS`

with **Redis Streams** between services in later phases, and connectors for Git, filesystem, browser, terminal.

### 2.2 Implemented today (monolithic process)

The **runtime is a single asyncio process** (`runtime/daemon.py`, class `KairoDaemon`) that:

- Owns an **in-process** `EventBus` (`runtime/event_bus.py`): asyncio pub/sub with typed events, not Redis Streams.
- Runs the **mic** in a **background thread** (`stt_service/mic_listener.py`).
- Drives a **session FSM** (`core/session/state_machine.py`).
- Routes cognition through **`DialoguePlanner`** then **`Reasoner`** (`assistant_core/`).
- Executes tools via **`IntentRoutedEvent`** → **`ToolExecutor`** (`core/registry/executor.py`).
- Persists **vector** and **behavioral** memory partly via bus events (`MemoryWriteEvent`, `ToolExecutionEvent`).

**Gaps still honest relative to the full target:**

- **VAD:** `sensors/voice/vad.py` exists (energy-based) but is **not wired** into the mic path; the mic uses RMS energy thresholds and silence timing.
- **Redis:** `memory/session_cache/redis_client.py` is **KV** session cache with in-memory fallback, not Streams.
- **IntentRouter:** `core/intent/router.py` is **not** wired in the daemon; the live path is **`Reasoner`** (tests may still cover `IntentRouter`).
- **Naming:** Product and `config/identity.yaml` use **Kairo**; legacy **`~/.jarvis/`** or **`config/jarvis.yaml`** may still exist on disk from older installs.

---

## 3. Repository map (top-level packages)

| Area | Path | Role |
|------|------|------|
| Entry / orchestration | `runtime/` | Daemon, health, in-process bus |
| Session & intent glue | `core/` | FSM, session context, pipelines, config loader, tool registry & executor |
| Voice ingest | `stt_service/`, `sensors/voice/` | Mic listener, Whisper engine, normalizer, optional wake-word streaming |
| Wake word (optional) | `sensors/wake/` | openWakeWord adapter + factory |
| Hearing output | `voice_service/` | Piper TTS |
| Cognition | `assistant_core/` | Planner, personality, reasoner, fast-path, proactive, Groq agent loop |
| LLM routing | `llm_router/` | `LocalChatProvider` protocol, Ollama provider, Groq provider, factory |
| World model / prefs | `context_service/`, `memory_service/`, `memory/` | Context detector, identity, preferences, session store, behavioral DB, Chroma vector |
| Execution surface | `tools/`, `adapters/` | Plugin tools, macOS/Windows adapters |
| Sensors | `sensors/` | Camera, face, gesture fusion, voice verify |

---

## 4. Voice path (detailed)

### 4.1 Mic modes (`stt_service/mic_listener.py`)

- **`IDLE`:** Mic open but not processing (e.g. during TTS or tool execution).
- **`WAKE_WORD`:** Listen for wake; either **STT + keyword** or **openWakeWord streaming** (see below).
- **`COMMAND`:** Capture utterance, run STT, publish **`VoiceTranscriptEvent`**.

### 4.2 Wake detection (two engines)

1. **`stt_keyword` (default)**  
   After energy gating, a short buffer is transcribed with the same **`WhisperEngine`** used for commands. Text is matched against **`wake_words`** from identity config (substring match). On match, publishes **`GestureEvent(WAKE_WORD_DETECTED)`**. Optional inline command after the keyword can publish a **`VoiceTranscriptEvent`** for the remainder.

2. **`openwakeword` (optional)**  
   Config: `wake.engine: openwakeword` plus model paths and thresholds in `WakeConfig`.  
   Implementation: `sensors/wake/openwakeword_stream.py` feeds 16 kHz mono PCM frames into the openWakeWord **`Model`**. On score above threshold, publishes the same **`WAKE_WORD_DETECTED`** gesture.  
   Dependency: optional extra **`pip install -e ".[wake]"`** (`openwakeword` in `pyproject.toml`).  
   If import or init fails, the daemon **logs a warning** and behavior falls back to **STT keyword** wake.

### 4.3 Speech-to-text (`stt_service/whisper_engine.py`)

- **Primary:** **faster-whisper** for `stt.model` (default **`large-v3`**) on CPU int8.
- **Fallback:** **pywhispercpp** (whisper.cpp bindings): tries the configured primary name, then **`stt.cpp_fallback_model`** (default **`small.en`**) if loading the primary name fails.

Model selection is **config-driven** via `KairoConfig.stt` (not hardcoded in the daemon).

### 4.4 Text normalization

Command text may pass through `sensors/voice/normalizer.py` and noise/music heuristics before becoming a **`VoiceTranscriptEvent`**.

---

## 5. Cognition path (planner → reasoner → LLM)

### 5.1 Dialogue planner (`assistant_core/dialogue_planner.py`)

For each transcript, the daemon calls:

`plan = await planner.plan(transcript, session_id)`

**`PlanOutput`** fields (all have safe defaults for backward compatibility):

| Field | Meaning |
|--------|--------|
| `transcript` | Effective user text (may be normalized or replaced by a smarter planner later). |
| `use_llm` | If `False`, **`Reasoner`** runs **Tier 1 fast-path only**; if no match, returns a **speak** response without calling the LLM tiers. |
| `allow_tool_execution` | If `False`, the daemon **downgrades** execute / speak-and-execute outcomes to **speak-only** (no `IntentRoutedEvent`). |
| `persist_memory` | Passed through to **`IntentRoutedEvent.persist_memory`**; when `False`, **`ToolExecutor`** still runs the tool but **skips** **`MemoryWriteEvent`** (vector ingestion path). |
| `tone_hint` | Appended into system / cloud prompts inside **`Reasoner`** so responses can follow a tone without changing planner/daemon structure. |

The stock **`DialoguePlanner`** is a **pass-through**: it only strips whitespace and leaves all flags at permissive defaults.

### 5.2 Reasoner (`assistant_core/reasoner.py`)

- **Tier 1:** `assistant_core/fast_path.py` — regex/keyword tool routing without LLM.
- **Tier 2:** Local **fast** Ollama model (`ollama.fast_model`) with tool JSON protocol.
- **Tier 3:** Groq **agent loop** when available; else local **deep** model (`ollama.model`).

**New keyword-only parameters:** `process(..., use_llm=True, tone_hint=None)`.

**LLM typing:** the reasoner depends on **`LocalChatProvider`** (`llm_router/protocol.py`), not a concrete Ollama class, so another backend can satisfy the same protocol later.

### 5.3 Local LLM provider abstraction

- **`LocalChatProvider`** (`llm_router/protocol.py`): structural protocol (`healthy`, `fast_model`, `deep_model`, `initialize`, `chat`).
- **`create_local_chat_provider(config.ollama)`** (`llm_router/providers.py`): returns **`OllamaProvider`** today. **Swap the implementation here** to add non-Ollama local backends without editing the daemon or planner.

### 5.4 Default local “brain” model

**Default deep model** in **`OllamaConfig`** and sample **`config/kairo.yaml`**: **`qwen:4b`**.

You must **`ollama pull`** that tag (or whatever exact name you standardize) on the machine.

**Fast** model remains configurable (default **`llama3.2:3b`** in repo samples) for Tier 2 latency.

---

## 6. Tool routing and memory side effects

### 6.1 Intent and execution

- **`Reasoner`** produces **`ReasonerResponse`** (`EXECUTE`, `SPEAK`, `SPEAK_AND_EXECUTE`).
- The daemon publishes **`IntentRoutedEvent`** with tool name, params, session id, confidence, and **`persist_memory`**.
- **`ToolExecutor`** (`core/registry/executor.py`) runs the tool, publishes **`ToolExecutionEvent`**, and **conditionally** publishes **`MemoryWriteEvent`**:

  - If **`persist_memory`** is **`False`**, the executor **does not** emit **`MemoryWriteEvent`** (so vector memory does not ingest that command).

**Note:** Other memory paths (e.g. behavioral logging subscribed to **`ToolExecutionEvent`**) may still run unless separately gated in the future.

### 6.2 Legacy tool dependency injection

Tools still receive **`_config`** and **`_executor`** inside **`params`** from the daemon for workspace modes and similar flows. A future refactor is to replace this with an explicit **`ToolContext`**; that is **not** done in the current upgrade.

---

## 7. Configuration reference

### 7.1 Files

- **`config/kairo.yaml`**: runtime YAML merged with Pydantic defaults (`core/config/loader.py`). If missing, **`config/jarvis.yaml`** is loaded with a deprecation warning.
- **`config/identity.yaml`**: assistant name, wake word **strings** (for STT keyword wake), voice model, owner, verification mode.

### 7.2 Important `KairoConfig` sections (`core/config/models.py`)

- **`ollama`:** host, port, **`model`** (deep), **`fast_model`**, **`embed_model`**.
- **`stt`:** **`model`**, **`cpp_fallback_model`**.
- **`wake`:** **`engine`** (`stt_keyword` | `openwakeword`), openWakeWord paths, threshold, framework.
- **`session`, `memory`, `redis`, `groq`, `proactive`, `workspace_modes`, …** unchanged in role from before.

---

## 8. Events (in-process bus)

Defined in **`runtime/event_bus.py`**. Examples:

- **`GestureEvent`**, **`VoiceTranscriptEvent`**, **`SessionStateChangedEvent`**
- **`IntentRoutedEvent`** (includes **`persist_memory: bool = True`**)
- **`ToolExecutionEvent`**, **`ToolCancelEvent`**, **`MemoryWriteEvent`**

Subscribers are registered in **`KairoDaemon._wire_subscriptions`**.

---

## 9. Architecture review themes (retained for planning)

These came from an explicit codebase review; they are **not all fixed** yet:

- **God-object daemon:** Most wiring lives in one class; long-term split into handlers or smaller services.
- **Dual intent story:** `IntentRouter` vs `Reasoner` — pick one primary path or delete dead code.
- **Memory split:** `memory_service/` vs `memory/` — consider a facade API.
- **Transport:** In-process bus vs future **Redis Streams** — introduce a **`Bus`** protocol when splitting processes.
- **Wake vs STT:** Dedicated wake models reduce false accepts and latency versus STT-only wake.

---

## 10. Operations

### 10.1 Run

- **`kairo`** → `runtime.daemon:main`; **`kairo-enroll-face`** / **`kairo-enroll-voice`** for enrollment (`pyproject.toml` `[project.scripts]`).

### 10.2 Tests

```bash
cd kairo
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

### 10.3 Optional wake dependency

```bash
pip install -e ".[wake]"
```

---

## 11. Changelog — agent-runtime configuration upgrade (this effort)

Summary of **code changes** aligned with the “speech and reasoning stack” upgrade:

1. **`SttConfig` / `WakeConfig`** added to **`KairoConfig`**; **`config/kairo.yaml`** holds **`stt`** and **`wake`** blocks (legacy **`jarvis.yaml`** still loadable).
2. **Daemon** builds **`WhisperEngine`** from **`config.stt`** (no hardcoded `small.en`).
3. **`sensors/wake/`** added: **`OpenWakeWordStreamDetector`**, **`try_create_openwakeword_stream`**; **`MicListener`** uses streaming wake when a detector is provided.
4. **`DialoguePlanner`** + **`PlanOutput`**; daemon calls **`plan()`** before **`Reasoner.process`**, honoring **`use_llm`**, **`allow_tool_execution`**, **`persist_memory`**, **`tone_hint`**.
5. **`LocalChatProvider`** protocol + **`create_local_chat_provider`**; daemon obtains LLM via factory; default Ollama **deep** model set to **`qwen:4b`**.
6. **`IntentRoutedEvent.persist_memory`** + executor gating of **`MemoryWriteEvent`**.
7. **Tests** extended: planner smoke test, executor memory skip, reasoner **`use_llm=False`** behavior.
8. **`pyproject.toml`** optional **`[wake]`** extra for **`openwakeword`**.
9. **Product rename:** `jarvis-runtime` → **`kairo-runtime`**; `JarvisConfig` / `JarvisDaemon` → **`KairoConfig`** / **`KairoDaemon`**; CLI **`kairo`**, **`kairo-enroll-face`**, **`kairo-enroll-voice`**; data dir **`~/.kairo/`**; config **`config/kairo.yaml`** (legacy **`jarvis.yaml`** still loaded if present).

---

## 12. Related documentation

- **`README.md`** — user-facing setup, roadmap phases, stack targets.
- **`docs/superpowers/specs/`** (repo root) — older companion design specs; may predate current Kairo naming and code.

---

## 13. How to use this doc

- **Before refactors:** skim §2 (target vs built), §4–6 (pipelines), §7 (config).
- **Before adding a new LLM backend:** implement **`LocalChatProvider`**, return it from **`create_local_chat_provider`** (or branch on config).
- **Before changing wake behavior:** read §4.2 and **`sensors/wake/`** + **`MicListener`**.
- **Before adding Redis Streams:** treat **`EventBus`** as the first transport to abstract; keep event types stable.

This file should be **updated when** behavior or defaults change materially (new bus, ToolContext refactor, identity rename, etc.).
