# Jarvis v2 — Personal Desktop Companion Runtime

## Overview

Refactor Jarvis from a command automation tool into a conversational desktop companion that speaks naturally, remembers the user across restarts, observes workspace context, and adapts over time.

## Decisions Made

- **Voice TTS**: Piper TTS, `en_US-amy-medium` (HER/Samantha-style — warm, conversational)
- **STT**: Whisper `medium.en` with Metal GPU acceleration (~0.4s latency on M1 Pro)
- **Personality**: Warm, empathetic, conversational. Observes first, understands second, executes third.
- **LLM**: Ollama llama3.1 (swappable via interface)
- **Browser**: Brave (configurable)
- **Hardware**: M1 Pro 16GB — include auto-unload + throttling optimizations
- **Keep**: YouTube tools, face verification, ChromeBridge, all existing tools
- **Remove**: Old greeting pipeline (absorbed into personality), macOS `say` command (replaced by Piper)

## Architecture

```
jarvis-runtime/
├── assistant_core/           # Brain — personality + reasoning
│   ├── personality.py        # Loads identity.yaml, shapes all responses
│   └── reasoner.py           # Conversational LLM with context + memory + tools
├── memory_service/           # All persistence, layered
│   ├── identity.py           # YAML read/write — name, traits, owner info
│   ├── preferences.py        # SQLite — user habits, aliases, repos, patterns
│   └── session_store.py      # SQLite — session summaries for cross-restart continuity
├── context_service/          # Desktop awareness
│   └── detector.py           # Active app, repo, file, terminal via AppleScript
├── voice_service/            # TTS output
│   ├── base.py               # Abstract engine interface
│   └── piper_engine.py       # Piper TTS → WAV file → afplay with mic muting
├── stt_service/              # Speech input
│   ├── base.py               # Abstract STT interface
│   ├── whisper_engine.py     # Whisper medium.en + Metal
│   └── mic_listener.py       # Mic thread, energy gating, mode switching, media gate
├── llm_router/               # LLM provider abstraction
│   ├── base.py               # Abstract LLM interface
│   ├── ollama_provider.py    # Ollama chat with conversation history
│   └── prompts/              # Jinja2 system prompt templates
├── tool_registry/            # Plugin tools
│   ├── base.py               # BaseTool ABC
│   ├── registry.py           # Discovery
│   └── tools/                # open_url, open_project, youtube_*, open_notes, etc.
├── session/                  # Session lifecycle
│   ├── state_machine.py      # FSM: SLEEP → WAKE_PENDING → ACTIVE → EXECUTING → IDLE
│   └── context.py            # Session context (ID, timestamps)
├── sensors/                  # Hardware sensors (kept)
│   ├── camera.py             # Shared camera thread (throttled to 1 FPS)
│   └── gesture/              # Face verification (security gate)
├── platform_adapters/
│   └── mac/
│       ├── adapter.py        # macOS platform adapter
│       ├── applescript.py    # AppleScript builders
│       ├── chrome_bridge.py  # Brave JS injection
│       └── process_manager.py
├── runtime/
│   ├── daemon.py             # Slim wiring — no business logic
│   └── event_bus.py          # Async pub/sub
└── config/
    ├── jarvis.yaml           # Runtime config (ollama, session, browser, etc.)
    └── identity.yaml         # Personality definition
```

## Module Contracts

### 1. identity.yaml (NEW)

```yaml
assistant:
  name: Jarvis
  personality: warm, conversational, empathetic
  style: "Speak like a close friend who happens to be brilliant. Short sentences. No robotic confirmations. Observe context before acting."
  voice: en_US-amy-medium

owner:
  name: Shubham
  preferences:
    ide: IntelliJ IDEA
    browser: Brave Browser
    music: YouTube
```

### 2. assistant_core/personality.py

- Loads `identity.yaml` at startup
- Builds the LLM system prompt dynamically by combining: identity traits + current workspace context + recent memory + available tools
- Shapes response tone (no "Done.", no "Opened X in Y" — instead natural conversational responses)
- Generates context-aware greetings on wake: "Looks like you're back in payments-service."

### 3. assistant_core/reasoner.py

The central brain. ALL user interactions flow through here.

```python
class Reasoner:
    async def process(self, transcript: str, session_id: str) -> ReasonerResponse:
        # 1. Load personality + context + memory into system prompt
        # 2. Send conversation to LLM
        # 3. Parse response: tool execution OR conversational reply
        # 4. Return structured response
```

`ReasonerResponse` has three modes:
- `EXECUTE`: tool_name + params (Jarvis does something)
- `SPEAK`: message (Jarvis says something conversational)
- `SPEAK_AND_EXECUTE`: message + tool (Jarvis acknowledges AND acts)

### 4. memory_service/identity.py

Reads/writes `identity.yaml`. Exposes `get_owner_name()`, `get_personality()`, `get_style()`.

### 5. memory_service/preferences.py

SQLite table `preferences`:
- `key TEXT PRIMARY KEY, value TEXT, updated_at TIMESTAMP`
- Stores: favorite repos, aliases ("payments" → "payments-service"), habits

SQLite table `aliases`:
- `alias TEXT PRIMARY KEY, expansion TEXT`
- User says "open payments" → resolves to "open payments-service project"

### 6. memory_service/session_store.py

SQLite table `sessions`:
- `session_id TEXT, started_at TIMESTAMP, ended_at TIMESTAMP, summary TEXT, tools_used TEXT`

On startup: load last session summary → inject into system prompt so Jarvis remembers what happened last time.

On session end: ask LLM to generate a 1-sentence summary of what happened → store it.

### 7. context_service/detector.py

AppleScript-based detection:

```python
class ContextDetector:
    async def get_context(self) -> WorkspaceContext:
        # Returns: active_app, active_window_title, repo_path, git_branch, open_file
```

- Active app: `tell application "System Events" to get name of first process whose frontmost is true`
- Window title: parse for repo name, file name
- Repo: check `~/.jarvis/known_repos.json` or parse window title
- Git branch: `git -C <repo_path> branch --show-current`

Polled every 10 seconds (not continuous). Results cached.

### 8. voice_service/piper_engine.py

```python
class PiperVoiceEngine(VoiceEngine):
    async def speak(self, text: str) -> None:
        # 1. Generate WAV: piper --model amy --output_file /tmp/jarvis_speech.wav
        # 2. Mute mic (via callback)
        # 3. Play: afplay /tmp/jarvis_speech.wav
        # 4. Wait 0.5s buffer
        # 5. Unmute mic (via callback)
```

Piper runs as a subprocess. Generates audio to a temp file, then plays it. This separation means:
- No speaker-to-mic feedback (mic is muted during playback)
- Clean audio lifecycle
- Easy to swap engines later

### 9. stt_service/mic_listener.py

Refactored from current `wake_word.py`:
- Same three modes: IDLE, WAKE_WORD, COMMAND
- Media-aware energy threshold: 600 normal, 2500 when media playing
- Uses `stt_service/whisper_engine.py` for transcription (medium.en + Metal)

### 10. stt_service/whisper_engine.py

```python
class WhisperEngine(STTEngine):
    def __init__(self, model: str = "medium.en"):
        # Load with Metal acceleration
    
    def transcribe(self, audio: np.ndarray) -> str:
        # Returns cleaned transcript
```

### 11. llm_router/ollama_provider.py

Refactored from current `intent/router.py`:
- Maintains per-session conversation history
- Sends full context (personality + workspace + memory + tools) as system prompt
- Parses structured JSON responses
- Extracts JSON from mixed text (existing `_extract_json`)
- `add_context()` for tool result injection
- `clear_session()` on session end

### 12. runtime/daemon.py

Slim wiring only. No business logic. Pseudocode:

```python
async def start():
    identity = IdentityMemory("config/identity.yaml")
    prefs = PreferencesMemory("~/.jarvis/preferences.db")
    sessions = SessionStore("~/.jarvis/sessions.db")
    context = ContextDetector()
    voice = PiperVoiceEngine(identity.voice_model)
    stt = WhisperEngine("medium.en")
    mic = MicListener(stt, event_bus)
    llm = OllamaProvider(config.ollama)
    personality = Personality(identity, prefs, sessions, context)
    reasoner = Reasoner(personality, llm, tool_registry)
    fsm = SessionStateMachine(event_bus)
    
    # Wire events
    event_bus.subscribe(VoiceTranscript, reasoner.process)
    event_bus.subscribe(ReasonerResponse, executor.handle)
    event_bus.subscribe(ToolResult, reasoner.inject_result)
    event_bus.subscribe(SessionStateChanged, mic.on_state)
    event_bus.subscribe(SessionStateChanged, voice.on_state)
```

## Interaction Flow

```
User speaks → mic_listener detects speech → whisper_engine transcribes
→ reasoner.process(transcript):
    1. personality builds system prompt (identity + context + memory + tools)
    2. ollama_provider.chat(messages) with full conversation history
    3. parse response → EXECUTE / SPEAK / SPEAK_AND_EXECUTE
→ if SPEAK: voice_service speaks (piper → wav → afplay, mic muted)
→ if EXECUTE: tool runs via executor → result injected into conversation
→ mic re-enables
```

## Conversation Examples

**Wake + context-aware greeting:**
```
User: "Jarvis"
Jarvis: "Hey — looks like you're in payments-service on the fix-timeout branch."
```

**Music request:**
```
User: "Play something chill"
Jarvis: [youtube_search("chill music")] — results appear on screen
Jarvis: "Found some options — which one looks good?"
User: "Third one"
Jarvis: [youtube_pick(3)] — video starts
```

**During music (media gate active):**
```
User: "Jarvis" → mic switches to COMMAND briefly
User: "Skip this one"
Jarvis: [youtube_control("next")] — skips silently
```

**Context-aware help:**
```
User: "What was I working on yesterday?"
Jarvis: (loads last session summary) "You were debugging the timeout issue in payments-service and opened the Stripe docs."
```

## Resource Optimization

- **Ollama**: `OLLAMA_KEEP_ALIVE=5m` — unloads model after 5min silence (saves 4.5GB)
- **Camera**: Throttled to 1 FPS (from ~30 FPS). Face check every 3 seconds.
- **Whisper**: Lazy-loaded on first speech detection. Medium.en + Metal.
- **Piper**: Loaded once, stays resident (~80MB).

## What Stays vs Changes

| Component | Status |
|---|---|
| Event bus | Stays (minor cleanup) |
| FSM states | Stays (same 5 states) |
| Tool registry + discovery | Stays |
| YouTube tools (5) | Stays |
| open_url, open_project, etc. | Stays |
| ChromeBridge | Stays |
| Face verification | Stays (throttled) |
| Platform adapter (macOS) | Stays |
| `say` command TTS | **Replaced by Piper** |
| `core/intent/router.py` | **Replaced by reasoner + ollama_provider** |
| `greeting_pipeline.py` | **Replaced by personality.py** |
| `sensors/voice/wake_word.py` | **Refactored into stt_service/mic_listener.py** |
| `sensors/voice/transcriber.py` | **Refactored into stt_service/whisper_engine.py** |
| `memory/` (current) | **Restructured into memory_service/** |
| `runtime/daemon.py` | **Slimmed to pure wiring** |
