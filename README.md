# KAIRO

KAIRO is a **local-first, voice-driven developer assistant** designed to act as a **persistent runtime companion**.

Unlike chatbot-style assistants, KAIRO continuously observes user workflow context and evolves a **structured memory model** over time.

## What KAIRO is (core design)

KAIRO is **not** a chatbot. It is a **runtime companion layer** that:

- understands what the user is doing
- remembers important patterns
- assists naturally via voice
- executes tasks autonomously when appropriate
- improves over time through structured memory

## Roadmap

### Phase 1

- Wake word detection ("Hey Kairo")
- Natural conversational interaction
- Local speech-to-text pipeline
- Local LLM reasoning engine
- Expressive text-to-speech output
- Basic tool execution (open apps, run commands)

### Phase 2

- Screen awareness without continuous screenshots
- Active window tracking
- Browser tab awareness
- Clipboard monitoring
- Semantic activity detection
- Structured memory engine

### Phase 3

- Long-term contextual learning
- Workflow prediction
- Connector system (Git, filesystem, browser, terminal)
- Redis Streams event bus
- Dialogue planner with tone-aware responses

## Primary stack (target)

| Concern | Choice |
|--------|--------|
| Wake word | openWakeWord / Porcupine |
| STT | faster-whisper large-v3 |
| LLM | Qwen 3 4B Instruct |
| TTS | Piper |
| Memory | SQLite (+ optional vector DB later) |
| Event bus | Redis Streams |
| Observer service | Windows system hooks |

---

## This repository

The **repository root** is the runtime source tree (Python package **`kairo-runtime`**, see `pyproject.toml`): runtime, core, adapters, memory, sensors, voice/STT, tests, and tools.

For as-built architecture, configuration, planner/reasoner flow, wake/STT options, and upgrades, see **[docs/PROJECT-CONTEXT.md](docs/PROJECT-CONTEXT.md)**.

**Data directory:** runtime state, logs, SQLite, Chroma, and enrollments live under **`~/.kairo/`** (migrate from `~/.jarvis/` manually if upgrading from an older install).

## Prerequisites

- macOS 13+ (primary target today)
- Python 3.11+
- Ollama installed and running (`brew install ollama && ollama serve`)
- Camera access for gesture/face detection (when those paths are enabled)

## Setup

```bash
git clone https://github.com/TonyStark0801/KAIRO.git
cd KAIRO
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Pull required Ollama models (defaults use Qwen + nomic embed)
ollama pull qwen3:4b-instruct-q4_K_M
ollama pull nomic-embed-text

# Enroll your face (one-time)
kairo-enroll-face
```

## Running

```bash
# Run directly
kairo

# Or install as launchd service (auto-start on login, auto-restart on crash)
bash adapters/macos/launchd/install_launchd.sh
```

## Configuration

- **`config/kairo.yaml`** — models, session timeouts, workspace modes, memory paths, STT/wake settings.
- **`config/identity.yaml`** — assistant name, wake phrases, owner, verification mode.

If you still have **`config/jarvis.yaml`**, the loader will load it with a deprecation warning; rename to **`kairo.yaml`** and update paths to **`~/.kairo/`**.

## Workspace modes

Voice: "Open office mode" / "Switch to focus mode" — runs tool sequences defined in **`config/kairo.yaml`**.

## Architecture (implementation)

- **Event-driven**: inter-module communication via async event bus
- **Sensor threads**: camera, face, gesture, voice in a thread pool
- **Explicit FSM**: session states (e.g. SLEEP → WAKE_PENDING → ACTIVE_SESSION → EXECUTING)
- **Dynamic tools**: plugins discovered from `tools/`
- **Platform adapters**: macOS via AppleScript; Windows adapter stub for future parity

## Tests

```bash
pytest tests/ -v
```

## Logs

Logs: **`~/.kairo/kairo.log`** (rotating, 10MB max).
