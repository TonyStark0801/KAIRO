# Jarvis Runtime

A production-grade macOS desktop assistant daemon. Activated by multi-modal gestures (face + clap + snap), accepts voice commands, routes them via local LLM (Ollama), and executes desktop actions.

## Prerequisites

- macOS 13+
- Python 3.11+
- Ollama installed and running (`brew install ollama && ollama serve`)
- Camera access for gesture/face detection

## Setup

```bash
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

Workspace modes execute a sequence of tool actions defined in `jarvis.yaml`.

## Architecture

- **Event-driven**: All inter-module communication via async event bus
- **Sensor threads**: Camera, face, gesture, voice run in ThreadPoolExecutor
- **Pure FSM**: Session states (SLEEP -> WAKE_PENDING -> ACTIVE_SESSION -> EXECUTING)
- **Dynamic tools**: Plugins auto-discovered from `tools/` directory
- **Platform adapters**: macOS adapter uses AppleScript; Windows stub ready for future

## Tests

```bash
pytest tests/ -v
```

## Logs

Logs are written to `~/.jarvis/jarvis.log` (rotating, 10MB max).
