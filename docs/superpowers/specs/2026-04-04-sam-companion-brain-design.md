# Sam Companion Brain — Design Spec

**Date:** 2026-04-04
**Status:** Draft
**Goal:** Transform Sam from a voice-controlled launcher into a proactive, conversational AI companion with internet access, real intelligence, memory, and adaptive personality.

---

## 1. Tier Architecture

Three tiers, unchanged at the edges, rebuilt at the core.

### Tier 1: Keyword Fast-Path (unchanged)
- Regex-matched commands: volume, next/pause/skip, open apps, open URLs
- Zero LLM calls, instant execution
- File: `assistant_core/fast_path.py`

### Tier 2: Local Fast Model (unchanged)
- `llama3.2:3b` via Ollama with compact prompt (`fast.j2`)
- Handles simple tool routing: "play lofi beats" → youtube_search
- Returns `escalate` for anything beyond its scope
- ~0.5s response time

### Tier 3: Cloud Agent Loop (NEW — replaces local 8b)
- **Provider:** Groq API (free tier, Llama 3.3 70B Versatile)
- **Swappable:** Provider interface allows OpenAI/Anthropic/Gemini swap via config
- **Agent loop:** LLM can call tools mid-conversation, read results, call more tools (max 3 iterations per user turn)
- **Function calling:** Uses Groq's native tool/function calling format
- **Fallback:** If Groq unreachable, falls back to local `llama3.1` with "I can't access the internet right now" caveat
- ~0.5-1s per LLM call, ~1-3s for multi-tool flows

### Escalation triggers (Tier 2 → Tier 3):
- Tier 2 returns `{"action": "escalate"}`
- Transcript matches deep trigger patterns (explain, analyze, tell me about, etc.)
- Conversation turn count >= 8 in session
- Any request that needs internet/tools beyond basic commands

### Config addition (`jarvis.yaml`):
```yaml
groq:
  api_key_env: GROQ_API_KEY
  model: llama-3.3-70b-versatile
  max_tokens: 300
  temperature: 0.7
```

---

## 2. Groq Cloud Provider

### New file: `llm_router/groq_provider.py`

Implements `LLMProvider` interface alongside `OllamaProvider`.

**Key methods:**
- `initialize()` — validate API key from env, test connection
- `chat(system_prompt, messages, tools=None)` — standard chat, returns text
- `chat_with_tools(system_prompt, messages, tools)` — function calling mode, returns text + tool_calls list

**Native function calling:** Groq supports OpenAI-compatible function calling. Tools are passed as JSON schemas. The model returns structured `tool_calls` — no JSON parsing gymnastics.

**Keepalive/warmup:** Not needed for cloud API.

**Dependency:** `groq` Python package (official SDK).

---

## 3. Agent Loop

### New file: `assistant_core/agent_loop.py`

Called by the reasoner when Tier 3 activates.

```
def run_agent_loop(transcript, conversation_history, tools, system_prompt) -> AgentResult:
    messages = build_messages(system_prompt, conversation_history, transcript)
    
    for iteration in range(MAX_ITERATIONS):  # MAX_ITERATIONS = 3
        response = groq.chat_with_tools(messages, tools)
        
        if response.tool_calls:
            # Speak interim message if provided ("Let me check...")
            if response.message:
                yield SpeakEvent(response.message)
            
            # Execute each tool call
            for tool_call in response.tool_calls:
                result = execute_tool(tool_call)
                messages.append(tool_result_message(tool_call, result))
        else:
            # No tools called — final response
            return AgentResult(message=response.message, tools_used=[...])
    
    # Hit max iterations — return whatever we have
    return AgentResult(message=response.message, tools_used=[...], capped=True)
```

**Key behaviors:**
- Yields `SpeakEvent` mid-loop so Sam can say "Let me check..." while searching
- Tool results injected as tool-role messages (Groq format)
- Max 3 iterations hard cap
- Returns all tools used for behavioral tracking

### AgentResult dataclass:
```python
@dataclass
class AgentResult:
    message: str           # Final spoken response
    tools_used: list[str]  # For behavioral tracking
    capped: bool = False   # True if hit max iterations
```

---

## 4. Internet Tools (Inner Tools)

Callable only by the Tier 3 agent loop. Not directly voice-triggered.

### 4.1 `web_search` — `tools/internet/web_search.py`
- DuckDuckGo Instant Answer API (free, no key)
- Returns: top 5 results with title + snippet + URL
- Fallback: DuckDuckGo HTML search scrape if API returns nothing

### 4.2 `web_browse` — `tools/internet/web_browse.py`
- Fetches a URL with `httpx`
- Extracts main content with `readability-lxml` (strips ads, nav, scripts)
- Returns: clean text, truncated to 2000 chars (keeps Groq context small)
- Dependency: `readability-lxml`

### 4.3 `weather` — `tools/internet/weather.py`
- Open-Meteo API (free, no key, no auth)
- Input: city name → geocode → weather
- Returns: current temp, condition, today's high/low, tomorrow forecast

### 4.4 `add_todo` — `tools/tasks/todo_tool.py`
- CRUD operations: add, list, complete, delete
- SQLite storage at `~/.jarvis/todos.db`
- Schema:
  ```sql
  CREATE TABLE todos (
      id INTEGER PRIMARY KEY,
      title TEXT NOT NULL,
      due_date TEXT,
      due_time TEXT,
      status TEXT DEFAULT 'pending',
      source TEXT,
      created_at TEXT,
      completed_at TEXT,
      context TEXT
  );
  ```
- Actions: `add`, `list`, `complete`, `delete`
- On `list`: also pushes macOS notification with formatted list
- Optional: sync to Apple Notes via AppleScript ("Sam's TODOs" note)

### 4.5 `google_calendar` — `tools/internet/calendar_tool.py` (stretch)
- Google Calendar API via OAuth 2.0
- One-time browser auth flow, token stored at `~/.jarvis/gcal_token.json`
- Actions: `list_events(days=1)`, `add_event(title, datetime, duration)`
- Deferred to Phase 2 if OAuth setup is too complex for initial build

---

## 5. Memory Injection

### Changes to `assistant_core/personality.py`

New method: `build_cloud_prompt(tool_metas, recent_commands, transcript)` — builds the rich system prompt for Tier 3.

**Prompt structure for Groq:**
```
You are Sam (Samantha), Tony's personal AI companion on macOS.

## Personality Mode: {mood_mode}
{mood_specific_instructions}

## What you know about Tony:
- Preferences: {from PreferencesMemory}
- Habits: {from BehavioralQuery — frequent tools, time patterns}
- Recent sessions: {last 3 session summaries from SessionStore}

## Relevant memories:
{top 3 vector search results from VectorMemoryClient, searched by current transcript}

## Current context (do not share externally):
- Active app: {app_name}
- Working on: {repo/file}
- Time: {time_of_day}

## Conversation rules:
- Be natural, warm, conversational. Not robotic.
- You can call tools to search the web, check weather, manage TODOs.
- Speak mid-thought if searching ("Let me check...")
- If Tony mentions plans/travel, offer to search for options.
- Auto-act on safe things (TODOs, reminders). Ask before risky things.
- Match Tony's energy. Casual → casual. Focused → professional.
```

### Mood detection: `assistant_core/mood.py`

Simple rule-based mood detector:

```python
def detect_mood(context: WorkspaceContext, transcript: str, hour: int) -> MoodMode:
    # WORK: IDE/terminal active during work hours
    if context.active_app in WORK_APPS and 9 <= hour <= 18:
        return MoodMode.WORK
    
    # SUPPORTIVE: frustration signals in transcript
    if any(w in transcript.lower() for w in FRUSTRATION_WORDS):
        return MoodMode.SUPPORTIVE
    
    # CASUAL: default
    return MoodMode.CASUAL
```

Three modes with different prompt injections:
- **CASUAL:** "Speak like a close friend. Playful, warm, uses humor."
- **WORK:** "Act as a senior SDE-3. Be precise, call out issues, suggest improvements. Professional but not stiff."
- **SUPPORTIVE:** "Be empathetic and supportive. Listen first. Don't try to fix everything immediately."

---

## 6. Voice — Edge TTS Streaming

### Rewrite: `voice_service/piper_engine.py`

**Priority chain:**
1. **Edge TTS streaming** — `edge_tts.Communicate().stream()` → pipe to `mpv --no-video -`
2. **Piper** (if installed) — local, fast, natural
3. **macOS `say`** — last resort fallback

**Edge TTS streaming flow:**
```python
async def _speak_edge_stream(self, text: str) -> None:
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    
    proc = await asyncio.create_subprocess_exec(
        "mpv", "--no-video", "--no-terminal", "-",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            proc.stdin.write(chunk["data"])
            await proc.stdin.drain()
    
    proc.stdin.close()
    await proc.wait()
```

**Time to first word:** ~300ms (vs 5-15s with full download).

**New dependency:** `mpv` via Homebrew (`brew install mpv`).

---

## 7. Proactive Engine

### New file: `assistant_core/proactive.py`

Runs on a 60-second interval timer in the daemon. Lightweight — no LLM calls for detection, only for message generation.

**Trigger checks (every 60s):**

| Trigger | Condition | Action |
|---------|-----------|--------|
| Morning briefing | First wake before 10am, session just started | Auto: speak calendar + TODOs + last session context |
| TODO reminder | Any TODO with `due_date` = today and `status` = pending | Auto: speak reminder |
| Long focus | Same app active for 2+ hours | Suggest: "Want some music?" |
| Evening wind-down | Time > 9pm, user active | Suggest: "Getting late, want chill music?" |
| Pattern suggestion | Behavioral DB shows repeated pattern not yet automated | Suggest: "Want me to do X automatically?" |

**How it fires:**
1. `ProactiveEngine.check()` runs every 60s
2. Returns `ProactiveSuggestion(type, message_template, auto_act: bool, tool_call: dict | None)`
3. Daemon receives it, if `auto_act` → execute tool + speak
4. If not `auto_act` → speak suggestion, wait for user response

**Cooldown:** Each trigger type has a cooldown (e.g., morning briefing once per day, TODO reminders max once per hour per item).

---

## 8. New Dependencies

```toml
# pyproject.toml additions
"groq>=0.9",
"readability-lxml>=0.8",
```

**System dependency:** `brew install mpv`

---

## 9. Config Changes

### `jarvis.yaml` additions:
```yaml
groq:
  api_key_env: GROQ_API_KEY
  model: llama-3.3-70b-versatile
  max_tokens: 300
  temperature: 0.7

proactive:
  enabled: true
  check_interval: 60
  morning_briefing: true
  todo_reminders: true
  focus_suggestions: true
```

---

## 10. Files Changed / Created

### New files:
- `llm_router/groq_provider.py` — Groq cloud LLM provider
- `assistant_core/agent_loop.py` — Tier 3 tool-calling loop
- `assistant_core/mood.py` — Context-based mood detection
- `assistant_core/proactive.py` — Proactive suggestion engine
- `tools/internet/web_search.py` — DuckDuckGo search
- `tools/internet/web_browse.py` — URL content extraction
- `tools/internet/weather.py` — Open-Meteo weather
- `tools/tasks/todo_tool.py` — TODO CRUD + notifications
- `tools/tasks/todo_store.py` — SQLite TODO storage
- `llm_router/prompts/cloud.j2` — Rich system prompt for Groq

### Modified files:
- `assistant_core/reasoner.py` — Wire Tier 3 to agent loop instead of local 8b
- `assistant_core/personality.py` — Add `build_cloud_prompt()`, inject behavioral + vector memory
- `voice_service/piper_engine.py` — Edge TTS streaming with mpv
- `runtime/daemon.py` — Wire Groq provider, proactive engine, TODO store
- `core/config/models.py` — Add `GroqConfig`, `ProactiveConfig`
- `config/jarvis.yaml` — Add groq + proactive sections
- `pyproject.toml` — Add groq, readability-lxml deps

### Unchanged:
- Tier 1 fast-path, Tier 2 local 3b, mic listener, wake word, FSM, event bus, all existing tools
