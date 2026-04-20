# Sam Companion Brain — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform Sam from a voice-controlled launcher into a conversational AI companion with Groq cloud brain, internet tools, memory injection, streaming voice, and proactive behavior.

**Architecture:** Hybrid 3-tier system. Tier 1 (keyword regex) and Tier 2 (local 3b) stay unchanged. Tier 3 becomes a Groq-powered agent loop with tool calling. Memory, mood, and context are injected into Tier 3 prompts. Edge TTS streaming replaces macOS `say`. A proactive engine runs on a 60s timer for suggestions and reminders.

**Tech Stack:** Python 3.11+, Groq SDK, edge-tts, mpv, DuckDuckGo search, Open-Meteo weather API, readability-lxml, aiosqlite, httpx.

---

### Task 1: Config — Add Groq and Proactive settings

**Files:**
- Modify: `jarvis-runtime/core/config/models.py`
- Modify: `jarvis-runtime/config/jarvis.yaml`
- Modify: `jarvis-runtime/pyproject.toml`

- [ ] **Step 1: Add GroqConfig and ProactiveConfig to models.py**

Add after `BrowserConfig`:

```python
class GroqConfig(BaseModel):
    api_key_env: str = "GROQ_API_KEY"
    model: str = "llama-3.3-70b-versatile"
    max_tokens: int = 300
    temperature: float = 0.7


class ProactiveConfig(BaseModel):
    enabled: bool = True
    check_interval: int = 60
    morning_briefing: bool = True
    todo_reminders: bool = True
    focus_suggestions: bool = True
```

Add to `JarvisConfig`:

```python
class JarvisConfig(BaseModel):
    # ... existing fields ...
    groq: GroqConfig = Field(default_factory=GroqConfig)
    proactive: ProactiveConfig = Field(default_factory=ProactiveConfig)
```

- [ ] **Step 2: Add groq + proactive sections to jarvis.yaml**

Append to `config/jarvis.yaml`:

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

- [ ] **Step 3: Add new dependencies to pyproject.toml**

Add to `dependencies` list:

```toml
"groq>=0.9",
"readability-lxml>=0.8",
"duckduckgo-search>=7.0",
```

- [ ] **Step 4: Install dependencies**

Run: `cd jarvis-runtime && pip install -e .`

- [ ] **Step 5: Install mpv for streaming audio**

Run: `brew install mpv`

---

### Task 2: Groq Cloud Provider

**Files:**
- Create: `jarvis-runtime/llm_router/groq_provider.py`

- [ ] **Step 1: Create groq_provider.py**

```python
"""Groq cloud LLM provider — smart brain with native function calling."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class GroqResponse:
    message: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class GroqProvider:
    def __init__(
        self,
        api_key_env: str = "GROQ_API_KEY",
        model: str = "llama-3.3-70b-versatile",
        max_tokens: int = 300,
        temperature: float = 0.7,
    ) -> None:
        self._api_key_env = api_key_env
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._healthy = False
        self._client = None

    @property
    def healthy(self) -> bool:
        return self._healthy

    async def initialize(self) -> bool:
        api_key = os.environ.get(self._api_key_env, "")
        if not api_key:
            logger.warning("Groq API key not found in env var %s", self._api_key_env)
            return False
        try:
            from groq import AsyncGroq
            self._client = AsyncGroq(api_key=api_key)
            # Test connection with a trivial call
            await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            self._healthy = True
            logger.info("Groq connected: model=%s", self._model)
            return True
        except Exception:
            logger.exception("Failed to connect to Groq")
            return False

    async def chat(self, system_prompt: str, messages: list[dict]) -> str:
        if not self._healthy or not self._client:
            return ""
        try:
            full_messages = [{"role": "system", "content": system_prompt}]
            full_messages.extend(messages[-20:])
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=full_messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            return response.choices[0].message.content or ""
        except Exception:
            logger.exception("Groq chat failed")
            return ""

    async def chat_with_tools(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
    ) -> GroqResponse:
        if not self._healthy or not self._client:
            return GroqResponse()
        try:
            import json
            full_messages = [{"role": "system", "content": system_prompt}]
            full_messages.extend(messages[-20:])

            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": full_messages,
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = await self._client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            msg = choice.message

            tool_calls = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    ))

            return GroqResponse(
                message=msg.content or "",
                tool_calls=tool_calls,
            )
        except Exception:
            logger.exception("Groq chat_with_tools failed")
            return GroqResponse()
```

- [ ] **Step 2: Verify import works**

Run: `cd jarvis-runtime && python -c "from llm_router.groq_provider import GroqProvider; print('OK')"`
Expected: `OK`

---

### Task 3: TODO Store + Tool

**Files:**
- Create: `jarvis-runtime/tools/tasks/__init__.py`
- Create: `jarvis-runtime/tools/tasks/todo_store.py`
- Create: `jarvis-runtime/tools/tasks/todo_tool.py`

- [ ] **Step 1: Create `tools/tasks/__init__.py`**

Empty file.

- [ ] **Step 2: Create todo_store.py**

```python
"""SQLite TODO storage — Sam's internal task list."""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

_DB_PATH = Path("~/.jarvis/todos.db").expanduser()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS todos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    due_date TEXT,
    due_time TEXT,
    status TEXT DEFAULT 'pending',
    source TEXT DEFAULT 'conversation',
    created_at TEXT NOT NULL,
    completed_at TEXT,
    context TEXT
)
"""


class TodoStore:
    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or str(_DB_PATH)
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> bool:
        try:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._db = await aiosqlite.connect(self._db_path)
            await self._db.execute(_SCHEMA)
            await self._db.commit()
            return True
        except Exception:
            logger.exception("TodoStore init failed")
            return False

    async def add(
        self, title: str, due_date: str | None = None, due_time: str | None = None,
        source: str = "conversation", context: dict | None = None,
    ) -> int:
        now = datetime.datetime.now().isoformat()
        ctx_json = json.dumps(context) if context else None
        async with self._db.execute(
            "INSERT INTO todos (title, due_date, due_time, source, created_at, context) VALUES (?, ?, ?, ?, ?, ?)",
            (title, due_date, due_time, source, now, ctx_json),
        ) as cursor:
            await self._db.commit()
            return cursor.lastrowid

    async def list_pending(self) -> list[dict]:
        async with self._db.execute(
            "SELECT id, title, due_date, due_time, created_at FROM todos WHERE status = 'pending' ORDER BY created_at DESC LIMIT 20"
        ) as cursor:
            rows = await cursor.fetchall()
            return [{"id": r[0], "title": r[1], "due_date": r[2], "due_time": r[3], "created_at": r[4]} for r in rows]

    async def list_due_today(self) -> list[dict]:
        today = datetime.date.today().isoformat()
        async with self._db.execute(
            "SELECT id, title, due_time FROM todos WHERE status = 'pending' AND due_date = ?", (today,)
        ) as cursor:
            rows = await cursor.fetchall()
            return [{"id": r[0], "title": r[1], "due_time": r[2]} for r in rows]

    async def complete(self, todo_id: int) -> bool:
        now = datetime.datetime.now().isoformat()
        async with self._db.execute(
            "UPDATE todos SET status = 'done', completed_at = ? WHERE id = ?", (now, todo_id),
        ) as cursor:
            await self._db.commit()
            return cursor.rowcount > 0

    async def delete(self, todo_id: int) -> bool:
        async with self._db.execute("DELETE FROM todos WHERE id = ?", (todo_id,)) as cursor:
            await self._db.commit()
            return cursor.rowcount > 0

    async def close(self) -> None:
        if self._db:
            await self._db.close()
```

- [ ] **Step 3: Create todo_tool.py**

```python
"""Tool: Manage TODOs — add, list, complete, delete."""

from __future__ import annotations

from typing import Any

from tools._base import BaseTool, ToolResult


class TodoTool(BaseTool):
    @property
    def name(self) -> str:
        return "manage_todos"

    @property
    def description(self) -> str:
        return "Manage Tony's TODO list. Actions: add, list, complete, delete."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "list", "complete", "delete"]},
                "title": {"type": "string", "description": "TODO title (for add)"},
                "due_date": {"type": "string", "description": "Due date YYYY-MM-DD (optional)"},
                "due_time": {"type": "string", "description": "Due time HH:MM (optional)"},
                "todo_id": {"type": "integer", "description": "TODO id (for complete/delete)"},
            },
            "required": ["action"],
        }

    async def execute(self, params: dict[str, Any], adapter) -> ToolResult:
        store = params.get("_todo_store")
        if not store:
            return ToolResult(success=False, message="TODO store not available")

        action = params.get("action", "list")

        if action == "add":
            title = params.get("title", "")
            if not title:
                return ToolResult(success=False, message="No title provided")
            todo_id = await store.add(
                title=title,
                due_date=params.get("due_date"),
                due_time=params.get("due_time"),
            )
            return ToolResult(success=True, message=f"Added: {title}", data={"id": todo_id})

        if action == "list":
            todos = await store.list_pending()
            if not todos:
                return ToolResult(success=True, message="No pending TODOs.")
            lines = [f"{t['id']}. {t['title']}" + (f" (due {t['due_date']})" if t['due_date'] else "") for t in todos]
            msg = "Your TODOs:\n" + "\n".join(lines)
            if adapter:
                await adapter.send_notification("Sam — TODOs", msg)
            return ToolResult(success=True, message=msg, data={"speak_result": True})

        if action == "complete":
            todo_id = params.get("todo_id")
            if todo_id and await store.complete(todo_id):
                return ToolResult(success=True, message="Marked as done.")
            return ToolResult(success=False, message="TODO not found.")

        if action == "delete":
            todo_id = params.get("todo_id")
            if todo_id and await store.delete(todo_id):
                return ToolResult(success=True, message="Deleted.")
            return ToolResult(success=False, message="TODO not found.")

        return ToolResult(success=False, message=f"Unknown action: {action}")
```

- [ ] **Step 4: Verify imports**

Run: `cd jarvis-runtime && python -c "from tools.tasks.todo_store import TodoStore; from tools.tasks.todo_tool import TodoTool; print('OK')"`

---

### Task 4: Internet Tools — Web Search, Browse, Weather

**Files:**
- Create: `jarvis-runtime/tools/internet/__init__.py`
- Create: `jarvis-runtime/tools/internet/web_search.py`
- Create: `jarvis-runtime/tools/internet/web_browse.py`
- Create: `jarvis-runtime/tools/internet/weather.py`

- [ ] **Step 1: Create `tools/internet/__init__.py`**

Empty file.

- [ ] **Step 2: Create web_search.py**

```python
"""Tool: Search the web via DuckDuckGo."""

from __future__ import annotations

import logging
from typing import Any

from tools._base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the internet. Returns top results with titles and snippets."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        }

    async def execute(self, params: dict[str, Any], adapter) -> ToolResult:
        query = params.get("query", "")
        if not query:
            return ToolResult(success=False, message="No search query provided")
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
            if not results:
                return ToolResult(success=True, message="No results found.", data={"results": []})
            lines = []
            for r in results:
                lines.append(f"**{r.get('title', '')}**")
                lines.append(r.get("body", ""))
                lines.append(r.get("href", ""))
                lines.append("")
            return ToolResult(success=True, message="\n".join(lines), data={"results": results})
        except Exception:
            logger.exception("Web search failed")
            return ToolResult(success=False, message="Search failed, try again.")
```

- [ ] **Step 3: Create web_browse.py**

```python
"""Tool: Browse a URL and extract readable text content."""

from __future__ import annotations

import logging
from typing import Any

from tools._base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_MAX_CHARS = 2000


class WebBrowseTool(BaseTool):
    @property
    def name(self) -> str:
        return "web_browse"

    @property
    def description(self) -> str:
        return "Fetch a URL and extract the main text content (strips ads/nav)."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL to fetch"},
            },
            "required": ["url"],
        }

    async def execute(self, params: dict[str, Any], adapter) -> ToolResult:
        url = params.get("url", "")
        if not url:
            return ToolResult(success=False, message="No URL provided")
        try:
            import httpx
            from readability import Document
            async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
            doc = Document(resp.text)
            import re
            text = re.sub(r"<[^>]+>", " ", doc.summary())
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > _MAX_CHARS:
                text = text[:_MAX_CHARS] + "..."
            return ToolResult(success=True, message=text, data={"title": doc.title(), "url": url})
        except Exception:
            logger.exception("Web browse failed for %s", url)
            return ToolResult(success=False, message=f"Couldn't fetch {url}")
```

- [ ] **Step 4: Create weather.py**

```python
"""Tool: Get weather via Open-Meteo (free, no API key)."""

from __future__ import annotations

import logging
from typing import Any

from tools._base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

_WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Rime fog", 51: "Light drizzle", 53: "Drizzle",
    55: "Heavy drizzle", 61: "Light rain", 63: "Rain", 65: "Heavy rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 80: "Rain showers",
    81: "Moderate showers", 82: "Heavy showers", 95: "Thunderstorm",
}


class WeatherTool(BaseTool):
    @property
    def name(self) -> str:
        return "weather"

    @property
    def description(self) -> str:
        return "Get current weather and forecast for a city."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. 'Delhi' or 'Mumbai'"},
            },
            "required": ["city"],
        }

    async def execute(self, params: dict[str, Any], adapter) -> ToolResult:
        city = params.get("city", "")
        if not city:
            return ToolResult(success=False, message="No city provided")
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                geo = await client.get(_GEOCODE_URL, params={"name": city, "count": 1})
                geo.raise_for_status()
                geo_data = geo.json()
                results = geo_data.get("results", [])
                if not results:
                    return ToolResult(success=False, message=f"Couldn't find {city}")
                lat, lon = results[0]["latitude"], results[0]["longitude"]
                place = results[0].get("name", city)

                weather = await client.get(_WEATHER_URL, params={
                    "latitude": lat, "longitude": lon,
                    "current": "temperature_2m,weather_code,wind_speed_10m",
                    "daily": "temperature_2m_max,temperature_2m_min,weather_code",
                    "timezone": "auto", "forecast_days": 2,
                })
                weather.raise_for_status()
                w = weather.json()

            current = w.get("current", {})
            daily = w.get("daily", {})
            temp = current.get("temperature_2m", "?")
            code = current.get("weather_code", 0)
            condition = _WMO_CODES.get(code, "Unknown")
            wind = current.get("wind_speed_10m", "?")

            today_max = daily.get("temperature_2m_max", [None])[0]
            today_min = daily.get("temperature_2m_min", [None])[0]
            tomorrow_max = daily.get("temperature_2m_max", [None, None])[1] if len(daily.get("temperature_2m_max", [])) > 1 else None
            tomorrow_code = daily.get("weather_code", [None, None])[1] if len(daily.get("weather_code", [])) > 1 else None

            msg = f"{place}: {temp}°C, {condition}, wind {wind} km/h."
            if today_max and today_min:
                msg += f" Today: {today_min}°–{today_max}°C."
            if tomorrow_max and tomorrow_code is not None:
                msg += f" Tomorrow: {tomorrow_max}°C, {_WMO_CODES.get(tomorrow_code, 'Unknown')}."

            return ToolResult(success=True, message=msg, data={"temp": temp, "condition": condition})
        except Exception:
            logger.exception("Weather lookup failed")
            return ToolResult(success=False, message="Couldn't get weather right now.")
```

- [ ] **Step 5: Verify all imports**

Run: `cd jarvis-runtime && python -c "from tools.internet.web_search import WebSearchTool; from tools.internet.web_browse import WebBrowseTool; from tools.internet.weather import WeatherTool; print('OK')"`

---

### Task 5: Mood Detection

**Files:**
- Create: `jarvis-runtime/assistant_core/mood.py`

- [ ] **Step 1: Create mood.py**

```python
"""Mood detection — adapts Sam's personality based on context."""

from __future__ import annotations

from enum import Enum


class MoodMode(Enum):
    CASUAL = "casual"
    WORK = "work"
    SUPPORTIVE = "supportive"


_WORK_APPS = {
    "IntelliJ IDEA", "IntelliJ IDEA CE", "WebStorm", "PyCharm",
    "Visual Studio Code", "Cursor", "Terminal", "iTerm2", "Warp",
}

_FRUSTRATION_WORDS = {
    "ugh", "damn", "frustrated", "annoying", "hate", "stuck",
    "broken", "why won't", "doesn't work", "fed up", "tired",
    "sad", "upset", "stressed", "overwhelmed", "depressed",
}

_MOOD_PROMPTS = {
    MoodMode.CASUAL: (
        "Speak like a close friend. Playful, warm, uses humor. "
        "Use contractions. Be yourself — confident, occasionally sarcastic, always kind."
    ),
    MoodMode.WORK: (
        "Act as a senior SDE-3 colleague. Be precise, direct, and helpful. "
        "Call out potential issues proactively. Suggest improvements. "
        "Professional but not stiff — you're a teammate, not a robot."
    ),
    MoodMode.SUPPORTIVE: (
        "Be empathetic and supportive. Listen first. Don't try to fix everything immediately. "
        "Acknowledge feelings. Be warm and serious — no jokes right now. "
        "Offer help gently: 'Want to talk about it?' or 'Can I help with anything?'"
    ),
}


def detect_mood(active_app: str, transcript: str, hour: int) -> MoodMode:
    lower = transcript.lower()
    if any(w in lower for w in _FRUSTRATION_WORDS):
        return MoodMode.SUPPORTIVE

    if active_app in _WORK_APPS and 9 <= hour <= 18:
        return MoodMode.WORK

    return MoodMode.CASUAL


def get_mood_prompt(mood: MoodMode) -> str:
    return _MOOD_PROMPTS.get(mood, _MOOD_PROMPTS[MoodMode.CASUAL])
```

---

### Task 6: Cloud Prompt Template + Memory Injection

**Files:**
- Create: `jarvis-runtime/llm_router/prompts/cloud.j2`
- Modify: `jarvis-runtime/assistant_core/personality.py`

- [ ] **Step 1: Create cloud.j2**

```jinja2
You are {{ assistant_name }} ({{ full_name }}), {{ owner_name }}'s personal AI companion running on macOS.

## Personality Mode: {{ mood_mode }}
{{ mood_prompt }}

## What you know about {{ owner_name }}:
{% if preferences %}
- Preferences: {{ preferences }}
{% endif %}
{% if habits %}
- Habits: {{ habits }}
{% endif %}
{% if session_summaries %}
- Recent sessions:
{% for s in session_summaries %}
  - {{ s }}
{% endfor %}
{% endif %}

{% if relevant_memories %}
## Relevant past interactions:
{% for m in relevant_memories %}
- {{ m }}
{% endfor %}
{% endif %}

{% if workspace_context %}
## Current context:
{{ owner_name }} is in: {{ workspace_context }}
{% endif %}

Time: {{ time_of_day }}

## Rules:
- Be natural and conversational. Match {{ owner_name }}'s energy.
- You have tools to search the web, check weather, manage TODOs, and control the Mac.
- When searching or looking things up, say "Let me check..." before calling a tool.
- For safe actions (TODOs, reminders, opening apps), just do it. For risky actions (bookings, sending), ask first.
- If {{ owner_name }} mentions plans or travel, proactively offer to search for options.
- Keep spoken responses concise — 1-3 sentences. This is voice, not text.
- NEVER say "How can I assist you?" or any robotic phrase.
```

- [ ] **Step 2: Add build_cloud_prompt to personality.py**

Add these imports at top of `personality.py`:

```python
from assistant_core.mood import MoodMode, detect_mood, get_mood_prompt
```

Add new method after `build_fast_prompt`:

```python
    async def build_cloud_prompt(
        self,
        tool_metas: list[ToolMeta],
        transcript: str,
        behavioral_query=None,
        vector_client=None,
    ) -> str:
        workspace_ctx = await self._context.get_context()
        workspace_summary = workspace_ctx.natural_summary()

        import datetime
        now = datetime.datetime.now()
        hour = now.hour

        mood = detect_mood(workspace_ctx.active_app, transcript, hour)
        mood_prompt = get_mood_prompt(mood)

        if 5 <= hour < 12:
            time_of_day = f"morning ({now.strftime('%I:%M %p')})"
        elif 12 <= hour < 17:
            time_of_day = f"afternoon ({now.strftime('%I:%M %p')})"
        elif 17 <= hour < 21:
            time_of_day = f"evening ({now.strftime('%I:%M %p')})"
        else:
            time_of_day = f"night ({now.strftime('%I:%M %p')})"

        # Behavioral patterns
        habits = ""
        if behavioral_query:
            try:
                recent_tools = await behavioral_query.get_recent_tools(5)
                patterns = await behavioral_query.get_time_of_day_pattern()
                if recent_tools:
                    habits = f"Frequently uses: {', '.join(recent_tools)}."
                if patterns:
                    for bucket, tools in patterns.items():
                        if tools:
                            habits += f" {bucket}: often uses {', '.join(tools[:3])}."
            except Exception:
                pass

        # Vector memory search
        relevant_memories = []
        if vector_client and vector_client.healthy:
            try:
                results = await vector_client.search(transcript, n_results=3)
                relevant_memories = [r["document"] for r in results if r.get("document")]
            except Exception:
                pass

        # Session summaries
        session_summaries = []
        try:
            recent = await self._session_store.get_recent_sessions(3)
            session_summaries = [s["summary"] for s in recent if s.get("summary")]
        except Exception:
            pass

        # Preferences
        prefs = ""
        try:
            all_prefs = await self._preferences.get_all()
            if all_prefs:
                prefs = ", ".join(f"{k}: {v}" for k, v in all_prefs.items())
        except Exception:
            pass

        template = self._jinja.get_template("cloud.j2")
        return template.render(
            assistant_name=self._identity.assistant_name,
            full_name=self._identity.full_name,
            owner_name=self._identity.owner_name,
            mood_mode=mood.value,
            mood_prompt=mood_prompt,
            preferences=prefs,
            habits=habits,
            session_summaries=session_summaries,
            relevant_memories=relevant_memories,
            workspace_context=workspace_summary,
            time_of_day=time_of_day,
        )
```

- [ ] **Step 3: Add `get_recent_sessions` to SessionStore if missing**

Check `memory_service/session_store.py`. If `get_recent_sessions` doesn't exist, add:

```python
    async def get_recent_sessions(self, limit: int = 3) -> list[dict]:
        if not self._db:
            return []
        async with self._db.execute(
            "SELECT session_id, summary, started_at, ended_at FROM sessions WHERE summary != '' ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [{"session_id": r[0], "summary": r[1], "started_at": r[2], "ended_at": r[3]} for r in rows]
```

- [ ] **Step 4: Add `get_all` to PreferencesMemory if missing**

Check `memory_service/preferences.py`. If `get_all` doesn't exist, add:

```python
    async def get_all(self) -> dict[str, str]:
        if not self._db:
            return {}
        async with self._db.execute("SELECT key, value FROM preferences") as cursor:
            rows = await cursor.fetchall()
            return {r[0]: r[1] for r in rows}
```

- [ ] **Step 5: Add `full_name` property to IdentityMemory if missing**

Check `memory_service/identity.py`. If `full_name` doesn't exist, add:

```python
    @property
    def full_name(self) -> str:
        return self._data.get("assistant", {}).get("full_name", self.assistant_name)
```

---

### Task 7: Agent Loop

**Files:**
- Create: `jarvis-runtime/assistant_core/agent_loop.py`

- [ ] **Step 1: Create agent_loop.py**

```python
"""Tier 3 Agent Loop — Groq with tool calling, max 3 iterations."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_router.groq_provider import GroqProvider
    from tools._base import BaseTool, ToolMeta

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = 3


@dataclass
class AgentResult:
    message: str = ""
    tools_used: list[str] = field(default_factory=list)
    capped: bool = False
    interim_messages: list[str] = field(default_factory=list)


def _tool_metas_to_groq_tools(metas: list[ToolMeta]) -> list[dict]:
    """Convert our ToolMeta list to Groq/OpenAI function calling format."""
    tools = []
    for m in metas:
        schema = dict(m.parameters_schema)
        # Strip internal fields that shouldn't be exposed to LLM
        props = schema.get("properties", {})
        props = {k: v for k, v in props.items() if not k.startswith("_")}
        schema["properties"] = props
        required = [r for r in schema.get("required", []) if not r.startswith("_")]
        schema["required"] = required

        tools.append({
            "type": "function",
            "function": {
                "name": m.name,
                "description": m.description,
                "parameters": schema,
            },
        })
    return tools


class AgentLoop:
    def __init__(
        self,
        groq: GroqProvider,
        tool_registry,
        adapter,
    ) -> None:
        self._groq = groq
        self._registry = tool_registry
        self._adapter = adapter

    async def run(
        self,
        system_prompt: str,
        conversation_history: list[dict[str, str]],
        tool_metas: list[ToolMeta],
        extra_params: dict[str, Any] | None = None,
    ) -> AgentResult:
        groq_tools = _tool_metas_to_groq_tools(tool_metas)
        messages = list(conversation_history)

        result = AgentResult()

        for iteration in range(_MAX_ITERATIONS):
            response = await self._groq.chat_with_tools(
                system_prompt, messages, groq_tools,
            )

            if not response.tool_calls:
                result.message = response.message
                return result

            if response.message:
                result.interim_messages.append(response.message)

            # Build assistant message with tool calls for history
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": response.message or ""}
            tc_list = []
            for tc in response.tool_calls:
                tc_list.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                })
            assistant_msg["tool_calls"] = tc_list
            messages.append(assistant_msg)

            for tc in response.tool_calls:
                tool = self._registry.get(tc.name)
                if tool is None:
                    tool_result_text = f"Tool '{tc.name}' not found."
                else:
                    params = dict(tc.arguments)
                    if extra_params:
                        params.update(extra_params)
                    try:
                        tool_result = await tool.execute(params, self._adapter)
                        tool_result_text = tool_result.message
                    except Exception as e:
                        tool_result_text = f"Tool error: {e}"

                result.tools_used.append(tc.name)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result_text,
                })

        # Hit max iterations
        final = await self._groq.chat_with_tools(system_prompt, messages, [])
        result.message = final.message or "I got a bit lost. Can you try again?"
        result.capped = True
        return result
```

---

### Task 8: Wire Tier 3 into Reasoner

**Files:**
- Modify: `jarvis-runtime/assistant_core/reasoner.py`

- [ ] **Step 1: Add cloud brain to Reasoner.__init__**

Add parameters for groq provider, agent loop, behavioral query, and vector client. Update the constructor:

```python
    def __init__(
        self,
        personality: Personality,
        llm: OllamaProvider,
        preferences: PreferencesMemory,
        session_store: SessionStore,
        groq: GroqProvider | None = None,
        agent_loop: AgentLoop | None = None,
        behavioral_query=None,
        vector_client=None,
    ) -> None:
        self._personality = personality
        self._llm = llm
        self._preferences = preferences
        self._session_store = session_store
        self._groq = groq
        self._agent_loop = agent_loop
        self._behavioral_query = behavioral_query
        self._vector_client = vector_client
        self._conversations: dict[str, list[dict[str, str]]] = {}
        self._session_tools: dict[str, list[str]] = {}
        self._cached_prompts: dict[str, str] = {}
        self._cached_fast_prompt: str | None = None
        self._media_playing = False
```

- [ ] **Step 2: Replace Tier 3 deep model path with Groq agent loop**

In the `process` method, replace the `use_deep` code path. When `use_deep` is True and Groq is healthy, use the agent loop. Otherwise fall back to local Ollama:

```python
        if use_deep:
            if self._groq and self._groq.healthy and self._agent_loop:
                # Tier 3: Cloud agent loop
                cloud_prompt = await self._personality.build_cloud_prompt(
                    tool_metas, resolved,
                    behavioral_query=self._behavioral_query,
                    vector_client=self._vector_client,
                )
                history.append({"role": "user", "content": resolved})
                agent_result = await self._agent_loop.run(
                    system_prompt=cloud_prompt,
                    conversation_history=history,
                    tool_metas=tool_metas,
                )
                for msg in agent_result.interim_messages:
                    # These will be spoken by the daemon
                    pass
                history.append({"role": "assistant", "content": agent_result.message})
                self._session_tools.setdefault(session_id, []).extend(agent_result.tools_used)
                tier = 3
                logger.info("Tier 3 (groq): tools=%s msg='%s'", agent_result.tools_used, agent_result.message[:80])
                return ReasonerResponse(
                    action=ReasonerAction.SPEAK,
                    message=agent_result.message,
                    tier=3,
                )
            else:
                # Fallback: local deep model
                if session_id not in self._cached_prompts:
                    self._cached_prompts[session_id] = await self._personality.build_system_prompt(
                        tool_metas, recent_commands,
                    )
                system_prompt = self._cached_prompts[session_id]
                model = self._llm.deep_model
                tier = 3
```

- [ ] **Step 3: Make interim messages available for speaking**

Add `interim_messages` to `ReasonerResponse`:

```python
@dataclass
class ReasonerResponse:
    action: ReasonerAction
    tool_name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    confidence: float = 0.0
    raw: str = ""
    tier: int = 0
    interim_messages: list[str] = field(default_factory=list)
```

Update the Groq path to pass interim messages:

```python
                return ReasonerResponse(
                    action=ReasonerAction.SPEAK,
                    message=agent_result.message,
                    tier=3,
                    interim_messages=agent_result.interim_messages,
                )
```

---

### Task 9: Voice — Edge TTS Streaming

**Files:**
- Modify: `jarvis-runtime/voice_service/piper_engine.py`

- [ ] **Step 1: Rewrite piper_engine.py with Edge TTS streaming**

```python
"""Voice engine — streaming Edge TTS (primary), Piper (secondary), say (fallback).

Edge TTS streams audio chunks to mpv for near-instant playback.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path

from voice_service.base import VoiceEngine

logger = logging.getLogger(__name__)

_POST_SPEECH_BUFFER = 0.2
_EDGE_VOICE = "en-US-AriaNeural"
_SAY_VOICE = "Samantha"
_SAY_RATE = "185"


class PiperVoiceEngine(VoiceEngine):
    def __init__(self, model: str = "en_US-amy-medium") -> None:
        self._model = model
        self._piper_path: str | None = None
        self._model_path: str | None = None
        self._mpv_path: str | None = None
        self._engine: str = "say"

    async def initialize(self) -> bool:
        self._mpv_path = shutil.which("mpv")

        # Check Edge TTS + mpv first (streaming)
        if self._mpv_path:
            try:
                import edge_tts  # noqa: F401
                self._engine = "edge_stream"
                logger.info("Voice engine: Edge TTS streaming (voice=%s)", _EDGE_VOICE)
                return True
            except ImportError:
                pass

        # Try Piper
        self._piper_path = shutil.which("piper")
        if self._piper_path:
            model_dir = Path("~/.local/share/piper-voices").expanduser()
            model_file = model_dir / f"{self._model}.onnx"
            if model_file.exists():
                self._model_path = str(model_file)
                self._engine = "piper"
                logger.info("Voice engine: Piper (model=%s)", self._model)
                return True

        # Edge TTS non-streaming (no mpv, use afplay)
        try:
            import edge_tts  # noqa: F401
            self._engine = "edge_download"
            logger.info("Voice engine: Edge TTS download (voice=%s, install mpv for streaming)", _EDGE_VOICE)
            return True
        except ImportError:
            pass

        self._engine = "say"
        logger.info("Voice engine: macOS say (fallback)")
        return True

    async def speak(self, text: str) -> None:
        if not text:
            return

        if self._mute_mic:
            self._mute_mic()

        try:
            if self._engine == "edge_stream":
                await self._speak_edge_stream(text)
            elif self._engine == "piper":
                await self._speak_piper(text)
            elif self._engine == "edge_download":
                await self._speak_edge_download(text)
            else:
                await self._speak_macos_say(text)

            await asyncio.sleep(_POST_SPEECH_BUFFER)
        except Exception:
            logger.exception("Speech failed, falling back to say")
            try:
                await self._speak_macos_say(text)
            except Exception:
                pass
        finally:
            if self._unmute_mic:
                self._unmute_mic()

    async def _speak_edge_stream(self, text: str) -> None:
        import edge_tts

        proc = await asyncio.create_subprocess_exec(
            self._mpv_path, "--no-video", "--no-terminal", "--really-quiet", "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            communicate = edge_tts.Communicate(text, _EDGE_VOICE)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio" and proc.stdin:
                    proc.stdin.write(chunk["data"])
                    await proc.stdin.drain()
        except Exception:
            logger.warning("Edge TTS stream error, falling back to say")
            if proc.stdin:
                proc.stdin.close()
            await proc.wait()
            await self._speak_macos_say(text)
            return

        if proc.stdin:
            proc.stdin.close()
        await proc.wait()

    async def _speak_edge_download(self, text: str) -> None:
        import edge_tts

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            mp3_path = tmp.name
        try:
            communicate = edge_tts.Communicate(text, _EDGE_VOICE)
            await communicate.save(mp3_path)
            play_proc = await asyncio.create_subprocess_exec(
                "afplay", mp3_path,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            await play_proc.communicate()
        except Exception:
            logger.warning("Edge TTS download failed, falling back to say")
            await self._speak_macos_say(text)
        finally:
            Path(mp3_path).unlink(missing_ok=True)

    async def _speak_piper(self, text: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            proc = await asyncio.create_subprocess_exec(
                self._piper_path, "--model", self._model_path, "--output_file", wav_path,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate(input=text.encode())
            if proc.returncode != 0:
                await self._speak_macos_say(text)
                return
            play_proc = await asyncio.create_subprocess_exec(
                "afplay", wav_path,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            await play_proc.communicate()
        finally:
            Path(wav_path).unlink(missing_ok=True)

    @staticmethod
    async def _speak_macos_say(text: str) -> None:
        escaped = text.replace('"', '\\"')
        proc = await asyncio.create_subprocess_exec(
            "say", "-v", _SAY_VOICE, "-r", _SAY_RATE, escaped,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
```

---

### Task 10: Proactive Engine

**Files:**
- Create: `jarvis-runtime/assistant_core/proactive.py`

- [ ] **Step 1: Create proactive.py**

```python
"""Proactive engine — checks triggers every 60s, fires suggestions or auto-actions."""

from __future__ import annotations

import asyncio
import datetime
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from context_service.detector import ContextDetector
    from memory.behavioral.query import BehavioralQuery
    from tools.tasks.todo_store import TodoStore

logger = logging.getLogger(__name__)


@dataclass
class ProactiveSuggestion:
    trigger: str
    message: str
    auto_act: bool = False
    tool_name: str | None = None
    tool_params: dict | None = None


class ProactiveEngine:
    def __init__(
        self,
        context_detector: ContextDetector,
        todo_store: TodoStore | None = None,
        behavioral_query: BehavioralQuery | None = None,
        check_interval: int = 60,
    ) -> None:
        self._context = context_detector
        self._todo_store = todo_store
        self._behavioral = behavioral_query
        self._interval = check_interval
        self._cooldowns: dict[str, float] = {}
        self._morning_done_today: str = ""
        self._last_app: str = ""
        self._last_app_since: float = 0

    def _cooled_down(self, trigger: str, cooldown_seconds: float) -> bool:
        now = time.time()
        last = self._cooldowns.get(trigger, 0)
        if now - last < cooldown_seconds:
            return False
        self._cooldowns[trigger] = now
        return True

    async def check(self) -> list[ProactiveSuggestion]:
        suggestions = []
        now = datetime.datetime.now()
        today = now.date().isoformat()

        ctx = await self._context.get_context()

        # Track app duration
        if ctx.active_app != self._last_app:
            self._last_app = ctx.active_app
            self._last_app_since = time.time()

        # Morning briefing (once per day, before 10am)
        if now.hour < 10 and self._morning_done_today != today:
            if self._cooled_down("morning", 86400):
                self._morning_done_today = today
                msg = await self._build_morning_briefing(ctx, now)
                if msg:
                    suggestions.append(ProactiveSuggestion(
                        trigger="morning_briefing", message=msg, auto_act=True,
                    ))

        # TODO reminders (due today)
        if self._todo_store and self._cooled_down("todo_reminder", 3600):
            due = await self._todo_store.list_due_today()
            if due:
                titles = ", ".join(t["title"] for t in due[:3])
                suggestions.append(ProactiveSuggestion(
                    trigger="todo_reminder",
                    message=f"Reminder: you have pending tasks today — {titles}.",
                    auto_act=True,
                ))

        # Long focus (same app > 2 hours)
        app_duration = time.time() - self._last_app_since
        if app_duration > 7200 and self._cooled_down("focus_break", 7200):
            suggestions.append(ProactiveSuggestion(
                trigger="focus_break",
                message=f"You've been in {ctx.active_app} for a while. Want some music or a break?",
                auto_act=False,
            ))

        # Evening wind-down
        if now.hour >= 21 and self._cooled_down("evening", 86400):
            suggestions.append(ProactiveSuggestion(
                trigger="evening",
                message="It's getting late. Want me to put on some chill music?",
                auto_act=False,
            ))

        return suggestions

    async def _build_morning_briefing(self, ctx, now: datetime.datetime) -> str:
        parts = [f"Morning, Tony."]

        if self._todo_store:
            due = await self._todo_store.list_due_today()
            if due:
                parts.append(f"You have {len(due)} task{'s' if len(due) > 1 else ''} due today.")

        if ctx.active_app:
            parts.append(f"Looks like you're starting in {ctx.active_app}.")

        return " ".join(parts)

    async def run_loop(self, stop_event: asyncio.Event, speak_callback) -> None:
        while not stop_event.is_set():
            try:
                suggestions = await self.check()
                for s in suggestions:
                    logger.info("Proactive [%s]: %s (auto=%s)", s.trigger, s.message[:60], s.auto_act)
                    if speak_callback:
                        await speak_callback(s.message)
            except Exception:
                logger.exception("Proactive check failed")

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self._interval)
                break
            except asyncio.TimeoutError:
                pass
```

---

### Task 11: Wire Everything into Daemon

**Files:**
- Modify: `jarvis-runtime/runtime/daemon.py`

- [ ] **Step 1: Add imports at top of daemon.py**

```python
from llm_router.groq_provider import GroqProvider
from assistant_core.agent_loop import AgentLoop
from assistant_core.proactive import ProactiveEngine
from tools.tasks.todo_store import TodoStore
```

- [ ] **Step 2: Add new instance variables in __init__**

```python
        self._groq: GroqProvider | None = None
        self._agent_loop: AgentLoop | None = None
        self._todo_store: TodoStore | None = None
        self._proactive: ProactiveEngine | None = None
        self._proactive_stop: asyncio.Event | None = None
```

- [ ] **Step 3: Add _init_groq method after _init_llm**

```python
    async def _init_groq(self) -> None:
        self._groq = GroqProvider(
            api_key_env=self._config.groq.api_key_env,
            model=self._config.groq.model,
            max_tokens=self._config.groq.max_tokens,
            temperature=self._config.groq.temperature,
        )
        if await self._groq.initialize():
            self._health.mark("groq", HealthStatus.HEALTHY)
        else:
            self._health.mark("groq", HealthStatus.DOWN, "No API key or connection failed")
            self._groq = None
```

- [ ] **Step 4: Add _init_todos method**

```python
    async def _init_todos(self) -> None:
        self._todo_store = TodoStore()
        if await self._todo_store.initialize():
            self._health.mark("todos", HealthStatus.HEALTHY)
        else:
            self._health.mark("todos", HealthStatus.DOWN)
            self._todo_store = None
```

- [ ] **Step 5: Update _init_brain to pass Groq + memory to Reasoner**

```python
    async def _init_brain(self) -> None:
        self._personality = Personality(
            self._identity, self._preferences,
            self._session_store, self._context_detector,
        )

        agent_loop = None
        if self._groq and self._groq.healthy:
            agent_loop = AgentLoop(self._groq, self._tool_registry, self._adapter)
            self._agent_loop = agent_loop

        self._reasoner = Reasoner(
            self._personality, self._llm,
            self._preferences, self._session_store,
            groq=self._groq,
            agent_loop=agent_loop,
            behavioral_query=self._behavioral_query,
            vector_client=self._vector_client,
        )
        self._health.mark("brain", HealthStatus.HEALTHY)
```

- [ ] **Step 6: Add _init_proactive method**

```python
    async def _init_proactive(self) -> None:
        if not self._config.proactive.enabled:
            return
        self._proactive = ProactiveEngine(
            context_detector=self._context_detector,
            todo_store=self._todo_store,
            behavioral_query=self._behavioral_query,
            check_interval=self._config.proactive.check_interval,
        )
        self._proactive_stop = asyncio.Event()
        asyncio.ensure_future(self._proactive.run_loop(self._proactive_stop, self._speak))
        logger.info("Proactive engine started (interval=%ds)", self._config.proactive.check_interval)
```

- [ ] **Step 7: Update start() method to call new init methods**

Add after `_init_llm`:

```python
        await self._init_groq()
        await self._init_todos()
```

Add after `_init_fsm`:

```python
        await self._init_proactive()
```

- [ ] **Step 8: Update _on_voice_transcript to speak interim messages**

In the `SPEAK` response handler, check for interim messages:

```python
        elif response.action == ReasonerAction.SPEAK:
            for interim in response.interim_messages:
                if interim:
                    await self._speak(interim)
            if response.message:
                await self._speak(response.message)
```

- [ ] **Step 9: Pass todo_store to tool params**

In `_dispatch_tool`, add:

```python
        params["_todo_store"] = self._todo_store
```

- [ ] **Step 10: Update stop() to clean up new resources**

```python
        if self._proactive_stop:
            self._proactive_stop.set()
        if self._todo_store:
            await self._todo_store.close()
```

---

### Task 12: Get Groq API Key

- [ ] **Step 1: Sign up at console.groq.com and get free API key**

Go to https://console.groq.com, create account, get API key.

- [ ] **Step 2: Set environment variable**

```bash
export GROQ_API_KEY="gsk_your_key_here"
```

Add to `~/.zshrc` for persistence:

```bash
echo 'export GROQ_API_KEY="gsk_your_key_here"' >> ~/.zshrc
```

---

### Task 13: Smoke Test

- [ ] **Step 1: Run Sam**

```bash
cd jarvis-runtime && .venv/bin/python -m runtime.daemon
```

Verify in logs:
- `Groq connected: model=llama-3.3-70b-versatile`
- `Voice engine: Edge TTS streaming`
- `Proactive engine started`

- [ ] **Step 2: Test Tier 1 (fast-path)**

Say: "volume up" — should execute instantly, no LLM call.

- [ ] **Step 3: Test Tier 2 (local 3b)**

Say: "play lofi beats" — should route to youtube_search via 3b model.

- [ ] **Step 4: Test Tier 3 (Groq conversation)**

Say: "What's the weather in Delhi?" — should escalate to Groq, call weather tool, speak result.

- [ ] **Step 5: Test multi-turn conversation**

Say: "I'm planning to go to Mumbai" — should get conversational response.
Then: "Check trains for me" — should call web_search and report results.

- [ ] **Step 6: Test TODO**

Say: "Remind me to call mom tomorrow" — should add TODO.
Say: "What's on my list?" — should read back TODOs.

- [ ] **Step 7: Test Edge TTS streaming**

Verify voice output starts within ~1 second of response, not 10+ seconds.
