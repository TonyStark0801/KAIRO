"""Personality builder — shapes all LLM interactions using identity, context, and memory."""

from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import Environment, FileSystemLoader

if TYPE_CHECKING:
    from context_service.detector import ContextDetector, WorkspaceContext
    from memory_service.identity import IdentityMemory
    from memory_service.preferences import PreferencesMemory
    from memory_service.session_store import SessionStore
    from tools._base import ToolMeta

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "llm_router" / "prompts"


class Personality:
    def __init__(
        self,
        identity: IdentityMemory,
        preferences: PreferencesMemory,
        session_store: SessionStore,
        context_detector: ContextDetector,
    ) -> None:
        self._identity = identity
        self._preferences = preferences
        self._session_store = session_store
        self._context = context_detector
        self._jinja = Environment(
            loader=FileSystemLoader(str(_PROMPTS_DIR)),
            autoescape=False,
        )

    async def build_system_prompt(
        self,
        tool_metas: list[ToolMeta],
        recent_commands: list[str] | None = None,
    ) -> str:
        # Flat name+description list — full parameter schemas are too many tokens
        # for a 3-4b model and cause the JSON-output hallucination / fence-confusion bug.
        tools_list = [{"name": t.name, "description": t.description} for t in tool_metas]

        workspace_ctx = await self._context.get_context()
        workspace_summary = workspace_ctx.natural_summary()

        now = datetime.datetime.now()
        # Full datetime string so the model can answer "what's the date" without guessing.
        current_datetime = now.strftime("%A, %B %d %Y, %I:%M %p")  # e.g. "Tuesday, April 21 2026, 02:08 AM"

        posix_user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
        home_path = str(Path.home())

        template = self._jinja.get_template("system.j2")
        return template.render(
            assistant_name=self._identity.assistant_name,
            owner_name=self._identity.owner_name,
            workspace_context=workspace_summary,
            current_datetime=current_datetime,
            tools_list=tools_list,
            recent_commands=recent_commands or [],
            posix_user=posix_user,
            home_path=home_path,
        )

    async def build_fast_prompt(self, tool_metas: list[ToolMeta]) -> str:
        """Compact prompt for the 3b fast model — ~150 tokens."""
        tool_names = ", ".join(t.name for t in tool_metas)
        template = self._jinja.get_template("fast.j2")
        return template.render(
            assistant_name=self._identity.assistant_name,
            owner_name=self._identity.owner_name,
            tool_names=tool_names,
        )

    async def build_cloud_prompt(
        self,
        tool_metas: list[ToolMeta],
        transcript: str,
        behavioral_query=None,
        vector_client=None,
    ) -> str:
        """Rich system prompt for Tier 3 cloud LLM — includes memory, mood, context."""
        from assistant_core.mood import MoodMode, detect_mood, get_mood_prompt

        workspace_ctx = await self._context.get_context()

        now = datetime.datetime.now()
        hour = now.hour
        if 5 <= hour < 12:
            time_of_day = f"morning ({now.strftime('%I:%M %p')})"
        elif 12 <= hour < 17:
            time_of_day = f"afternoon ({now.strftime('%I:%M %p')})"
        elif 17 <= hour < 21:
            time_of_day = f"evening ({now.strftime('%I:%M %p')})"
        else:
            time_of_day = f"night ({now.strftime('%I:%M %p')})"

        mood = detect_mood(workspace_ctx.active_app, transcript, hour)
        mood_prompt = get_mood_prompt(mood)

        # Gather preferences
        prefs = ""
        try:
            all_prefs = await self._preferences.get_all()
            if all_prefs:
                prefs = ", ".join(f"{k}: {v}" for k, v in all_prefs.items())
        except Exception:
            pass

        # Gather habits from behavioral query
        habits = ""
        if behavioral_query:
            try:
                recent_tools = await behavioral_query.get_recent_tools(5)
                patterns = await behavioral_query.get_time_of_day_pattern()
                parts = []
                if recent_tools:
                    parts.append(f"Frequent tools: {', '.join(recent_tools)}")
                if patterns:
                    for period, tools in patterns.items():
                        if tools:
                            parts.append(f"{period}: uses {', '.join(tools[:3])}")
                habits = ". ".join(parts)
            except Exception:
                pass

        # Gather session summaries
        session_summaries = []
        try:
            recent = await self._session_store.get_recent_sessions(3)
            session_summaries = [s.get("summary", "") for s in recent if s.get("summary")]
        except Exception:
            pass

        # Gather vector memories
        vector_memories = []
        if vector_client:
            try:
                results = await vector_client.search(transcript, n_results=3)
                vector_memories = [r["document"] for r in results if r.get("document")]
            except Exception:
                pass

        tools = [{"name": t.name, "description": t.description} for t in tool_metas]

        template = self._jinja.get_template("cloud.j2")
        return template.render(
            assistant_name=self._identity.assistant_name,
            assistant_full_name=self._identity.assistant_full_name,
            owner_name=self._identity.owner_name,
            mood_mode=mood.value,
            mood_prompt=mood_prompt,
            preferences=prefs,
            habits=habits,
            session_summaries=session_summaries,
            vector_memories=vector_memories,
            active_app=workspace_ctx.active_app,
            working_on=workspace_ctx.summary(),
            time_of_day=time_of_day,
            tools=tools,
        )

    async def generate_greeting(self) -> str:
        """Short spoken greeting for session start — keep it under 6 words."""
        name = self._identity.owner_name

        now = datetime.datetime.now()
        hour = now.hour
        if 5 <= hour < 12:
            return f"Morning, {name}."
        elif 17 <= hour < 21:
            return f"Hey {name}."
        elif hour >= 21 or hour < 5:
            return f"Hey {name}."
        return f"Hey {name}."
