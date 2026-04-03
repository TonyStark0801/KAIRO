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
    def __init__(self, adapter: PlatformAdapter, behavioral_query: BehavioralQuery, session_cache: SessionCache, process_manager: ProcessManager | None = None) -> None:
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
        if 5 <= hour < 12: time_greeting = "Good morning"
        elif 12 <= hour < 17: time_greeting = "Good afternoon"
        elif 17 <= hour < 21: time_greeting = "Good evening"
        else: time_greeting = "Hey"
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
        if len(greeting.split()) > 40:
            greeting = f"{time_greeting}! Happy {day_name}.{tool_hint} Ready."
        if len(greeting.split()) > 40:
            greeting = f"{time_greeting}! Ready."
        return greeting
