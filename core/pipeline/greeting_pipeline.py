"""Greeting pipeline — silent. No speech on wake, just a notification."""
from __future__ import annotations
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
        self._greeted_sessions: set[str] = set()

    async def on_state_changed(self, event: SessionStateChangedEvent) -> None:
        if event.new_state == SessionState.SLEEP:
            self._greeted_sessions.discard(event.session_id)
            return
        if event.new_state != SessionState.ACTIVE_SESSION:
            return
        if event.old_state != SessionState.WAKE_PENDING:
            return
        if event.session_id in self._greeted_sessions:
            return
        self._greeted_sessions.add(event.session_id)
        try:
            await self._adapter.send_notification("Kairo", "Listening...")
        except Exception:
            logger.exception("Greeting notification failed")
