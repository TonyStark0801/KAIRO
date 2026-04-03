"""Wake pipeline — initializes session context on WAKE_PENDING."""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from core.session.session_context import SessionContext
from runtime.event_bus import SessionState, SessionStateChangedEvent

if TYPE_CHECKING:
    from memory.session_cache.redis_client import SessionCache

logger = logging.getLogger(__name__)


class WakePipeline:
    def __init__(self, session_cache: SessionCache) -> None:
        self._cache = session_cache
        self._current_context: SessionContext | None = None

    @property
    def current_context(self) -> SessionContext | None:
        return self._current_context

    async def on_state_changed(self, event: SessionStateChangedEvent) -> None:
        if event.new_state == SessionState.WAKE_PENDING:
            ctx = SessionContext()
            self._current_context = ctx
            await self._cache.set_session_start(ctx.session_id)
            await self._cache.set_session_state(ctx.session_id, "WAKE_PENDING")
            logger.info("Session context created: %s", ctx.session_id)
        elif event.new_state == SessionState.SLEEP:
            self._current_context = None
