"""Context observer — polls workspace context and emits ContextChangedEvent.

Runs as an asyncio task inside the daemon event loop. Polls
ContextDetector every N seconds and publishes a ContextChangedEvent only
when the active app, window title, or browser URL has actually changed.
This avoids flooding the bus with identical snapshots.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from runtime.event_bus import ActivityType, ContextChangedEvent

if TYPE_CHECKING:
    from context_service.detector import ContextDetector
    from runtime.event_bus import EventBus
    from sensors.observer.activity_classifier import ActivityClassifier

logger = logging.getLogger(__name__)


class ContextObserver:
    """Async polling observer for Phase 2 context awareness."""

    def __init__(
        self,
        event_bus: EventBus,
        detector: ContextDetector,
        classifier: ActivityClassifier,
        poll_interval_sec: float = 0.8,
    ) -> None:
        self._bus = event_bus
        self._detector = detector
        self._classifier = classifier
        self._interval = max(0.2, poll_interval_sec)

        # Track previous state to detect changes
        self._last_app: str = ""
        self._last_window: str = ""
        self._last_url: str = ""
        self._last_activity: ActivityType = ActivityType.UNKNOWN

    async def run(self) -> None:
        """Polling loop — run with asyncio.ensure_future() from the daemon."""
        logger.info("Context observer started (interval=%.1fs)", self._interval)
        while True:
            try:
                ctx = await self._detector.get_context()
                await self._maybe_emit(ctx)
            except asyncio.CancelledError:
                logger.info("Context observer stopped")
                break
            except Exception:
                logger.debug("Context observer poll error", exc_info=True)
            await asyncio.sleep(self._interval)

    async def _maybe_emit(self, ctx) -> None:
        """Emit ContextChangedEvent only when something meaningful changed."""
        changed = (
            ctx.active_app != self._last_app
            or ctx.window_title != self._last_window
            or ctx.browser_url != self._last_url
        )
        if not changed:
            return

        activity = self._classifier.classify(ctx)

        # Log the transition so it's visible in kairo.log without noise
        if ctx.active_app != self._last_app:
            logger.info(
                "Context: %s → %s (activity=%s)",
                self._last_app or "none",
                ctx.active_app,
                activity.name,
            )
        elif ctx.browser_url != self._last_url:
            logger.debug(
                "Browser tab: %s | activity=%s",
                (ctx.browser_tab_title or ctx.browser_url)[:60],
                activity.name,
            )

        self._last_app = ctx.active_app
        self._last_window = ctx.window_title
        self._last_url = ctx.browser_url
        self._last_activity = activity

        event = ContextChangedEvent(
            app=ctx.active_app,
            window_title=ctx.window_title,
            browser_url=ctx.browser_url,
            browser_tab_title=ctx.browser_tab_title,
            activity_type=activity,
            timestamp=ctx.timestamp,
        )
        await self._bus.publish(event)

    @property
    def current_activity(self) -> ActivityType:
        return self._last_activity
