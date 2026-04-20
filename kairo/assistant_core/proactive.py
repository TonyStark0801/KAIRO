"""Proactive engine — timer-based suggestions and auto-actions.

Runs every N seconds, checks conditions, yields suggestions for the daemon to act on.
No LLM calls for detection — only for message generation if needed.
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ProactiveSuggestion:
    trigger_type: str
    message: str
    auto_act: bool = False
    tool_name: str | None = None
    tool_params: dict[str, Any] = field(default_factory=dict)


class ProactiveEngine:
    def __init__(
        self,
        config=None,
        todo_store=None,
        context_detector=None,
        session_store=None,
    ) -> None:
        self._config = config
        self._todo_store = todo_store
        self._context = context_detector
        self._session_store = session_store

        self._cooldowns: dict[str, datetime.datetime] = {}
        self._morning_done_today: str = ""
        self._focus_app_start: datetime.datetime | None = None
        self._focus_app_name: str = ""

    def _on_cooldown(self, trigger: str, minutes: int = 60) -> bool:
        """Check if a trigger is on cooldown."""
        last = self._cooldowns.get(trigger)
        if last is None:
            return False
        return (datetime.datetime.now() - last).total_seconds() < minutes * 60

    def _set_cooldown(self, trigger: str) -> None:
        self._cooldowns[trigger] = datetime.datetime.now()

    async def check(self) -> list[ProactiveSuggestion]:
        """Run all checks. Returns list of suggestions (usually 0-1)."""
        suggestions: list[ProactiveSuggestion] = []
        now = datetime.datetime.now()

        if self._config and not self._config.enabled:
            return suggestions

        # Morning briefing
        if self._config and self._config.morning_briefing:
            s = await self._check_morning_briefing(now)
            if s:
                suggestions.append(s)

        # TODO reminders
        if self._config and self._config.todo_reminders and self._todo_store:
            s = await self._check_todo_reminders(now)
            if s:
                suggestions.append(s)

        # Focus suggestion
        if self._config and self._config.focus_suggestions and self._context:
            s = await self._check_focus(now)
            if s:
                suggestions.append(s)

        # Evening wind-down
        s = await self._check_evening(now)
        if s:
            suggestions.append(s)

        return suggestions

    async def _check_morning_briefing(self, now: datetime.datetime) -> ProactiveSuggestion | None:
        today = now.strftime("%Y-%m-%d")
        if self._morning_done_today == today:
            return None
        if not (6 <= now.hour <= 10):
            return None

        self._morning_done_today = today

        parts = ["Good morning, Tony!"]

        # Add TODO count
        if self._todo_store:
            try:
                todos = await self._todo_store.get_due_today()
                if todos:
                    parts.append(f"You have {len(todos)} thing{'s' if len(todos) > 1 else ''} due today.")
                    for t in todos[:3]:
                        parts.append(f"  - {t['title']}")
            except Exception:
                pass

        # Add last session context
        if self._session_store:
            try:
                last = await self._session_store.get_last_session()
                if last and last.get("summary"):
                    parts.append(f"Last time: {last['summary']}")
            except Exception:
                pass

        return ProactiveSuggestion(
            trigger_type="morning_briefing",
            message=" ".join(parts),
            auto_act=True,
        )

    async def _check_todo_reminders(self, now: datetime.datetime) -> ProactiveSuggestion | None:
        if self._on_cooldown("todo_reminder", minutes=60):
            return None

        try:
            todos = await self._todo_store.get_due_today()
            pending = [t for t in todos if t.get("status") == "pending"]
            if not pending:
                return None

            self._set_cooldown("todo_reminder")
            if len(pending) == 1:
                msg = f"Reminder: you still have '{pending[0]['title']}' due today."
            else:
                msg = f"Reminder: you have {len(pending)} things due today."

            return ProactiveSuggestion(
                trigger_type="todo_reminder",
                message=msg,
                auto_act=True,
            )
        except Exception:
            return None

    async def _check_focus(self, now: datetime.datetime) -> ProactiveSuggestion | None:
        if self._on_cooldown("focus", minutes=120):
            return None

        try:
            ctx = await self._context.get_context()
            current_app = ctx.active_app

            if current_app == self._focus_app_name and self._focus_app_start:
                elapsed = (now - self._focus_app_start).total_seconds()
                if elapsed > 7200:  # 2 hours
                    self._set_cooldown("focus")
                    return ProactiveSuggestion(
                        trigger_type="focus",
                        message=f"You've been in {current_app} for over 2 hours. Want some music or a break?",
                        auto_act=False,
                    )
            else:
                self._focus_app_name = current_app
                self._focus_app_start = now

        except Exception:
            pass
        return None

    async def _check_evening(self, now: datetime.datetime) -> ProactiveSuggestion | None:
        if now.hour < 21:
            return None
        if self._on_cooldown("evening", minutes=180):
            return None

        self._set_cooldown("evening")
        return ProactiveSuggestion(
            trigger_type="evening",
            message="It's getting late. Want me to put on some chill music?",
            auto_act=False,
        )
