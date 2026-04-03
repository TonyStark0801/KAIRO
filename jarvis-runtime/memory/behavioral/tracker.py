"""Behavioral tracker — records tool executions in SQLite."""
from __future__ import annotations
import json
import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config.models import MemoryConfig
    from runtime.event_bus import EventBus, ToolExecutionEvent

logger = logging.getLogger(__name__)


class BehavioralTracker:
    def __init__(self, config: MemoryConfig, event_bus: EventBus) -> None:
        self._config = config
        self._bus = event_bus
        self._db = None
        self._healthy = False

    @property
    def healthy(self) -> bool:
        return self._healthy

    async def initialize(self) -> bool:
        try:
            import aiosqlite
            from pathlib import Path
            db_path = Path(self._config.behavioral_db)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = await aiosqlite.connect(str(db_path))
            await self._db.execute(
                """CREATE TABLE IF NOT EXISTS commands (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    tool_name TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    success INTEGER NOT NULL
                )"""
            )
            await self._db.commit()
            self._healthy = True
            return True
        except Exception:
            logger.exception("Failed to initialize behavioral store")
            return False

    async def record(self, event: ToolExecutionEvent) -> None:
        if not self._healthy or self._db is None:
            return
        try:
            params_json = json.dumps(event.result.data) if event.result else "{}"
            await self._db.execute(
                "INSERT INTO commands (timestamp, tool_name, params_json, session_id, success) VALUES (?, ?, ?, ?, ?)",
                (time.time(), event.tool_name, params_json, event.session_id, int(event.success)),
            )
            await self._db.commit()
        except Exception:
            logger.exception("Failed to record command")

    async def on_tool_execution(self, event: ToolExecutionEvent) -> None:
        await self.record(event)

    async def get_frequent_tools(self, limit: int = 5) -> list[dict]:
        if not self._healthy or self._db is None:
            return []
        try:
            cursor = await self._db.execute(
                "SELECT tool_name, COUNT(*) as cnt FROM commands WHERE success = 1 GROUP BY tool_name ORDER BY cnt DESC LIMIT ?", (limit,))
            rows = await cursor.fetchall()
            return [{"tool_name": r[0], "count": r[1]} for r in rows]
        except Exception:
            logger.exception("Failed to query frequent tools")
            return []

    async def get_time_of_day_pattern(self) -> dict[str, list[str]]:
        if not self._healthy or self._db is None:
            return {}
        try:
            cursor = await self._db.execute("SELECT tool_name, timestamp FROM commands WHERE success = 1 ORDER BY timestamp DESC LIMIT 100")
            rows = await cursor.fetchall()
            import datetime
            patterns: dict[str, list[str]] = {"morning": [], "afternoon": [], "evening": [], "night": []}
            for tool_name, ts in rows:
                hour = datetime.datetime.fromtimestamp(ts).hour
                if 5 <= hour < 12: period = "morning"
                elif 12 <= hour < 17: period = "afternoon"
                elif 17 <= hour < 21: period = "evening"
                else: period = "night"
                if tool_name not in patterns[period]:
                    patterns[period].append(tool_name)
            return patterns
        except Exception:
            logger.exception("Failed to query time patterns")
            return {}

    async def close(self) -> None:
        if self._db:
            await self._db.close()
