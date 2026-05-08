"""Session store — persists session summaries across restarts."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path("~/.kairo/sessions.db").expanduser()


class SessionStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path).expanduser() if db_path else _DEFAULT_DB_PATH
        self._db = None
        self._healthy = False

    @property
    def healthy(self) -> bool:
        return self._healthy

    async def initialize(self) -> bool:
        try:
            import aiosqlite
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = await aiosqlite.connect(str(self._db_path))
            await self._db.execute(
                """CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    started_at REAL NOT NULL,
                    ended_at REAL,
                    summary TEXT DEFAULT '',
                    tools_used TEXT DEFAULT '[]',
                    context_snapshot TEXT DEFAULT '{}'
                )"""
            )
            await self._db.commit()
            self._healthy = True
            logger.info("Session store initialized at %s", self._db_path)
            return True
        except Exception:
            logger.exception("Failed to initialize session store")
            return False

    async def start_session(self, session_id: str, context: dict[str, Any] | None = None) -> None:
        if not self._healthy or self._db is None:
            return
        try:
            await self._db.execute(
                "INSERT OR REPLACE INTO sessions (session_id, started_at, context_snapshot) VALUES (?, ?, ?)",
                (session_id, time.time(), json.dumps(context or {})),
            )
            await self._db.commit()
        except Exception:
            logger.exception("Failed to start session record")

    async def end_session(self, session_id: str, summary: str, tools_used: list[str]) -> None:
        if not self._healthy or self._db is None:
            return
        try:
            await self._db.execute(
                "UPDATE sessions SET ended_at = ?, summary = ?, tools_used = ? WHERE session_id = ?",
                (time.time(), summary, json.dumps(tools_used), session_id),
            )
            await self._db.commit()
        except Exception:
            logger.exception("Failed to end session record")

    async def get_last_session(self) -> dict[str, Any] | None:
        if not self._healthy or self._db is None:
            return None
        try:
            cursor = await self._db.execute(
                "SELECT session_id, started_at, ended_at, summary, tools_used, context_snapshot "
                "FROM sessions WHERE summary != '' ORDER BY started_at DESC LIMIT 1"
            )
            row = await cursor.fetchone()
            if not row:
                return None
            return {
                "session_id": row[0],
                "started_at": row[1],
                "ended_at": row[2],
                "summary": row[3],
                "tools_used": json.loads(row[4]) if row[4] else [],
                "context_snapshot": json.loads(row[5]) if row[5] else {},
            }
        except Exception:
            logger.exception("Failed to get last session")
            return None

    async def get_recent_sessions(self, limit: int = 5) -> list[dict[str, Any]]:
        if not self._healthy or self._db is None:
            return []
        try:
            cursor = await self._db.execute(
                "SELECT session_id, started_at, ended_at, summary, tools_used "
                "FROM sessions WHERE summary != '' ORDER BY started_at DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
            return [
                {
                    "session_id": r[0],
                    "started_at": r[1],
                    "ended_at": r[2],
                    "summary": r[3],
                    "tools_used": json.loads(r[4]) if r[4] else [],
                }
                for r in rows
            ]
        except Exception:
            logger.exception("Failed to get recent sessions")
            return []

    async def close(self) -> None:
        if self._db:
            await self._db.close()
