"""Preferences memory — SQLite-backed user habits, aliases, and patterns."""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path("~/.kairo/preferences.db").expanduser()


class PreferencesMemory:
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
                """CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )"""
            )
            await self._db.execute(
                """CREATE TABLE IF NOT EXISTS aliases (
                    alias TEXT PRIMARY KEY,
                    expansion TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )"""
            )
            await self._db.commit()
            self._healthy = True
            logger.info("Preferences DB initialized at %s", self._db_path)
            return True
        except Exception:
            logger.exception("Failed to initialize preferences DB")
            return False

    async def get(self, key: str, default: str = "") -> str:
        if not self._healthy or self._db is None:
            return default
        try:
            cursor = await self._db.execute(
                "SELECT value FROM preferences WHERE key = ?", (key,)
            )
            row = await cursor.fetchone()
            return row[0] if row else default
        except Exception:
            logger.exception("Failed to get preference %s", key)
            return default

    async def set(self, key: str, value: str) -> None:
        if not self._healthy or self._db is None:
            return
        try:
            await self._db.execute(
                "INSERT OR REPLACE INTO preferences (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, time.time()),
            )
            await self._db.commit()
        except Exception:
            logger.exception("Failed to set preference %s", key)

    async def get_all(self) -> dict[str, str]:
        if not self._healthy or self._db is None:
            return {}
        try:
            cursor = await self._db.execute("SELECT key, value FROM preferences")
            rows = await cursor.fetchall()
            return {r[0]: r[1] for r in rows}
        except Exception:
            logger.exception("Failed to get all preferences")
            return {}

    async def resolve_alias(self, text: str) -> str:
        if not self._healthy or self._db is None:
            return text
        try:
            cursor = await self._db.execute("SELECT alias, expansion FROM aliases")
            rows = await cursor.fetchall()
            for alias, expansion in rows:
                if alias.lower() in text.lower():
                    text = text.lower().replace(alias.lower(), expansion)
            return text
        except Exception:
            return text

    async def set_alias(self, alias: str, expansion: str) -> None:
        if not self._healthy or self._db is None:
            return
        try:
            await self._db.execute(
                "INSERT OR REPLACE INTO aliases (alias, expansion, updated_at) VALUES (?, ?, ?)",
                (alias.lower(), expansion, time.time()),
            )
            await self._db.commit()
        except Exception:
            logger.exception("Failed to set alias %s", alias)

    async def get_aliases(self) -> dict[str, str]:
        if not self._healthy or self._db is None:
            return {}
        try:
            cursor = await self._db.execute("SELECT alias, expansion FROM aliases")
            rows = await cursor.fetchall()
            return {r[0]: r[1] for r in rows}
        except Exception:
            return {}

    async def close(self) -> None:
        if self._db:
            await self._db.close()
