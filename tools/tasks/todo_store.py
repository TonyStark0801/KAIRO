"""SQLite-backed TODO storage for Kairo (macOS desktop assistant)."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path("~/.kairo/todos.db").expanduser()

_SCHEMA = """CREATE TABLE IF NOT EXISTS todos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    due_date TEXT,
    due_time TEXT,
    status TEXT DEFAULT 'pending',
    source TEXT DEFAULT 'voice',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    context TEXT
);"""


def _row_to_dict(row: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "id": row[0],
        "title": row[1],
        "due_date": row[2],
        "due_time": row[3],
        "status": row[4],
        "source": row[5],
        "created_at": row[6],
        "completed_at": row[7],
        "context": row[8],
    }


class TodoStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path).expanduser() if db_path else _DEFAULT_DB_PATH
        self._db: aiosqlite.Connection | None = None
        self._healthy = False

    @property
    def healthy(self) -> bool:
        return self._healthy

    async def initialize(self) -> bool:
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = await aiosqlite.connect(str(self._db_path))
            await self._db.execute(_SCHEMA)
            await self._db.commit()
            self._healthy = True
            logger.info("Todo store initialized at %s", self._db_path)
            return True
        except Exception:
            logger.exception("Failed to initialize todo store")
            self._db = None
            self._healthy = False
            return False

    async def add(
        self,
        title: str,
        due_date: str | None = None,
        due_time: str | None = None,
        context: str = "",
    ) -> int:
        if not self._healthy or self._db is None:
            raise RuntimeError("Todo store is not initialized")
        cursor = await self._db.execute(
            """INSERT INTO todos (title, due_date, due_time, context)
               VALUES (?, ?, ?, ?)""",
            (title, due_date, due_time, context or ""),
        )
        await self._db.commit()
        return int(cursor.lastrowid)

    async def list_todos(self, status: str = "pending") -> list[dict[str, Any]]:
        if not self._healthy or self._db is None:
            return []
        try:
            cursor = await self._db.execute(
                "SELECT id, title, due_date, due_time, status, source, "
                "created_at, completed_at, context FROM todos WHERE status = ? "
                "ORDER BY id ASC",
                (status,),
            )
            rows = await cursor.fetchall()
            return [_row_to_dict(r) for r in rows]
        except Exception:
            logger.exception("Failed to list todos")
            return []

    async def complete(self, todo_id: int) -> bool:
        if not self._healthy or self._db is None:
            return False
        try:
            cursor = await self._db.execute(
                """UPDATE todos SET status = 'completed', completed_at = CURRENT_TIMESTAMP
                   WHERE id = ? AND status != 'completed'""",
                (todo_id,),
            )
            await self._db.commit()
            return cursor.rowcount > 0
        except Exception:
            logger.exception("Failed to complete todo %s", todo_id)
            return False

    async def delete(self, todo_id: int) -> bool:
        if not self._healthy or self._db is None:
            return False
        try:
            cursor = await self._db.execute("DELETE FROM todos WHERE id = ?", (todo_id,))
            await self._db.commit()
            return cursor.rowcount > 0
        except Exception:
            logger.exception("Failed to delete todo %s", todo_id)
            return False

    async def get_due_today(self) -> list[dict[str, Any]]:
        if not self._healthy or self._db is None:
            return []
        today = date.today().isoformat()
        try:
            cursor = await self._db.execute(
                "SELECT id, title, due_date, due_time, status, source, "
                "created_at, completed_at, context FROM todos "
                "WHERE due_date = ? ORDER BY id ASC",
                (today,),
            )
            rows = await cursor.fetchall()
            return [_row_to_dict(r) for r in rows]
        except Exception:
            logger.exception("Failed to get todos due today")
            return []

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
        self._healthy = False
