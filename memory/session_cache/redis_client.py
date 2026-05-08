"""Session cache — Redis with in-memory dict fallback."""
from __future__ import annotations
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class SessionCache:
    def __init__(self, redis_enabled: bool = False, redis_url: str = "redis://localhost:6379/0") -> None:
        self._redis_enabled = redis_enabled
        self._redis_url = redis_url
        self._redis = None
        self._fallback: dict[str, str] = {}
        self._using_fallback = True

    @property
    def healthy(self) -> bool:
        return True

    async def initialize(self) -> bool:
        if not self._redis_enabled:
            logger.info("Redis disabled — using in-memory session cache")
            self._using_fallback = True
            return True
        try:
            import aioredis
            self._redis = await aioredis.from_url(self._redis_url)
            await self._redis.ping()
            self._using_fallback = False
            logger.info("Connected to Redis at %s", self._redis_url)
            return True
        except Exception:
            logger.warning("Redis unavailable — falling back to in-memory cache")
            self._using_fallback = True
            return True

    async def set(self, key: str, value: Any) -> None:
        encoded = json.dumps(value)
        if self._using_fallback:
            self._fallback[key] = encoded
        else:
            try:
                await self._redis.set(key, encoded)
            except Exception:
                logger.warning("Redis set failed, using fallback")
                self._fallback[key] = encoded

    async def get(self, key: str) -> Any | None:
        if self._using_fallback:
            raw = self._fallback.get(key)
        else:
            try:
                raw = await self._redis.get(key)
                if isinstance(raw, bytes):
                    raw = raw.decode()
            except Exception:
                raw = self._fallback.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    async def append_command(self, session_id: str, command: dict) -> None:
        key = f"session:{session_id}:commands"
        commands = await self.get(key) or []
        commands.append(command)
        commands = commands[-10:]
        await self.set(key, commands)

    async def get_recent_commands(self, session_id: str, limit: int = 5) -> list[dict]:
        key = f"session:{session_id}:commands"
        commands = await self.get(key) or []
        return commands[-limit:]

    async def set_session_state(self, session_id: str, state: str) -> None:
        await self.set(f"session:{session_id}:state", state)

    async def set_session_start(self, session_id: str) -> None:
        await self.set(f"session:{session_id}:start", time.time())

    async def close(self) -> None:
        if self._redis and not self._using_fallback:
            await self._redis.close()
