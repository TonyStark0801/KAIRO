"""Subsystem health tracker — tracks init status with retry logic."""
from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = auto()
    DEGRADED = auto()
    DOWN = auto()


@dataclass
class SubsystemHealth:
    name: str
    status: HealthStatus
    message: str = ""


class HealthTracker:
    def __init__(self) -> None:
        self._subsystems: dict[str, SubsystemHealth] = {}

    def get_status(self) -> dict[str, SubsystemHealth]:
        return dict(self._subsystems)

    def mark(self, name: str, status: HealthStatus, message: str = "") -> None:
        self._subsystems[name] = SubsystemHealth(name=name, status=status, message=message)
        level = logging.INFO if status == HealthStatus.HEALTHY else logging.WARNING
        logger.log(level, "Subsystem %s: %s %s", name, status.name, message)

    async def init_with_retry(self, name: str, init_fn: Callable[[], Awaitable[bool]], max_retries: int = 3) -> bool:
        for attempt in range(1, max_retries + 1):
            try:
                success = await init_fn()
                if success:
                    self.mark(name, HealthStatus.HEALTHY)
                    return True
            except Exception as e:
                logger.warning("Subsystem %s init attempt %d/%d failed: %s", name, attempt, max_retries, e)
            if attempt < max_retries:
                backoff = 2 ** (attempt - 1)
                await asyncio.sleep(backoff)
        self.mark(name, HealthStatus.DOWN, f"Failed after {max_retries} attempts")
        return False
