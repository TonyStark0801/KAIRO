"""Read-only query interface for behavioral data."""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memory.behavioral.tracker import BehavioralTracker


class BehavioralQuery:
    def __init__(self, tracker: BehavioralTracker) -> None:
        self._tracker = tracker

    async def get_recent_tools(self, limit: int = 3) -> list[str]:
        frequent = await self._tracker.get_frequent_tools(limit)
        return [entry["tool_name"] for entry in frequent]

    async def get_time_of_day_pattern(self) -> dict[str, list[str]]:
        return await self._tracker.get_time_of_day_pattern()
