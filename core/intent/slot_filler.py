"""Slot filler — fills missing tool parameters from context."""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SlotFillingResult:
    params: dict[str, Any]
    unfilled: list[str] = field(default_factory=list)


class SlotFiller:
    async def fill(self, params: dict[str, Any], parameters_schema: dict[str, Any], recent_commands: list[dict], time_patterns: dict[str, list[str]]) -> SlotFillingResult:
        required = parameters_schema.get("required", [])
        filled = dict(params)
        unfilled = []
        for param_name in required:
            if param_name in filled and filled[param_name]:
                continue
            value = self._fill_from_recent(param_name, recent_commands)
            if value is not None:
                filled[param_name] = value
                continue
            unfilled.append(param_name)
        return SlotFillingResult(params=filled, unfilled=unfilled)

    @staticmethod
    def _fill_from_recent(param_name: str, recent_commands: list[dict]) -> Any | None:
        for cmd in reversed(recent_commands):
            cmd_params = cmd.get("params", {})
            if param_name in cmd_params:
                return cmd_params[param_name]
        return None
