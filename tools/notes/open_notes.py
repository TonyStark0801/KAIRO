"""Tool: Open Apple Notes."""
from __future__ import annotations
from typing import Any
from tools._base import BaseTool, ToolResult


class OpenNotesTool(BaseTool):
    @property
    def name(self) -> str:
        return "open_notes"

    @property
    def description(self) -> str:
        return "Opens Apple Notes application"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, params: dict[str, Any], adapter) -> ToolResult:
        try:
            await adapter.open_application("Notes")
            return ToolResult(success=True, message="Opened Notes")
        except Exception as e:
            return ToolResult(success=False, message=str(e))
