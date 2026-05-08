"""Tool: Switch to a specific IntelliJ window."""
from __future__ import annotations
from typing import Any
from tools._base import BaseTool, ToolResult


class SwitchWindowTool(BaseTool):
    @property
    def name(self) -> str:
        return "switch_intellij_window"

    @property
    def description(self) -> str:
        return "Switches to a specific IntelliJ IDEA window by title pattern"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"title_pattern": {"type": "string", "description": "Window title to match"}},
            "required": ["title_pattern"],
        }

    async def execute(self, params: dict[str, Any], adapter) -> ToolResult:
        try:
            pattern = params.get("title_pattern", "")
            success = await adapter.switch_window("IntelliJ IDEA", pattern)
            if success:
                return ToolResult(success=True, message=f"Switched to window: {pattern}")
            return ToolResult(success=False, message=f"Window not found: {pattern}")
        except Exception as e:
            return ToolResult(success=False, message=str(e))
