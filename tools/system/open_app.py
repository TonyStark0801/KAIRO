"""Tool: Open any macOS application by name."""
from __future__ import annotations
from typing import Any
from tools._base import BaseTool, ToolResult


class OpenAppTool(BaseTool):
    @property
    def name(self) -> str:
        return "open_app"

    @property
    def description(self) -> str:
        return "Opens a macOS application by name"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"app_name": {"type": "string", "description": "Application name, e.g. 'IntelliJ IDEA'"}},
            "required": ["app_name"],
        }

    async def execute(self, params: dict[str, Any], adapter) -> ToolResult:
        app_name = params.get("app_name", "")
        if not app_name:
            return ToolResult(success=False, message="No app name provided")
        try:
            await adapter.open_application(app_name)
            return ToolResult(success=True, message=f"Opened {app_name}")
        except Exception as e:
            return ToolResult(success=False, message=str(e))
