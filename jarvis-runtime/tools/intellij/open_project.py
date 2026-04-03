"""Tool: Open an IntelliJ project."""
from __future__ import annotations
from typing import Any
from tools._base import BaseTool, ToolResult


class OpenProjectTool(BaseTool):
    @property
    def name(self) -> str:
        return "open_project"

    @property
    def description(self) -> str:
        return "Opens a project in IntelliJ IDEA by name"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"project": {"type": "string", "description": "Project name from config"}},
            "required": ["project"],
        }

    async def execute(self, params: dict[str, Any], adapter) -> ToolResult:
        try:
            project = params.get("project", "")
            await adapter.open_application("IntelliJ IDEA")
            return ToolResult(success=True, message=f"Opened project: {project}")
        except Exception as e:
            return ToolResult(success=False, message=str(e))
