"""Tool: Open a URL in a browser."""
from __future__ import annotations
from typing import Any
from tools._base import BaseTool, ToolResult


class OpenUrlTool(BaseTool):
    @property
    def name(self) -> str:
        return "open_url"

    @property
    def description(self) -> str:
        return "Opens a URL in the specified browser"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to open"},
                "browser": {"type": "string", "description": "Browser name", "default": "Chrome"},
            },
            "required": ["url"],
        }

    async def execute(self, params: dict[str, Any], adapter) -> ToolResult:
        try:
            url = params["url"]
            browser = params.get("browser", "Chrome")
            success = await adapter.open_url_in_browser(url, browser)
            if success:
                return ToolResult(success=True, message=f"Opened {url} in {browser}")
            return ToolResult(success=False, message=f"Failed to open {url}")
        except Exception as e:
            return ToolResult(success=False, message=str(e))
