"""Tool: Click the Nth YouTube search result to start playing it."""
from __future__ import annotations

import asyncio
from typing import Any

from tools._base import BaseTool, ToolResult


_CLICK_RESULT_JS = """
(function(idx) {
    var items = document.querySelectorAll('ytd-video-renderer');
    if (idx < 1 || idx > items.length) return 'ERROR:INDEX_OUT_OF_RANGE';
    var link = items[idx-1].querySelector('a#video-title');
    if (!link) return 'ERROR:NO_LINK';
    link.click();
    return 'OK:' + (link.textContent || '').trim();
})(INDEX_PLACEHOLDER)
"""


class YouTubePickTool(BaseTool):
    @property
    def name(self) -> str:
        return "youtube_pick"

    @property
    def description(self) -> str:
        return "Clicks the Nth search result on the current YouTube search page to play it. Use after youtube_search."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "Which result to play (1-5)"},
            },
            "required": ["index"],
        }

    async def execute(self, params: dict[str, Any], adapter) -> ToolResult:
        index = params.get("index", 1)
        try:
            from adapters.macos.chrome_bridge import ChromeBridge
            config = params.get("_config", {})
            app_name = config.get("browser", {}).get("app_name", "Brave Browser")
            bridge = ChromeBridge(app_name=app_name)

            js = _CLICK_RESULT_JS.replace("INDEX_PLACEHOLDER", str(int(index)))
            result = await bridge.execute_js(js, url_pattern="youtube.com")

            if result.startswith("ERROR:"):
                return ToolResult(success=False, message=f"Could not pick result {index}: {result}")

            title = result.replace("OK:", "", 1).strip()
            await asyncio.sleep(1.0)
            return ToolResult(
                success=True,
                message=f"Now playing: {title}" if title else f"Playing result {index}",
                data={"title": title, "speak_result": False},
            )
        except Exception as e:
            return ToolResult(success=False, message=f"Failed to pick result: {e}")
