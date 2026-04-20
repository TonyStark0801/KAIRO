"""Tool: Search YouTube and auto-play the first result — one-step, like Alexa."""
from __future__ import annotations

import asyncio
from typing import Any

from tools._base import BaseTool, ToolResult

_AUTO_PLAY_FIRST_JS = """
(function() {
    var vid = document.querySelector('ytd-video-renderer a#video-title');
    if (vid) {
        var title = (vid.textContent || '').trim();
        vid.click();
        return 'playing: ' + title;
    }
    var thumb = document.querySelector('ytd-video-renderer #thumbnail');
    if (thumb) { thumb.click(); return 'playing first result'; }
    return 'NO_RESULTS';
})()
"""

_READ_RESULTS_JS = """
(function() {
    var results = [];
    var items = document.querySelectorAll('ytd-video-renderer, ytd-playlist-renderer');
    for (var i = 0; i < Math.min(items.length, 5); i++) {
        var titleEl = items[i].querySelector('#video-title, #title a, a#video-title');
        if (titleEl) {
            var text = (titleEl.textContent || titleEl.innerText || '').trim();
            if (text) results.push((i+1) + '. ' + text);
        }
    }
    return results.join('\\n') || 'No results found';
})()
"""


class YouTubeSearchTool(BaseTool):
    @property
    def name(self) -> str:
        return "youtube_search"

    @property
    def description(self) -> str:
        return "Searches YouTube and auto-plays the first result. Use for any 'play X' request."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for on YouTube"},
                "auto_play": {
                    "type": "boolean",
                    "description": "Auto-play first result (default true). Set false to just show results.",
                    "default": True,
                },
            },
            "required": ["query"],
        }

    async def execute(self, params: dict[str, Any], adapter) -> ToolResult:
        query = params.get("query", "")
        auto_play = params.get("auto_play", True)
        if not query:
            return ToolResult(success=False, message="No search query provided")

        try:
            from adapters.macos.chrome_bridge import ChromeBridge
            config = params.get("_config", {})
            app_name = config.get("browser", {}).get("app_name", "Brave Browser")
            bridge = ChromeBridge(app_name=app_name)

            search_url = bridge.build_search_url(query)
            ok = await bridge.ensure_youtube_tab(search_url)
            if not ok:
                return ToolResult(success=False, message="Could not open YouTube")

            await asyncio.sleep(2.5)

            if auto_play:
                result = await bridge.execute_js(_AUTO_PLAY_FIRST_JS, url_pattern="youtube.com")
                if result and result != "NO_RESULTS":
                    return ToolResult(
                        success=True,
                        message=result,
                        data={"speak_result": False, "auto_played": True},
                    )
                return ToolResult(
                    success=True,
                    message=f"No results for '{query}'.",
                    data={"speak_result": True, "auto_played": False},
                )

            raw = await bridge.execute_js(_READ_RESULTS_JS, url_pattern="youtube.com")
            if not raw or raw == "No results found":
                return ToolResult(
                    success=True,
                    message=f"No results for '{query}'.",
                    data={"results": [], "speak_result": True},
                )

            lines = [l.strip() for l in raw.split("\n") if l.strip()]
            summary = f"Found {len(lines)} results for '{query}'. Which one?"
            return ToolResult(
                success=True,
                message=summary,
                data={"results": lines, "speak_result": True},
            )
        except Exception as e:
            return ToolResult(success=False, message=f"YouTube search failed: {e}")
