"""Tool: Search for and play a YouTube playlist."""
from __future__ import annotations

import asyncio
import urllib.parse
from typing import Any

from tools._base import BaseTool, ToolResult


_CLICK_FIRST_PLAYLIST_JS = """
(function() {
    var pl = document.querySelector('ytd-playlist-renderer a#thumbnail');
    if (pl) { pl.click(); return 'OK:playlist'; }
    var vid = document.querySelector('ytd-video-renderer a#video-title');
    if (vid) { vid.click(); return 'OK:video'; }
    return 'ERROR:NOTHING_FOUND';
})()
"""


class YouTubePlaylistTool(BaseTool):
    @property
    def name(self) -> str:
        return "youtube_playlist"

    @property
    def description(self) -> str:
        return "Searches YouTube for a playlist by name and starts playing it."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Playlist name to search for"},
            },
            "required": ["name"],
        }

    async def execute(self, params: dict[str, Any], adapter) -> ToolResult:
        name = params.get("name", "")
        if not name:
            return ToolResult(success=False, message="No playlist name provided")

        try:
            from adapters.macos.chrome_bridge import ChromeBridge
            config = params.get("_config", {})
            app_name = config.get("browser", {}).get("app_name", "Brave Browser")
            bridge = ChromeBridge(app_name=app_name)

            encoded = urllib.parse.quote_plus(f"{name} playlist")
            url = f"https://www.youtube.com/results?search_query={encoded}&sp=EgIQAw%253D%253D"
            await bridge.ensure_youtube_tab(url)
            await asyncio.sleep(2.5)

            result = await bridge.execute_js(_CLICK_FIRST_PLAYLIST_JS, url_pattern="youtube.com")
            if result.startswith("ERROR:"):
                return ToolResult(success=False, message=f"No playlist found for '{name}'")

            await asyncio.sleep(1.5)
            return ToolResult(
                success=True,
                message=f"Playing playlist: {name}",
                data={"speak_result": False},
            )
        except Exception as e:
            return ToolResult(success=False, message=f"Playlist failed: {e}")
