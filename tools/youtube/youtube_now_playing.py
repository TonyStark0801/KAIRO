"""Tool: Read what's currently playing on YouTube."""
from __future__ import annotations

from typing import Any

from tools._base import BaseTool, ToolResult


_NOW_PLAYING_JS = """
(function() {
    var v = document.querySelector('video');
    if (!v) return 'No video playing';
    var title = document.querySelector('yt-formatted-string.ytd-watch-metadata, h1.ytd-watch-metadata yt-formatted-string');
    var channel = document.querySelector('#channel-name a, ytd-channel-name a');
    var titleText = title ? (title.textContent || '').trim() : 'Unknown';
    var channelText = channel ? (channel.textContent || '').trim() : '';
    var cur = Math.floor(v.currentTime);
    var dur = Math.floor(v.duration) || 0;
    var mm1 = Math.floor(cur/60); var ss1 = cur%60;
    var mm2 = Math.floor(dur/60); var ss2 = dur%60;
    var time = mm1 + ':' + (ss1<10?'0':'') + ss1 + ' / ' + mm2 + ':' + (ss2<10?'0':'') + ss2;
    var state = v.paused ? 'Paused' : 'Playing';
    var info = state + ': ' + titleText;
    if (channelText) info += ' by ' + channelText;
    info += ', ' + time;
    return info;
})()
"""


class YouTubeNowPlayingTool(BaseTool):
    @property
    def name(self) -> str:
        return "youtube_now_playing"

    @property
    def description(self) -> str:
        return "Returns what is currently playing on YouTube — title, channel, and progress."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, params: dict[str, Any], adapter) -> ToolResult:
        try:
            from adapters.macos.chrome_bridge import ChromeBridge
            config = params.get("_config", {})
            app_name = config.get("browser", {}).get("app_name", "Brave Browser")
            bridge = ChromeBridge(app_name=app_name)

            result = await bridge.execute_js(_NOW_PLAYING_JS, url_pattern="youtube.com")
            return ToolResult(
                success=True,
                message=result or "No video playing",
                data={"speak_result": True},
            )
        except Exception as e:
            return ToolResult(success=False, message=f"Could not read YouTube state: {e}")
