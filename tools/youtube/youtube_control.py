"""Tool: Control YouTube playback — play, pause, next, prev, mute, unmute."""
from __future__ import annotations

from typing import Any

from tools._base import BaseTool, ToolResult


_CONTROL_JS = {
    "play": "document.querySelector('video').play(); 'playing'",
    "pause": "document.querySelector('video').pause(); 'paused'",
    "next": "(function(){ var btn = document.querySelector('.ytp-next-button'); if(btn){btn.click(); return 'next'} return 'no next button' })()",
    "prev": "(function(){ var v = document.querySelector('video'); if(v){v.currentTime=0; return 'restarted'} return 'no video' })()",
    "mute": "(function(){ var v = document.querySelector('video'); if(v){v.muted=true; return 'muted'} return 'no video' })()",
    "unmute": "(function(){ var v = document.querySelector('video'); if(v){v.muted=false; return 'unmuted'} return 'no video' })()",
}

_VALID_ACTIONS = list(_CONTROL_JS.keys())


class YouTubeControlTool(BaseTool):
    @property
    def name(self) -> str:
        return "youtube_control"

    @property
    def description(self) -> str:
        return f"Controls YouTube playback. Actions: {', '.join(_VALID_ACTIONS)}"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": f"One of: {', '.join(_VALID_ACTIONS)}",
                    "enum": _VALID_ACTIONS,
                },
            },
            "required": ["action"],
        }

    async def execute(self, params: dict[str, Any], adapter) -> ToolResult:
        action = params.get("action", "").lower().strip()
        if action not in _CONTROL_JS:
            return ToolResult(success=False, message=f"Unknown action: {action}. Use: {', '.join(_VALID_ACTIONS)}")

        try:
            from adapters.macos.chrome_bridge import ChromeBridge
            config = params.get("_config", {})
            app_name = config.get("browser", {}).get("app_name", "Brave Browser")
            bridge = ChromeBridge(app_name=app_name)

            result = await bridge.execute_js(_CONTROL_JS[action], url_pattern="youtube.com")
            return ToolResult(
                success=True,
                message=f"YouTube: {result}",
                data={"action": action, "speak_result": False},
            )
        except Exception as e:
            return ToolResult(success=False, message=f"YouTube control failed: {e}")
