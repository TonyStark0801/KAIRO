"""Tool: Play music via Apple Music."""
from __future__ import annotations
from typing import Any
from tools._base import BaseTool, ToolResult


class PlayMusicTool(BaseTool):
    @property
    def name(self) -> str:
        return "play_music"

    @property
    def description(self) -> str:
        return "Plays music or a playlist in Apple Music"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Playlist or song name to play"}},
            "required": [],
        }

    async def execute(self, params: dict[str, Any], adapter) -> ToolResult:
        try:
            query = params.get("query", "")
            script = 'tell application "Music"\n  activate\n  play\nend tell'
            if query:
                script = f'tell application "Music"\n  activate\n  play (first playlist whose name contains "{query}")\nend tell'
            await adapter.run_script(script)
            return ToolResult(success=True, message=f"Playing music: {query or 'resumed'}")
        except Exception as e:
            return ToolResult(success=False, message=str(e))
