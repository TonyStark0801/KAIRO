"""Tool: Control macOS system volume via AppleScript."""
from __future__ import annotations

import asyncio
from typing import Any

from tools._base import BaseTool, ToolResult

_VOLUME_STEP = 15

_VALID_ACTIONS = {"up", "down", "mute", "unmute", "set", "get"}


def _build_script(action: str, step: int = _VOLUME_STEP, level: int = 50) -> str:
    if action == "up":
        return (
            "set curVol to output volume of (get volume settings)\n"
            f"set newVol to curVol + {step}\n"
            "if newVol > 100 then set newVol to 100\n"
            "set volume output volume newVol"
        )
    if action == "down":
        return (
            "set curVol to output volume of (get volume settings)\n"
            f"set newVol to curVol - {step}\n"
            "if newVol < 0 then set newVol to 0\n"
            "set volume output volume newVol"
        )
    if action == "mute":
        return "set volume output muted true"
    if action == "unmute":
        return "set volume output muted false"
    if action == "set":
        return f"set volume output volume {max(0, min(100, level))}"
    return "return output volume of (get volume settings)"


class SystemVolumeTool(BaseTool):
    @property
    def name(self) -> str:
        return "system_volume"

    @property
    def description(self) -> str:
        return "Controls macOS system volume. Actions: up, down, mute, unmute, set (with level 0-100), get"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "One of: up, down, mute, unmute, set, get",
                    "enum": sorted(_VALID_ACTIONS),
                },
                "level": {
                    "type": "integer",
                    "description": "Volume level 0-100 (only for 'set' action)",
                },
            },
            "required": ["action"],
        }

    async def execute(self, params: dict[str, Any], adapter) -> ToolResult:
        action = params.get("action", "").lower().strip()
        if action not in _VALID_ACTIONS:
            return ToolResult(
                success=False,
                message=f"Unknown action: {action}. Use: {', '.join(sorted(_VALID_ACTIONS))}",
            )

        level = int(params.get("level", 50))
        script = _build_script(action, step=_VOLUME_STEP, level=level)

        try:
            proc = await asyncio.create_subprocess_exec(
                "osascript", "-e", script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                return ToolResult(
                    success=False,
                    message=f"Volume control failed: {stderr.decode().strip()}",
                )

            result = stdout.decode().strip()
            messages = {
                "up": "Turned it up.",
                "down": "Turned it down.",
                "mute": "Muted.",
                "unmute": "Unmuted.",
                "set": f"Volume set to {level}.",
                "get": f"Volume is at {result}.",
            }

            return ToolResult(
                success=True,
                message=messages.get(action, "Done."),
                data={"action": action, "speak_result": action == "get"},
            )
        except Exception as e:
            return ToolResult(success=False, message=f"Volume control failed: {e}")
