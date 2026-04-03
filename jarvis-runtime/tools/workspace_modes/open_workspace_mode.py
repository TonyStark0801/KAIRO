"""Tool: Execute a workspace mode — runs a sequence of tool steps."""
from __future__ import annotations
import asyncio
import logging
from typing import Any
from tools._base import BaseTool, ToolResult

logger = logging.getLogger(__name__)
_STEP_DELAY = 0.5


class OpenWorkspaceModeTool(BaseTool):
    @property
    def name(self) -> str:
        return "open_workspace_mode"

    @property
    def description(self) -> str:
        return "Activates a named workspace mode, executing each configured step sequentially"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"mode": {"type": "string", "description": "Workspace mode name from config"}},
            "required": ["mode"],
        }

    async def execute(self, params: dict[str, Any], adapter) -> ToolResult:
        try:
            mode_name = params.get("mode", "")
            config = params.get("_config", {})
            executor = params.get("_executor")

            modes = config.get("workspace_modes", {})
            mode = modes.get(mode_name)
            if mode is None:
                return ToolResult(success=False, message=f"Unknown workspace mode: {mode_name}")

            steps = mode.get("steps", [])
            succeeded = 0
            failed = 0
            for step in steps:
                tool_name = step.get("tool", "")
                tool_params = step.get("params", {})
                if executor is not None:
                    result = await executor.execute_tool(tool_name=tool_name, params=tool_params, adapter=adapter)
                    if result.success:
                        succeeded += 1
                    else:
                        failed += 1
                        logger.warning("Workspace step %s failed: %s", tool_name, result.message)
                await asyncio.sleep(_STEP_DELAY)

            summary = f"Workspace '{mode_name}': {succeeded} succeeded"
            if failed:
                summary += f", {failed} failed"
            return ToolResult(success=True, message=summary)
        except Exception as e:
            return ToolResult(success=False, message=str(e))
