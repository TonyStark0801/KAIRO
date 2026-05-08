"""Tool: Write text content to a file on the user's filesystem.

Used when the user says "write to file", "save that", "save the solution", etc.
The actual file write happens in the daemon's synthesis closure so it can access
the cached last code block (_last_code_block) when the LLM doesn't embed content
in params (which a 3b model often can't do reliably for large code).

This file exists so WriteFileTool appears in tool_metas → system prompt.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from adapters.base.platform_adapter import PlatformAdapter
from tools._base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_MAX_CONTENT_BYTES = 500_000  # 500 KB hard cap
_BLOCKED_PREFIXES = ("/etc", "/usr", "/bin", "/sbin", "/System", "/Library/System")


class WriteFileTool(BaseTool):
    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "Write text content to a file. "
            "Use when: user says 'write to file', 'save the solution', 'save that code', "
            "'create a file with', 'export to'. "
            "Path defaults to ~/Desktop/<inferred-name>.<ext> if not specified. "
            "Content defaults to the last code block if not provided."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Destination file path, e.g. '~/Desktop/Solution.java'. "
                        "Infer from context: Java code → .java, Python → .py, etc."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": (
                        "Text to write. Leave empty to use the last cached code block."
                    ),
                },
            },
            "required": ["path"],
        }

    async def execute(self, params: dict[str, Any], adapter: PlatformAdapter) -> ToolResult:
        """Fallback direct execution — daemon's synthesis closure is preferred."""
        raw_path = (params.get("path") or "~/Desktop/solution.txt").strip()
        content = (params.get("content") or "").strip()

        if not content:
            return ToolResult(
                success=False,
                message="No content to write — ask me to generate the solution first.",
            )

        path = Path(raw_path.replace("~", str(Path.home()))).expanduser().resolve()

        for fp in _BLOCKED_PREFIXES:
            if str(path).startswith(fp):
                return ToolResult(success=False, message=f"Writing to {fp} is not allowed.")

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            logger.info("write_file (direct): %d chars → %s", len(content), path)
        except Exception as e:
            logger.exception("write_file direct execute failed: %s", path)
            return ToolResult(success=False, message=f"Failed to write file: {e}")

        return ToolResult(
            success=True,
            message=f"Written to {path.name}.",
            data={"speak_result": True, "path": str(path)},
        )
