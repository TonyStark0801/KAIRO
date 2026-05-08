"""Tool: Run a read-only shell command and return its output.

Designed for developer assistants — allows inspection commands (ls, find, cat,
git status, etc.) but blocks destructive operations.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shlex
from pathlib import Path
from typing import Any

from adapters.base.platform_adapter import PlatformAdapter
from tools._base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# Commands that are always blocked regardless of arguments.
_BLOCKED_COMMANDS = {
    "rm", "rmdir", "sudo", "su", "chmod", "chown", "chflags",
    "mv", "dd", "mkfs", "fdisk", "kill", "killall", "pkill",
    "reboot", "shutdown", "halt", "poweroff",
    "curl", "wget",      # no arbitrary downloads
    "python", "python3", "node", "ruby", "perl",  # no arbitrary code exec
    "bash", "sh", "zsh", "fish",                   # no shell spawn
    "eval", "exec",
}

# Destructive flag patterns — block even if the base command is allowed.
_BLOCKED_PATTERNS = [
    "> /",          # redirect to root path
    ">/",
    "| rm",
    ";rm",
    "&&rm",
    "| sudo",
]

_COMMAND_TIMEOUT = 10  # seconds
_MAX_OUTPUT_CHARS = 1200  # truncate before feeding to LLM


def _is_safe(command: str) -> tuple[bool, str]:
    """Return (safe, reason). Blocks obviously destructive commands."""
    stripped = command.strip()
    if not stripped:
        return False, "empty command"

    # Check blocked patterns in raw string
    lower = stripped.lower()
    for pat in _BLOCKED_PATTERNS:
        if pat in lower:
            return False, f"blocked pattern: {pat}"

    # Extract base command (first token)
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        tokens = stripped.split()

    if not tokens:
        return False, "empty command"

    base = Path(tokens[0]).name.lower()  # handles /usr/bin/rm → rm
    if base in _BLOCKED_COMMANDS:
        return False, f"blocked command: {base}"

    return True, ""


class TerminalCommandTool(BaseTool):
    @property
    def name(self) -> str:
        return "terminal_command"

    @property
    def description(self) -> str:
        return (
            "Run a read-only shell command and return its output. "
            "Use for: listing directory contents (ls), finding files (find), "
            "reading files (cat/head/tail), git status, grep, pwd. "
            "Does NOT persist working directory between calls — use absolute paths."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to run (e.g. 'ls ~/IdeaProjects/MyProject')",
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory (optional, defaults to user home)",
                },
            },
            "required": ["command"],
        }

    async def execute(self, params: dict[str, Any], adapter: PlatformAdapter) -> ToolResult:
        command = (params.get("command") or "").strip()
        cwd_param = (params.get("cwd") or "").strip()

        if not command:
            return ToolResult(success=False, message="No command provided.")

        safe, reason = _is_safe(command)
        if not safe:
            logger.warning("terminal_command blocked: %s — %s", command, reason)
            return ToolResult(
                success=False,
                message=f"That command is blocked for safety ({reason}).",
            )

        # Resolve working directory
        home = Path.home()
        if cwd_param:
            cwd = Path(cwd_param.replace("~", str(home))).expanduser()
        else:
            cwd = home

        if not cwd.exists():
            return ToolResult(success=False, message=f"Directory not found: {cwd}")

        logger.info("terminal_command: %r (cwd=%s)", command, cwd)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
                env={**os.environ, "TERM": "dumb"},
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=_COMMAND_TIMEOUT
            )
        except asyncio.TimeoutError:
            return ToolResult(success=False, message=f"Command timed out after {_COMMAND_TIMEOUT}s.")
        except Exception as e:
            logger.exception("terminal_command failed: %r", command)
            return ToolResult(success=False, message=f"Command failed: {e}")

        output = stdout.decode(errors="replace").strip()
        err = stderr.decode(errors="replace").strip()

        if not output and err:
            return ToolResult(success=False, message=f"Command error: {err[:400]}")

        if not output:
            return ToolResult(success=True, message="Command ran with no output.")

        # Truncate for LLM context
        if len(output) > _MAX_OUTPUT_CHARS:
            output = output[:_MAX_OUTPUT_CHARS] + f"\n... (truncated, {len(output)} chars total)"

        return ToolResult(
            success=True,
            message=output,
            data={"speak_result": False, "raw_output": True},
        )
