"""Tool: Fuzzy-find a folder by name across common project roots.

Bridges noisy STT to real filesystem paths. The LLM calls this instead of
guessing paths — it returns scored matches so the LLM can either use the
top hit directly (high confidence) or ask the user to confirm (low confidence).
"""
from __future__ import annotations

import difflib
import logging
import os
from pathlib import Path
from typing import Any

from adapters.base.platform_adapter import PlatformAdapter
from tools._base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_DEFAULT_ROOTS = ["~", "~/IdeaProjects", "~/Projects", "~/Desktop", "~/Documents"]
_MAX_DEPTH = 3
_SKIP_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__", ".idea",
    ".gradle", "build", "dist", "target", ".cache", "Library",
    ".Trash", ".npm", ".pyenv", ".rbenv", ".nvm",
}
_TOP_N = 5


def _scan(root: Path, max_depth: int) -> list[Path]:
    """Yield directories under root up to max_depth, skipping hidden/noisy ones."""
    results: list[Path] = []

    def _walk(path: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            with os.scandir(path) as it:
                for entry in it:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    name = entry.name
                    if name.startswith(".") or name in _SKIP_DIRS:
                        continue
                    results.append(Path(entry.path))
                    _walk(Path(entry.path), depth + 1)
        except (PermissionError, OSError):
            return

    _walk(root, 1)
    return results


class FindFolderTool(BaseTool):
    @property
    def name(self) -> str:
        return "find_folder"

    @property
    def description(self) -> str:
        return (
            "Fuzzy-find a folder by name across common roots (~, ~/IdeaProjects, "
            "~/Projects, ~/Desktop, ~/Documents). Returns top 5 matches with "
            "similarity scores 0-1. Use this INSTEAD of guessing paths when the "
            "user mentions a folder by name — especially when STT is noisy."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Folder name to search for (e.g. 'jarvis', 'codejam')",
                },
                "roots": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional override of search roots. Defaults to common project dirs.",
                },
            },
            "required": ["query"],
        }

    async def execute(self, params: dict[str, Any], adapter: PlatformAdapter) -> ToolResult:
        query = (params.get("query") or "").strip().lower()
        if not query:
            return ToolResult(success=False, message="No query provided.")

        roots = params.get("roots") or _DEFAULT_ROOTS
        home = Path.home()

        candidates: list[Path] = []
        seen: set[Path] = set()
        for root_str in roots:
            root = Path(root_str.replace("~", str(home))).expanduser()
            if not root.exists() or not root.is_dir():
                continue
            for path in _scan(root, _MAX_DEPTH):
                if path in seen:
                    continue
                seen.add(path)
                candidates.append(path)

        if not candidates:
            return ToolResult(
                success=True,
                message=f"No folders found under {roots}.",
                data={"matches": [], "speak_result": False, "raw_output": True},
            )

        scored: list[tuple[float, Path]] = []
        for path in candidates:
            name = path.name.lower()
            ratio = difflib.SequenceMatcher(None, query, name).ratio()
            if query in name:
                ratio = max(ratio, 0.85)
            if name == query:
                ratio = 1.0
            scored.append((ratio, path))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:_TOP_N]

        matches = [{"path": str(p), "score": round(s, 3), "name": p.name} for s, p in top]

        if matches and matches[0]["score"] >= 0.75:
            summary = f"Top match: {matches[0]['path']} (score {matches[0]['score']})"
        elif matches and matches[0]["score"] >= 0.6:
            names = ", ".join(m["name"] for m in matches[:3])
            summary = f"Possible matches: {names} (best score {matches[0]['score']})"
        else:
            summary = f"No strong matches for '{query}' (best score {matches[0]['score'] if matches else 0})"

        logger.info("find_folder query=%r → %d matches, top=%s", query, len(matches), matches[0] if matches else None)

        return ToolResult(
            success=True,
            message=summary,
            data={"matches": matches, "speak_result": False, "raw_output": True},
        )
