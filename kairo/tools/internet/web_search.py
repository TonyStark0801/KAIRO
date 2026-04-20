"""Tool: DuckDuckGo web search."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from adapters.base.platform_adapter import PlatformAdapter
from tools._base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web using DuckDuckGo. Returns top results with titles and snippets."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        }

    async def execute(
        self, params: dict[str, Any], adapter: PlatformAdapter
    ) -> ToolResult:
        query = (params.get("query") or "").strip()
        if not query:
            return ToolResult(success=False, message="No search query provided")

        try:
            results = await self._run_search(query)
        except Exception as e:
            logger.exception("web_search failed for query=%r", query)
            return ToolResult(success=False, message=f"Search failed: {e}")

        if not results:
            return ToolResult(
                success=True,
                message="No results found.",
                data={"speak_result": False},
            )

        lines: list[str] = []
        for i, r in enumerate(results, start=1):
            title = str(r.get("title", "") or "")
            snippet = str(
                r.get("body", "") or r.get("snippet", "") or ""
            )
            url = str(r.get("href", "") or r.get("url", "") or "")
            lines.append(f"{i}. {title}\n   {snippet}\n   {url}")
        formatted = "\n\n".join(lines)

        return ToolResult(
            success=True,
            message=formatted,
            data={"speak_result": False},
        )

    async def _run_search(self, query: str) -> list[dict[str, Any]]:
        """Search via ddgs (new package name) or duckduckgo_search (legacy)."""
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        def _sync() -> list[dict[str, Any]]:
            return list(DDGS().text(query, max_results=5))

        return await asyncio.to_thread(_sync)
