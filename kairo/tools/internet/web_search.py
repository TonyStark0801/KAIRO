"""Tool: web search — SearxNG primary, DDG fallback.

SearxNG: self-hosted meta-search aggregator (Google/Bing/DDG etc.).
  docker run -d -p 8888:8080 --name searxng searxng/searxng
Free, unlimited, no API key. Used for most voice queries.
DDG: last-resort fallback if SearxNG is down.

For synthesis-style queries (where you want a direct answer, not snippets), use deep_search (Tavily).
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import yaml

from adapters.base.platform_adapter import PlatformAdapter
from tools._base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_CONFIG_PATH = os.path.expanduser("~/Jarvis/kairo/config/kairo.yaml")


def _load_search_config() -> dict[str, Any]:
    try:
        with open(_CONFIG_PATH) as f:
            return (yaml.safe_load(f) or {}).get("search", {})
    except Exception:
        return {}


class WebSearchTool(BaseTool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for general information. Returns ranked snippets that you must synthesize. Use for news, reference lookups, how-to, factual queries."

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

        cfg = _load_search_config()
        results: list[dict[str, Any]] = []

        # Tier 1: SearxNG
        searxng_url = cfg.get("searxng_url")
        if searxng_url:
            try:
                results = await self._searxng(searxng_url, query)
                if results:
                    logger.info("web_search via SearxNG: %d results", len(results))
            except Exception:
                logger.exception("SearxNG search failed, falling back to DDG")

        # Tier 2: DDG fallback
        if not results and cfg.get("enable_ddg_fallback", True):
            try:
                results = await self._ddg(query)
                if results:
                    logger.info("web_search via DDG fallback: %d results", len(results))
            except Exception:
                logger.exception("DDG fallback also failed")

        if not results:
            return ToolResult(
                success=True,
                message="No results found.",
                data={"speak_result": False},
            )

        lines: list[str] = []
        for i, r in enumerate(results[:5], start=1):
            title = str(r.get("title") or "")
            snippet = str(r.get("body") or r.get("snippet") or r.get("content") or "")
            url = str(r.get("href") or r.get("url") or "")
            lines.append(f"{i}. {title}\n   {snippet}\n   {url}")

        return ToolResult(
            success=True,
            message="\n\n".join(lines),
            data={"speak_result": False},
        )

    @staticmethod
    async def _searxng(base_url: str, query: str) -> list[dict[str, Any]]:
        import httpx

        async with httpx.AsyncClient(timeout=6.0) as client:
            resp = await client.get(
                f"{base_url.rstrip('/')}/search",
                params={"q": query, "format": "json", "safesearch": "0"},
                headers={"User-Agent": "KAIRO/1.0"},
            )
            resp.raise_for_status()
            data = resp.json()
        return data.get("results", []) or []

    @staticmethod
    async def _ddg(query: str) -> list[dict[str, Any]]:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        def _sync() -> list[dict[str, Any]]:
            return list(DDGS().text(query, max_results=5))

        return await asyncio.to_thread(_sync)
