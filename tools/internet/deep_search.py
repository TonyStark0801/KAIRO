"""Tool: deep_search — Tavily-powered synthesis search.

Tavily returns a pre-synthesized direct answer (plus source URLs) instead of
raw snippets. Ideal for voice queries where the user wants a spoken answer,
not a list of links.

Free tier: 1000 requests/month. Key via TAVILY_API_KEY env var.
Falls back to web_search (SearxNG) if no key or Tavily call fails.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import yaml

from adapters.base.platform_adapter import PlatformAdapter
from tools._base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_CONFIG_PATH = os.path.expanduser("~/Jarvis/kairo/config/kairo.yaml")


def _get_tavily_key() -> str | None:
    try:
        with open(_CONFIG_PATH) as f:
            cfg = (yaml.safe_load(f) or {}).get("search", {})
        env_var = cfg.get("tavily_api_key_env", "TAVILY_API_KEY")
    except Exception:
        env_var = "TAVILY_API_KEY"
    return os.environ.get(env_var) or None


class DeepSearchTool(BaseTool):
    @property
    def name(self) -> str:
        return "deep_search"

    @property
    def description(self) -> str:
        return "Deep search with AI-synthesized direct answer. Use ONLY for questions that need a current synthesized fact — latest news, current events, time-sensitive answers, 'what is X happening now'. Uses paid API credits (1000/mo), so prefer web_search for general lookups."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language question"},
            },
            "required": ["query"],
        }

    async def execute(
        self, params: dict[str, Any], adapter: PlatformAdapter
    ) -> ToolResult:
        query = (params.get("query") or "").strip()
        if not query:
            return ToolResult(success=False, message="No query provided")

        api_key = _get_tavily_key()
        if not api_key:
            logger.info("TAVILY_API_KEY not set — falling back to web_search")
            from tools.internet.web_search import WebSearchTool
            return await WebSearchTool().execute(params, adapter)

        try:
            answer, sources = await self._tavily(api_key, query)
        except Exception as e:
            logger.exception("Tavily failed, falling back to web_search")
            from tools.internet.web_search import WebSearchTool
            fallback = await WebSearchTool().execute(params, adapter)
            return fallback

        if not answer:
            return ToolResult(
                success=True,
                message="No answer found.",
                data={"speak_result": False},
            )

        src_lines = "\n".join(f"- {s}" for s in sources[:3])
        message = f"ANSWER: {answer}\n\nSOURCES:\n{src_lines}" if src_lines else f"ANSWER: {answer}"
        return ToolResult(
            success=True,
            message=message,
            data={"speak_result": False, "synthesized_answer": answer},
        )

    @staticmethod
    async def _tavily(api_key: str, query: str) -> tuple[str, list[str]]:
        import httpx

        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "include_answer": True,
                    "search_depth": "basic",
                    "max_results": 5,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        answer = (data.get("answer") or "").strip()
        sources = [r.get("url", "") for r in data.get("results", []) if r.get("url")]
        return answer, sources
