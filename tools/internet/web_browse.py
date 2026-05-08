"""Tool: Fetch and extract main text from a web page."""
from __future__ import annotations

import html
import logging
import re
from typing import Any

import httpx
from adapters.base.platform_adapter import PlatformAdapter
from tools._base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")


class WebBrowseTool(BaseTool):
    @property
    def name(self) -> str:
        return "web_browse"

    @property
    def description(self) -> str:
        return "Read the content of a web page. Returns clean text extracted from the URL."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch"},
            },
            "required": ["url"],
        }

    async def execute(
        self, params: dict[str, Any], adapter: PlatformAdapter
    ) -> ToolResult:
        url = (params.get("url") or "").strip()
        if not url:
            return ToolResult(success=False, message="No URL provided")
        if not url.startswith(("http://", "https://")):
            return ToolResult(success=False, message=f"Not a valid URL: {url}")

        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
                raw_html = response.text

            from readability import Document

            summary_html = Document(raw_html).summary()
            plain = _TAG_RE.sub(" ", summary_html)
            plain = html.unescape(plain)
            plain = re.sub(r"\s+", " ", plain).strip()
            if len(plain) > 2000:
                plain = plain[:2000] + "…"

            return ToolResult(
                success=True,
                message=plain,
                data={"speak_result": False},
            )
        except Exception as e:
            logger.exception("web_browse failed for url=%r", url)
            return ToolResult(
                success=False,
                message=f"Could not read page: {e}",
            )
