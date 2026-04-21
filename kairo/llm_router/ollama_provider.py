"""Ollama LLM provider — single local model (qwen2.5:3b) as the unified brain.

No more 3-tier escalation. One model, one client, one connection pool.
Client is created once in initialize() and reused — no per-call leaks.
"""

from __future__ import annotations

import logging
from typing import Any

from llm_router.base import LLMProvider

logger = logging.getLogger(__name__)

# 10 turns = 20 messages (user + assistant pairs). Enough context without
# hammering RAM. Older turns are already summarized into session memory.
_MAX_HISTORY = 10


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 11434,
        model: str = "qwen2.5:3b",
        fast_model: str = "qwen2.5:3b",
    ) -> None:
        self._host = host
        self._port = port
        self._model = model
        self._fast_model = fast_model
        self._healthy = False
        self._client: Any = None  # created once in initialize(), reused across calls

    @property
    def healthy(self) -> bool:
        return self._healthy

    @property
    def fast_model(self) -> str:
        return self._fast_model

    @property
    def deep_model(self) -> str:
        return self._model

    async def initialize(self) -> bool:
        try:
            import ollama
            # Create client once — reused for all subsequent calls.
            # Avoids creating a new httpx connection pool per chat() call (ResourceWarning source).
            self._client = ollama.AsyncClient(host=f"http://{self._host}:{self._port}")
            await self._client.list()
            self._healthy = True
            logger.info(
                "Ollama connected: %s:%d model=%s",
                self._host, self._port, self._model,
            )
            await self._warmup()
            return True
        except Exception:
            logger.exception("Failed to connect to Ollama")
            self._client = None
            return False

    async def close(self) -> None:
        """Explicitly close the underlying httpx client on SIGINT / shutdown."""
        self._healthy = False
        if self._client is not None:
            try:
                # ollama.AsyncClient wraps httpx.AsyncClient; close() drains the connection pool.
                await self._client._client.aclose()
                logger.info("Ollama client closed cleanly")
            except Exception:
                pass
            self._client = None

    async def _warmup(self) -> None:
        """Pre-load the model so the first real call isn't penalised by a cold load."""
        try:
            await self._client.chat(
                model=self._model,
                messages=[{"role": "user", "content": "hi"}],
                options={"num_predict": 1},
                keep_alive="5m",
            )
            logger.info("Warmed up model: %s", self._model)
        except Exception:
            logger.warning("Warmup failed for %s (will load on first use)", self._model)

    async def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        model_override: str | None = None,
    ) -> str:
        if not self._healthy or self._client is None:
            return ""

        model = model_override or self._model

        full_messages = [{"role": "system", "content": system_prompt}]
        # Trim to last N turns. Each turn = 1 user + 1 assistant message.
        trimmed = messages[-_MAX_HISTORY:] if len(messages) > _MAX_HISTORY else messages
        full_messages.extend(trimmed)

        try:
            response = await self._client.chat(
                model=model,
                messages=full_messages,
                keep_alive="5m",
            )
            return response["message"]["content"]
        except Exception:
            logger.exception("Ollama chat failed (model=%s)", model)
            return ""
