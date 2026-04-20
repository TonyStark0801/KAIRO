"""Ollama LLM provider — supports fast (3b) and deep (8b) models."""

from __future__ import annotations

import logging

from llm_router.base import LLMProvider

logger = logging.getLogger(__name__)

_MAX_HISTORY = 20


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 11434,
        model: str = "qwen3:4b-instruct-q4_K_M",
        fast_model: str = "llama3.2:3b",
    ) -> None:
        self._host = host
        self._port = port
        self._model = model
        self._fast_model = fast_model
        self._healthy = False

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
            client = ollama.AsyncClient(host=f"http://{self._host}:{self._port}")
            await client.list()
            self._healthy = True
            logger.info(
                "Ollama connected: %s:%d fast=%s deep=%s",
                self._host, self._port, self._fast_model, self._model,
            )
            # Warm up both models so first call isn't slow
            await self._warmup(client)
            return True
        except Exception:
            logger.exception("Failed to connect to Ollama")
            return False

    async def _warmup(self, client) -> None:
        """Pre-load only the fast model — deep model uses Groq cloud now."""
        import ollama
        try:
            await client.chat(
                model=self._fast_model,
                messages=[{"role": "user", "content": "hi"}],
                options={"num_predict": 1},
            )
            logger.info("Warmed up model: %s", self._fast_model)
        except Exception:
            logger.warning("Warmup failed for %s (will load on first use)", self._fast_model)
        # Unload the deep model if it was previously cached
        try:
            await client.chat(
                model=self._model,
                messages=[{"role": "user", "content": ""}],
                keep_alive=0,
            )
            logger.info("Unloaded deep model %s to save memory (Groq handles Tier 3)", self._model)
        except Exception:
            pass

    async def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        model_override: str | None = None,
    ) -> str:
        if not self._healthy:
            return ""

        model = model_override or self._model

        full_messages = [{"role": "system", "content": system_prompt}]
        trimmed = messages[-_MAX_HISTORY:] if len(messages) > _MAX_HISTORY else messages
        full_messages.extend(trimmed)

        try:
            import ollama
            client = ollama.AsyncClient(host=f"http://{self._host}:{self._port}")
            response = await client.chat(
                model=model,
                messages=full_messages,
                keep_alive="10m",
            )
            return response["message"]["content"]
        except Exception:
            logger.exception("Ollama chat failed (model=%s)", model)
            return ""
