"""Text embedder via Ollama embeddings API (nomic-embed-text)."""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config.models import OllamaConfig

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, config: OllamaConfig) -> None:
        self._config = config
        self._healthy = False

    @property
    def healthy(self) -> bool:
        return self._healthy

    async def initialize(self) -> bool:
        try:
            import ollama
            response = await ollama.AsyncClient(
                host=f"http://{self._config.host}:{self._config.port}"
            ).embeddings(model=self._config.embed_model, prompt="test")
            if response and "embedding" in response:
                self._healthy = True
                return True
            logger.warning("Embedding model %s returned empty response", self._config.embed_model)
            return False
        except Exception:
            logger.exception("Failed to initialize embedding model %s — run 'ollama pull %s'", self._config.embed_model, self._config.embed_model)
            return False

    async def embed(self, text: str) -> list[float] | None:
        if not self._healthy:
            return None
        try:
            import ollama
            response = await ollama.AsyncClient(
                host=f"http://{self._config.host}:{self._config.port}"
            ).embeddings(model=self._config.embed_model, prompt=text)
            return response.get("embedding")
        except Exception:
            logger.exception("Embedding failed for text: %s...", text[:50])
            return None
