"""Factory for local chat providers — keeps daemon and planner free of vendor imports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_router.ollama_provider import OllamaProvider
from llm_router.protocol import LocalChatProvider

if TYPE_CHECKING:
    from core.config.models import OllamaConfig


def create_local_chat_provider(ollama: OllamaConfig) -> LocalChatProvider:
    """Construct the default local LLM stack (Ollama). Swap implementation here later."""
    return OllamaProvider(
        host=ollama.host,
        port=ollama.port,
        model=ollama.model,
        fast_model=ollama.fast_model,
    )
