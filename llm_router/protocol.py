"""Structural protocol for local chat LLMs (Ollama today; others later)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LocalChatProvider(Protocol):
    """Minimal surface the Reasoner needs from a local (or local-style) LLM backend."""

    @property
    def healthy(self) -> bool: ...

    @property
    def fast_model(self) -> str: ...

    @property
    def deep_model(self) -> str: ...

    async def initialize(self) -> bool: ...

    async def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        model_override: str | None = None,
    ) -> str: ...
