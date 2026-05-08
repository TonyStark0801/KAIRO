"""Abstract LLM provider interface — swap model backends without touching callers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class ResponseAction(Enum):
    EXECUTE = auto()
    SPEAK = auto()
    SPEAK_AND_EXECUTE = auto()


@dataclass
class LLMResponse:
    action: ResponseAction
    tool_name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    confidence: float = 0.0
    raw: str = ""


class LLMProvider(ABC):
    @abstractmethod
    async def initialize(self) -> bool:
        """Connect to the LLM backend. Return True if healthy."""

    @abstractmethod
    async def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        model_override: str | None = None,
    ) -> str:
        """Send messages with system prompt, return raw text response."""

    @property
    @abstractmethod
    def healthy(self) -> bool: ...
