"""Groq cloud LLM provider — OpenAI-compatible API via official Groq SDK."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_MAX_HISTORY = 20


@dataclass
class GroqToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class GroqToolResponse:
    message: str = ""
    tool_calls: list[GroqToolCall] = field(default_factory=list)


class GroqProvider:
    def __init__(
        self,
        api_key_env: str = "GROQ_API_KEY",
        model: str = "llama-3.3-70b-versatile",
        max_tokens: int = 300,
        temperature: float = 0.7,
    ) -> None:
        self._api_key_env = api_key_env
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._healthy = False
        self._client: Any = None

    @property
    def healthy(self) -> bool:
        return self._healthy

    async def initialize(self) -> bool:
        self._healthy = False
        self._client = None
        api_key = os.environ.get(self._api_key_env, "").strip()
        if not api_key:
            logger.warning("Groq API key not found in env var %s", self._api_key_env)
            return False
        try:
            from groq import AsyncGroq

            client = AsyncGroq(api_key=api_key)
            await client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
            )
            self._client = client
            self._healthy = True
            logger.info("Groq connected: model=%s", self._model)
            return True
        except Exception:
            logger.exception("Failed to connect to Groq")
            self._healthy = False
            self._client = None
            return False

    async def chat(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        model_override: str | None = None,
    ) -> str:
        if not self._healthy or not self._client:
            return ""
        model = model_override or self._model
        full_messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        trimmed = messages[-_MAX_HISTORY:] if len(messages) > _MAX_HISTORY else messages
        full_messages.extend(trimmed)
        try:
            import groq  # noqa: F401 — lazy import; package required at runtime

            response = await self._client.chat.completions.create(
                model=model,
                messages=full_messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            msg = response.choices[0].message
            return msg.content or ""
        except Exception:
            logger.exception("Groq chat failed (model=%s)", model)
            return ""

    async def chat_with_tools(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ) -> GroqToolResponse:
        if not self._healthy or not self._client:
            return GroqToolResponse()
        full_messages: list[dict[str, Any]] = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        trimmed = messages[-_MAX_HISTORY:] if len(messages) > _MAX_HISTORY else messages
        full_messages.extend(trimmed)
        try:
            import groq  # noqa: F401 — lazy import; package required at runtime

            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": full_messages,
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = await self._client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            msg = choice.message

            tool_calls: list[GroqToolCall] = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    raw_args = tc.function.arguments if tc.function else ""
                    args: dict[str, Any] = {}
                    if raw_args:
                        try:
                            parsed = json.loads(raw_args)
                            args = parsed if isinstance(parsed, dict) else {}
                        except json.JSONDecodeError:
                            logger.warning(
                                "Groq tool call arguments JSON decode failed (name=%s)",
                                tc.function.name if tc.function else "?",
                            )
                    name = tc.function.name if tc.function else ""
                    tool_calls.append(
                        GroqToolCall(
                            id=tc.id or "",
                            name=name,
                            arguments=args,
                        )
                    )

            return GroqToolResponse(
                message=msg.content or "",
                tool_calls=tool_calls,
            )
        except Exception as exc:
            # If tool calling failed (model used wrong format), retry without tools
            if "tool_use_failed" in str(exc) or "400" in str(exc):
                logger.warning("Groq tool calling failed, retrying without tools")
                try:
                    no_tool_kwargs = {
                        "model": self._model,
                        "messages": full_messages,
                        "max_tokens": self._max_tokens,
                        "temperature": self._temperature,
                    }
                    response = await self._client.chat.completions.create(**no_tool_kwargs)
                    msg = response.choices[0].message
                    return GroqToolResponse(message=msg.content or "")
                except Exception:
                    logger.exception("Groq fallback chat also failed")
            else:
                logger.exception("Groq chat_with_tools failed (model=%s)", self._model)
            return GroqToolResponse()
