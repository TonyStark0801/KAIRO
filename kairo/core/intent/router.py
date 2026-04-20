"""Conversational intent router — multi-turn dialogue with Ollama.

Instead of one-shot tool matching, this router maintains per-session
conversation history so the LLM can ask clarifying questions, remember
context, and build up enough information before executing a tool.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_MAX_HISTORY = 20


class RouterAction(Enum):
    EXECUTE = auto()
    CLARIFY = auto()
    CHAT = auto()


@dataclass
class RouterResponse:
    action: RouterAction
    tool_name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    confidence: float = 0.0
    raw_response: str = ""


@dataclass
class IntentResult:
    """Backward-compatible result for old ``route()`` callers."""
    tool_name: str | None
    params: dict[str, Any]
    confidence: float
    raw_response: str


class IntentRouter:
    def __init__(self, host: str, port: int, model: str) -> None:
        self._host = host
        self._port = port
        self._model = model
        self._jinja = Environment(
            loader=FileSystemLoader(str(_PROMPTS_DIR)), autoescape=False,
        )
        self._healthy = False
        self._conversations: dict[str, list[dict[str, str]]] = {}

    @property
    def healthy(self) -> bool:
        return self._healthy

    async def initialize(self) -> bool:
        try:
            import ollama
            client = ollama.AsyncClient(
                host=f"http://{self._host}:{self._port}",
            )
            await client.list()
            self._healthy = True
            return True
        except Exception:
            logger.exception("Failed to connect to Ollama")
            return False

    # ------------------------------------------------------------------
    # Conversation management
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self, tool_metas: list, recent_commands: list[str],
    ) -> str:
        tools_json = json.dumps(
            [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters_schema,
                }
                for t in tool_metas
            ],
            indent=2,
        )
        template = self._jinja.get_template("system.j2")
        return template.render(
            tools_json=tools_json, recent_commands=recent_commands,
        )

    async def chat(
        self,
        transcript: str,
        session_id: str,
        tool_metas: list,
        recent_commands: list[str],
    ) -> RouterResponse:
        if not self._healthy:
            return RouterResponse(
                action=RouterAction.CHAT,
                message="I'm having trouble thinking right now.",
                raw_response="unhealthy",
            )

        if session_id not in self._conversations:
            system_prompt = self._build_system_prompt(
                tool_metas, recent_commands,
            )
            self._conversations[session_id] = [
                {"role": "system", "content": system_prompt},
            ]

        history = self._conversations[session_id]
        history.append({"role": "user", "content": transcript})

        if len(history) > _MAX_HISTORY + 1:
            system_msg = history[0]
            history[:] = [system_msg] + history[-((_MAX_HISTORY)):]

        try:
            import ollama
            client = ollama.AsyncClient(
                host=f"http://{self._host}:{self._port}",
            )
            response = await client.chat(
                model=self._model, messages=history,
            )
            raw = response["message"]["content"]
            history.append({"role": "assistant", "content": raw})
            return self._parse_response(raw)
        except Exception:
            logger.exception("Ollama chat failed")
            return RouterResponse(
                action=RouterAction.CHAT,
                message="Sorry, something went wrong.",
                raw_response="error",
            )

    def add_context(self, session_id: str, context: str) -> None:
        """Inject tool execution results so the LLM has feedback."""
        if session_id in self._conversations:
            self._conversations[session_id].append(
                {"role": "system", "content": context},
            )

    def clear_session(self, session_id: str) -> None:
        self._conversations.pop(session_id, None)

    # ------------------------------------------------------------------
    # Backward-compatible one-shot route (used by tests)
    # ------------------------------------------------------------------

    async def route(
        self, transcript: str, tool_metas: list, recent_commands: list[str],
    ) -> IntentResult:
        resp = await self.chat(
            transcript, "__legacy__", tool_metas, recent_commands,
        )
        self.clear_session("__legacy__")
        if resp.action == RouterAction.EXECUTE:
            return IntentResult(
                tool_name=resp.tool_name,
                params=resp.params,
                confidence=resp.confidence,
                raw_response=resp.raw_response,
            )
        return IntentResult(
            tool_name=None, params={}, confidence=0.0,
            raw_response=resp.raw_response,
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from LLM output that may contain text around it."""
        text = text.strip()
        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        start = text.find("{")
        if start == -1:
            return text
        end = text.rfind("}")
        if end == -1:
            return text
        return text[start:end + 1]

    def _parse_response(self, raw: str) -> RouterResponse:
        try:
            cleaned = self._extract_json(raw)
            data = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.info("LLM returned non-JSON, treating as chat: %s", raw[:200])
            return RouterResponse(
                action=RouterAction.CHAT,
                message=raw.strip()[:200],
                raw_response=raw,
            )

        action_str = str(data.get("action", "")).lower()

        if action_str == "execute":
            return RouterResponse(
                action=RouterAction.EXECUTE,
                tool_name=data.get("tool"),
                params=data.get("params", {}),
                message=data.get("speak", ""),
                confidence=float(data.get("confidence", 0.9)),
                raw_response=raw,
            )
        if action_str == "clarify":
            return RouterResponse(
                action=RouterAction.CLARIFY,
                message=data.get("message", "Could you say that again?"),
                raw_response=raw,
            )
        if action_str == "chat":
            return RouterResponse(
                action=RouterAction.CHAT,
                message=data.get("message", ""),
                raw_response=raw,
            )

        if "tool" in data and data["tool"]:
            return RouterResponse(
                action=RouterAction.EXECUTE,
                tool_name=data["tool"],
                params=data.get("params", {}),
                message=data.get("speak", ""),
                confidence=float(data.get("confidence", 0.9)),
                raw_response=raw,
            )

        msg = data.get("message", data.get("speak", "I'm not sure what you need."))
        return RouterResponse(
            action=RouterAction.CHAT, message=msg, raw_response=raw,
        )
