"""Reasoner — 2-tier brain: keyword fast-path + single local LLM.

Tier 1: Keyword fast-path  → instant (0ms)
Tier 2: Local model        → handles everything including web_search via inline loop

No Groq. No escalation. One model does it all.
If the model emits web_search, we execute it locally (DDGS) and re-call the model
once for synthesis. search_fn is injected at construction so Reasoner stays
decoupled from the tool registry.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Awaitable, Callable, TYPE_CHECKING

from assistant_core.fast_path import FastPathResult, try_fast_path

if TYPE_CHECKING:
    from assistant_core.personality import Personality
    from llm_router.protocol import LocalChatProvider
    from memory.behavioral.query import BehavioralQuery
    from memory.vector.client import VectorMemoryClient
    from memory_service.preferences import PreferencesMemory
    from memory_service.session_store import SessionStore
    from tools._base import ToolMeta

logger = logging.getLogger(__name__)


class ReasonerAction(Enum):
    EXECUTE = auto()
    SPEAK = auto()
    SPEAK_AND_EXECUTE = auto()


@dataclass
class ReasonerResponse:
    action: ReasonerAction
    tool_name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    confidence: float = 0.0
    raw: str = ""
    tier: int = 0
    interim_messages: list[str] = field(default_factory=list)


class Reasoner:
    def __init__(
        self,
        personality: Personality,
        llm: LocalChatProvider,
        preferences: PreferencesMemory,
        session_store: SessionStore,
        synthesis_tools: dict[str, Callable[[dict], Awaitable[str]]] | None = None,
        behavioral_query: BehavioralQuery | None = None,
        vector_client: VectorMemoryClient | None = None,
        # Kept for backwards-compat — silently ignored.
        groq: Any = None,
        agent_loop: Any = None,
        search_fn: Any = None,
    ) -> None:
        self._personality = personality
        self._llm = llm
        self._preferences = preferences
        self._session_store = session_store
        # synthesis_tools: {tool_name: async fn(params) -> str}
        # Tools whose raw output should be fed back to the LLM for a spoken synthesis response.
        # Examples: web_search (DDGS results), terminal_command (ls/find output).
        self._synthesis_tools: dict[str, Callable[[dict], Awaitable[str]]] = synthesis_tools or {}
        self._behavioral_query = behavioral_query
        self._vector_client = vector_client
        self._conversations: dict[str, list[dict[str, str]]] = {}
        self._session_tools: dict[str, list[str]] = {}
        self._cached_system_prompt: str | None = None  # single prompt, not per-session
        self._media_playing = False

    def set_media_playing(self, playing: bool) -> None:
        self._media_playing = playing

    async def process(
        self,
        transcript: str,
        session_id: str,
        tool_metas: list[ToolMeta],
        recent_commands: list[str] | None = None,
        *,
        use_llm: bool = True,
        tone_hint: str | None = None,
    ) -> ReasonerResponse:
        if not self._llm.healthy:
            return ReasonerResponse(
                action=ReasonerAction.SPEAK,
                message="I'm having trouble thinking right now.",
            )

        resolved = await self._preferences.resolve_alias(transcript)

        # --- Tier 1: keyword fast-path ---
        fast = try_fast_path(resolved, media_playing=self._media_playing)
        if fast:
            logger.info("Tier 1 (fast-path): %s → %s", resolved, fast.tool_name)
            self._session_tools.setdefault(session_id, []).append(fast.tool_name)
            return ReasonerResponse(
                action=ReasonerAction.EXECUTE,
                tool_name=fast.tool_name,
                params=fast.params,
                message=fast.message,
                confidence=1.0,
                tier=1,
            )

        if not use_llm:
            return ReasonerResponse(
                action=ReasonerAction.SPEAK,
                message="I can only handle quick commands for that — language reasoning is off for this turn.",
                tier=0,
            )

        # --- Tier 2: single local LLM handles everything ---
        if session_id not in self._conversations:
            self._conversations[session_id] = []

        history = self._conversations[session_id]
        history.append({"role": "user", "content": resolved})

        # Build (or reuse) the system prompt. Invalidated externally via invalidate_prompt_cache().
        if self._cached_system_prompt is None:
            self._cached_system_prompt = await self._personality.build_system_prompt(tool_metas)
        system_prompt = self._with_tone(self._cached_system_prompt, tone_hint)

        raw = await self._llm.chat(system_prompt, history)
        if not raw:
            return ReasonerResponse(action=ReasonerAction.SPEAK, message="Sorry, something went wrong.")

        response = self._parse(raw, session_id)
        response.tier = 2

        # --- Synthesis tool loop ---
        # For tools that return raw data (search results, terminal output, etc.),
        # execute them locally, inject the result, re-call the LLM once for a
        # spoken synthesis. One round-trip max — no recursion risk on a 3b model.
        tool_name = response.tool_name or ""
        if (
            response.action in (ReasonerAction.EXECUTE, ReasonerAction.SPEAK_AND_EXECUTE)
            and tool_name in self._synthesis_tools
        ):
            synthesis_fn = self._synthesis_tools[tool_name]
            logger.info("Synthesis loop: tool=%s params=%s", tool_name, response.params)
            try:
                raw_result = await synthesis_fn(response.params)
            except Exception:
                logger.exception("Synthesis tool %r failed", tool_name)
                raw_result = f"Tool '{tool_name}' failed."

            self._session_tools.setdefault(session_id, []).append(tool_name)
            history.append({"role": "assistant", "content": raw})
            history.append({
                "role": "system",
                "content": (
                    f"Result from {tool_name}:\n{raw_result}\n\n"
                    "Using only the above result, give a concise spoken answer to the user's question. "
                    "Do not make up information not present in the result."
                ),
            })

            synthesis_raw = await self._llm.chat(system_prompt, history)
            if synthesis_raw:
                history.append({"role": "assistant", "content": synthesis_raw})
                synthesis = self._parse(synthesis_raw, session_id)
                synthesis.tier = 2
                logger.info("Synthesis response (%s): '%s'", tool_name, synthesis.message[:80])
                return synthesis

            return ReasonerResponse(
                action=ReasonerAction.SPEAK,
                message="I got the results but had trouble summarising them.",
                tier=2,
            )

        logger.info("Tier 2 (%s): action=%s tool=%s", self._llm.fast_model, response.action.name, response.tool_name)
        history.append({"role": "assistant", "content": raw})
        return response

    @staticmethod
    def _with_tone(system_prompt: str, tone_hint: str | None) -> str:
        if not tone_hint:
            return system_prompt
        return f"{system_prompt}\n\nTone for this reply: {tone_hint}"

    def invalidate_prompt_cache(self) -> None:
        """Force rebuild of the system prompt on the next process() call.

        Call this whenever context changes (active app switch, new session context, etc.)
        so the model gets an up-to-date environment snapshot.
        """
        self._cached_system_prompt = None

    def inject_tool_result(self, session_id: str, tool_name: str, result_text: str) -> None:
        if session_id in self._conversations:
            self._conversations[session_id].append(
                {"role": "system", "content": f"Tool '{tool_name}' result: {result_text}"}
            )
        self._session_tools.setdefault(session_id, []).append(tool_name)

    def clear_session(self, session_id: str) -> None:
        self._conversations.pop(session_id, None)
        self._session_tools.pop(session_id, None)
        # Note: _cached_system_prompt is shared across sessions (not per-session).
        # Use invalidate_prompt_cache() explicitly if you want to force a rebuild.

    def get_session_tools(self, session_id: str) -> list[str]:
        return self._session_tools.get(session_id, [])

    async def generate_session_summary(self, session_id: str) -> str:
        history = self._conversations.get(session_id, [])
        if len(history) < 2:
            return ""

        summary_prompt = (
            "Summarize this conversation in one short sentence for future reference. "
            "Focus on what the user wanted and what was done."
        )
        messages = history + [{"role": "user", "content": summary_prompt}]

        try:
            raw = await self._llm.chat(
                "You are a session summarizer. Output only a single sentence summary.",
                messages,
                model_override=self._llm.fast_model,
            )
            return raw.strip()[:200]
        except Exception:
            logger.exception("Failed to generate session summary")
            return ""

    def _parse(self, raw: str, session_id: str) -> ReasonerResponse:
        try:
            cleaned = self._extract_json(raw)
            data = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.info("LLM returned non-JSON, treating as speech: %s", raw[:200])
            return ReasonerResponse(
                action=ReasonerAction.SPEAK,
                message=raw.strip()[:200],
                raw=raw,
            )

        # Safety net: if message field contains nested JSON, unwrap it
        data = self._unwrap_nested_json(data)

        action_str = str(data.get("action", "")).lower()

        if action_str == "execute":
            return ReasonerResponse(
                action=ReasonerAction.EXECUTE,
                tool_name=data.get("tool"),
                params=data.get("params", {}),
                message=data.get("message", ""),
                confidence=float(data.get("confidence", 0.9)),
                raw=raw,
            )

        if action_str == "speak_and_execute":
            return ReasonerResponse(
                action=ReasonerAction.SPEAK_AND_EXECUTE,
                tool_name=data.get("tool"),
                params=data.get("params", {}),
                message=data.get("message", ""),
                confidence=float(data.get("confidence", 0.9)),
                raw=raw,
            )

        if action_str == "speak":
            msg = data.get("message", "")
            # If the "speak" message itself looks like a JSON action, unwrap it
            if isinstance(msg, str) and msg.strip().startswith("{"):
                try:
                    inner = json.loads(msg)
                    if isinstance(inner, dict) and "action" in inner:
                        return self._parse(msg, session_id)
                except (json.JSONDecodeError, ValueError):
                    pass
            return ReasonerResponse(
                action=ReasonerAction.SPEAK,
                message=msg,
                raw=raw,
            )

        if action_str == "clarify":
            return ReasonerResponse(
                action=ReasonerAction.SPEAK,
                message=data.get("message", "Could you say that again?"),
                raw=raw,
            )

        if "tool" in data and data["tool"]:
            return ReasonerResponse(
                action=ReasonerAction.EXECUTE,
                tool_name=data["tool"],
                params=data.get("params", {}),
                message=data.get("message", ""),
                confidence=float(data.get("confidence", 0.9)),
                raw=raw,
            )

        msg = data.get("message", data.get("speak", "I'm not sure what you need."))
        return ReasonerResponse(action=ReasonerAction.SPEAK, message=msg, raw=raw)

    @staticmethod
    def _unwrap_nested_json(data: dict) -> dict:
        """If the 3b model wrapped its JSON response inside a speak message, unwrap it."""
        if data.get("action") == "speak":
            msg = data.get("message", "")
            if isinstance(msg, str) and msg.strip().startswith("{"):
                try:
                    inner = json.loads(msg.strip())
                    if isinstance(inner, dict) and "action" in inner:
                        return inner
                except (json.JSONDecodeError, ValueError):
                    pass
        return data

    @staticmethod
    def _extract_json(text: str) -> str:
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
