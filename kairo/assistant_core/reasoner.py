"""Reasoner — 3-tier brain with fast-path, small model, and cloud agent loop.

Tier 1: Keyword fast-path  → instant (0ms)
Tier 2: Fast model (3b)    → intent routing (~0.5-1s)
Tier 3: Cloud agent loop   → Groq 70B with tool calling (~1-3s)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, TYPE_CHECKING

from assistant_core.fast_path import FastPathResult, try_fast_path

if TYPE_CHECKING:
    from assistant_core.agent_loop import AgentLoop
    from assistant_core.personality import Personality
    from llm_router.groq_provider import GroqProvider
    from llm_router.protocol import LocalChatProvider
    from memory.behavioral.query import BehavioralQuery
    from memory.vector.client import VectorMemoryClient
    from memory_service.preferences import PreferencesMemory
    from memory_service.session_store import SessionStore
    from tools._base import ToolMeta

logger = logging.getLogger(__name__)

_DEEP_TRIGGERS = re.compile(
    r"\b(explain\s+\w|why do you|what do you think|tell me about\s+\w"
    r"|help me understand|how does\s+\w|what happened|analyze\s+\w|compare\s+\w)\b",
    re.IGNORECASE,
)

_ESCALATE_TURN_THRESHOLD = 8

_CLOUD_ONLY_TOOLS = {"web_search", "web_browse", "weather", "manage_todos"}


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
        groq: GroqProvider | None = None,
        agent_loop: AgentLoop | None = None,
        behavioral_query: BehavioralQuery | None = None,
        vector_client: VectorMemoryClient | None = None,
    ) -> None:
        self._personality = personality
        self._llm = llm
        self._preferences = preferences
        self._session_store = session_store
        self._groq = groq
        self._agent_loop = agent_loop
        self._behavioral_query = behavioral_query
        self._vector_client = vector_client
        self._conversations: dict[str, list[dict[str, str]]] = {}
        self._session_tools: dict[str, list[str]] = {}
        self._cached_prompts: dict[str, str] = {}
        self._cached_fast_prompt: str | None = None
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

        # --- Prepare conversation history ---
        if session_id not in self._conversations:
            self._conversations[session_id] = []

        history = self._conversations[session_id]
        history.append({"role": "user", "content": resolved})

        # --- Decide: Tier 2 (fast) or Tier 3 (deep) ---
        use_deep = self._should_use_deep_model(resolved, session_id)

        if use_deep:
            return await self._run_tier3(
                resolved, session_id, tool_metas, history, tone_hint=tone_hint,
            )

        # --- Tier 2: fast model (exclude cloud-only tools) ---
        tier2_metas = [t for t in tool_metas if t.name not in _CLOUD_ONLY_TOOLS]
        if self._cached_fast_prompt is None:
            self._cached_fast_prompt = await self._personality.build_fast_prompt(tier2_metas)
        system_prompt = self._with_tone(self._cached_fast_prompt, tone_hint)
        model = self._llm.fast_model

        raw = await self._llm.chat(system_prompt, history, model_override=model)
        if not raw:
            return ReasonerResponse(
                action=ReasonerAction.SPEAK,
                message="Sorry, something went wrong.",
            )

        response = self._parse(raw, session_id)
        response.tier = 2

        # --- Tier 2 → Tier 3 escalation ---
        # Escalate if: explicit escalate, OR speak (Tier 2 should never speak — all conversation goes to Tier 3)
        if self._needs_escalation(response, raw) or response.action == ReasonerAction.SPEAK:
            logger.info("Tier 2 → Tier 3 escalation (action=%s)", response.action.name)
            return await self._run_tier3(
                resolved, session_id, tool_metas, history, tone_hint=tone_hint,
            )

        logger.info("Tier 2 (%s): action=%s tool=%s",
                     model, response.action.name, response.tool_name)

        history.append({"role": "assistant", "content": raw})
        return response

    @staticmethod
    def _with_tone(system_prompt: str, tone_hint: str | None) -> str:
        if not tone_hint:
            return system_prompt
        return f"{system_prompt}\n\nTone for this reply: {tone_hint}"

    async def _run_tier3(
        self,
        transcript: str,
        session_id: str,
        tool_metas: list[ToolMeta],
        history: list[dict[str, str]],
        *,
        tone_hint: str | None = None,
    ) -> ReasonerResponse:
        """Tier 3: Cloud agent loop (Groq) or local deep fallback."""
        if self._groq and self._groq.healthy and self._agent_loop:
            try:
                cloud_prompt = await self._personality.build_cloud_prompt(
                    tool_metas, transcript,
                    behavioral_query=self._behavioral_query,
                    vector_client=self._vector_client,
                )
                if tone_hint:
                    cloud_prompt = f"{cloud_prompt}\n\nTone: {tone_hint}"
                agent_result = await self._agent_loop.run(cloud_prompt, history)

                for tool_name in agent_result.tools_used:
                    self._session_tools.setdefault(session_id, []).append(tool_name)

                history.append({"role": "assistant", "content": agent_result.message})
                logger.info("Tier 3 (cloud): msg='%s' tools=%s",
                            agent_result.message[:80], agent_result.tools_used)

                return ReasonerResponse(
                    action=ReasonerAction.SPEAK,
                    message=agent_result.message,
                    tier=3,
                    interim_messages=agent_result.interim_messages,
                )
            except Exception:
                logger.exception("Cloud agent loop failed, falling back to local")

        # Fallback: local deep model
        if session_id not in self._cached_prompts:
            self._cached_prompts[session_id] = await self._personality.build_system_prompt(
                tool_metas,
            )
        deep_prompt = self._with_tone(self._cached_prompts[session_id], tone_hint)
        raw = await self._llm.chat(deep_prompt, history, model_override=self._llm.deep_model)
        if raw:
            response = self._parse(raw, session_id)
            response.tier = 3
            history.append({"role": "assistant", "content": raw})
            logger.info("Tier 3 (local fallback): action=%s tool=%s",
                         response.action.name, response.tool_name)
            return response

        return ReasonerResponse(
            action=ReasonerAction.SPEAK,
            message="I'm having trouble right now. Let me try again later.",
            tier=3,
        )

    def _should_use_deep_model(self, transcript: str, session_id: str) -> bool:
        if _DEEP_TRIGGERS.search(transcript):
            return True

        turn_count = len(self._conversations.get(session_id, []))
        if turn_count >= _ESCALATE_TURN_THRESHOLD:
            return True

        return False

    def _needs_escalation(self, response: ReasonerResponse, raw: str) -> bool:
        try:
            cleaned = self._extract_json(raw)
            data = json.loads(cleaned)
            if str(data.get("action", "")).lower() == "escalate":
                return True
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        return False

    def inject_tool_result(self, session_id: str, tool_name: str, result_text: str) -> None:
        if session_id in self._conversations:
            self._conversations[session_id].append(
                {"role": "system", "content": f"Tool '{tool_name}' result: {result_text}"}
            )
        self._session_tools.setdefault(session_id, []).append(tool_name)

    def clear_session(self, session_id: str) -> None:
        self._conversations.pop(session_id, None)
        self._session_tools.pop(session_id, None)
        self._cached_prompts.pop(session_id, None)

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
