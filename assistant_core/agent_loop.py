"""Agent loop — iterative tool-calling for Tier 3 cloud LLM."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from adapters.base.platform_adapter import PlatformAdapter
    from core.registry.tool_registry import ToolRegistry
    from llm_router.groq_provider import GroqProvider

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = 3

_CLOUD_TOOLS = {"web_search", "web_browse", "weather", "manage_todos"}


@dataclass
class AgentResult:
    message: str = ""
    tools_used: list[str] = field(default_factory=list)
    interim_messages: list[str] = field(default_factory=list)
    capped: bool = False


class AgentLoop:
    def __init__(
        self,
        groq: GroqProvider,
        tool_registry: ToolRegistry,
        adapter: PlatformAdapter,
    ) -> None:
        self._groq = groq
        self._registry = tool_registry
        self._adapter = adapter

    def _build_tool_schemas(self) -> list[dict[str, Any]]:
        """Only expose internet/task tools to the cloud LLM — keeps schemas small and reliable."""
        schemas = []
        for meta in self._registry.list_all():
            if meta.name not in _CLOUD_TOOLS:
                continue
            schemas.append({
                "type": "function",
                "function": {
                    "name": meta.name,
                    "description": meta.description,
                    "parameters": meta.parameters_schema,
                },
            })
        return schemas

    async def run(
        self,
        system_prompt: str,
        conversation: list[dict[str, str]],
    ) -> AgentResult:
        """Run the agent loop. Returns final response after tool iterations."""
        tools = self._build_tool_schemas()
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation)

        result = AgentResult()
        last_message = ""

        for iteration in range(_MAX_ITERATIONS):
            response = await self._groq.chat_with_tools(
                system_prompt="",  # already in messages
                messages=messages,
                tools=tools,
            )

            if not response.tool_calls:
                # No tools — this is the final response
                result.message = response.message or last_message
                return result

            # LLM wants to call tools
            if response.message:
                result.interim_messages.append(response.message)
                last_message = response.message

            # Build assistant message with tool_calls for Groq format
            assistant_msg = {"role": "assistant", "content": response.message or ""}
            tool_calls_payload = []
            for tc in response.tool_calls:
                tool_calls_payload.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                })
            assistant_msg["tool_calls"] = tool_calls_payload
            messages.append(assistant_msg)

            # Execute each tool and append results
            for tc in response.tool_calls:
                tool = self._registry.get(tc.name)
                if tool is None:
                    tool_result_text = f"Tool '{tc.name}' not found."
                    logger.warning("Agent loop: unknown tool %s", tc.name)
                else:
                    try:
                        tool_result = await tool.execute(tc.arguments, self._adapter)
                        tool_result_text = tool_result.message
                        result.tools_used.append(tc.name)
                        logger.info(
                            "Agent loop: %s → %s",
                            tc.name,
                            "OK" if tool_result.success else "FAIL",
                        )
                    except Exception:
                        logger.exception("Agent loop: tool %s crashed", tc.name)
                        tool_result_text = f"Tool '{tc.name}' failed with an error."

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result_text,
                })

        # Hit max iterations
        result.message = last_message or "I ran out of steps, but here's what I found so far."
        result.capped = True
        return result
