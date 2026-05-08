"""Tests for the Reasoner — central brain of the assistant."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from assistant_core.reasoner import Reasoner, ReasonerAction, ReasonerResponse
from tools._base import ToolMeta


@pytest.fixture
def tool_metas():
    return [
        ToolMeta(
            name="youtube_search",
            description="Search YouTube",
            parameters_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        ),
        ToolMeta(
            name="open_url",
            description="Opens a URL",
            parameters_schema={"type": "object", "properties": {"url": {"type": "string"}}},
        ),
    ]


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.healthy = True
    llm.fast_model = "fast-m"
    llm.deep_model = "deep-m"
    llm.chat = AsyncMock(return_value="")
    return llm


@pytest.fixture
def mock_personality():
    p = MagicMock()
    p.build_system_prompt = AsyncMock(return_value="You are Kairo.")
    # Reasoner.process() awaits build_fast_prompt (Tier 2) and build_cloud_prompt
    # (Tier 3). All async methods on the personality must be AsyncMock or the
    # test fails with "MagicMock object can't be awaited".
    p.build_fast_prompt = AsyncMock(return_value="You are Kairo (fast).")
    p.build_cloud_prompt = AsyncMock(return_value="You are Kairo (cloud).")
    p.generate_greeting = AsyncMock(return_value="Hey, what's up?")
    return p


@pytest.fixture
def mock_preferences():
    p = MagicMock()
    p.resolve_alias = AsyncMock(side_effect=lambda t: t)
    return p


@pytest.fixture
def mock_session_store():
    return MagicMock()


@pytest.fixture
def reasoner(mock_personality, mock_llm, mock_preferences, mock_session_store):
    return Reasoner(mock_personality, mock_llm, mock_preferences, mock_session_store)


@pytest.mark.asyncio
async def test_execute_response(reasoner, mock_llm, tool_metas):
    mock_llm.chat = AsyncMock(return_value=json.dumps({
        "action": "execute",
        "tool": "youtube_search",
        "params": {"query": "chill vibes"},
        "confidence": 0.9,
    }))
    resp = await reasoner.process("play chill vibes", "s1", tool_metas)
    assert resp.action == ReasonerAction.EXECUTE
    assert resp.tool_name == "youtube_search"
    assert resp.params == {"query": "chill vibes"}


@pytest.mark.asyncio
async def test_speak_response(reasoner, mock_llm, tool_metas):
    mock_llm.chat = AsyncMock(return_value=json.dumps({
        "action": "speak",
        "message": "Hey, what's up?",
    }))
    resp = await reasoner.process("hello", "s1", tool_metas)
    assert resp.action == ReasonerAction.SPEAK
    assert "what's up" in resp.message


@pytest.mark.asyncio
async def test_speak_and_execute_response(reasoner, mock_llm, tool_metas):
    mock_llm.chat = AsyncMock(return_value=json.dumps({
        "action": "speak_and_execute",
        "tool": "youtube_search",
        "params": {"query": "lofi"},
        "message": "On it.",
    }))
    resp = await reasoner.process("play some lofi", "s1", tool_metas)
    assert resp.action == ReasonerAction.SPEAK_AND_EXECUTE
    assert resp.tool_name == "youtube_search"
    assert resp.message == "On it."


@pytest.mark.asyncio
async def test_plain_text_becomes_speak(reasoner, mock_llm, tool_metas):
    mock_llm.chat = AsyncMock(return_value="I'm not sure what you mean.")
    resp = await reasoner.process("asdfgh", "s1", tool_metas)
    assert resp.action == ReasonerAction.SPEAK


@pytest.mark.asyncio
async def test_markdown_fences_stripped(reasoner, mock_llm, tool_metas):
    wrapped = '```json\n{"action": "execute", "tool": "open_url", "params": {"url": "https://google.com"}}\n```'
    mock_llm.chat = AsyncMock(return_value=wrapped)
    resp = await reasoner.process("open google", "s1", tool_metas)
    assert resp.action == ReasonerAction.EXECUTE
    assert resp.tool_name == "open_url"


@pytest.mark.asyncio
async def test_conversation_history_maintained(reasoner, mock_llm, tool_metas):
    # Use EXECUTE responses so Tier 2 doesn't escalate to Tier 3 (which would
    # consume extra llm.chat calls and break the side_effect sequence).
    mock_llm.chat = AsyncMock(side_effect=[
        json.dumps({"action": "execute", "tool": "youtube_search", "params": {"query": "something"}}),
        json.dumps({"action": "execute", "tool": "youtube_search", "params": {"query": "blinding lights"}}),
    ])
    await reasoner.process("play something", "s1", tool_metas)
    await reasoner.process("blinding lights", "s1", tool_metas)
    # Each turn appends 1 user + 1 assistant entry = 4 total after 2 turns
    assert len(reasoner._conversations["s1"]) == 4


@pytest.mark.asyncio
async def test_clear_session(reasoner, mock_llm, tool_metas):
    mock_llm.chat = AsyncMock(return_value=json.dumps({"action": "speak", "message": "Hi"}))
    await reasoner.process("hi", "s1", tool_metas)
    assert "s1" in reasoner._conversations
    reasoner.clear_session("s1")
    assert "s1" not in reasoner._conversations


@pytest.mark.asyncio
async def test_inject_tool_result(reasoner, mock_llm, tool_metas):
    mock_llm.chat = AsyncMock(return_value=json.dumps({"action": "speak", "message": "ok"}))
    await reasoner.process("test", "s1", tool_metas)
    reasoner.inject_tool_result("s1", "youtube_search", "Found 5 results")
    history = reasoner._conversations["s1"]
    assert any("youtube_search" in msg["content"] for msg in history if msg["role"] == "system")


@pytest.mark.asyncio
async def test_unhealthy_llm_returns_speak(reasoner, mock_llm, tool_metas):
    mock_llm.healthy = False
    resp = await reasoner.process("hello", "s1", tool_metas)
    assert resp.action == ReasonerAction.SPEAK
    assert "trouble" in resp.message


@pytest.mark.asyncio
async def test_use_llm_false_skips_llm(reasoner, mock_llm, tool_metas):
    mock_llm.chat = AsyncMock(return_value=json.dumps({
        "action": "speak", "message": "should not run",
    }))
    resp = await reasoner.process("hello", "s1", tool_metas, use_llm=False)
    assert resp.action == ReasonerAction.SPEAK
    assert "quick commands" in resp.message.lower() or "reasoning" in resp.message.lower()
    mock_llm.chat.assert_not_called()


@pytest.mark.asyncio
async def test_alias_resolution(reasoner, mock_llm, mock_preferences, tool_metas):
    mock_preferences.resolve_alias = AsyncMock(return_value="open payments-service project")
    mock_llm.chat = AsyncMock(return_value=json.dumps({
        "action": "execute",
        "tool": "open_url",
        "params": {"url": "https://example.com"},
    }))
    await reasoner.process("open payments", "s1", tool_metas)
    mock_preferences.resolve_alias.assert_called_once_with("open payments")
