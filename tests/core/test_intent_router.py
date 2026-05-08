"""Tests for conversational intent router."""
import json
from unittest.mock import AsyncMock, patch

import pytest

from core.intent.router import IntentResult, IntentRouter, RouterAction, RouterResponse
from tools._base import ToolMeta


@pytest.fixture
def tool_metas():
    return [
        ToolMeta(
            name="open_project",
            description="Opens a project in IntelliJ",
            parameters_schema={
                "type": "object",
                "properties": {"project": {"type": "string"}},
                "required": ["project"],
            },
        ),
        ToolMeta(
            name="open_url",
            description="Opens a URL in a browser",
            parameters_schema={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        ),
    ]


# ------------------------------------------------------------------
# Backward-compatible route() tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_parses_execute_response(tool_metas):
    payload = json.dumps({
        "action": "execute",
        "tool": "open_project",
        "params": {"project": "office"},
        "confidence": 0.95,
        "speak": "Opening office project",
    })
    mock_response = {"message": {"content": payload}}
    with patch("ollama.AsyncClient") as MockClient:
        instance = MockClient.return_value
        instance.chat = AsyncMock(return_value=mock_response)
        router = IntentRouter(host="localhost", port=11434, model="llama3.1")
        router._healthy = True
        result = await router.route("open the office project", tool_metas, [])
    assert isinstance(result, IntentResult)
    assert result.tool_name == "open_project"
    assert result.params == {"project": "office"}
    assert result.confidence == 0.95


@pytest.mark.asyncio
async def test_router_clarify_maps_to_none_tool(tool_metas):
    payload = json.dumps({
        "action": "clarify",
        "message": "Which project do you want to open?",
    })
    mock_response = {"message": {"content": payload}}
    with patch("ollama.AsyncClient") as MockClient:
        instance = MockClient.return_value
        instance.chat = AsyncMock(return_value=mock_response)
        router = IntentRouter(host="localhost", port=11434, model="llama3.1")
        router._healthy = True
        result = await router.route("open a project", tool_metas, [])
    assert result.tool_name is None


@pytest.mark.asyncio
async def test_router_handles_plain_text(tool_metas):
    mock_response = {"message": {"content": "I don't understand that request."}}
    with patch("ollama.AsyncClient") as MockClient:
        instance = MockClient.return_value
        instance.chat = AsyncMock(return_value=mock_response)
        router = IntentRouter(host="localhost", port=11434, model="llama3.1")
        router._healthy = True
        result = await router.route("gibberish", tool_metas, [])
    assert result.tool_name is None


# ------------------------------------------------------------------
# Multi-turn chat() tests
# ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_returns_execute(tool_metas):
    payload = json.dumps({
        "action": "execute",
        "tool": "open_url",
        "params": {"url": "https://youtube.com", "browser": "Chrome"},
        "confidence": 0.9,
        "speak": "Opening YouTube",
    })
    mock_response = {"message": {"content": payload}}
    with patch("ollama.AsyncClient") as MockClient:
        instance = MockClient.return_value
        instance.chat = AsyncMock(return_value=mock_response)
        router = IntentRouter(host="localhost", port=11434, model="llama3.1")
        router._healthy = True
        resp = await router.chat("open YouTube", "sess1", tool_metas, [])
    assert resp.action == RouterAction.EXECUTE
    assert resp.tool_name == "open_url"
    assert resp.message == "Opening YouTube"


@pytest.mark.asyncio
async def test_chat_returns_clarify(tool_metas):
    payload = json.dumps({
        "action": "clarify",
        "message": "What would you like to play?",
    })
    mock_response = {"message": {"content": payload}}
    with patch("ollama.AsyncClient") as MockClient:
        instance = MockClient.return_value
        instance.chat = AsyncMock(return_value=mock_response)
        router = IntentRouter(host="localhost", port=11434, model="llama3.1")
        router._healthy = True
        resp = await router.chat("play music", "sess1", tool_metas, [])
    assert resp.action == RouterAction.CLARIFY
    assert "play" in resp.message.lower()


@pytest.mark.asyncio
async def test_chat_maintains_conversation_history(tool_metas):
    responses = [
        {"message": {"content": json.dumps({"action": "clarify", "message": "Which song?"})}},
        {"message": {"content": json.dumps({"action": "execute", "tool": "open_url", "params": {"url": "https://youtube.com/results?search_query=rock"}, "speak": "Playing rock"})}},
    ]
    with patch("ollama.AsyncClient") as MockClient:
        instance = MockClient.return_value
        instance.chat = AsyncMock(side_effect=responses)
        router = IntentRouter(host="localhost", port=11434, model="llama3.1")
        router._healthy = True

        resp1 = await router.chat("play music", "sess1", tool_metas, [])
        assert resp1.action == RouterAction.CLARIFY

        resp2 = await router.chat("rock music on youtube", "sess1", tool_metas, [])
        assert resp2.action == RouterAction.EXECUTE

    assert len(router._conversations["sess1"]) == 5  # system + 2 user + 2 assistant


@pytest.mark.asyncio
async def test_clear_session(tool_metas):
    payload = json.dumps({"action": "chat", "message": "Hi!"})
    mock_response = {"message": {"content": payload}}
    with patch("ollama.AsyncClient") as MockClient:
        instance = MockClient.return_value
        instance.chat = AsyncMock(return_value=mock_response)
        router = IntentRouter(host="localhost", port=11434, model="llama3.1")
        router._healthy = True
        await router.chat("hello", "sess1", tool_metas, [])
    assert "sess1" in router._conversations
    router.clear_session("sess1")
    assert "sess1" not in router._conversations


@pytest.mark.asyncio
async def test_old_format_backward_compat(tool_metas):
    """Old-format JSON without 'action' key still works."""
    payload = json.dumps({
        "tool": "open_project",
        "params": {"project": "office"},
        "confidence": 0.95,
    })
    mock_response = {"message": {"content": payload}}
    with patch("ollama.AsyncClient") as MockClient:
        instance = MockClient.return_value
        instance.chat = AsyncMock(return_value=mock_response)
        router = IntentRouter(host="localhost", port=11434, model="llama3.1")
        router._healthy = True
        result = await router.route("open office", tool_metas, [])
    assert result.tool_name == "open_project"


@pytest.mark.asyncio
async def test_markdown_fences_stripped(tool_metas):
    wrapped = '```json\n{"action": "execute", "tool": "open_url", "params": {"url": "https://google.com"}, "speak": "Done"}\n```'
    mock_response = {"message": {"content": wrapped}}
    with patch("ollama.AsyncClient") as MockClient:
        instance = MockClient.return_value
        instance.chat = AsyncMock(return_value=mock_response)
        router = IntentRouter(host="localhost", port=11434, model="llama3.1")
        router._healthy = True
        result = await router.route("open google", tool_metas, [])
    assert result.tool_name == "open_url"
