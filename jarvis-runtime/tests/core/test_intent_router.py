"""Tests for intent router."""
import json
from unittest.mock import AsyncMock, patch
import pytest
from core.intent.router import IntentResult, IntentRouter
from tools._base import ToolMeta


@pytest.fixture
def tool_metas():
    return [
        ToolMeta(name="open_project", description="Opens a project in IntelliJ",
                 parameters_schema={"type": "object", "properties": {"project": {"type": "string"}}, "required": ["project"]}),
        ToolMeta(name="open_url", description="Opens a URL in a browser",
                 parameters_schema={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}),
    ]


@pytest.mark.asyncio
async def test_router_parses_valid_response(tool_metas):
    mock_response = {"message": {"content": json.dumps({"tool": "open_project", "params": {"project": "office"}, "confidence": 0.95})}}
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
async def test_router_returns_none_on_low_confidence(tool_metas):
    mock_response = {"message": {"content": json.dumps({"tool": "open_project", "params": {"project": "office"}, "confidence": 0.3})}}
    with patch("ollama.AsyncClient") as MockClient:
        instance = MockClient.return_value
        instance.chat = AsyncMock(return_value=mock_response)
        router = IntentRouter(host="localhost", port=11434, model="llama3.1")
        router._healthy = True
        result = await router.route("something vague", tool_metas, [])
    assert result.tool_name is None


@pytest.mark.asyncio
async def test_router_handles_invalid_json(tool_metas):
    mock_response = {"message": {"content": "I don't understand that request."}}
    with patch("ollama.AsyncClient") as MockClient:
        instance = MockClient.return_value
        instance.chat = AsyncMock(return_value=mock_response)
        router = IntentRouter(host="localhost", port=11434, model="llama3.1")
        router._healthy = True
        result = await router.route("gibberish", tool_metas, [])
    assert result.tool_name is None
