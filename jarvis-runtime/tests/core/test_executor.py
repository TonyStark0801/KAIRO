"""Tests for tool executor."""
import asyncio
from unittest.mock import AsyncMock, MagicMock
import pytest
import pytest_asyncio
from runtime.event_bus import EventBus, IntentRoutedEvent, MemoryWriteEvent, ToolExecutionEvent, ToolResult
from core.registry.executor import ToolExecutor


@pytest_asyncio.fixture
async def bus():
    b = EventBus()
    await b.start()
    yield b
    await b.stop()


@pytest.fixture
def mock_registry():
    registry = MagicMock()
    tool = AsyncMock()
    tool.name = "test_tool"
    tool.execute = AsyncMock(return_value=ToolResult(success=True, message="ok"))
    registry.get.return_value = tool
    return registry


@pytest.fixture
def mock_adapter():
    return AsyncMock()


@pytest.mark.asyncio
async def test_executor_publishes_tool_execution_event(bus, mock_registry, mock_adapter):
    events = []
    bus.subscribe(ToolExecutionEvent, lambda e: events.append(e))
    await asyncio.sleep(0.05)
    executor = ToolExecutor(bus, mock_registry, mock_adapter, timeout=30)
    event = IntentRoutedEvent(tool_name="test_tool", params={"a": 1}, confidence=0.9, session_id="s1")
    await executor.on_intent_routed(event)
    await asyncio.sleep(0.1)
    assert len(events) == 1
    assert events[0].success is True
    assert events[0].tool_name == "test_tool"


@pytest.mark.asyncio
async def test_executor_publishes_memory_write_event(bus, mock_registry, mock_adapter):
    events = []
    bus.subscribe(MemoryWriteEvent, lambda e: events.append(e))
    await asyncio.sleep(0.05)
    executor = ToolExecutor(bus, mock_registry, mock_adapter, timeout=30)
    event = IntentRoutedEvent(tool_name="test_tool", params={}, confidence=0.9, session_id="s1")
    await executor.on_intent_routed(event)
    await asyncio.sleep(0.1)
    assert len(events) == 1
    assert events[0].tool_name == "test_tool"


@pytest.mark.asyncio
async def test_executor_handles_timeout(bus, mock_adapter):
    registry = MagicMock()
    tool = AsyncMock()
    tool.name = "slow_tool"
    async def slow_execute(params, adapter):
        await asyncio.sleep(10)
        return ToolResult(success=True, message="done")
    tool.execute = slow_execute
    registry.get.return_value = tool
    events = []
    bus.subscribe(ToolExecutionEvent, lambda e: events.append(e))
    await asyncio.sleep(0.05)
    executor = ToolExecutor(bus, registry, mock_adapter, timeout=0.1)
    event = IntentRoutedEvent(tool_name="slow_tool", params={}, confidence=0.9, session_id="s1")
    await executor.on_intent_routed(event)
    await asyncio.sleep(0.5)
    assert len(events) == 1
    assert events[0].success is False
    assert "timed out" in events[0].result.message


@pytest.mark.asyncio
async def test_executor_handles_tool_not_found(bus, mock_adapter):
    registry = MagicMock()
    registry.get.return_value = None
    events = []
    bus.subscribe(ToolExecutionEvent, lambda e: events.append(e))
    await asyncio.sleep(0.05)
    executor = ToolExecutor(bus, registry, mock_adapter, timeout=30)
    event = IntentRoutedEvent(tool_name="missing", params={}, confidence=0.9, session_id="s1")
    await executor.on_intent_routed(event)
    await asyncio.sleep(0.1)
    assert len(events) == 1
    assert events[0].success is False
