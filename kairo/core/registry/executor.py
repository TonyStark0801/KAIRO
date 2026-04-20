"""Tool executor — runs tools with timeout, publishes results."""
from __future__ import annotations
import asyncio
import logging
import time
from typing import TYPE_CHECKING
from runtime.event_bus import IntentRoutedEvent, MemoryWriteEvent, ToolCancelEvent, ToolExecutionEvent, ToolResult

if TYPE_CHECKING:
    from adapters.base.platform_adapter import PlatformAdapter
    from core.registry.tool_registry import ToolRegistry
    from runtime.event_bus import EventBus

logger = logging.getLogger(__name__)


class ToolExecutor:
    def __init__(self, event_bus: EventBus, registry: ToolRegistry, adapter: PlatformAdapter, timeout: float = 30.0) -> None:
        self._bus = event_bus
        self._registry = registry
        self._adapter = adapter
        self._timeout = timeout
        self._current_task: asyncio.Task | None = None

    async def on_intent_routed(self, event: IntentRoutedEvent) -> None:
        tool = self._registry.get(event.tool_name)
        if tool is None:
            await self._publish_result(
                event.tool_name,
                ToolResult(success=False, message=f"Tool not found: {event.tool_name}"),
                event.session_id,
                event.params,
                persist_memory=event.persist_memory,
            )
            return
        try:
            self._current_task = asyncio.current_task()
            result = await asyncio.wait_for(tool.execute(event.params, self._adapter), timeout=self._timeout)
        except asyncio.TimeoutError:
            result = ToolResult(success=False, message=f"Tool '{event.tool_name}' timed out")
        except asyncio.CancelledError:
            result = ToolResult(success=False, message="cancelled by user")
        except Exception as e:
            logger.exception("Unexpected executor error for %s", event.tool_name)
            result = ToolResult(success=False, message=str(e))
        finally:
            self._current_task = None
        await self._publish_result(
            event.tool_name, result, event.session_id, event.params,
            persist_memory=event.persist_memory,
        )

    async def on_cancel(self, event: ToolCancelEvent) -> None:
        if self._current_task is not None and not self._current_task.done():
            self._current_task.cancel()
            logger.info("Tool execution cancelled: %s", event.reason)

    async def execute_tool(self, tool_name: str, params: dict, adapter=None) -> ToolResult:
        tool = self._registry.get(tool_name)
        if tool is None:
            return ToolResult(success=False, message=f"Tool not found: {tool_name}")
        try:
            return await tool.execute(params, adapter or self._adapter)
        except Exception as e:
            return ToolResult(success=False, message=str(e))

    async def _publish_result(
        self,
        tool_name: str,
        result: ToolResult,
        session_id: str,
        params: dict,
        *,
        persist_memory: bool = True,
    ) -> None:
        await self._bus.publish(ToolExecutionEvent(tool_name=tool_name, success=result.success, result=result, session_id=session_id))
        if persist_memory:
            await self._bus.publish(MemoryWriteEvent(tool_name=tool_name, command_text=f"{tool_name} {params}", params=params, session_id=session_id, timestamp=time.time()))
