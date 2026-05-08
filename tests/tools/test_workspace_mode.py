"""Tests for workspace mode tool."""
from unittest.mock import AsyncMock

import pytest
from tools._base import ToolResult
from tools.workspace_modes.open_workspace_mode import OpenWorkspaceModeTool


@pytest.mark.asyncio
async def test_workspace_mode_executes_steps_in_order():
    tool = OpenWorkspaceModeTool()
    mock_executor = AsyncMock()
    mock_executor.execute_tool = AsyncMock(return_value=ToolResult(success=True, message="ok"))

    config = {
        "workspace_modes": {
            "office": {
                "description": "Office mode",
                "steps": [
                    {"tool": "open_project", "params": {"project": "office"}},
                    {"tool": "open_url", "params": {"url": "https://mail.google.com", "browser": "Chrome"}},
                ],
            }
        }
    }
    params = {"mode": "office", "_config": config, "_executor": mock_executor}
    adapter = AsyncMock()

    result = await tool.execute(params, adapter)
    assert result.success is True
    assert mock_executor.execute_tool.call_count == 2
    calls = mock_executor.execute_tool.call_args_list
    assert calls[0][1]["tool_name"] == "open_project"
    assert calls[1][1]["tool_name"] == "open_url"


@pytest.mark.asyncio
async def test_workspace_mode_unknown_mode():
    tool = OpenWorkspaceModeTool()
    params = {"mode": "nonexistent", "_config": {"workspace_modes": {}}}
    adapter = AsyncMock()
    result = await tool.execute(params, adapter)
    assert result.success is False


@pytest.mark.asyncio
async def test_workspace_mode_partial_failure():
    tool = OpenWorkspaceModeTool()
    mock_executor = AsyncMock()
    mock_executor.execute_tool = AsyncMock(
        side_effect=[ToolResult(success=True, message="ok"), ToolResult(success=False, message="failed")]
    )
    config = {
        "workspace_modes": {
            "test": {"description": "Test", "steps": [{"tool": "a", "params": {}}, {"tool": "b", "params": {}}]}
        }
    }
    params = {"mode": "test", "_config": config, "_executor": mock_executor}
    adapter = AsyncMock()
    result = await tool.execute(params, adapter)
    assert result.success is True
    assert "1 failed" in result.message
