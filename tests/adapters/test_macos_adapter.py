"""Tests for macOS adapter and AppleScript builders."""

from unittest.mock import AsyncMock, patch

import pytest

from adapters.macos.applescript import (
    build_open_app_script,
    build_switch_window_script,
    build_notification_script,
    build_open_url_script,
    build_say_script,
)
from adapters.macos.adapter import MacOSAdapter


def test_build_open_app_script():
    script = build_open_app_script("IntelliJ IDEA")
    assert 'tell application "IntelliJ IDEA"' in script
    assert "activate" in script


def test_build_switch_window_script():
    script = build_switch_window_script("Safari", "GitHub")
    assert 'tell application' in script or 'tell process' in script
    assert "GitHub" in script


def test_build_notification_script():
    script = build_notification_script("Hello", "World")
    assert "display notification" in script


def test_build_open_url_script():
    script = build_open_url_script("https://example.com", "Chrome")
    assert "https://example.com" in script
    assert "Chrome" in script


def test_build_say_script():
    script = build_say_script("Hello Shubham")
    assert "Hello Shubham" in script


@pytest.mark.asyncio
async def test_adapter_run_script():
    adapter = MacOSAdapter()
    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"result", b"")
        mock_proc.returncode = 0
        mock_exec.return_value = mock_proc

        result = await adapter.run_script('tell application "Finder" to activate')
        assert result == "result"
        mock_exec.assert_called_once()


@pytest.mark.asyncio
async def test_adapter_open_application():
    adapter = MacOSAdapter()
    with patch.object(adapter, "run_script", new_callable=AsyncMock, return_value=""):
        result = await adapter.open_application("Safari")
        assert result is True


@pytest.mark.asyncio
async def test_adapter_send_notification():
    adapter = MacOSAdapter()
    with patch.object(adapter, "run_script", new_callable=AsyncMock, return_value=""):
        await adapter.send_notification("Test", "Body")
