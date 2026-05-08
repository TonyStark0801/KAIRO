"""Tests for context detector."""

from unittest.mock import AsyncMock, patch

import pytest

from context_service.detector import ContextDetector, WorkspaceContext


def test_workspace_context_summary():
    ctx = WorkspaceContext(
        active_app="IntelliJ IDEA",
        repo_path="/Users/me/Projects/payments-service",
        git_branch="fix-timeout",
        open_file="PaymentController.java",
    )
    summary = ctx.summary()
    assert "payments-service" in summary
    assert "fix-timeout" in summary

    natural = ctx.natural_summary()
    assert "payments-service" in natural
    assert "fix-timeout" in natural


def test_empty_context_summary():
    ctx = WorkspaceContext()
    assert ctx.summary() == "no workspace detected"
    assert ctx.natural_summary() == ""


@pytest.mark.asyncio
async def test_get_context_caching():
    detector = ContextDetector()
    # Patch _run_osascript (used for window detection) AND _get_browser_tab
    # (Phase 2 addition — also calls osascript when active app is a browser).
    with patch.object(detector, "_run_osascript", new_callable=AsyncMock) as mock_osa, \
         patch.object(detector, "_get_browser_tab", new_callable=AsyncMock) as mock_tab:
        mock_osa.return_value = "Brave Browser|YouTube - Google Chrome"
        mock_tab.return_value = ("https://youtube.com", "YouTube")
        ctx1 = await detector.get_context()
        ctx2 = await detector.get_context()
        # _run_osascript should only fire once — second call hits the cache
        assert mock_osa.call_count == 1
        assert ctx1.active_app == "Brave Browser"
        assert ctx1.browser_url == "https://youtube.com"


@pytest.mark.asyncio
async def test_ide_context_parsing():
    detector = ContextDetector()
    with patch.object(detector, "_run_osascript", new_callable=AsyncMock) as mock_osa:
        mock_osa.return_value = "IntelliJ IDEA|PaymentController.java – payments-service"
        with patch.object(detector, "_get_git_branch", new_callable=AsyncMock) as mock_git:
            mock_git.return_value = "main"
            ctx = await detector.get_context()
            assert ctx.active_app == "IntelliJ IDEA"
            assert ctx.open_file == "PaymentController.java"
