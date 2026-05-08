"""Tests for ContextObserver — async polling and event emission."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from runtime.event_bus import ActivityType, ContextChangedEvent, EventBus
from sensors.observer.activity_classifier import ActivityClassifier
from sensors.observer.context_observer import ContextObserver


@dataclass
class _Ctx:
    active_app: str = ""
    window_title: str = ""
    browser_url: str = ""
    browser_tab_title: str = ""
    repo_path: str = ""
    git_branch: str = ""
    open_file: str = ""
    timestamp: float = field(default_factory=time.time)


@pytest_asyncio.fixture
async def bus():
    b = EventBus()
    await b.start()
    yield b
    await b.stop()


@pytest.fixture
def mock_detector():
    d = MagicMock()
    d.get_context = AsyncMock(return_value=_Ctx(active_app="Terminal", window_title="bash"))
    return d


@pytest.fixture
def classifier():
    return ActivityClassifier()


# ── Emission tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_emits_event_on_first_poll(bus, mock_detector, classifier):
    """First poll always emits because previous state is empty."""
    events = []
    bus.subscribe(ContextChangedEvent, lambda e: events.append(e))
    await asyncio.sleep(0.05)

    observer = ContextObserver(bus, mock_detector, classifier, poll_interval_sec=0.05)
    await observer._maybe_emit(mock_detector.get_context.return_value)
    await asyncio.sleep(0.05)

    assert len(events) == 1
    assert events[0].app == "Terminal"
    assert events[0].activity_type == ActivityType.TERMINAL


@pytest.mark.asyncio
async def test_no_event_when_context_unchanged(bus, mock_detector, classifier):
    observer = ContextObserver(bus, mock_detector, classifier, poll_interval_sec=0.05)

    events = []
    bus.subscribe(ContextChangedEvent, lambda e: events.append(e))
    await asyncio.sleep(0.05)

    ctx = _Ctx(active_app="Terminal", window_title="bash")
    await observer._maybe_emit(ctx)
    await asyncio.sleep(0.05)
    count_after_first = len(events)

    # Same context again — should NOT emit
    await observer._maybe_emit(ctx)
    await asyncio.sleep(0.05)

    assert len(events) == count_after_first  # no new event


@pytest.mark.asyncio
async def test_emits_on_app_change(bus, mock_detector, classifier):
    observer = ContextObserver(bus, mock_detector, classifier, poll_interval_sec=0.05)

    events = []
    bus.subscribe(ContextChangedEvent, lambda e: events.append(e))
    await asyncio.sleep(0.05)

    await observer._maybe_emit(_Ctx(active_app="Terminal"))
    await asyncio.sleep(0.05)
    first_count = len(events)

    await observer._maybe_emit(_Ctx(active_app="Brave Browser"))
    await asyncio.sleep(0.05)

    assert len(events) == first_count + 1
    assert events[-1].app == "Brave Browser"


@pytest.mark.asyncio
async def test_emits_on_browser_url_change(bus, mock_detector, classifier):
    observer = ContextObserver(bus, mock_detector, classifier, poll_interval_sec=0.05)

    events = []
    bus.subscribe(ContextChangedEvent, lambda e: events.append(e))
    await asyncio.sleep(0.05)

    await observer._maybe_emit(_Ctx(active_app="Brave Browser", browser_url="https://github.com"))
    await asyncio.sleep(0.05)
    first_count = len(events)

    await observer._maybe_emit(
        _Ctx(active_app="Brave Browser", browser_url="https://stackoverflow.com")
    )
    await asyncio.sleep(0.05)

    assert len(events) == first_count + 1
    assert events[-1].browser_url == "https://stackoverflow.com"


@pytest.mark.asyncio
async def test_run_loop_cancels_cleanly(bus, mock_detector, classifier):
    observer = ContextObserver(bus, mock_detector, classifier, poll_interval_sec=0.05)
    task = asyncio.ensure_future(observer.run())
    await asyncio.sleep(0.15)  # a few polls
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    # Should complete without exception


@pytest.mark.asyncio
async def test_current_activity_updates(bus, mock_detector, classifier):
    observer = ContextObserver(bus, mock_detector, classifier)

    assert observer.current_activity == ActivityType.UNKNOWN  # initial state

    await observer._maybe_emit(_Ctx(active_app="PyCharm"))
    await asyncio.sleep(0.05)
    assert observer.current_activity == ActivityType.CODING
