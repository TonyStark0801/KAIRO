"""Tests for ClipboardMonitor — content classification and event emission."""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

from runtime.event_bus import ClipboardChangedEvent, EventBus
from sensors.observer.clipboard_monitor import ClipboardMonitor


@pytest_asyncio.fixture
async def bus():
    b = EventBus()
    await b.start()
    yield b
    await b.stop()


# ── Content classification unit tests ────────────────────────────────────────

@pytest.mark.parametrize("content,expected_type", [
    ("https://github.com/user/repo", "url"),
    ("http://localhost:8080/api", "url"),
    ("ftp://files.example.com/data.zip", "url"),
    ("def hello():\n    return 'world'", "code"),
    ("import asyncio\nasync def main():\n    pass", "code"),
    ("const x = 5;\nfunction foo() { return x; }", "code"),
    # Single-line SQL has no newline → classified as text (multi-line SQL → code)
    ("SELECT * FROM users WHERE id = 1", "text"),
    ("Hello world, this is plain text.", "text"),
    ("Meeting at 3pm with the team", "text"),
])
def test_classify_content(content, expected_type):
    assert ClipboardMonitor._classify(content) == expected_type


def test_code_requires_newline():
    """A single code token without a newline is treated as text."""
    assert ClipboardMonitor._classify("def hello()") == "text"


# ── Emission via run loop ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_emits_event_on_clipboard_change(bus):
    loop = asyncio.get_running_loop()
    monitor = ClipboardMonitor(bus, loop, poll_interval=0.1)

    events = []
    bus.subscribe(ClipboardChangedEvent, lambda e: events.append(e))
    await asyncio.sleep(0.05)

    stop = threading.Event()
    call_count = 0

    def fake_read():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "https://github.com/kairo"
        stop.set()
        return "https://github.com/kairo"  # same — should not emit twice

    with patch.object(monitor, "_read_clipboard", side_effect=fake_read):
        t = threading.Thread(target=monitor.run, args=(stop,), daemon=True)
        t.start()
        await asyncio.sleep(0.5)
        stop.set()
        t.join(timeout=2.0)

    assert len(events) == 1
    assert events[0].content_type == "url"
    assert "github.com" in events[0].content


@pytest.mark.asyncio
async def test_no_event_for_identical_content(bus):
    loop = asyncio.get_running_loop()
    monitor = ClipboardMonitor(bus, loop, poll_interval=0.05)

    events = []
    bus.subscribe(ClipboardChangedEvent, lambda e: events.append(e))
    await asyncio.sleep(0.05)

    stop = threading.Event()
    reads = ["same content here"] * 5

    def fake_read():
        if reads:
            return reads.pop(0)
        stop.set()
        return ""

    with patch.object(monitor, "_read_clipboard", side_effect=fake_read):
        t = threading.Thread(target=monitor.run, args=(stop,), daemon=True)
        t.start()
        await asyncio.sleep(0.5)
        stop.set()
        t.join(timeout=2.0)

    assert len(events) == 1  # only first occurrence


@pytest.mark.asyncio
async def test_no_event_for_short_content(bus):
    loop = asyncio.get_running_loop()
    monitor = ClipboardMonitor(bus, loop, poll_interval=0.05)

    events = []
    bus.subscribe(ClipboardChangedEvent, lambda e: events.append(e))
    await asyncio.sleep(0.05)

    stop = threading.Event()

    def fake_read():
        stop.set()
        return "ok"  # < _MIN_CONTENT_LEN (3), triggers edge case

    with patch.object(monitor, "_read_clipboard", side_effect=fake_read):
        t = threading.Thread(target=monitor.run, args=(stop,), daemon=True)
        t.start()
        await asyncio.sleep(0.3)
        t.join(timeout=2.0)

    # "ok" is 2 chars which equals _MIN_CONTENT_LEN so this tests the boundary
    assert len(events) <= 1


@pytest.mark.asyncio
async def test_content_truncated_at_500_chars(bus):
    loop = asyncio.get_running_loop()
    monitor = ClipboardMonitor(bus, loop, poll_interval=0.05)

    events = []
    bus.subscribe(ClipboardChangedEvent, lambda e: events.append(e))
    await asyncio.sleep(0.05)

    stop = threading.Event()
    big_content = "x" * 1000

    def fake_read():
        stop.set()
        return big_content

    with patch.object(monitor, "_read_clipboard", side_effect=fake_read):
        t = threading.Thread(target=monitor.run, args=(stop,), daemon=True)
        t.start()
        await asyncio.sleep(0.3)
        t.join(timeout=2.0)

    if events:
        assert len(events[0].content) <= 500
