"""Tests for session store."""

import pytest

from memory_service.session_store import SessionStore


@pytest.mark.asyncio
async def test_session_lifecycle(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    await store.initialize()
    assert store.healthy

    await store.start_session("s1", {"app": "IntelliJ"})
    await store.end_session("s1", "User worked on payments service", ["open_project", "youtube_search"])

    last = await store.get_last_session()
    assert last is not None
    assert last["session_id"] == "s1"
    assert "payments" in last["summary"]
    assert "open_project" in last["tools_used"]

    await store.close()


@pytest.mark.asyncio
async def test_no_sessions_returns_none(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    await store.initialize()
    last = await store.get_last_session()
    assert last is None
    await store.close()


@pytest.mark.asyncio
async def test_recent_sessions(tmp_path):
    store = SessionStore(tmp_path / "sessions.db")
    await store.initialize()

    for i in range(3):
        await store.start_session(f"s{i}")
        await store.end_session(f"s{i}", f"Session {i} summary", [])

    recent = await store.get_recent_sessions(limit=2)
    assert len(recent) == 2
    await store.close()
