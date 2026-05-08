"""Tests for the session state machine."""

import asyncio

import pytest
import pytest_asyncio

from runtime.event_bus import (
    EventBus,
    GestureEvent,
    GestureType,
    IntentRoutedEvent,
    SessionState,
    SessionStateChangedEvent,
    ToolExecutionEvent,
    ToolResult,
)
from core.session.state_machine import InvalidTransitionError, SessionStateMachine


@pytest_asyncio.fixture
async def bus():
    b = EventBus()
    await b.start()
    yield b
    await b.stop()


@pytest_asyncio.fixture
async def fsm(bus):
    sm = SessionStateMachine(bus)
    return sm


@pytest.mark.asyncio
async def test_initial_state_is_sleep(fsm):
    assert fsm.state == SessionState.SLEEP


@pytest.mark.asyncio
async def test_sleep_to_wake_pending_on_face_verified(fsm, bus):
    events = []
    bus.subscribe(SessionStateChangedEvent, lambda e: events.append(e))
    await asyncio.sleep(0.05)

    await fsm.handle_event(
        GestureEvent(type=GestureType.FACE_VERIFIED, timestamp=1.0)
    )
    await asyncio.sleep(0.05)

    assert fsm.state == SessionState.WAKE_PENDING
    assert len(events) == 1
    assert events[0].old_state == SessionState.SLEEP
    assert events[0].new_state == SessionState.WAKE_PENDING


@pytest.mark.asyncio
async def test_wake_pending_to_active_on_all_signals(fsm, bus):
    await fsm.handle_event(GestureEvent(type=GestureType.FACE_VERIFIED, timestamp=1.0))
    await fsm.handle_event(GestureEvent(type=GestureType.ALL_SIGNALS_CONFIRMED, timestamp=2.0))
    assert fsm.state == SessionState.ACTIVE_SESSION


@pytest.mark.asyncio
async def test_wake_pending_to_sleep_on_timeout(fsm, bus):
    await fsm.handle_event(GestureEvent(type=GestureType.FACE_VERIFIED, timestamp=1.0))
    await fsm.handle_event(GestureEvent(type=GestureType.WAKE_TIMEOUT, timestamp=4.0))
    assert fsm.state == SessionState.SLEEP


@pytest.mark.asyncio
async def test_active_to_executing_on_intent(fsm, bus):
    await fsm.handle_event(GestureEvent(type=GestureType.FACE_VERIFIED, timestamp=1.0))
    await fsm.handle_event(GestureEvent(type=GestureType.ALL_SIGNALS_CONFIRMED, timestamp=2.0))
    await fsm.handle_event(IntentRoutedEvent(tool_name="test", params={}, confidence=0.9, session_id="s1"))
    assert fsm.state == SessionState.EXECUTING


@pytest.mark.asyncio
async def test_executing_to_active_on_success(fsm, bus):
    await fsm.handle_event(GestureEvent(type=GestureType.FACE_VERIFIED, timestamp=1.0))
    await fsm.handle_event(GestureEvent(type=GestureType.ALL_SIGNALS_CONFIRMED, timestamp=2.0))
    await fsm.handle_event(IntentRoutedEvent(tool_name="t", params={}, confidence=0.9, session_id="s1"))
    await fsm.handle_event(ToolExecutionEvent(tool_name="t", success=True, result=ToolResult(success=True, message="ok"), session_id="s1"))
    assert fsm.state == SessionState.ACTIVE_SESSION


@pytest.mark.asyncio
async def test_executing_to_active_on_failure(fsm, bus):
    await fsm.handle_event(GestureEvent(type=GestureType.FACE_VERIFIED, timestamp=1.0))
    await fsm.handle_event(GestureEvent(type=GestureType.ALL_SIGNALS_CONFIRMED, timestamp=2.0))
    await fsm.handle_event(IntentRoutedEvent(tool_name="t", params={}, confidence=0.9, session_id="s1"))
    await fsm.handle_event(ToolExecutionEvent(tool_name="t", success=False, result=ToolResult(success=False, message="error"), session_id="s1"))
    assert fsm.state == SessionState.ACTIVE_SESSION


@pytest.mark.asyncio
async def test_idle_timeout_transition(fsm, bus):
    await fsm.handle_event(GestureEvent(type=GestureType.FACE_VERIFIED, timestamp=1.0))
    await fsm.handle_event(GestureEvent(type=GestureType.ALL_SIGNALS_CONFIRMED, timestamp=2.0))
    await fsm.trigger_idle_timeout()
    assert fsm.state == SessionState.SLEEP


@pytest.mark.asyncio
async def test_irrelevant_event_is_noop(fsm, bus):
    await fsm.handle_event(GestureEvent(type=GestureType.DOUBLE_CLAP, timestamp=1.0))
    assert fsm.state == SessionState.SLEEP


@pytest.mark.asyncio
async def test_invalid_transition_raises(fsm, bus):
    # SessionStateChangedEvent has no handler in any FSM state and is NOT in
    # _NOOP_EVENTS, so it must raise InvalidTransitionError.
    # (IntentRoutedEvent in SLEEP is a deliberate noop — not an invalid transition.)
    with pytest.raises(InvalidTransitionError):
        await fsm.handle_event(
            SessionStateChangedEvent(
                old_state=SessionState.SLEEP,
                new_state=SessionState.ACTIVE_SESSION,
                session_id="s1",
            )
        )
