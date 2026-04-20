import pytest

from assistant_core.dialogue_planner import DialoguePlanner


@pytest.mark.asyncio
async def test_default_planner_passthrough():
    p = DialoguePlanner()
    out = await p.plan("  open notes  ", "s1")
    assert out.transcript == "open notes"
    assert out.use_llm is True
    assert out.allow_tool_execution is True
    assert out.persist_memory is True
    assert out.tone_hint is None
