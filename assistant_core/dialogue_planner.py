"""Dialogue planner — policy layer between raw STT and the Reasoner.

Default plan is permissive (backward compatible). Subclass or replace to add
gating, safety, or tone rules without changing Reasoner internals.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlanOutput:
    """Planner decision for one user transcript."""

    transcript: str
    use_llm: bool = True
    allow_tool_execution: bool = True
    persist_memory: bool = True
    tone_hint: str | None = None


class DialoguePlanner:
    async def plan(self, transcript: str, session_id: str = "") -> PlanOutput:
        _ = session_id
        return PlanOutput(transcript=transcript.strip())
