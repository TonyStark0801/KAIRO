"""Session context — holds per-session state."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class SessionContext:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    start_time: float = field(default_factory=time.time)
    workspace_mode: str | None = None
    command_history: list[str] = field(default_factory=list)
