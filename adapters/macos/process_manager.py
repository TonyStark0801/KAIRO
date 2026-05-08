"""Process manager — queries running applications via AppleScript."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adapters.macos.adapter import MacOSAdapter

logger = logging.getLogger(__name__)


class ProcessManager:
    def __init__(self, adapter: MacOSAdapter) -> None:
        self._adapter = adapter

    async def get_open_intellij_projects(self) -> list[str]:
        script = (
            'tell application "System Events"\n'
            '  if exists (process "IntelliJ IDEA") then\n'
            '    tell process "IntelliJ IDEA"\n'
            '      set windowNames to name of every window\n'
            '    end tell\n'
            '    return windowNames\n'
            '  else\n'
            '    return ""\n'
            '  end if\n'
            'end tell'
        )
        raw = await self._adapter.run_script(script)
        if not raw:
            return []
        return [w.strip() for w in raw.split(",") if w.strip()]

    async def is_app_running(self, app_name: str) -> bool:
        script = (
            f'tell application "System Events"\n'
            f'  return exists (process "{app_name}")\n'
            f'end tell'
        )
        result = await self._adapter.run_script(script)
        return result.lower() == "true"
