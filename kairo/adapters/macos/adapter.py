"""macOS platform adapter — all OS interaction goes through here."""

from __future__ import annotations

import asyncio
import logging

from adapters.base.platform_adapter import AppWindow, PlatformAdapter
from adapters.macos.applescript import (
    build_get_active_workspace_script,
    build_list_running_apps_script,
    build_notification_script,
    build_open_app_script,
    build_open_url_script,
    build_play_audio_script,
    build_say_script,
    build_switch_window_script,
)

logger = logging.getLogger(__name__)


class MacOSAdapter(PlatformAdapter):

    async def run_script(self, script: str) -> str:
        proc = await asyncio.create_subprocess_exec(
            "osascript", "-e", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.warning("osascript failed (rc=%d): %s", proc.returncode, stderr.decode().strip())
        return stdout.decode().strip()

    async def open_application(self, app_name: str) -> bool:
        try:
            await self.run_script(build_open_app_script(app_name))
            return True
        except Exception:
            logger.exception("Failed to open %s", app_name)
            return False

    async def switch_window(self, app_name: str, title_pattern: str | None = None) -> bool:
        try:
            await self.run_script(build_switch_window_script(app_name, title_pattern))
            return True
        except Exception:
            logger.exception("Failed to switch window for %s", app_name)
            return False

    async def get_running_apps(self) -> list[AppWindow]:
        raw = await self.run_script(build_list_running_apps_script())
        if not raw:
            return []
        names = [n.strip() for n in raw.split(",")]
        return [AppWindow(app_name=n, title="", pid=0) for n in names if n]

    async def open_url_in_browser(self, url: str, browser: str = "Safari") -> bool:
        try:
            await self.run_script(build_open_url_script(url, browser))
            return True
        except Exception:
            logger.exception("Failed to open URL %s", url)
            return False

    async def send_notification(self, title: str, body: str) -> None:
        await self.run_script(build_notification_script(title, body))

    async def play_audio_file(self, path: str) -> None:
        await self.run_script(build_play_audio_script(path))

    async def get_active_workspace(self) -> str | None:
        result = await self.run_script(build_get_active_workspace_script())
        return result or None
