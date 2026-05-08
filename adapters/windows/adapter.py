"""Windows adapter stub — all methods raise NotImplementedError."""

from __future__ import annotations

from adapters.base.platform_adapter import AppWindow, PlatformAdapter


class WindowsAdapter(PlatformAdapter):

    async def open_application(self, app_name: str) -> bool:
        raise NotImplementedError("Windows adapter not yet implemented")

    async def switch_window(self, app_name: str, title_pattern: str | None = None) -> bool:
        raise NotImplementedError("Windows adapter: switch_window not implemented")

    async def run_script(self, script: str) -> str:
        raise NotImplementedError("Windows adapter: run_script not implemented")

    async def get_running_apps(self) -> list[AppWindow]:
        raise NotImplementedError("Windows adapter: get_running_apps not implemented")

    async def open_url_in_browser(self, url: str, browser: str = "Safari") -> bool:
        raise NotImplementedError("Windows adapter: open_url_in_browser not implemented")

    async def send_notification(self, title: str, body: str) -> None:
        raise NotImplementedError("Windows adapter: send_notification not implemented")

    async def play_audio_file(self, path: str) -> None:
        raise NotImplementedError("Windows adapter: play_audio_file not implemented")

    async def get_active_workspace(self) -> str | None:
        raise NotImplementedError("Windows adapter: get_active_workspace not implemented")
