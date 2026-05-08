"""Abstract base class for platform adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AppWindow:
    app_name: str
    title: str
    pid: int


class PlatformAdapter(ABC):

    @abstractmethod
    async def open_application(self, app_name: str) -> bool: ...

    @abstractmethod
    async def switch_window(
        self, app_name: str, title_pattern: str | None = None
    ) -> bool: ...

    @abstractmethod
    async def run_script(self, script: str) -> str: ...

    @abstractmethod
    async def get_running_apps(self) -> list[AppWindow]: ...

    @abstractmethod
    async def open_url_in_browser(self, url: str, browser: str = "Safari") -> bool: ...

    @abstractmethod
    async def send_notification(self, title: str, body: str) -> None: ...

    @abstractmethod
    async def play_audio_file(self, path: str) -> None: ...

    @abstractmethod
    async def get_active_workspace(self) -> str | None: ...
