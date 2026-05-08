"""Context detector — observes what the user is working on via AppleScript."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 10.0
_CACHE_TTL = 8.0

_FRONTMOST_APP_SCRIPT = (
    'tell application "System Events" to get name of first process whose frontmost is true'
)
_ACTIVE_WINDOW_SCRIPT = """
tell application "System Events"
    set frontApp to first process whose frontmost is true
    set appName to name of frontApp
    try
        set winTitle to name of front window of frontApp
    on error
        set winTitle to ""
    end try
    return appName & "|" & winTitle
end tell
"""

_IDE_APPS = {"IntelliJ IDEA", "IntelliJ IDEA CE", "WebStorm", "PyCharm", "Visual Studio Code", "Cursor"}

# Browser-specific AppleScript fragments for tab URL + title.
# Firefox doesn't expose URL via AppleScript, so we fall back to window title only.
_BROWSER_TAB_SCRIPTS: dict[str, str] = {
    "Brave Browser": """
tell application "Brave Browser"
    set tabURL to URL of active tab of front window
    set tabTitle to title of active tab of front window
    return tabURL & "||" & tabTitle
end tell""",
    "Google Chrome": """
tell application "Google Chrome"
    set tabURL to URL of active tab of front window
    set tabTitle to title of active tab of front window
    return tabURL & "||" & tabTitle
end tell""",
    "Chromium": """
tell application "Chromium"
    set tabURL to URL of active tab of front window
    set tabTitle to title of active tab of front window
    return tabURL & "||" & tabTitle
end tell""",
    "Safari": """
tell application "Safari"
    set tabURL to URL of current tab of front window
    set tabTitle to name of current tab of front window
    return tabURL & "||" & tabTitle
end tell""",
    "Firefox": """
tell application "Firefox"
    set winTitle to name of front window
    return "about:firefox||" & winTitle
end tell""",
    "Arc": """
tell application "Arc"
    set tabURL to URL of active tab of front window
    set tabTitle to title of active tab of front window
    return tabURL & "||" & tabTitle
end tell""",
}

_BROWSER_APPS = set(_BROWSER_TAB_SCRIPTS.keys())


@dataclass
class WorkspaceContext:
    active_app: str = ""
    window_title: str = ""
    repo_path: str = ""
    git_branch: str = ""
    open_file: str = ""
    # Phase 2: browser awareness
    browser_url: str = ""
    browser_tab_title: str = ""
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        parts = []
        if self.active_app:
            parts.append(f"app={self.active_app}")
        if self.repo_path:
            repo_name = Path(self.repo_path).name
            parts.append(f"repo={repo_name}")
        if self.git_branch:
            parts.append(f"branch={self.git_branch}")
        if self.open_file:
            parts.append(f"file={self.open_file}")
        return ", ".join(parts) if parts else "no workspace detected"

    def natural_summary(self) -> str:
        parts = []
        if self.repo_path:
            repo_name = Path(self.repo_path).name
            parts.append(repo_name)
        if self.git_branch and self.git_branch != "main" and self.git_branch != "master":
            parts.append(f"on the {self.git_branch} branch")
        if self.open_file:
            parts.append(f"editing {Path(self.open_file).name}")
        if parts:
            return " ".join(parts)
        if self.active_app:
            return self.active_app
        return ""


class ContextDetector:
    def __init__(self) -> None:
        self._cache: WorkspaceContext | None = None
        self._cache_time: float = 0
        self._known_repos: dict[str, str] = {}

    async def get_context(self) -> WorkspaceContext:
        now = time.time()
        if self._cache and (now - self._cache_time) < _CACHE_TTL:
            return self._cache

        ctx = WorkspaceContext()
        try:
            raw = await self._run_osascript(_ACTIVE_WINDOW_SCRIPT)
            if "|" in raw:
                app_name, window_title = raw.split("|", 1)
                ctx.active_app = app_name.strip()
                ctx.window_title = window_title.strip()
            else:
                ctx.active_app = raw.strip()

            if ctx.active_app in _IDE_APPS and ctx.window_title:
                self._parse_ide_context(ctx)

            if ctx.repo_path:
                branch = await self._get_git_branch(ctx.repo_path)
                if branch:
                    ctx.git_branch = branch

            # Phase 2: browser tab awareness
            if ctx.active_app in _BROWSER_APPS:
                url, tab_title = await self._get_browser_tab(ctx.active_app)
                ctx.browser_url = url
                ctx.browser_tab_title = tab_title

        except Exception:
            logger.debug("Context detection failed", exc_info=True)

        ctx.timestamp = now
        self._cache = ctx
        self._cache_time = now
        return ctx

    def _parse_ide_context(self, ctx: WorkspaceContext) -> None:
        title = ctx.window_title
        if " – " in title:
            parts = title.split(" – ")
            if len(parts) >= 2:
                file_part = parts[0].strip()
                project_part = parts[-1].strip()
                if "/" in file_part or "." in file_part:
                    ctx.open_file = file_part
                ctx.repo_path = self._resolve_repo(project_part)
        elif " - " in title:
            parts = title.split(" - ")
            if len(parts) >= 2:
                file_part = parts[0].strip()
                project_part = parts[1].strip()
                if "." in file_part:
                    ctx.open_file = file_part
                ctx.repo_path = self._resolve_repo(project_part)

    def _resolve_repo(self, project_name: str) -> str:
        if project_name in self._known_repos:
            return self._known_repos[project_name]
        common_dirs = [
            Path.home() / "Projects" / project_name,
            Path.home() / "projects" / project_name,
            Path.home() / "repos" / project_name,
            Path.home() / "workspace" / project_name,
            Path.home() / "code" / project_name,
        ]
        for d in common_dirs:
            if d.is_dir():
                self._known_repos[project_name] = str(d)
                return str(d)
        return project_name

    async def _get_browser_tab(self, app_name: str) -> tuple[str, str]:
        """Return (url, tab_title) for the active browser tab, or ("", "") on failure."""
        script = _BROWSER_TAB_SCRIPTS.get(app_name)
        if not script:
            return "", ""
        try:
            raw = await self._run_osascript(script)
            if "||" in raw:
                url, title = raw.split("||", 1)
                return url.strip(), title.strip()
            return raw.strip(), ""
        except Exception:
            logger.debug("Browser tab detection failed for %s", app_name, exc_info=True)
            return "", ""

    async def _get_git_branch(self, repo_path: str) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "-C", repo_path, "branch", "--show-current",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=3.0)
            return stdout.decode().strip()
        except Exception:
            return ""

    @staticmethod
    async def _run_osascript(script: str) -> str:
        proc = await asyncio.create_subprocess_exec(
            "osascript", "-e", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        return stdout.decode().strip()
