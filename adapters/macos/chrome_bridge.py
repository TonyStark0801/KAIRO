"""ChromeBridge — AppleScript + JavaScript injection for Chromium browsers.

Talks to the user's existing Brave/Chrome via osascript, injecting JavaScript
to read DOM state and interact with pages (YouTube, etc.).

Requires: Browser > View > Developer > "Allow JavaScript from Apple Events"
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.parse

logger = logging.getLogger(__name__)

_DEFAULT_APP = "Brave Browser"


class ChromeBridge:
    def __init__(self, app_name: str = _DEFAULT_APP) -> None:
        self._app = app_name

    async def _run_osascript(self, script: str) -> str:
        proc = await asyncio.create_subprocess_exec(
            "osascript", "-e", script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            err = stderr.decode().strip()
            logger.warning("ChromeBridge osascript failed (rc=%d): %s", proc.returncode, err)
            raise RuntimeError(err)
        return stdout.decode().strip()

    async def execute_js(self, js_code: str, url_pattern: str | None = None) -> str:
        """Run JavaScript in a browser tab. If url_pattern is given, finds
        the first tab whose URL contains that pattern."""
        escaped_js = js_code.replace("\\", "\\\\").replace('"', '\\"')

        if url_pattern:
            script = (
                f'tell application "{self._app}"\n'
                f'  set targetTab to null\n'
                f'  repeat with w in windows\n'
                f'    repeat with t in tabs of w\n'
                f'      if URL of t contains "{url_pattern}" then\n'
                f'        set targetTab to t\n'
                f'        exit repeat\n'
                f'      end if\n'
                f'    end repeat\n'
                f'    if targetTab is not null then exit repeat\n'
                f'  end repeat\n'
                f'  if targetTab is null then\n'
                f'    return "ERROR:NO_TAB"\n'
                f'  end if\n'
                f'  execute targetTab javascript "{escaped_js}"\n'
                f'end tell'
            )
        else:
            script = (
                f'tell application "{self._app}"\n'
                f'  execute front window\'s active tab javascript "{escaped_js}"\n'
                f'end tell'
            )
        return await self._run_osascript(script)

    async def _ensure_window(self) -> None:
        """Make sure the browser is running and has at least one window."""
        check_script = (
            f'tell application "{self._app}"\n'
            f'  activate\n'
            f'  if (count of windows) is 0 then\n'
            f'    make new window\n'
            f'    delay 1\n'
            f'  end if\n'
            f'end tell'
        )
        try:
            await self._run_osascript(check_script)
        except RuntimeError:
            open_script = (
                f'tell application "{self._app}" to activate\n'
                f'delay 2\n'
                f'tell application "{self._app}" to make new window'
            )
            try:
                await self._run_osascript(open_script)
                await asyncio.sleep(1.5)
            except RuntimeError:
                logger.warning("Could not open %s window", self._app)

    async def navigate(self, url: str, reuse_pattern: str | None = None) -> bool:
        """Navigate to a URL. If reuse_pattern is given, reuses an existing tab
        whose URL contains that pattern instead of opening a new one."""
        await self._ensure_window()

        if reuse_pattern:
            script = (
                f'tell application "{self._app}"\n'
                f'  activate\n'
                f'  set found to false\n'
                f'  repeat with w in windows\n'
                f'    set tabIdx to 0\n'
                f'    repeat with t in tabs of w\n'
                f'      set tabIdx to tabIdx + 1\n'
                f'      if URL of t contains "{reuse_pattern}" then\n'
                f'        set active tab index of w to tabIdx\n'
                f'        set index of w to 1\n'
                f'        set URL of t to "{url}"\n'
                f'        set found to true\n'
                f'        exit repeat\n'
                f'      end if\n'
                f'    end repeat\n'
                f'    if found then exit repeat\n'
                f'  end repeat\n'
                f'  if not found then\n'
                f'    tell front window to make new tab with properties {{URL:"{url}"}}\n'
                f'  end if\n'
                f'end tell'
            )
        else:
            script = (
                f'tell application "{self._app}"\n'
                f'  activate\n'
                f'  tell front window to make new tab with properties {{URL:"{url}"}}\n'
                f'end tell'
            )
        try:
            await self._run_osascript(script)
            return True
        except RuntimeError:
            return False

    async def get_active_tab(self) -> dict[str, str]:
        script = (
            f'tell application "{self._app}"\n'
            f'  set t to active tab of front window\n'
            f'  return (URL of t) & "|||" & (title of t)\n'
            f'end tell'
        )
        try:
            result = await self._run_osascript(script)
            parts = result.split("|||", 1)
            return {"url": parts[0], "title": parts[1] if len(parts) > 1 else ""}
        except RuntimeError:
            return {"url": "", "title": ""}

    async def ensure_youtube_tab(self, search_url: str | None = None) -> bool:
        """Make sure a YouTube tab exists. Navigates it to search_url if given,
        or opens a new one."""
        url = search_url or "https://www.youtube.com"
        return await self.navigate(url, reuse_pattern="youtube.com")

    @staticmethod
    def build_search_url(query: str) -> str:
        encoded = urllib.parse.quote_plus(query)
        return f"https://www.youtube.com/results?search_query={encoded}"
