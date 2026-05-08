"""Activity classifier — infers what the user is doing from workspace context.

Rule-based; no LLM needed. Fast, deterministic, zero latency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from runtime.event_bus import ActivityType

if TYPE_CHECKING:
    from context_service.detector import WorkspaceContext

# App → ActivityType mapping (ordered: more specific first)
_CODING_APPS = frozenset({
    "IntelliJ IDEA", "IntelliJ IDEA CE", "IntelliJ IDEA Ultimate",
    "PyCharm", "PyCharm CE", "WebStorm", "GoLand", "CLion", "Rider",
    "Visual Studio Code", "Cursor", "Zed", "Xcode",
    "Android Studio", "Fleet",
})

_TERMINAL_APPS = frozenset({
    "Terminal", "iTerm2", "iTerm", "Warp", "Alacritty",
    "Hyper", "Ghostty", "kitty",
})

_BROWSER_APPS = frozenset({
    "Brave Browser", "Google Chrome", "Chromium",
    "Safari", "Firefox", "Arc",
})

_COMMS_APPS = frozenset({
    "Slack", "Discord", "Zoom", "Microsoft Teams",
    "Messages", "Mail", "Spark", "Superhuman",
    "Telegram", "WhatsApp",
})

_WRITING_APPS = frozenset({
    "Pages", "Microsoft Word", "LibreOffice Writer",
    "Notion", "Obsidian", "Bear", "Typora",
    "TextEdit", "Ulysses",
})

_READING_APPS = frozenset({
    "Preview", "Adobe Acrobat Reader", "Kindle", "Books",
})

# Browser URL domains that indicate coding/dev work
_DEV_URL_DOMAINS = frozenset({
    "github.com", "gitlab.com", "bitbucket.org",
    "stackoverflow.com", "superuser.com",
    "docs.python.org", "docs.rust-lang.org", "docs.oracle.com",
    "developer.mozilla.org", "developer.apple.com",
    "pkg.go.dev", "crates.io", "npmjs.com", "pypi.org",
    "readthedocs.io", "readthedocs.org",
})

_COMMS_URL_DOMAINS = frozenset({
    "mail.google.com", "outlook.live.com", "outlook.office.com",
    "web.telegram.org", "web.whatsapp.com",
    "app.slack.com", "discord.com",
})


class ActivityClassifier:
    """Classify user activity from a WorkspaceContext snapshot."""

    def classify(self, ctx: WorkspaceContext) -> ActivityType:
        app = ctx.active_app

        if app in _CODING_APPS:
            return ActivityType.CODING

        if app in _TERMINAL_APPS:
            return ActivityType.TERMINAL

        if app in _COMMS_APPS:
            return ActivityType.COMMUNICATING

        if app in _WRITING_APPS:
            return ActivityType.WRITING

        if app in _READING_APPS:
            return ActivityType.READING

        if app in _BROWSER_APPS:
            return self._classify_browser(ctx)

        if not app:
            return ActivityType.IDLE

        return ActivityType.UNKNOWN

    @staticmethod
    def _classify_browser(ctx: WorkspaceContext) -> ActivityType:
        url = ctx.browser_url.lower()
        title = (ctx.browser_tab_title or ctx.window_title).lower()

        if not url or url.startswith("about:"):
            return ActivityType.BROWSING

        # Strip scheme for domain matching
        for scheme in ("https://", "http://"):
            if url.startswith(scheme):
                url = url[len(scheme):]
                break
        # lstrip strips *characters*, not the literal prefix — use removeprefix instead
        raw_domain = url.split("/")[0]
        domain = raw_domain.removeprefix("www.")

        if any(dev in domain for dev in _DEV_URL_DOMAINS):
            return ActivityType.CODING

        if any(comms in domain for comms in _COMMS_URL_DOMAINS):
            return ActivityType.COMMUNICATING

        # Title-based heuristics (useful when URL is empty, e.g. Firefox)
        writing_hints = ("docs.google.com", "notion.so", "coda.io", "quip.com")
        if any(h in domain for h in writing_hints):
            return ActivityType.WRITING

        return ActivityType.BROWSING
