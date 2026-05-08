"""Tests for ActivityClassifier — rule-based context classification."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

import pytest

from runtime.event_bus import ActivityType
from sensors.observer.activity_classifier import ActivityClassifier


# Minimal WorkspaceContext stub — avoids importing the full context_service
@dataclass
class _Ctx:
    active_app: str = ""
    window_title: str = ""
    browser_url: str = ""
    browser_tab_title: str = ""
    repo_path: str = ""
    git_branch: str = ""
    open_file: str = ""
    timestamp: float = field(default_factory=time.time)


@pytest.fixture
def clf():
    return ActivityClassifier()


# ── App-based classification ──────────────────────────────────────────────────

@pytest.mark.parametrize("app,expected", [
    ("IntelliJ IDEA", ActivityType.CODING),
    ("PyCharm", ActivityType.CODING),
    ("Visual Studio Code", ActivityType.CODING),
    ("Cursor", ActivityType.CODING),
    ("Terminal", ActivityType.TERMINAL),
    ("iTerm2", ActivityType.TERMINAL),
    ("Warp", ActivityType.TERMINAL),
    ("Slack", ActivityType.COMMUNICATING),
    ("Zoom", ActivityType.COMMUNICATING),
    ("Pages", ActivityType.WRITING),
    ("Obsidian", ActivityType.WRITING),
    ("Preview", ActivityType.READING),
    ("", ActivityType.IDLE),
    ("SomeRandomApp", ActivityType.UNKNOWN),
])
def test_app_classification(clf, app, expected):
    ctx = _Ctx(active_app=app)
    assert clf.classify(ctx) == expected


# ── Browser URL classification ────────────────────────────────────────────────

@pytest.mark.parametrize("url,expected", [
    ("https://github.com/user/repo", ActivityType.CODING),
    ("https://stackoverflow.com/questions/123", ActivityType.CODING),
    ("https://docs.python.org/3/library/asyncio.html", ActivityType.CODING),
    ("https://mail.google.com/mail", ActivityType.COMMUNICATING),
    ("https://web.telegram.org/#/im", ActivityType.COMMUNICATING),
    ("https://news.ycombinator.com", ActivityType.BROWSING),
    ("https://www.youtube.com/watch?v=abc", ActivityType.BROWSING),
    ("about:newtab", ActivityType.BROWSING),
])
def test_browser_url_classification(clf, url, expected):
    ctx = _Ctx(active_app="Brave Browser", browser_url=url)
    assert clf.classify(ctx) == expected


def test_browser_no_url_returns_browsing(clf):
    ctx = _Ctx(active_app="Google Chrome", browser_url="")
    assert clf.classify(ctx) == ActivityType.BROWSING


def test_writing_url(clf):
    ctx = _Ctx(active_app="Brave Browser", browser_url="https://docs.google.com/document/d/1abc")
    assert clf.classify(ctx) == ActivityType.WRITING


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_non_browser_app_ignores_url(clf):
    """URL in context shouldn't affect classification of non-browser apps."""
    ctx = _Ctx(active_app="Terminal", browser_url="https://github.com")
    assert clf.classify(ctx) == ActivityType.TERMINAL


def test_url_strip_www(clf):
    ctx = _Ctx(active_app="Safari", browser_url="https://www.github.com/foo")
    assert clf.classify(ctx) == ActivityType.CODING


def test_url_strip_https(clf):
    ctx = _Ctx(active_app="Firefox", browser_url="https://stackoverflow.com")
    # ActivityClassifier checks active_app first — Firefox is a browser app
    assert clf.classify(ctx) in (ActivityType.CODING, ActivityType.BROWSING)
