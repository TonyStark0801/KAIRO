"""Tests for identity memory."""

import tempfile
from pathlib import Path

import pytest

from memory_service.identity import IdentityMemory


def test_load_default_identity(tmp_path):
    yaml_content = """
assistant:
  name: Kairo
  personality: warm
  style: conversational
  voice_model: en_US-amy-medium
  wake_words:
    - kairo

owner:
  name: Shubham
  preferences:
    ide: IntelliJ IDEA
    browser: Brave Browser
"""
    path = tmp_path / "identity.yaml"
    path.write_text(yaml_content)
    identity = IdentityMemory(path)
    identity.load()

    assert identity.assistant_name == "Kairo"
    assert identity.owner_name == "Shubham"
    assert identity.personality == "warm"
    assert identity.voice_model == "en_US-amy-medium"
    assert "kairo" in identity.wake_words
    assert identity.get_owner_pref("ide") == "IntelliJ IDEA"
    assert identity.get_owner_pref("browser") == "Brave Browser"


def test_missing_file_uses_defaults():
    identity = IdentityMemory("/nonexistent/path.yaml")
    identity.load()
    assert identity.assistant_name == "Kairo"
    # _defaults() hardcodes owner name as "Tony" — "User" is only the fallback
    # when the owner.name key is missing entirely, which doesn't happen with defaults.
    assert identity.owner_name == "Tony"


def test_update_and_save(tmp_path):
    path = tmp_path / "identity.yaml"
    identity = IdentityMemory(path)
    identity.load()
    identity.update_owner_preference("terminal", "iTerm2")
    assert identity.get_owner_pref("terminal") == "iTerm2"

    reloaded = IdentityMemory(path)
    reloaded.load()
    assert reloaded.get_owner_pref("terminal") == "iTerm2"
