"""Tests for preferences memory."""

import pytest

from memory_service.preferences import PreferencesMemory


@pytest.mark.asyncio
async def test_set_and_get(tmp_path):
    prefs = PreferencesMemory(tmp_path / "prefs.db")
    await prefs.initialize()
    assert prefs.healthy

    await prefs.set("favorite_music", "lofi")
    result = await prefs.get("favorite_music")
    assert result == "lofi"

    await prefs.close()


@pytest.mark.asyncio
async def test_get_default(tmp_path):
    prefs = PreferencesMemory(tmp_path / "prefs.db")
    await prefs.initialize()

    result = await prefs.get("nonexistent", "default_val")
    assert result == "default_val"

    await prefs.close()


@pytest.mark.asyncio
async def test_aliases(tmp_path):
    prefs = PreferencesMemory(tmp_path / "prefs.db")
    await prefs.initialize()

    await prefs.set_alias("payments", "payments-service")
    resolved = await prefs.resolve_alias("open payments project")
    assert "payments-service" in resolved

    aliases = await prefs.get_aliases()
    assert "payments" in aliases

    await prefs.close()


@pytest.mark.asyncio
async def test_get_all(tmp_path):
    prefs = PreferencesMemory(tmp_path / "prefs.db")
    await prefs.initialize()

    await prefs.set("key1", "val1")
    await prefs.set("key2", "val2")

    all_prefs = await prefs.get_all()
    assert all_prefs == {"key1": "val1", "key2": "val2"}

    await prefs.close()
