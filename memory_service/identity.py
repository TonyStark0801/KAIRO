"""Identity memory — reads/writes the assistant and owner identity from YAML."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_IDENTITY_PATH = Path(__file__).resolve().parent.parent / "config" / "identity.yaml"


class IdentityMemory:
    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else _DEFAULT_IDENTITY_PATH
        self._data: dict[str, Any] = {}
        self._loaded = False

    def load(self) -> None:
        try:
            with open(self._path) as f:
                self._data = yaml.safe_load(f) or {}
            self._loaded = True
            logger.info("Identity loaded from %s", self._path)
        except FileNotFoundError:
            logger.warning("Identity file not found at %s — using defaults", self._path)
            self._data = self._defaults()
            self._loaded = True
        except Exception:
            logger.exception("Failed to load identity")
            self._data = self._defaults()
            self._loaded = True

    @staticmethod
    def _defaults() -> dict[str, Any]:
        return {
            "assistant": {
                "name": "Kairo",
                "full_name": "KAIRO",
                "personality": "warm, witty, conversational, empathetic",
                "style": "Speak like a close friend. Short sentences. Natural phrasing.",
                "voice_model": "en_US-amy-medium",
                "wake_words": ["kairo", "hey kairo", "kyro"],
            },
            "owner": {
                "name": "Tony",
                "preferences": {},
            },
        }

    @property
    def assistant_name(self) -> str:
        return self._data.get("assistant", {}).get("name", "Kairo")

    @property
    def assistant_full_name(self) -> str:
        return self._data.get("assistant", {}).get("full_name", self.assistant_name)

    @property
    def personality(self) -> str:
        return self._data.get("assistant", {}).get("personality", "")

    @property
    def style(self) -> str:
        return self._data.get("assistant", {}).get("style", "")

    @property
    def voice_model(self) -> str:
        return self._data.get("assistant", {}).get("voice_model", "en_US-amy-medium")

    @property
    def wake_words(self) -> list[str]:
        return self._data.get("assistant", {}).get("wake_words", ["kairo", "hey kairo"])

    @property
    def owner_name(self) -> str:
        return self._data.get("owner", {}).get("name", "User")

    @property
    def owner_preferences(self) -> dict[str, str]:
        return self._data.get("owner", {}).get("preferences", {})

    @property
    def verification_mode(self) -> str:
        return self._data.get("security", {}).get("verification_mode", "any")

    def get_owner_pref(self, key: str, default: str = "") -> str:
        return self.owner_preferences.get(key, default)

    def save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w") as f:
                yaml.dump(self._data, f, default_flow_style=False, sort_keys=False)
            logger.info("Identity saved to %s", self._path)
        except Exception:
            logger.exception("Failed to save identity")

    def update_owner_preference(self, key: str, value: str) -> None:
        self._data.setdefault("owner", {}).setdefault("preferences", {})[key] = value
        self.save()
