"""Configuration loader — reads kairo.yaml at startup (legacy jarvis.yaml if present)."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from core.config.models import KairoConfig

logger = logging.getLogger(__name__)


def _resolve_default_path() -> Path:
    base = Path(__file__).resolve().parent.parent.parent / "config"
    kairo_p = base / "kairo.yaml"
    legacy = base / "jarvis.yaml"
    if kairo_p.exists():
        return kairo_p
    if legacy.exists():
        logger.warning(
            "Loading legacy config/jarvis.yaml — rename to config/kairo.yaml and update paths to ~/.kairo/"
        )
        return legacy
    return kairo_p


def load_config(path: str | Path | None = None) -> KairoConfig:
    if path is None:
        path = _resolve_default_path()
    else:
        path = Path(path)

    if not path.exists():
        logger.warning("Config not found at %s — using defaults", path)
        return KairoConfig()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return KairoConfig(**raw)
