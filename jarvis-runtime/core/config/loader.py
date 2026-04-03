"""Configuration loader — reads jarvis.yaml once at startup."""

from __future__ import annotations

from pathlib import Path

import yaml

from core.config.models import JarvisConfig


def load_config(path: str | Path | None = None) -> JarvisConfig:
    if path is None:
        path = Path(__file__).resolve().parent.parent.parent / "config" / "jarvis.yaml"
    path = Path(path)
    if not path.exists():
        return JarvisConfig()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return JarvisConfig(**raw)
