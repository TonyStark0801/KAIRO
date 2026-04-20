"""Pydantic v2 configuration models for Kairo runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class OllamaConfig(BaseModel):
    host: str = "localhost"
    port: int = 11434
    model: str = "qwen:4b"
    fast_model: str = "qwen:4b"
    embed_model: str = "nomic-embed-text"


class SessionConfig(BaseModel):
    idle_timeout_seconds: int = Field(default=30, alias="idle_timeout")
    wake_window_seconds: int = Field(default=3, alias="wake_window")
    tool_timeout_seconds: int = Field(default=30, alias="tool_timeout")

    model_config = {"populate_by_name": True}


class MemoryConfig(BaseModel):
    chroma_path: str = "~/.kairo/chroma"
    behavioral_db: str = "~/.kairo/behavior.db"

    @field_validator("chroma_path", "behavioral_db")
    @classmethod
    def expand_user(cls, v: str) -> str:
        return str(Path(v).expanduser())


class RedisConfig(BaseModel):
    enabled: bool = Field(default=False, alias="redis_enabled")
    url: str = "redis://localhost:6379/0"

    model_config = {"populate_by_name": True}


class SttConfig(BaseModel):
    """Speech-to-text configuration.

    engine choices (in priority order if "auto"):
      mlx-whisper    — Apple Silicon Neural Engine (fastest on macOS M-series)
      faster-whisper — CTranslate2 int8, CPU/CUDA (Windows/Linux/Intel Mac)
      pywhispercpp   — whisper.cpp bindings, last resort
      auto           — tries mlx → faster-whisper → pywhispercpp

    Two models are used:
      model      — for full command transcription (small.en recommended)
      wake_model — for wake-word detection only (tiny.en recommended — runs
                   in ~80ms, just needs to recognise "kairo")
    """

    engine: str = "auto"
    model: str = "small.en"       # command transcription
    wake_model: str = "tiny.en"   # wake-word detection — ultra-lightweight
    cpp_fallback_model: str = "tiny.en"


class WakeConfig(BaseModel):
    """Wake detection: STT keyword (default) or streaming openWakeWord models."""

    engine: str = Field(default="stt_keyword", description="stt_keyword | openwakeword")
    openwakeword_models: list[str] = Field(default_factory=list)
    openwakeword_threshold: float = 0.5
    openwakeword_inference_framework: str = "tflite"


class ProjectEntry(BaseModel):
    name: str
    path: str
    intellij_module: str | None = None


class WorkspaceStep(BaseModel):
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)


class WorkspaceModeEntry(BaseModel):
    description: str = ""
    steps: list[WorkspaceStep] = Field(default_factory=list)


class PathsConfig(BaseModel):
    face_embedding_path: str = "~/.kairo/face_embedding.npy"

    @field_validator("face_embedding_path")
    @classmethod
    def expand_face_path(cls, v: str) -> str:
        return str(Path(v).expanduser())


class BrowserConfig(BaseModel):
    app_name: str = "Brave Browser"


class GroqConfig(BaseModel):
    api_key_env: str = "GROQ_API_KEY"
    model: str = "llama-3.3-70b-versatile"
    max_tokens: int = 1024
    temperature: float = 0.7


class ProactiveConfig(BaseModel):
    enabled: bool = True
    check_interval: int = 60
    morning_briefing: bool = True
    todo_reminders: bool = True
    focus_suggestions: bool = True


class ObserverConfig(BaseModel):
    """Phase 2 context observer — polls active window, browser tab, clipboard."""

    platform: str = "macos"
    poll_interval_ms: int = Field(default=800, ge=200, le=5000)
    capture_frontmost_window: bool = True
    capture_browser_tab: bool = True
    capture_clipboard: bool = True

    @property
    def poll_interval_sec(self) -> float:
        return self.poll_interval_ms / 1000.0


class KairoConfig(BaseModel):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    wake: WakeConfig = Field(default_factory=WakeConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    projects: dict[str, ProjectEntry] = Field(default_factory=dict)
    workspace_modes: dict[str, WorkspaceModeEntry] = Field(default_factory=dict)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    groq: GroqConfig = Field(default_factory=GroqConfig)
    proactive: ProactiveConfig = Field(default_factory=ProactiveConfig)
    observer: ObserverConfig = Field(default_factory=ObserverConfig)
