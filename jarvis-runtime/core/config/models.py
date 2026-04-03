"""Pydantic v2 configuration models for Jarvis runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class OllamaConfig(BaseModel):
    host: str = "localhost"
    port: int = 11434
    model: str = "llama3.1"
    embed_model: str = "nomic-embed-text"


class SessionConfig(BaseModel):
    idle_timeout_seconds: int = Field(default=30, alias="idle_timeout")
    wake_window_seconds: int = Field(default=3, alias="wake_window")
    tool_timeout_seconds: int = Field(default=30, alias="tool_timeout")

    model_config = {"populate_by_name": True}


class MemoryConfig(BaseModel):
    chroma_path: str = "~/.jarvis/chroma"
    behavioral_db: str = "~/.jarvis/behavior.db"

    @field_validator("chroma_path", "behavioral_db")
    @classmethod
    def expand_user(cls, v: str) -> str:
        return str(Path(v).expanduser())


class RedisConfig(BaseModel):
    enabled: bool = Field(default=False, alias="redis_enabled")
    url: str = "redis://localhost:6379/0"

    model_config = {"populate_by_name": True}


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
    face_embedding_path: str = "~/.jarvis/face_embedding.npy"

    @field_validator("face_embedding_path")
    @classmethod
    def expand_face_path(cls, v: str) -> str:
        return str(Path(v).expanduser())


class JarvisConfig(BaseModel):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    projects: dict[str, ProjectEntry] = Field(default_factory=dict)
    workspace_modes: dict[str, WorkspaceModeEntry] = Field(default_factory=dict)
    paths: PathsConfig = Field(default_factory=PathsConfig)
