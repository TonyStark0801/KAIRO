"""Base class for all tool plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from adapters.base.platform_adapter import PlatformAdapter


@dataclass
class ToolMeta:
    name: str
    description: str
    parameters_schema: dict[str, Any]


@dataclass
class ToolResult:
    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]: ...

    def get_meta(self) -> ToolMeta:
        return ToolMeta(
            name=self.name,
            description=self.description,
            parameters_schema=self.parameters_schema,
        )

    @abstractmethod
    async def execute(
        self, params: dict[str, Any], adapter: PlatformAdapter
    ) -> ToolResult: ...
