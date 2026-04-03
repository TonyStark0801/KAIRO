"""Dynamic tool registry — discovers and registers BaseTool subclasses."""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools._base import BaseTool, ToolMeta

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            logger.warning("Duplicate tool name %r — overwriting", tool.name)
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_all(self) -> list[ToolMeta]:
        return [t.get_meta() for t in self._tools.values()]

    def discover(self, tools_package_path: str | Path | None = None) -> None:
        from tools._base import BaseTool as BaseToolCls

        if tools_package_path is None:
            tools_package_path = (
                Path(__file__).resolve().parent.parent.parent / "tools"
            )
        else:
            tools_package_path = Path(tools_package_path)

        for importer, modname, ispkg in pkgutil.walk_packages(
            [str(tools_package_path)], prefix="tools."
        ):
            if modname == "tools._base":
                continue
            try:
                module = importlib.import_module(modname)
            except Exception:
                logger.exception("Failed to import tool module %s", modname)
                continue

            for _name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, BaseToolCls)
                    and obj is not BaseToolCls
                    and not inspect.isabstract(obj)
                ):
                    try:
                        instance = obj()
                        self.register(instance)
                    except Exception:
                        logger.exception("Failed to instantiate tool %s", _name)
