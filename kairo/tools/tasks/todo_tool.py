"""Tool: manage TODO list via TodoStore."""

from __future__ import annotations

from typing import Any

from adapters.base.platform_adapter import PlatformAdapter
from tools._base import BaseTool, ToolResult
from tools.tasks.todo_store import TodoStore


def _format_todo_list(todos: list[dict[str, Any]]) -> str:
    if not todos:
        return "You have no pending todos."
    lines: list[str] = []
    for t in todos:
        parts = [f"#{t['id']}: {t['title']}"]
        if t.get("due_date"):
            parts.append(f"due {t['due_date']}")
        if t.get("due_time"):
            parts.append(f"at {t['due_time']}")
        lines.append(" — ".join(parts) if len(parts) > 1 else parts[0])
    return "\n".join(lines)


class ManageTodosTool(BaseTool):
    @property
    def name(self) -> str:
        return "manage_todos"

    @property
    def description(self) -> str:
        return "Manage Tony's TODO list — add, list, complete, or delete items"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "complete", "delete"],
                },
                "title": {"type": "string", "description": "Title for new todo"},
                "due_date": {"type": "string", "description": "Due date (YYYY-MM-DD)"},
                "due_time": {"type": "string", "description": "Due time (HH:MM)"},
                "todo_id": {"type": "integer", "description": "ID for complete/delete"},
                "context": {"type": "string", "description": "Additional context"},
            },
            "required": ["action"],
        }

    async def execute(
        self, params: dict[str, Any], adapter: PlatformAdapter
    ) -> ToolResult:
        store = TodoStore()
        try:
            if not await store.initialize():
                return ToolResult(
                    success=False,
                    message="Could not open the todo database.",
                )

            action = params.get("action")
            if action == "add":
                title = (params.get("title") or "").strip()
                if not title:
                    return ToolResult(
                        success=False,
                        message="A title is required to add a todo.",
                    )
                due_date = params.get("due_date")
                due_time = params.get("due_time")
                context = params.get("context") or ""
                todo_id = await store.add(
                    title,
                    due_date if due_date else None,
                    due_time if due_time else None,
                    context,
                )
                return ToolResult(
                    success=True,
                    message=f"Added todo number {todo_id}.",
                    data={"todo_id": todo_id},
                )

            if action == "list":
                todos = await store.list_todos(status="pending")
                text = _format_todo_list(todos)
                return ToolResult(
                    success=True,
                    message=text,
                    data={"speak_result": True},
                )

            if action == "complete":
                todo_id = params.get("todo_id")
                if todo_id is None:
                    return ToolResult(
                        success=False,
                        message="todo_id is required to complete a todo.",
                    )
                ok = await store.complete(int(todo_id))
                if not ok:
                    return ToolResult(
                        success=False,
                        message=f"No pending todo found with id {todo_id}.",
                    )
                return ToolResult(
                    success=True,
                    message=f"Marked todo {todo_id} as completed.",
                )

            if action == "delete":
                todo_id = params.get("todo_id")
                if todo_id is None:
                    return ToolResult(
                        success=False,
                        message="todo_id is required to delete a todo.",
                    )
                ok = await store.delete(int(todo_id))
                if not ok:
                    return ToolResult(
                        success=False,
                        message=f"No todo found with id {todo_id}.",
                    )
                return ToolResult(
                    success=True,
                    message=f"Deleted todo {todo_id}.",
                )

            return ToolResult(
                success=False,
                message=f"Unknown action: {action!r}.",
            )
        finally:
            await store.close()
