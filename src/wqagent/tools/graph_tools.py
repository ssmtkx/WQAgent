"""Tools the graph nodes use for structured output and plan bookkeeping.

None of them touches the filesystem. The planner and the verifier bind a single
one of these with a forced ``tool_choice`` so the model answers with a *tool
call* whose arguments are the structured data the node needs (far more reliable
than parsing free-form JSON), and the actor calls ``todo_update`` to keep the
plan's statuses accurate while it works.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, Field

from wqagent.graph.state import TODO_STATUSES, TodoItem

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.tools import StructuredTool

__all__ = [
    "VerificationCheckItem",
    "VerificationReportInput",
    "VerificationReportTool",
    "TodoUpdateInput",
    "TodoUpdateItem",
    "TodoUpdateTool",
    "TodoWriteInput",
    "TodoWriteItem",
    "TodoWriteTool",
    "apply_todo_updates",
    "parse_todo_updates",
]

_STATUS_HELP = " | ".join(TODO_STATUSES)


# -- planner: structured plan ------------------------------------------------
class TodoWriteItem(BaseModel):
    """One planned todo item."""

    id: str = Field(description="Short stable id, e.g. 't1'.")
    content: str = Field(description="One concrete, verifiable step.")
    status: str = Field(default="pending", description=f"One of: {_STATUS_HELP}.")
    note: str = Field(default="", description="Optional detail, e.g. the file involved.")


class TodoWriteInput(BaseModel):
    """The plan the planner node records."""

    plan_summary: str = Field(description="Short paragraph describing the approach.")
    todos: list[TodoWriteItem] = Field(description="1-6 steps in execution order.")
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        description="What must be true when the task is done.",
    )
    verification_commands: list[str] = Field(
        default_factory=list,
        description="Shell commands that prove it (tests, builds); they run in the workspace root.",
    )


class TodoWriteTool:
    """Structured-output tool the planner binds to emit the plan."""

    name: str = "todo_write"
    description: str = (
        "Record the plan: a short summary, the todo list, the acceptance criteria and "
        "the shell commands that verify the result. Call this exactly once."
    )
    args_schema: type[BaseModel] = TodoWriteInput

    def run(
        self,
        plan_summary: str,
        todos: list[dict[str, Any]],
        acceptance_criteria: list[str] | None = None,
        verification_commands: list[str] | None = None,
    ) -> str:
        statuses = ", ".join(f"{todo.get('id')}={todo.get('status', 'pending')}" for todo in todos)
        return (
            f"Plan recorded: {len(todos)} todos ({statuses}); "
            f"{len(acceptance_criteria or [])} acceptance criteria; "
            f"{len(verification_commands or [])} verification commands."
        )

    def as_tool(self) -> "StructuredTool":
        """Wrap this implementation in a LangChain ``StructuredTool``."""
        from langchain_core.tools import StructuredTool

        return StructuredTool.from_function(
            func=self.run,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
        )


# -- actor: keep the plan up to date -----------------------------------------
class TodoUpdateItem(BaseModel):
    """A status change for one existing todo."""

    id: str = Field(description="Id of the todo to update.")
    status: str = Field(description=f"One of: {_STATUS_HELP}.")
    note: str = Field(default="", description="Short note about what happened.")


class TodoUpdateInput(BaseModel):
    """Status updates the actor emits while it works."""

    updates: list[TodoUpdateItem] = Field(description="One entry per todo whose status changed.")


def _as_items(raw: Any) -> list[Any]:
    """Iterate whatever the model sent, without ever raising."""
    if isinstance(raw, Mapping):
        return [raw]
    if raw is None or isinstance(raw, (bool, int, float, str)):
        return []
    try:
        return list(raw)
    except TypeError:  # not iterable at all
        return []


def parse_todo_updates(
    raw: Iterable[Mapping[str, Any]] | None,
) -> tuple[list[TodoUpdateItem], list[str]]:
    """Validate raw tool arguments; return ``(updates, problems)``.

    Never raises: entries that cannot be parsed or carry an unknown status are
    reported in *problems* instead.
    """
    updates: list[TodoUpdateItem] = []
    problems: list[str] = []
    for index, item in enumerate(_as_items(raw)):
        try:
            update = TodoUpdateItem(**dict(item))
        except Exception as exc:  # noqa: BLE001 - report, do not raise at the node boundary
            problems.append(f"update #{index}: {exc}")
            continue
        if update.status not in TODO_STATUSES:
            problems.append(
                f"update #{index} ({update.id}): unknown status {update.status!r}; "
                f"expected one of {_STATUS_HELP}"
            )
            continue
        updates.append(update)
    return updates, problems


def apply_todo_updates(
    todos: Sequence[Mapping[str, Any]], raw_updates: Iterable[Mapping[str, Any]] | None
) -> tuple[list[TodoItem], list[str]]:
    """Apply status updates by id; return ``(new todos, notes)``.

    Unknown ids are reported in the notes instead of being added, so the plan can
    only be changed through the planner.
    """
    by_id = {str(todo.get("id")): index for index, todo in enumerate(todos)}
    updated: list[TodoItem] = [
        TodoItem(
            id=str(todo.get("id", "")),
            content=str(todo.get("content", "")),
            status=str(todo.get("status", "pending")),
            note=str(todo.get("note", "")),
        )
        for todo in todos
    ]
    updates, notes = parse_todo_updates(raw_updates)
    for update in updates:
        index = by_id.get(update.id)
        if index is None:
            notes.append(f"unknown todo id {update.id!r}; ignored")
            continue
        updated[index]["status"] = update.status
        if update.note:
            updated[index]["note"] = update.note
    return updated, notes


class TodoUpdateTool:
    """Tool the actor calls to move plan items through their statuses."""

    name: str = "todo_update"
    description: str = (
        "Change the status of plan items while you work (mark a todo in_progress when "
        "you start it, completed when it is done and verified, blocked when you cannot "
        "continue). Pass one entry per changed todo."
    )
    args_schema: type[BaseModel] = TodoUpdateInput

    def run(self, updates: list[dict[str, Any]]) -> str:
        parsed, problems = parse_todo_updates(updates)
        if not parsed and problems:
            return "Error: " + "; ".join(problems)
        summary = ", ".join(f"{update.id} -> {update.status}" for update in parsed)
        text = f"Recorded {len(parsed)} todo update(s): {summary}."
        if problems:
            text += " Ignored: " + "; ".join(problems)
        return text

    def as_tool(self) -> "StructuredTool":
        """Wrap this implementation in a LangChain ``StructuredTool``."""
        from langchain_core.tools import StructuredTool

        return StructuredTool.from_function(
            func=self.run,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
        )


# -- verifier: structured verdict --------------------------------------------
class VerificationCheckItem(BaseModel):
    """One acceptance criterion or command the verifier looked at."""

    name: str = Field(description="The criterion or command being checked.")
    passed: bool = Field(description="Whether it holds.")
    detail: str = Field(default="", description="The evidence you saw.")


class VerificationReportInput(BaseModel):
    """The verdict the verifier node records."""

    passed: bool = Field(
        description="True only when every acceptance criterion holds and every verification command succeeded."
    )
    reason: str = Field(description="One short paragraph explaining the verdict.")
    checks: list[VerificationCheckItem] = Field(
        default_factory=list, description="One entry per acceptance criterion or command."
    )
    recommended_next_instruction: str = Field(
        default="",
        description="A concrete instruction for the actor when passed is false; empty when passed is true.",
    )


class VerificationReportTool:
    """Structured-output tool the verifier binds to emit its verdict."""

    name: str = "verification_report"
    description: str = (
        "Report the verification verdict: whether the task passed, why, the individual "
        "checks you performed, and the next instruction for the actor if it failed."
    )
    args_schema: type[BaseModel] = VerificationReportInput

    def run(
        self,
        passed: bool,
        reason: str,
        checks: list[dict[str, Any]] | None = None,
        recommended_next_instruction: str = "",
    ) -> str:
        verdict = "passed" if passed else "failed"
        return (
            f"Verification {verdict}: {reason} "
            f"({len(checks or [])} checks recorded; next: {recommended_next_instruction or 'n/a'})"
        )

    def as_tool(self) -> "StructuredTool":
        """Wrap this implementation in a LangChain ``StructuredTool``."""
        from langchain_core.tools import StructuredTool

        return StructuredTool.from_function(
            func=self.run,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
        )
