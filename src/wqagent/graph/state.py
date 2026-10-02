"""Shared state of the WQ Agent LangGraph workflow.

Every node of the graph reads and writes a single ``WQGraphState`` dict. LangGraph
derives the graph's channels from this schema:

* ``messages`` uses :func:`~langgraph.graph.message.add_messages` as its reducer,
  so a node returning ``{"messages": [response]}`` *appends* to the conversation
  instead of replacing it.
* every other key is last-value-wins: a node returns only the keys it changed,
  and keys it omits keep their previous value. Those channels hand the *same*
  object to the next node, so nodes should build and return new lists/dicts for
  ``todos`` and friends rather than mutating what they received.

Shape of the workflow this schema is meant for::

    plan    -> plan_summary, todos, acceptance_criteria, verification_commands
    act     -> messages (via add_messages), todos updates
    verify  -> verification_results, passed
    decide  -> attempts < max_attempts ? back to act : final_answer
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from wqagent.core.state import RuntimeState

__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "TODO_STATUSES",
    "TodoItem",
    "VerificationCheck",
    "VerificationResult",
    "WQGraphState",
    "make_todo",
    "new_graph_state",
    "todo_progress",
]

#: Allowed values for :attr:`TodoItem.status`.
TODO_STATUSES: tuple[str, ...] = ("pending", "in_progress", "completed", "blocked")

#: Default retry budget of the act/verify loop.
DEFAULT_MAX_ATTEMPTS = 3


class TodoItem(TypedDict):
    """One item of the plan the actor works through."""

    id: str
    content: str
    status: str  # "pending" | "in_progress" | "completed" | "blocked"
    note: str


class VerificationResult(TypedDict):
    """Outcome of running one verification command."""

    command: str
    ok: bool
    exit_code: int | None
    stdout: str
    stderr: str


class VerificationCheck(TypedDict):
    """One acceptance criterion the verifier judged."""

    name: str
    passed: bool
    detail: str


class WQGraphState(TypedDict, total=False):
    """Shared state of the WQ Agent graph; every key is optional.

    Attributes:
        task: The user's request, verbatim.
        runtime: Workspace-scoped :class:`~wqagent.core.state.RuntimeState` shared
            by every tool call in the run.
        messages: Conversation history; merged with ``add_messages``.
        plan_summary: Human-readable summary of the plan.
        todos: The plan, item by item.
        acceptance_criteria: What "done" means for this task.
        verification_commands: Commands the verify node runs to check the result.
        verification_results: What those commands actually returned.
        verification_checks: The verifier's per-criterion verdicts.
        passed: Whether verification succeeded.
        attempts: How many act/verify rounds have run so far.
        max_attempts: Budget for those rounds.
        last_actor_summary: What the actor reported at the end of its last turn.
        last_error: Why the last verification failed (empty once it passes).
        final_answer: The answer reported back to the user.
    """

    task: str
    runtime: RuntimeState
    messages: Annotated[list[BaseMessage], add_messages]
    plan_summary: str
    todos: list[TodoItem]
    acceptance_criteria: list[str]
    verification_commands: list[str]
    verification_results: list[VerificationResult]
    verification_checks: list[VerificationCheck]
    passed: bool
    attempts: int
    max_attempts: int
    last_actor_summary: str
    last_error: str
    final_answer: str


def make_todo(
    id: str,  # noqa: A002 - matches the TodoItem field name
    content: str,
    *,
    status: str = "pending",
    note: str = "",
) -> TodoItem:
    """Build a :class:`TodoItem`, rejecting statuses outside :data:`TODO_STATUSES`."""
    if status not in TODO_STATUSES:
        raise ValueError(
            f"unknown todo status {status!r}; expected one of {', '.join(TODO_STATUSES)}"
        )
    return TodoItem(id=id, content=content, status=status, note=note)


def todo_progress(todos: list[TodoItem]) -> str:
    """One-line progress report, e.g. ``"2/3 todos completed (1 blocked)"``."""
    total = len(todos)
    done = sum(1 for todo in todos if todo.get("status") == "completed")
    blocked = sum(1 for todo in todos if todo.get("status") == "blocked")
    summary = f"{done}/{total} todos completed"
    return f"{summary} ({blocked} blocked)" if blocked else summary


def new_graph_state(
    task: str,
    runtime: RuntimeState,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    **extra: Any,
) -> WQGraphState:
    """Build the state a graph run starts from.

    Args:
        task: The user's request.
        runtime: Runtime state handed to every tool.
        max_attempts: Retry budget of the act/verify loop.
        **extra: Additional keys merged over the defaults, e.g.
            ``acceptance_criteria=["tests pass"]``.

    Raises:
        TypeError: *runtime* is not a :class:`RuntimeState`.
    """
    if not isinstance(runtime, RuntimeState):
        raise TypeError(f"runtime must be a RuntimeState, got {type(runtime).__name__}")
    state: WQGraphState = {
        "task": task,
        "runtime": runtime,
        "messages": [],
        "plan_summary": "",
        "todos": [],
        "acceptance_criteria": [],
        "verification_commands": [],
        "verification_results": [],
        "verification_checks": [],
        "passed": False,
        "attempts": 0,
        "max_attempts": max(1, int(max_attempts)),
        "last_actor_summary": "",
        "last_error": "",
        "final_answer": "",
    }
    state.update(extra)
    return state
