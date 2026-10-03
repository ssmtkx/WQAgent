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

    planner  -> plan_summary, todos, acceptance_criteria, verification_commands
    delegate -> research_notes, sources, code_agent_summary, agent_handoffs
    verify   -> verification_results, verification_checks, passed
    decide   -> attempts < max_attempts ? back to planner : final_answer
    context  -> context_*, compression_events, memory_snapshot, history_summary

The ``context_*`` / ``memory_snapshot`` / ``history_summary`` fields describe the
layered memory of a run; :mod:`wqagent.graph.memory` builds the snapshot and
:data:`~wqagent.graph.memory.DEFAULT_CONTEXT_TOKEN_LIMIT` is the default budget.
"""

from __future__ import annotations

from typing import Annotated, Any, Mapping, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from wqagent.core.state import RuntimeState
from wqagent.graph.memory import (
    DEFAULT_CONTEXT_TOKEN_LIMIT,
    CompressionEvent,
    LayeredMemory,
    context_token_limit_from_env,
)

__all__ = [
    "AgentHandoff",
    "CompressionEvent",
    "DEFAULT_MAX_ATTEMPTS",
    "LayeredMemory",
    "SourceItem",
    "TODO_MARKS",
    "TODO_STATUSES",
    "TodoItem",
    "VerificationCheck",
    "VerificationResult",
    "WQGraphState",
    "make_todo",
    "new_graph_state",
    "normalise_todos",
    "render_todos",
    "todo_progress",
]

#: Allowed values for :attr:`TodoItem.status`.
TODO_STATUSES: tuple[str, ...] = ("pending", "in_progress", "completed", "blocked")

#: Checkbox marks used when rendering a todo list into a prompt.
TODO_MARKS: dict[str, str] = {
    "pending": "[ ]",
    "in_progress": "[~]",
    "completed": "[x]",
    "blocked": "[!]",
}

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


class SourceItem(TypedDict, total=False):
    """One web source kept from a searchAgent handoff."""

    title: str
    url: str
    content: str
    score: float


class AgentHandoff(TypedDict, total=False):
    """One delegation from the planner/supervisor to a specialist agent."""

    from_agent: str
    to_agent: str
    instruction: str
    result: str


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
        attempts: How many planner/verify rounds have run so far.
        max_attempts: Budget for those rounds.
        last_actor_summary: The last supervisor summary for this round (the actor's
            report in the pre-stage-3 workflow).
        last_error: Why the last verification failed (empty once it passes).
        research_notes: External findings gathered by searchAgent.
        sources: The web sources those findings came from.
        agent_handoffs: Every delegation the planner made, in order.
        code_agent_summary: What codeAgent reported after implementing.
        context_summary: Compressed history of this run (replaces older turns).
        context_token_count: Size of the current context, in estimated tokens.
        context_token_limit: The budget ``context_token_count`` is compared against.
        context_should_compress: Whether the context is over that budget.
        context_next_node: Which node the compressor decided should run next.
        compression_events: One record per compression, newest last.
        memory_snapshot: The layered memory last assembled for a node
            (:func:`wqagent.graph.memory.build_layered_memory`); empty until built.
        history_summary: Compressed history carried in the state; when empty, the
            snapshot falls back to ``HISTORY_SUMMARY.md`` in the workspace.
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
    research_notes: str
    sources: list[SourceItem]
    agent_handoffs: list[AgentHandoff]
    code_agent_summary: str
    context_summary: str
    context_token_count: int
    context_token_limit: int
    context_should_compress: bool
    context_next_node: str
    compression_events: list[CompressionEvent]
    memory_snapshot: LayeredMemory
    history_summary: str
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


def normalise_todos(value: Any) -> list[TodoItem]:
    """Coerce anything list-like of mappings into valid :class:`TodoItem` dicts.

    A single object becomes a one-item list; scalars and junk are skipped, so a
    hand-written or model-produced value can be used without raising.
    """
    if isinstance(value, Mapping):
        items: list[Any] = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = []
    todos: list[TodoItem] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        todos.append(
            TodoItem(
                id=str(item.get("id", "")),
                content=str(item.get("content", "")),
                status=str(item.get("status", "pending")),
                note=str(item.get("note", "")),
            )
        )
    return todos


def render_todos(todos: Sequence[Mapping[str, Any]]) -> str:
    """Render a todo list as ``[x] t1: content  (note)`` lines."""
    lines: list[str] = []
    for todo in todos or []:
        mark = TODO_MARKS.get(str(todo.get("status")), TODO_MARKS["pending"])
        note = str(todo.get("note") or "")
        lines.append(
            f"{mark} {todo.get('id')}: {todo.get('content')}" + (f"  ({note})" if note else "")
        )
    return "\n".join(lines) or "(no todos)"


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
        "research_notes": "",
        "sources": [],
        "agent_handoffs": [],
        "code_agent_summary": "",
        "context_summary": "",
        "context_token_count": 0,
        "context_token_limit": context_token_limit_from_env(DEFAULT_CONTEXT_TOKEN_LIMIT),
        "context_should_compress": False,
        "context_next_node": "",
        "compression_events": [],
        "memory_snapshot": {},
        "history_summary": "",
        "final_answer": "",
    }
    state.update(extra)
    return state
