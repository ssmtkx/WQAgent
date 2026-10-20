"""The implementation specialist (``codeAgent``).

:func:`run_code_agent` is a ReAct agent carrying the workspace toolset plus
``todo_update``: it implements one planner instruction, keeps the todo list up to
date *while* it works, and reports what changed::

    result = run_code_agent(state, "t2: add the slugify helper and run pytest")
    if result["ok"]:
        state["todos"] = result["todos"]

Todo changes are applied as they happen (and handed to the optional ``persist``
hook), so a run that dies half-way still leaves the plan accurate. The
layered-memory snapshot the agent starts from is built by
:func:`build_memory_snapshot` - today from what the state already carries, later
from the real memory layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from wqagent.core.agent import last_ai_text, react_loop, write_event
from wqagent.core.state import RuntimeState
from wqagent.graph.memory import (
    build_layered_memory,
    format_layered_memory_for_prompt,
    memory_event,
)
from wqagent.graph.state import (
    TodoItem,
    normalise_todos,
    render_todos,
    todo_progress,
)
from wqagent.prompts.stage3 import CODE_AGENT_PROMPT
from wqagent.providers import create_model
from wqagent.tools.graph_tools import TodoUpdateTool, apply_todo_updates
from wqagent.tools.registry import build_tools

__all__ = [
    "DEFAULT_MAX_LOOPS",
    "MemorySnapshot",
    "build_code_messages",
    "build_memory_snapshot",
    "run_code_agent",
]

#: How many ReAct rounds the agent may use by default.
DEFAULT_MAX_LOOPS = 10

#: Event sink: called with ``tool_call`` / ``todo_update`` / ``tool_result`` events.
EventWriter = Callable[[dict[str, Any]], None]

#: Called with the full todo list after every applied ``todo_update``.
TodoPersister = Callable[[list[TodoItem]], None]


@dataclass(frozen=True)
class MemorySnapshot:
    """What the agent is told about the work done so far.

    Two layers today: ``session`` (this run - plan progress, the last actor
    report, the last verification error) and ``notes`` (durable findings such as
    the search agent's research notes). The real layered memory - compressed
    history, retrieved file excerpts, notepad contents - plugs into
    :func:`build_memory_snapshot` later without changing this signature.
    """

    session: str = ""
    notes: str = ""

    def __bool__(self) -> bool:
        return bool(self.session.strip() or self.notes.strip())

    def render(self) -> str:
        """Render the layers as prompt sections."""
        parts: list[str] = []
        if self.session.strip():
            parts.append(f"Session context:\n{self.session.strip()}")
        if self.notes.strip():
            parts.append(f"Memory notes:\n{self.notes.strip()}")
        return "\n\n".join(parts)


def build_memory_snapshot(
    state: Mapping[str, Any] | None = None,
    memory: MemorySnapshot | str | None = None,
) -> MemorySnapshot:
    """Assemble the memory snapshot the agent starts from.

    Args:
        state: The graph state; its ``todos`` / ``plan_summary`` /
            ``last_actor_summary`` / ``last_error`` become the session layer and
            ``research_notes`` the durable layer.
        memory: An explicit snapshot (or a plain string used as the notes layer);
            when given it wins over what *state* carries.
    """
    if isinstance(memory, MemorySnapshot):
        return memory
    source = state if isinstance(state, Mapping) else {}

    if isinstance(memory, str) and memory.strip():
        notes = memory.strip()
    else:
        notes = str(source.get("research_notes") or "").strip()

    parts: list[str] = []
    todos = normalise_todos(source.get("todos"))
    if todos:
        parts.append(f"Plan progress - {todo_progress(todos)}\n{render_todos(todos)}")
    for label, key in (
        ("Plan", "plan_summary"),
        ("Last actor report", "last_actor_summary"),
        ("Last verification error", "last_error"),
    ):
        value = str(source.get(key) or "").strip()
        if value:
            parts.append(f"{label}: {value}")
    return MemorySnapshot(session="\n".join(parts), notes=notes)


def _code_agent_input(
    state: Mapping[str, Any] | None,
    instruction: str,
    memory: Mapping[str, Any] | None,
    *,
    session: MemorySnapshot | str | None = None,
) -> str:
    """The agent's human message: task + instruction + session + layered memory.

    The layered memory already carries the plan progress, the last implementer
    report and the last failure, so the agent-local session snapshot is only
    rendered when the caller passes one explicitly (``memory=``/``session=``).
    """
    source = state if isinstance(state, Mapping) else {}
    task = str(source.get("task") or "").strip() or "(no task given)"
    request = str(instruction or "").strip()

    lines = [f"Task:\n{task}"]
    if request:
        lines.append(f"Planner instruction:\n{request}")
    if session is not None:
        rendered = build_memory_snapshot(source, session).render()
        if rendered:
            lines.append(rendered)
    if memory:
        lines.append(
            "Layered memory (rules, current task state, durable notes and compressed "
            "history):\n" + format_layered_memory_for_prompt(memory)
        )
    lines.append(
        "Implement the instruction, keep the todo list up to date as you go, and end with "
        "a concise summary of the files you changed and the checks you ran."
    )
    return "\n\n".join(lines)


def build_code_messages(
    state: Mapping[str, Any] | None,
    instruction: str,
    *,
    memory: MemorySnapshot | str | None = None,
    layered: Mapping[str, Any] | None = None,
) -> list[BaseMessage]:
    """Build the opening messages: prompt + task + instruction + session + memory.

    ``memory`` is the agent-local session/notes snapshot; ``layered`` is the
    runtime's three-layer memory (see :mod:`wqagent.graph.memory`).
    """
    return [
        SystemMessage(content=CODE_AGENT_PROMPT),
        HumanMessage(content=_code_agent_input(state, instruction, layered, session=memory)),
    ]


def run_code_agent(
    state: Mapping[str, Any] | None,
    instruction: str,
    *,
    writer: EventWriter | None = None,
    max_loops: int = DEFAULT_MAX_LOOPS,
    model: Any | None = None,
    tools: Sequence[Any] | None = None,
    memory: MemorySnapshot | str | None = None,
    persist: TodoPersister | None = None,
) -> dict[str, Any]:
    """Implement *instruction* inside the workspace and report what changed.

    Args:
        state: The graph state holding ``task``, ``runtime``, ``todos`` and the
            memory the agent should start from.
        instruction: What to implement (usually one todo from the plan).
        writer: Event sink; defaults to LangGraph's custom stream writer when the
            agent runs inside a graph.
        max_loops: Maximum tool rounds.
        model: Chat model; defaults to ``create_model()`` (``.env``).
        tools: Tool override; defaults to ``build_tools(runtime) + [todo_update]``.
        memory: Explicit memory snapshot (see :func:`build_memory_snapshot`).
        persist: Called with the todo list after every applied ``todo_update`` -
            the hook that keeps the plan durable.

    Returns:
        ``{"ok": bool, "error": str, "summary": str, "todos": [...],
        "messages": [...], "tool_events": [...]}`` - never raises: a model or tool
        failure is reported while the work already done is kept.
    """
    source = state if isinstance(state, Mapping) else {}
    todos = normalise_todos(source.get("todos"))

    runtime = source.get("runtime")
    if not isinstance(runtime, RuntimeState):
        return _failure(
            build_code_messages(source, instruction, memory=memory),
            todos,
            "state has no 'runtime' RuntimeState; build the state with new_graph_state()",
        )

    chat = model if model is not None else create_model()
    tool_list = (
        list(tools)
        if tools is not None
        else [*build_tools(runtime), TodoUpdateTool().as_tool()]
    )

    tool_events: list[dict[str, Any]] = []
    error = ""

    def emit(event: dict[str, Any]) -> None:
        tool_events.append(event)
        write_event(writer, event)

    # the runtime's three-layer memory rides along in the opening message
    layered: dict[str, Any] = {}
    try:
        layered = dict(build_layered_memory(source, node="codeAgent"))
    except Exception:  # noqa: BLE001 - memory must never break the agent
        layered = {}
    emit(memory_event(layered, node="codeAgent"))
    messages = build_code_messages(source, instruction, memory=memory, layered=layered)

    def on_event(kind: str, payload: dict[str, Any]) -> None:
        nonlocal todos
        loop = int(payload.get("loop") or 0)
        if kind == "tool_call":
            name = str(payload.get("name") or "")
            args = dict(payload.get("args") or {})
            emit({"type": "tool_call", "name": name, "args": args, "loop": loop})
            if name == TodoUpdateTool.name:
                todos, notes = apply_todo_updates(todos, args.get("updates"))
                emit(
                    {
                        "type": "todo_update",
                        "updates": args.get("updates"),
                        "todos": [dict(todo) for todo in todos],
                        "notes": notes,
                        "loop": loop,
                    }
                )
                if persist is not None:
                    try:
                        persist([dict(todo) for todo in todos])
                    except Exception:  # pragma: no cover - persistence is best effort
                        pass
        elif kind == "tool_result":
            result = payload.get("result")
            data = (
                dict(result)
                if isinstance(result, Mapping)
                else {"ok": False, "error": str(result or "unexpected tool result")}
            )
            # same shape as the ReAct loop's tool_result: the envelope dict
            emit(
                {
                    "type": "tool_result",
                    "name": str(payload.get("name") or ""),
                    "result": data,
                    "diff": payload.get("diff"),
                    "loop": loop,
                }
            )

    try:
        for _ in react_loop(
            messages, model=chat, tools=tool_list, max_loops=max_loops, on_event=on_event
        ):
            pass
    except Exception as exc:  # noqa: BLE001 - partial progress is still worth keeping
        error = f"{type(exc).__name__}: {exc}"

    summary = last_ai_text(messages) or _digest(tool_events, todos)
    return {
        "ok": not error and bool(summary or tool_events),
        "error": error,
        "summary": summary,
        "todos": todos,
        "messages": messages,
        "tool_events": tool_events,
    }


def _failure(
    messages: list[BaseMessage], todos: list[TodoItem], error: str
) -> dict[str, Any]:
    return {
        "ok": False,
        "error": error,
        "summary": "",
        "todos": todos,
        "messages": messages,
        "tool_events": [],
    }


def _digest(events: Sequence[Mapping[str, Any]], todos: Sequence[TodoItem]) -> str:
    """A fallback summary when the model never wrote one (e.g. it hit max_loops)."""
    files: list[str] = []
    checks = 0
    for event in events:
        if event.get("type") != "tool_call":
            continue
        name = str(event.get("name") or "")
        args = event.get("args") if isinstance(event.get("args"), Mapping) else {}
        if name in {"file_write", "file_edit"}:
            path = str(args.get("file_path") or "")
            if path and path not in files:
                files.append(path)
        elif name == "bash":
            checks += 1
    lines: list[str] = []
    if files:
        lines.append("Files changed: " + ", ".join(files))
    if checks:
        lines.append(f"Checks run: {checks}")
    if todos:
        lines.append(todo_progress(todos))
    return "\n".join(lines)
