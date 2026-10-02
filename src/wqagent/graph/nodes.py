"""LangGraph nodes of the WQ Agent workflow: plan -> act -> verify -> route.

* :func:`planner_node` - turns the task into todos, acceptance criteria and
  verification commands, and revises them after a failed verification.
* :func:`actor_node` - runs the ReAct loop with the workspace tools plus
  ``todo_update``, forwarding one event per step.
* :func:`verifier_node` - runs the verification commands, inspects the workspace
  with read-only tools and records a structured verdict.
* :func:`verifier_route` - stops, or goes back to the planner for another attempt.

Nodes take their model from :func:`get_model` (lazily ``create_model()`` from
``.env``); pass ``model=...`` to a single node, or call :func:`set_model` once to
pin one for the whole graph.

Example wiring::

    graph = StateGraph(WQGraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("actor", actor_node)
    graph.add_node("verifier", verifier_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "actor")
    graph.add_edge("actor", "verifier")
    graph.add_conditional_edges("verifier", verifier_route, {"planner": "planner", "final": END})
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable, Mapping, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from wqagent.core.agent import message_text, react_loop
from wqagent.core.state import RuntimeState
from wqagent.graph.state import (
    DEFAULT_MAX_ATTEMPTS,
    TODO_STATUSES,
    TodoItem,
    VerificationCheck,
    VerificationResult,
    WQGraphState,
    make_todo,
)
from wqagent.prompts.stage1 import ACTOR_PROMPT
from wqagent.prompts.stage2 import FINAL_PROMPT, PLANNER_PROMPT, VERIFIER_PROMPT
from wqagent.providers import create_model
from wqagent.tools.bash_tool import BashTool
from wqagent.tools.graph_tools import (
    TodoUpdateTool,
    TodoWriteTool,
    VerificationReportTool,
    apply_todo_updates,
)
from wqagent.tools.registry import build_read_only_tools, build_tools

__all__ = [
    "DEFAULT_ACTOR_LOOPS",
    "DEFAULT_VERIFIER_LOOPS",
    "VERIFY_OUTPUT_CHARS",
    "actor_node",
    "final_node",
    "format_final_answer",
    "get_model",
    "planner_node",
    "run_verification_commands",
    "set_model",
    "verifier_node",
    "verifier_route",
]

#: ReAct loops the actor may use per attempt.
DEFAULT_ACTOR_LOOPS = 10
#: Read-only inspection loops the verifier may use before reporting.
DEFAULT_VERIFIER_LOOPS = 6
#: stdout/stderr characters kept per verification command in the state.
VERIFY_OUTPUT_CHARS = 4_000

_TODO_MARK = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]", "blocked": "[!]"}

_MODEL: Any | None = None


def get_model() -> Any:
    """Return the chat model the nodes use, creating it from ``.env`` on first use."""
    global _MODEL
    if _MODEL is None:
        _MODEL = create_model()
    return _MODEL


def set_model(model: Any | None) -> None:
    """Pin the model used by every node (``None`` restores lazy ``create_model()``)."""
    global _MODEL
    _MODEL = model


# -- state helpers -----------------------------------------------------------
def _runtime(state: Mapping[str, Any]) -> RuntimeState:
    runtime = state.get("runtime")
    if not isinstance(runtime, RuntimeState):
        raise ValueError(
            "WQGraphState has no 'runtime' RuntimeState; build the state with new_graph_state()"
        )
    return runtime


def _todos(state: Mapping[str, Any]) -> list[TodoItem]:
    """Normalised copy of the plan (never the objects held by the state)."""
    return [
        TodoItem(
            id=str(todo.get("id", "")),
            content=str(todo.get("content", "")),
            status=str(todo.get("status", "pending")),
            note=str(todo.get("note", "")),
        )
        for todo in (state.get("todos") or [])
    ]


def _text(value: Any) -> str:
    """A usable string from model output - anything that is not a string is dropped."""
    return value.strip() if isinstance(value, str) else ""


def _flag(value: Any) -> bool:
    """A strict boolean from model output.

    Only real booleans, 1/0 and the obvious words count - so ``"passed": "false"``
    is a failure rather than a truthy string.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "pass", "passed", "ok"}
    return False


def _as_items(value: Any) -> list[Any]:
    """Iterate whatever the model sent, without ever raising.

    A single object becomes a one-item list (models often drop the array), a
    scalar or string is treated as junk, and anything uniterable becomes empty.
    """
    if isinstance(value, Mapping):
        return [value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return []
    try:
        return list(value)
    except TypeError:  # not iterable at all
        return []


def _text_list(value: Any) -> list[str]:
    """Coerce whatever the model returned into a list of non-empty strings.

    Structured output is never trusted: a string, a list, a stray number or a
    nested object all end up as ``list[str]`` (or an empty list) instead of
    raising inside a node.
    """
    if value is None or isinstance(value, (bool, int, float, Mapping)):
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    result: list[str] = []
    for item in _as_items(value):
        if item is None or isinstance(item, (Mapping, list, tuple, set)):
            continue  # nested junk is dropped, not stringified
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def _render_todos(todos: Sequence[TodoItem]) -> str:
    lines = []
    for todo in todos:
        mark = _TODO_MARK.get(str(todo.get("status")), "[ ]")
        note = str(todo.get("note") or "")
        lines.append(f"{mark} {todo.get('id')}: {todo.get('content')}" + (f"  ({note})" if note else ""))
    return "\n".join(lines) or "(no todos)"


def _coerce_todos(raw: Any, *, fallback: Sequence[TodoItem] = ()) -> list[TodoItem]:
    """Turn tool arguments into valid todos, falling back to the previous plan.

    Tolerates a single object instead of a list, non-object entries, missing ids,
    unknown statuses and empty content - anything unusable is skipped rather than
    raising.
    """
    todos: list[TodoItem] = []
    for index, item in enumerate(_as_items(raw)):
        if not isinstance(item, Mapping):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        status = str(item.get("status") or "pending").strip()
        if status not in TODO_STATUSES:
            status = "pending"
        todos.append(
            make_todo(
                str(item.get("id") or f"t{index + 1}"),
                content,
                status=status,
                note=str(item.get("note") or ""),
            )
        )
    return todos or _todos({"todos": list(fallback)})


def _coerce_checks(raw: Any) -> list[VerificationCheck]:
    """Normalise the verifier's ``checks``; unusable entries are dropped."""
    checks: list[VerificationCheck] = []
    for item in _as_items(raw):
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        checks.append(
            VerificationCheck(
                name=name,
                passed=_flag(item.get("passed")),
                detail=_text(item.get("detail")),
            )
        )
    return checks


# -- model plumbing ----------------------------------------------------------
def _first_tool_args(response: Any, name: str) -> dict[str, Any]:
    for call in getattr(response, "tool_calls", None) or []:
        if not name or str(call.get("name")) == name:
            return dict(call.get("args") or {})
    return {}


def _balanced_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` block of *text*, ignoring braces in strings."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object in *text*, tolerating ```json fences and prose."""
    body = (text or "").strip()
    if "```" in body:
        start = body.find("```")
        end = body.find("```", start + 3)
        if end > start:
            block = body[start + 3 : end]
            body = block.split("\n", 1)[1].strip() if "\n" in block else block.strip()
    while body:
        candidate = _balanced_object(body)
        if candidate is None:
            return {}
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            body = body[body.find(candidate) + len(candidate) :]
            continue
        return value if isinstance(value, dict) else {}
    return {}


#: Used when a model cannot do forced tool calls and has to answer with JSON.
JSON_INSTRUCTION = "Answer with a single JSON object and nothing else. Required keys: {keys}."


def _request_structured(
    chat: Any, tool: Any, messages: list[BaseMessage], *, keys: str = ""
) -> dict[str, Any]:
    """Ask *chat* for one structured answer and return its arguments.

    Tries a forced tool call first (``tool_choice=<tool>``) and then a plain JSON
    answer, so models without tool-choice support still work. Returns ``{}`` when
    neither yields an object - callers must handle that.
    """
    spec = tool.as_tool()
    try:
        try:
            bound = chat.bind_tools([spec], tool_choice=spec.name)
        except TypeError:  # model/runtime without tool_choice support
            bound = chat.bind_tools([spec])
        response = bound.invoke(messages)
        args = _first_tool_args(response, spec.name)
        if args:
            return args
        parsed = _json_object(message_text(response))
        if parsed:
            return parsed
    except Exception:  # noqa: BLE001 - fall through to the JSON attempt
        pass
    try:
        fallback = [
            *messages,
            HumanMessage(content=JSON_INSTRUCTION.format(keys=keys or spec.name)),
        ]
        response = chat.invoke(fallback)
    except Exception:  # noqa: BLE001 - the node reports the empty answer
        return {}
    return _json_object(message_text(response))


# -- event plumbing ----------------------------------------------------------
def _stream_writer() -> Callable[[Any], None] | None:
    """LangGraph's custom stream writer (``stream_mode="custom"``), if available."""
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:  # noqa: BLE001 - not running inside a graph
        return None
    return writer if callable(writer) else None


def _forward(
    event: dict[str, Any], node: str, on_event: Callable[[dict[str, Any]], None] | None
) -> None:
    """Hand one ReAct event to the node's callback and to the graph's stream.

    The streamed copy is tagged with ``node``: LangGraph's custom chunks are
    otherwise anonymous, and the consumer needs to know who produced them.
    """
    if on_event is not None:
        try:
            on_event(event)
        except Exception:  # pragma: no cover - a broken callback must not kill the run
            pass
    writer = _stream_writer()
    if writer is not None:
        try:
            writer({**event, "node": node})
        except Exception:  # pragma: no cover - streaming is best effort
            pass


# -- planner -----------------------------------------------------------------
def planner_node(state: WQGraphState, *, model: Any | None = None) -> dict[str, Any]:
    """Create the plan, or revise it after the verifier failed.

    Returns the ``plan_summary``, ``todos``, ``acceptance_criteria`` and
    ``verification_commands`` updates. When a plan already exists and nothing
    failed, the current plan is returned unchanged (no model call).
    """
    _runtime(state)  # fail fast if the graph state was not built properly
    existing = _todos(state)
    last_error = str(state.get("last_error") or "").strip()
    if existing and not last_error:
        return {
            "plan_summary": str(state.get("plan_summary") or ""),
            "todos": existing,
            "acceptance_criteria": _text_list(state.get("acceptance_criteria")),
            "verification_commands": _text_list(state.get("verification_commands")),
        }

    chat = model if model is not None else get_model()
    args = _request_structured(
        chat,
        TodoWriteTool(),
        [
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(content=_planner_instruction(state, existing, last_error)),
        ],
        keys="plan_summary, todos, acceptance_criteria, verification_commands",
    )

    return {
        "plan_summary": _text(args.get("plan_summary"))
        or str(state.get("plan_summary") or ""),
        "todos": _coerce_todos(args.get("todos"), fallback=existing),
        "acceptance_criteria": _text_list(args.get("acceptance_criteria"))
        or _text_list(state.get("acceptance_criteria")),
        "verification_commands": _text_list(args.get("verification_commands"))
        or _text_list(state.get("verification_commands")),
    }


def _planner_instruction(
    state: Mapping[str, Any], existing: Sequence[TodoItem], last_error: str
) -> str:
    lines = [f"Task:\n{str(state.get('task') or '').strip()}"]
    if existing:
        lines.append(f"Current plan:\n{_render_todos(existing)}")
    if last_error:
        lines.append(
            "The last verification FAILED with:\n"
            f"{last_error}\n\n"
            "Revise the plan so the next attempt fixes exactly this. Keep completed todos "
            "completed and rewrite or add the todos that the fix requires."
        )
    else:
        criteria = _text_list(state.get("acceptance_criteria"))
        commands = _text_list(state.get("verification_commands"))
        if criteria:
            lines.append("Acceptance criteria already recorded:\n" + "\n".join(f"- {c}" for c in criteria))
        if commands:
            lines.append("Verification commands already recorded:\n" + "\n".join(f"- {c}" for c in commands))
        lines.append("Produce the plan for this task.")
    return "\n\n".join(lines)


# -- actor -------------------------------------------------------------------
def actor_node(
    state: WQGraphState,
    *,
    model: Any | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    max_loops: int = DEFAULT_ACTOR_LOOPS,
) -> dict[str, Any]:
    """Run the ReAct actor over the current plan, one event per step.

    Returns the messages produced by this attempt (appended to the graph's
    transcript by the ``add_messages`` reducer), the ``last_actor_summary`` and the
    plan with the statuses the actor recorded through ``todo_update``.
    """
    runtime = _runtime(state)
    chat = model if model is not None else get_model()
    tools = [*build_tools(runtime), TodoUpdateTool().as_tool()]

    prior = list(state.get("messages") or [])
    instruction = HumanMessage(content=_actor_instruction(state))
    messages: list[BaseMessage] = [SystemMessage(content=ACTOR_PROMPT), *prior, instruction]
    first_new = 1 + len(prior)  # keep the system prompt and the replayed history out of the delta

    events: list[dict[str, Any]] = []
    for event in react_loop(messages, model=chat, tools=tools, max_loops=max_loops):
        events.append(event)
        _forward(event, "actor", on_event)

    updates: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") == "tool_call" and event.get("name") == TodoUpdateTool.name:
            raw = dict(event.get("args") or {}).get("updates")
            if isinstance(raw, list):
                updates.extend(item for item in raw if isinstance(item, Mapping))
    todos, notes = apply_todo_updates(_todos(state), updates)

    final_event = next(
        (event for event in reversed(events) if event.get("type") == "final_answer"), {}
    )
    summary = (
        _last_ai_text(messages[first_new:])
        or str(final_event.get("content") or "").strip()
        or str(state.get("last_actor_summary") or "")
    )
    if notes:
        summary = f"{summary}\n\n[plan updates ignored: {'; '.join(notes)}]".strip()

    return {
        "messages": messages[first_new:],
        "todos": todos,
        "last_actor_summary": summary,
    }


def _last_ai_text(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = message_text(message).strip()
            if text:
                return text
    return ""


def _actor_instruction(state: Mapping[str, Any]) -> str:
    lines = [
        "Execute the plan below, working only inside the workspace.",
        f"Task:\n{str(state.get('task') or '').strip()}",
    ]
    plan_summary = str(state.get("plan_summary") or "").strip()
    if plan_summary:
        lines.append(f"Approach:\n{plan_summary}")
    todos = _todos(state)
    if todos:
        lines.append(
            "Plan (call the todo_update tool as you start and finish items):\n"
            f"{_render_todos(todos)}"
        )
    criteria = _text_list(state.get("acceptance_criteria"))
    if criteria:
        lines.append("Acceptance criteria:\n" + "\n".join(f"- {item}" for item in criteria))
    commands = _text_list(state.get("verification_commands"))
    if commands:
        lines.append(
            "These commands are run afterwards to check your work - make sure they pass:\n"
            + "\n".join(f"- {command}" for command in commands)
        )
    last_error = str(state.get("last_error") or "").strip()
    if last_error:
        lines.append(f"The previous attempt failed with:\n{last_error}")
    lines.append("Finish with a short summary of the files you changed and the commands you ran.")
    return "\n\n".join(lines)


# -- verifier ----------------------------------------------------------------
def run_verification_commands(
    runtime: RuntimeState,
    commands: Iterable[str],
    *,
    timeout_seconds: float | None = None,
) -> list[VerificationResult]:
    """Run every verification command in the workspace and record the outcome."""
    tool = BashTool(runtime)
    results: list[VerificationResult] = []
    for raw in commands:
        command = str(raw).strip()
        if not command:
            continue
        outcome = tool.execute(command, timeout_seconds)
        stderr = outcome.stderr
        if outcome.blocked_by is not None:
            stderr = f"refused by the safety filter (pattern {outcome.blocked_by!r})"
        elif outcome.error is not None:
            stderr = outcome.error
        results.append(
            VerificationResult(
                command=command,
                ok=outcome.ok,
                exit_code=outcome.exit_code,
                stdout=runtime.clip(outcome.stdout, label="stdout", limit=VERIFY_OUTPUT_CHARS),
                stderr=runtime.clip(stderr, label="stderr", limit=VERIFY_OUTPUT_CHARS),
            )
        )
    return results


def verifier_node(
    state: WQGraphState,
    *,
    model: Any | None = None,
    max_loops: int = DEFAULT_VERIFIER_LOOPS,
) -> dict[str, Any]:
    """Verify the work: run the commands, inspect read-only, report a verdict.

    Returns ``passed``, ``attempts`` (+1), ``verification_results``,
    ``verification_checks``, ``last_error`` (only set while failing) and ``todos``
    (all completed once verification passes).

    A ``passed`` verdict from the model is overruled whenever a verification
    command exited non-zero: the commands are the hard evidence.
    """
    runtime = _runtime(state)
    chat = model if model is not None else get_model()
    results = run_verification_commands(runtime, _text_list(state.get("verification_commands")))

    messages: list[BaseMessage] = [
        SystemMessage(content=VERIFIER_PROMPT),
        HumanMessage(content=_verifier_instruction(state, results)),
    ]
    # Read-only inspection: the verifier may look, never touch.
    for _ in react_loop(
        messages, model=chat, tools=build_read_only_tools(runtime), max_loops=max_loops
    ):
        pass

    report = _request_structured(
        chat,
        VerificationReportTool(),
        messages,
        keys="passed, reason, checks, recommended_next_instruction",
    )
    passed = _flag(report.get("passed"))
    reason = _text(report.get("reason"))
    instruction = _text(report.get("recommended_next_instruction"))
    checks = _coerce_checks(report.get("checks"))

    failed = [result for result in results if not result["ok"]]
    if passed and failed:
        passed = False
        overruled = "verification refused by the safety filter" if any(
            "refused by the safety filter" in str(result["stderr"]) for result in failed
        ) else "verification command failed"
        reason = f"{reason} (overruled: {overruled}: {', '.join(str(r['command']) for r in failed)})".strip()
    if not report:
        passed = False
        reason = reason or "the verifier returned no verdict"

    attempts = int(state.get("attempts") or 0) + 1
    todos = _todos(state)
    if passed:
        todos = [
            make_todo(todo["id"], todo["content"], status="completed", note=todo["note"] or "verified")
            for todo in todos
        ]

    return {
        "passed": passed,
        "attempts": attempts,
        "verification_results": results,
        "verification_checks": checks,
        "last_error": "" if passed else _compose_error(reason, instruction, results),
        "todos": todos,
    }


def _verifier_instruction(
    state: Mapping[str, Any], results: Sequence[VerificationResult]
) -> str:
    lines = [f"Task:\n{str(state.get('task') or '').strip()}"]
    plan_summary = str(state.get("plan_summary") or "").strip()
    if plan_summary:
        lines.append(f"Plan summary:\n{plan_summary}")
    todos = _todos(state)
    if todos:
        lines.append(f"Todos as the actor left them:\n{_render_todos(todos)}")
    criteria = _text_list(state.get("acceptance_criteria"))
    if criteria:
        lines.append("Acceptance criteria:\n" + "\n".join(f"- {item}" for item in criteria))

    if results:
        blocks = []
        for result in results:
            block = (
                f"- `{result['command']}` -> exit_code={result['exit_code']} ok={result['ok']}"
            )
            stdout = str(result["stdout"]).strip()
            stderr = str(result["stderr"]).strip()
            if stdout:
                block += f"\n  stdout: {stdout}"
            if stderr:
                block += f"\n  stderr: {stderr}"
            blocks.append(block)
        lines.append("Verification command results:\n" + "\n".join(blocks))
    else:
        lines.append(
            "No verification commands were defined for this task; judge from the files and "
            "the acceptance criteria."
        )

    summary = str(state.get("last_actor_summary") or "").strip()
    if summary:
        lines.append(f"The actor reports:\n{summary}")
    lines.append(
        "Inspect the workspace with the read-only tools, then report with the "
        "`verification_report` tool."
    )
    return "\n\n".join(lines)


def _compose_error(
    reason: str, instruction: str, results: Sequence[VerificationResult]
) -> str:
    lines = [reason or "verification failed"]
    for result in results:
        if result["ok"]:
            continue
        lines.append(f"command failed: {result['command']} (exit code {result['exit_code']})")
        detail = (str(result["stderr"]).strip() or str(result["stdout"]).strip()).splitlines()
        lines.extend(f"  {line}" for line in detail[:4])
    if instruction:
        lines.append(f"Recommended next instruction: {instruction}")
    return "\n".join(lines)


# -- routing -----------------------------------------------------------------
def verifier_route(state: WQGraphState) -> str:
    """Route after verification: stop on success or when the budget is spent."""
    if state.get("passed"):
        return "final"
    attempts = int(state.get("attempts") or 0)
    max_attempts = int(state.get("max_attempts") or DEFAULT_MAX_ATTEMPTS)
    if attempts >= max_attempts:
        return "final"
    return "planner"


# -- final -------------------------------------------------------------------
def format_final_answer(state: Mapping[str, Any]) -> str:
    """Render the run's outcome as a deterministic Markdown report.

    This is the guaranteed answer: it is built from the state alone, so it is
    always available even when no model can be reached.
    """
    task = str(state.get("task") or "").strip() or "(no task recorded)"
    passed = bool(state.get("passed"))
    attempts = int(state.get("attempts") or 0)
    max_attempts = int(state.get("max_attempts") or DEFAULT_MAX_ATTEMPTS)
    lines = [
        f"**Task:** {task}",
        f"**Status:** {'completed' if passed else 'not completed'} "
        f"(verification attempts: {attempts}/{max_attempts})",
    ]

    plan_summary = str(state.get("plan_summary") or "").strip()
    if plan_summary:
        lines.append(f"**Approach:** {plan_summary}")

    todos = _todos(state)
    if todos:
        lines.append("**Plan:**")
        for todo in todos:
            mark = _TODO_MARK.get(str(todo.get("status")), "[ ]")
            note = str(todo.get("note") or "")
            lines.append(
                f"- {mark} {todo.get('id')}: {todo.get('content')}"
                + (f" — {note}" if note else "")
            )

    criteria = _text_list(state.get("acceptance_criteria"))
    checks = {
        str(check.get("name")): check
        for check in (state.get("verification_checks") or [])
        if isinstance(check, Mapping)
    }
    if criteria:
        lines.append("**Acceptance criteria:**")
        for item in criteria:
            check = checks.get(item)
            if check is None:
                lines.append(f"- {item}")
                continue
            verdict = "PASS" if check.get("passed") else "FAIL"
            detail = str(check.get("detail") or "").strip()
            lines.append(f"- {verdict} — {item}" + (f" ({detail})" if detail else ""))

    results = [r for r in (state.get("verification_results") or []) if isinstance(r, Mapping)]
    if results:
        lines.append("**Verification commands:**")
        for result in results:
            outcome = "ok" if result.get("ok") else "failed"
            lines.append(f"- `{result.get('command')}` -> exit code {result.get('exit_code')} ({outcome})")

    actor_summary = str(state.get("last_actor_summary") or "").strip()
    if actor_summary:
        lines.append(f"**What the actor reported:**\n{actor_summary}")

    last_error = str(state.get("last_error") or "").strip()
    if not passed and last_error:
        lines.append(f"**Why it is not done:**\n{last_error}")
    if not passed:
        remaining = [todo for todo in todos if str(todo.get("status")) != "completed"]
        if remaining:
            lines.append("**Remaining work:**")
            lines.extend(f"- {todo.get('id')}: {todo.get('content')}" for todo in remaining)
    return "\n\n".join(lines).strip()


def final_node(
    state: WQGraphState,
    *,
    model: Any | None = None,
    summarize: bool = True,
) -> dict[str, Any]:
    """Write the user-facing ``final_answer``.

    A deterministic report is always built from the state; when *summarize* is
    true and a model is reachable, the model rewrites it (``FINAL_PROMPT``) into a
    concise answer. Any model failure keeps the report, so this node never raises
    and ``final_answer`` is never empty.
    """
    report = format_final_answer(state)
    if not summarize:
        return {"final_answer": report}
    try:
        chat = model if model is not None else get_model()
        response = chat.invoke(
            [
                SystemMessage(content=FINAL_PROMPT),
                HumanMessage(content=_final_instruction(report)),
            ]
        )
        text = message_text(response).strip()
    except Exception:  # noqa: BLE001 - the deterministic report is always a valid answer
        text = ""
    return {"final_answer": text or report}


def _final_instruction(report: str) -> str:
    return (
        "Write the final answer for the user from these facts only.\n\n"
        f"--- facts ---\n{report}\n--- end of facts ---\n\n"
        "Short, no tool calls."
    )
