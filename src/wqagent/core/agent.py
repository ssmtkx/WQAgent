"""WQ Agent's entry point and its ReAct engine.

* :func:`stream_agent_events` - **the agent entry point**. It no longer drives a
  bare ReAct loop: it streams the compiled LangGraph workflow
  (``planner -> context_monitor -> verifier -> final`` with context compression
  wired in, the planner delegating to the specialist agents) via
  ``build_complex_workflow().stream(inputs, stream_mode=["updates", "custom"])``
  and translates the graph chunks into one flat event stream for the CLI.
* :func:`react_loop` - the ReAct engine used *inside* that workflow by the actor
  node (:func:`wqagent.graph.nodes.actor_node`): think, call tools, observe.
* :class:`Agent` - object wrapper around :func:`react_loop` (the engine-level API).

The graph modules import this module, so the imports of
:mod:`wqagent.graph.workflow` / :mod:`wqagent.graph.state` inside
:func:`stream_agent_events` are deliberately local.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from wqagent.core.checkpoint import CheckpointManager
from wqagent.core.diff import capture as capture_diff
from wqagent.core.diff import finish as finish_diff
from wqagent.core.session import (
    append_assistant_turn,
    append_user_turn,
    build_session_context,
    load_or_create_session,
    save_session,
    session_path,
    summary_path,
)
from wqagent.core.state import RuntimeState, create_runtime
from wqagent.core.trace import TraceRecorder
from wqagent.prompts.stage1 import ACTOR_PROMPT

__all__ = [
    "ACTOR_PROMPT",
    "Agent",
    "AgentResult",
    "ToolCallRecord",
    "DEFAULT_MAX_LOOPS",
    "DEFAULT_MAX_STEPS",
    "SYSTEM_PROMPT",
    "default_system_prompt",
    "execute_tool",
    "last_ai_text",
    "message_text",
    "react_loop",
    "stream_agent_events",
    "stream_session_events",
    "stream_writer",
    "write_event",
]

#: Loop budget of :func:`stream_agent_events` (and of the ``wq`` default).
DEFAULT_MAX_LOOPS = 10
#: Custom events that earn a checkpoint of their own, between node boundaries.
CHECKPOINT_CUSTOM_EVENTS: frozenset[str] = frozenset({"handoff", "context_compressed"})
#: Node-update keys left out of the *event logs* - the transcript is written in full
#: to ``state.json`` (strict checkpoints), and repeating it per node would dwarf the
#: logs. The event stream handed to callers always carries the complete update.
LOG_OMITTED_UPDATE_KEYS: tuple[str, ...] = ("messages",)


def _log_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """The form of a node event written to the logs (see :data:`LOG_OMITTED_UPDATE_KEYS`)."""
    update = event.get("update")
    if not isinstance(update, Mapping):
        return dict(event)
    if not any(key in update for key in LOG_OMITTED_UPDATE_KEYS):
        return dict(event)
    slim = {key: value for key, value in update.items() if key not in LOG_OMITTED_UPDATE_KEYS}
    messages = update.get("messages")
    if isinstance(messages, (list, tuple)):
        slim["message_count"] = len(messages)
    return {**dict(event), "update": slim}

#: Loop budget of :class:`Agent`, which ships a richer default prompt.
DEFAULT_MAX_STEPS = 30

#: System prompt used by :class:`Agent`; :func:`stream_agent_events` uses
#: :data:`~wqagent.prompts.stage1.ACTOR_PROMPT` instead.
SYSTEM_PROMPT = """You are WQ Agent, an autonomous coding and research assistant working inside a single workspace directory.

Workspace root: {workspace}

How to work:
1. Every path you pass to a tool is resolved inside the workspace. Absolute paths outside it and `..` escapes are rejected - prefer paths relative to the workspace root.
2. Inspect before you change: `grep` to locate code, `file_read` to read it.
3. `file_write` creates or fully overwrites a file; `file_edit` replaces a snippet that must match exactly once. Prefer `file_edit` for small changes.
4. `bash` runs shell commands in the workspace root with a timeout - use it for builds, tests and git.
5. A tool result starting with `Error:` means the call failed. Read the message, fix your arguments, and continue.
6. Do not ask the user questions; decide, act, and verify your own work.
7. When the task is done, reply with a short report: what changed or what you found, the paths involved, and the important command output. No tool calls in that final reply.

Tools: {tools}"""

EventCallback = Callable[[str, dict[str, Any]], None]


@dataclass
class ToolCallRecord:
    """One executed tool call, kept for the run summary."""

    step: int
    name: str
    args: dict[str, Any]
    output: str
    ok: bool = True


@dataclass
class AgentResult:
    """Outcome of :meth:`Agent.run`."""

    text: str
    steps: int
    messages: list[BaseMessage] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    stopped_early: bool = False

    @property
    def completed(self) -> bool:
        """True when the model produced a final answer instead of hitting the loop budget."""
        return not self.stopped_early


# -- helpers -----------------------------------------------------------------
def message_text(message: Any) -> str:
    """Extract plain text from a message whose ``content`` may be str or blocks."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def last_ai_text(messages: Sequence[BaseMessage]) -> str:
    """The text of the last AI message that has any (skips tool-call-only turns)."""
    for message in reversed(list(messages)):
        if isinstance(message, AIMessage):
            text = message_text(message).strip()
            if text:
                return text
    return ""


def stream_writer() -> Callable[[Any], None] | None:
    """LangGraph's custom stream writer (``stream_mode="custom"``), if available.

    Returns ``None`` when not running inside a graph, so callers can treat
    streaming as best effort.
    """
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:  # noqa: BLE001 - not running inside a graph
        return None
    return writer if callable(writer) else None


def write_event(
    writer: Callable[[dict[str, Any]], None] | None, event: dict[str, Any]
) -> None:
    """Send one event to *writer*, falling back to the graph's custom stream.

    Writers are best effort: a failing sink must never break a run.
    """
    sink = writer if writer is not None else stream_writer()
    if sink is None:
        return
    try:
        sink(event)
    except Exception:  # pragma: no cover - streaming is best effort
        pass


def default_system_prompt(state: RuntimeState, tools: Sequence[Any] = ()) -> str:
    """System prompt describing this workspace and the available tools."""
    names = ", ".join(str(getattr(tool, "name", tool)) for tool in tools) or "(none)"
    return SYSTEM_PROMPT.format(workspace=state.workspace, tools=names)


def execute_tool(
    name: str, args: Mapping[str, Any], tools: Mapping[str, Any]
) -> dict[str, Any]:
    """Run one tool call and always return a JSON-serialisable result dict.

    Returns ``{"ok": True, "tool": name, "result": <text>}`` on success and
    ``{"ok": False, "tool": name, "error": <message>}`` when the tool is unknown,
    raised, or answered with an ``Error:`` message. Never raises: failures are
    handed back to the model as feedback.
    """
    tool = tools.get(name)
    if tool is None:
        available = ", ".join(sorted(tools)) or "(none)"
        return {
            "ok": False,
            "tool": name,
            "error": f"unknown tool {name!r}. Available tools: {available}",
        }
    try:
        raw = tool.invoke(dict(args))
    except Exception as exc:  # noqa: BLE001 - tool bugs become model-visible feedback
        return {"ok": False, "tool": name, "error": f"{type(exc).__name__}: {exc}"}

    output = (
        raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, default=str)
    )
    # Our tools report handled failures as "Error: ..." strings instead of raising.
    if output.lstrip().startswith("Error:"):
        return {"ok": False, "tool": name, "error": output.lstrip()[len("Error:") :].strip()}
    return {"ok": True, "tool": name, "result": output}


def _publish(on_event: EventCallback | None, event: dict[str, Any]) -> dict[str, Any]:
    """Forward *event* to the optional callback and return it unchanged."""
    if on_event is not None:
        payload = {key: value for key, value in event.items() if key != "type"}
        try:
            on_event(str(event["type"]), payload)
        except Exception:  # pragma: no cover - a broken callback must not kill the run
            pass
    return event


def _pump_graph(
    app: Any,
    inputs: Mapping[str, Any],
    merged: dict[str, Any],
    *,
    manager: CheckpointManager,
    tracer: TraceRecorder,
    on_event: EventCallback | None,
    progress: dict[str, str],
) -> Iterator[dict[str, Any]]:
    """Stream one compiled graph into the flat event protocol.

    Used for both the entry graph and the main workflow, so the two bookkeep
    identically: node updates are merged into *merged*, checkpoints are written, the
    trace is flushed, and every event reaches the caller. ``progress["latest_node"]``
    tracks where the run is (the entry and main graphs share one progress dict).
    """
    for mode, chunk in app.stream(inputs, stream_mode=["updates", "custom"]):
        if mode == "custom":
            event = dict(chunk or {})
            if event.get("type") == "final_answer":
                # The actor round's own end marker - only the `final` node closes
                # the run, and that answer is emitted once at the end.
                continue
            if not event.get("transient"):  # token deltas are for the UI only
                if manager.strict:
                    manager.append_event(event)
                tracer.record_custom_event(event)
            yield _publish(on_event, event)
            if _custom_event_needs_checkpoint(event):
                # after the event, so the checkpoint is "everything up to here"
                saved = manager.save(
                    merged,
                    status="running",
                    latest_node=progress["latest_node"],
                    event=event,
                )
                if saved is not None:
                    tracer.record_custom_event(saved)
                    tracer.snapshot(status="running", latest_node=progress["latest_node"])
                    yield _publish(on_event, saved)
            continue

        for node, update in dict(chunk or {}).items():
            patch = dict(update or {})
            merged.update(patch)
            progress["latest_node"] = str(node)
            node_event = {"type": "node_end", "node": str(node), "update": patch}
            logged = _log_event(node_event)
            tracer.record_graph_update(logged)
            saved = manager.save(
                merged, status="running", latest_node=progress["latest_node"], event=logged
            )
            if saved is not None:
                tracer.record_custom_event(saved)
            # flush the trace at every node boundary, before the caller is told the node
            # finished: a run that is killed (or that stalls inside a model call) must
            # still leave a usable trace behind
            tracer.snapshot(status="running", latest_node=progress["latest_node"])
            yield _publish(on_event, node_event)
            if saved is not None:
                yield _publish(on_event, saved)


def _close_run(
    manager: CheckpointManager,
    tracer: TraceRecorder,
    state: Mapping[str, Any],
    *,
    status: str,
    latest_node: str,
) -> None:
    """Record the end of a run that did not finish normally (best effort).

    Called from ``except`` blocks, so it never yields and never raises: the original
    exception must stay the one the caller sees.
    """
    try:
        saved = manager.save(state, status=status, latest_node=latest_node)
        if saved is not None:
            tracer.record_custom_event(saved)
        tracer.end(status=status, latest_node=latest_node, final_state=state)
    except Exception:  # noqa: BLE001 - bookkeeping must not mask the real error
        pass


def react_loop(
    messages: list[BaseMessage],
    *,
    model: Any,
    tools: Sequence[Any],
    max_loops: int = DEFAULT_MAX_LOOPS,
    on_event: EventCallback | None = None,
) -> Iterator[dict[str, Any]]:
    """Think, act, observe - yielding one event per step.

    Unlike :func:`stream_agent_events` this takes a caller-built message list:
    put the system prompt and the task (or the whole transcript) in *messages*,
    and the loop appends to that very list as it runs, so the caller keeps the
    full conversation. Graph nodes use this to seed the actor with the workflow's
    history. Yields the same events as :func:`stream_agent_events`.
    """
    tool_list = list(tools)
    tools_by_name = {tool.name: tool for tool in tool_list}
    bound = (
        model.bind_tools(tool_list)
        if tool_list and hasattr(model, "bind_tools")
        else model
    )

    last_content = ""
    loops_run = 0
    for loop in range(1, max(1, int(max_loops)) + 1):
        loops_run = loop
        response = bound.invoke(messages)
        messages.append(response)

        content = message_text(response)
        if content.strip():
            last_content = content
        calls = list(getattr(response, "tool_calls", None) or [])
        yield _publish(
            on_event,
            {
                "type": "ai_message",
                "content": content,
                "tool_calls": [str(call.get("name") or "") for call in calls],
                "loop": loop,
            },
        )

        if not calls:
            yield _publish(
                on_event,
                {
                    "type": "final_answer",
                    "content": content.strip() or last_content,
                    "loop": loop,
                    "stopped_early": False,
                },
            )
            return

        for index, call in enumerate(calls):
            name = str(call.get("name") or "")
            args = dict(call.get("args") or {})
            yield _publish(
                on_event, {"type": "tool_call", "name": name, "args": args, "loop": loop}
            )
            # read the file before/after the call so the UI can show a real diff;
            # the model only ever sees `result`, never the patch
            snapshot = capture_diff(tools_by_name.get(name), args)
            result = execute_tool(name, args, tools_by_name)
            change = finish_diff(snapshot)
            messages.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False, default=str),
                    tool_call_id=str(call.get("id") or f"call_{loop}_{index}"),
                )
            )
            yield _publish(
                on_event,
                {
                    "type": "tool_result",
                    "name": name,
                    "result": result,
                    "diff": change.as_dict() if change is not None else None,
                    "loop": loop,
                },
            )

    yield _publish(
        on_event,
        {
            "type": "final_answer",
            "content": (
                f"Stopped after {loops_run} steps without a final answer. "
                "Re-run with a higher --max-steps value, or split the task into smaller ones."
            ),
            "loop": loops_run,
            "stopped_early": True,
        },
    )


def _custom_event_needs_checkpoint(event: Mapping[str, Any]) -> bool:
    """True for custom events worth a checkpoint of their own.

    Node boundaries are checkpointed anyway; in between, a *delegation* and a
    *context compression* change enough state to be worth saving immediately (a
    crash right after either would otherwise lose it).
    """
    return str(event.get("type") or "") in CHECKPOINT_CUSTOM_EVENTS


def stream_agent_events(
    task: str,
    *,
    workspace: str | Path | None = None,
    max_attempts: int | None = None,
    max_loops: int | None = None,
    model: Any | None = None,
    state: RuntimeState | None = None,
    inputs: Any | None = None,
    graph: Any | None = None,
    approval_mode: str | None = "inline",
    approval_handler: Any | None = None,
    checkpoint_mode: str | None = "light",
    resume_workspace: str | Path | None = None,
    trace_mode: str | None = "on",
    entry_router: bool | None = False,
    resume_event: Mapping[str, Any] | None = None,
    on_event: EventCallback | None = None,
) -> Iterator[dict[str, Any]]:
    """Run *task* through the compiled workflow, yielding one flat event stream.

    Instead of driving a bare ReAct loop, this streams
    :func:`wqagent.graph.workflow.build_complex_workflow` with
    ``stream_mode=["updates", "custom"]``:

    * ``custom`` chunks are the actor's per-step ReAct events - forwarded as they
      are (they already carry ``node``);
    * ``updates`` chunks are each node's state update - emitted as ``node_end``
      events so the CLI can report planner / verifier / final;
    * every event is recorded by a :class:`~wqagent.core.trace.TraceRecorder`, and
      checkpoints are written by a :class:`~wqagent.core.checkpoint.CheckpointManager`
      at the start, on every node boundary (plus delegations and compressions), and
      at the end;
    * the run is closed with a single ``final_answer`` built from the ``final``
      node's text plus the verdict the verifier recorded.

    Events (flat - each dict has ``type`` plus its own fields):
        ``{"type": "ai_message", "node": str, "content": str, "tool_calls": [str], "loop": int}``
        ``{"type": "tool_call", "node": str, "name": str, "args": dict, "loop": int}``
        ``{"type": "tool_result", "node": str, "name": str, "result": dict, "loop": int}``
        ``{"type": "handoff"|"memory"|"context_compressed"|"checkpoint_saved"|..., ...}``
        ``{"type": "intent", "intent_route": "chat"|"workflow", "intent_reason": str, "intent_confidence": float}``
        ``{"type": "chat_response", "chat_response": str, "final_answer": str}``
        ``{"type": "node_end", "node": str, "update": dict}``
        ``{"type": "trace_saved", "trace_id": str, "path": str, ...}``
        ``{"type": "final_answer", "content": str, "passed": bool, "attempts": int, "stopped_early": bool, "route": "chat"|"workflow"}``
            Always the last event.

    Args:
        task: What the agent should do, in natural language.
        workspace: Workspace root; created automatically when missing.
        max_attempts: Plan/delegate/verify rounds before the run stops (default 3).
        max_loops: Tool rounds the planner may use per attempt (default 10).
        model: Chat model to use; defaults to the nodes' ``create_model()``
            (``.env`` configuration).
        state: Reuse an existing :class:`RuntimeState` (CLI timeouts, output budget);
            takes precedence over ``workspace`` and keeps its own modes.
        inputs: A ready-made ``WQGraphState``; skips building one from *task*.
        graph: A prebuilt workflow app; skips ``build_complex_workflow``.
        approval_mode: ``inline`` / ``auto`` / ``deny`` for risky shell commands
            (only used when this function builds the runtime; ``None`` -> env).
        approval_handler: Consulted in ``inline`` mode; ``None`` denies.
        checkpoint_mode: ``light`` (metadata + recovery guide + git snapshot),
            ``strict`` (plus full state and every event) or ``off``.
        resume_workspace: Workspace to resume: its checkpoint is loaded, its workspace
            snapshot restored and, when no checkpoint exists, ``LookupError`` is
            raised. Takes precedence over ``workspace``.
        trace_mode: ``summary`` (alias ``on``), ``full`` or ``off``.
        entry_router: Run :func:`~wqagent.graph.workflow.build_entry_workflow` first, so
            conversation ("你好") is answered by one cheap model call instead of the
            whole planner/verifier pipeline. Off by default (the library keeps its
            existing behaviour); the CLI turns it on and ``--no-router`` turns it off.
        resume_event: The event ``load_resume_inputs`` returned (used for the trace
            and to label the run as resumed).
        on_event: Optional ``(event_type, payload)`` callback mirroring the events.

    Raises:
        ValueError: When no workspace, state or inputs say where to work.
        LookupError: When ``resume_workspace`` holds no usable checkpoint.
    """
    # Imported lazily on purpose: the workflow's nodes import this module for
    # `react_loop`, so a module-level import here would be circular.
    from wqagent.graph.state import DEFAULT_MAX_ATTEMPTS, new_graph_state
    from wqagent.graph.workflow import build_complex_workflow, build_entry_workflow

    provided_inputs = inputs is not None
    # 1. where to work: an existing state/inputs wins over a freshly built runtime
    if inputs is None:
        runtime = state
        if runtime is None:
            if resume_workspace is not None:
                runtime = create_runtime(
                    resume_workspace,
                    approval_mode=approval_mode,
                    approval_handler=approval_handler,
                    checkpoint_mode=checkpoint_mode,
                    trace_mode=trace_mode,
                    resume_from=resume_workspace,
                )
                inputs, loaded = CheckpointManager.load_resume_inputs(
                    runtime,
                    task or None,
                    max_attempts=max_attempts or DEFAULT_MAX_ATTEMPTS,
                )
                if inputs is None:
                    reasons = "; ".join(str(item) for item in (loaded.get("warnings") or []))
                    raise LookupError(
                        f"no checkpoint to resume in {runtime.workspace}"
                        f" ({reasons or 'not found'})"
                    )
                resume_event = loaded
            else:
                if workspace is None:
                    raise ValueError(
                        "pass workspace=, state=, inputs= or resume_workspace= so the "
                        "agent knows where to work"
                    )
                runtime = create_runtime(
                    workspace,
                    approval_mode=approval_mode,
                    approval_handler=approval_handler,
                    checkpoint_mode=checkpoint_mode,
                    trace_mode=trace_mode,
                )
        if inputs is None:
            inputs = new_graph_state(
                task, runtime, max_attempts=max_attempts or DEFAULT_MAX_ATTEMPTS
            )

    app = (
        graph
        if graph is not None
        else build_complex_workflow(model=model, planner_loops=max_loops)
    )

    # 2. one checkpoint manager and one tracer for the run
    manager = CheckpointManager(inputs.get("runtime"), str(inputs.get("task") or task))
    tracer = TraceRecorder(inputs.get("runtime"), str(inputs.get("task") or task))
    merged: dict[str, Any] = dict(inputs)
    progress = {"latest_node": "start"}

    # 3. the opening events: trace the run and save the starting checkpoint
    tracer.start(merged, resumed=resume_event is not None, resume_event=resume_event)
    tracer.snapshot(status="started", latest_node=progress["latest_node"])
    opening = manager.save(merged, status="started", latest_node=progress["latest_node"])
    if opening is not None:
        tracer.record_custom_event(opening)
        yield _publish(on_event, opening)

    router = bool(entry_router)
    route = "workflow"
    try:
        # 4. the entry graph: conversation or work? (`workflow` ends it immediately)
        if router:
            entry_app = build_entry_workflow(model=model)
            for event in _pump_graph(
                entry_app,
                inputs,
                merged,
                manager=manager,
                tracer=tracer,
                on_event=on_event,
                progress=progress,
            ):
                yield event
            route = str(merged.get("intent_route") or "workflow")
            if route not in {"chat", "workflow"}:
                route = "workflow"

        # 5. the main workflow (skipped entirely for the chat route)
        if route != "chat":
            for event in _pump_graph(
                app,
                inputs,
                merged,
                manager=manager,
                tracer=tracer,
                on_event=on_event,
                progress=progress,
            ):
                yield event
    except KeyboardInterrupt:
        # an interrupted run is still resumable: close both records, then let the
        # interrupt travel on (the CLI turns it into its own exit code)
        _close_run(
            manager, tracer, merged, status="interrupted", latest_node=progress["latest_node"]
        )
        raise
    except Exception:
        # a crash (API error, bad tool, ...) must leave the same trail as an interrupt
        _close_run(
            manager, tracer, merged, status="failed", latest_node=progress["latest_node"]
        )
        raise

    # 6. the closing events: final checkpoint, then the trace summary, then the answer
    if route == "chat":
        final_event = {
            "type": "final_answer",
            "content": str(merged.get("chat_response") or merged.get("final_answer") or ""),
            "passed": True,
            "attempts": 0,
            "stopped_early": False,
            "route": "chat",
            "intent_route": "chat",
            "intent_reason": str(merged.get("intent_reason") or ""),
            "intent_confidence": float(merged.get("intent_confidence") or 0.0),
        }
        status = "passed"
    else:
        passed = bool(merged.get("passed"))
        attempts = int(merged.get("attempts") or 0)
        final_event = {
            "type": "final_answer",
            "content": str(merged.get("final_answer") or ""),
            "passed": passed,
            "attempts": attempts,
            "stopped_early": not passed,
            "route": "workflow",
            "intent_route": str(merged.get("intent_route") or ""),
        }
        status = "passed" if passed else "failed"
    # the closing checkpoint is written (and counted) before the trace is closed, so
    # every counter covers the whole run
    saved = manager.save(merged, status=status, latest_node="final")
    if saved is not None:
        tracer.record_custom_event(saved)
    tracer.record_custom_event(final_event)
    trace = tracer.end(status=status, latest_node="final", final_state=merged)
    if saved is not None:
        yield _publish(on_event, saved)
    if trace is not None:
        yield _publish(
            on_event,
            {
                "type": "trace_saved",
                "trace_id": trace["trace_id"],
                "path": str(tracer.trace_path),
                "mode": trace["mode"],
                "status": trace["status"],
                "duration_ms": trace["duration_ms"],
                "tool_calls": trace["tool_calls"],
                "failed_tool_calls": trace["failed_tool_calls"],
                "approval_count": trace["approval_count"],
                "checkpoint_count": trace["checkpoint_count"],
                "handoff_count": trace["handoff_count"],
                "node_visits": trace["node_visits"],
                "timeline_total": trace["timeline_total"],
            },
        )
    yield _publish(on_event, final_event)


SESSION_CONTEXT_HEADER = "--- session context ---"
SESSION_CONTEXT_FOOTER = "--- end of session context ---"


def _summary_line(text: str, limit: int = 200) -> str:
    """The first sentence of *text*, for the turn summary (empty when there is none)."""
    body = " ".join(str(text or "").split())
    if not body:
        return ""
    for separator in (". ", "。", "! ", "? ", "\n"):
        index = body.find(separator)
        if 0 < index < limit:
            return body[: index + len(separator.strip())].strip()
    return body[:limit].strip()


def stream_session_events(
    task: str,
    *,
    session_workspace: str | Path | None = None,
    workspace: str | Path | None = None,
    session: dict[str, Any] | None = None,
    save: bool = True,
    model: Any | None = None,
    max_attempts: int | None = None,
    max_loops: int | None = None,
    approval_mode: str | None = None,
    approval_handler: Any | None = None,
    checkpoint_mode: str | None = None,
    trace_mode: str | None = None,
    entry_router: bool = True,
    graph: Any | None = None,
    on_event: EventCallback | None = None,
) -> Iterator[dict[str, Any]]:
    """Run one **session turn**: conversation state in, the usual event stream out.

    The steps, in order:

    1. load ``.wqagent/session/session.json`` (or start a session) in
       *session_workspace*;
    2. record the user's turn (``append_user_turn``), which is what makes the turn
       number in the context block advance;
    3. build the session context - session id, turn number, the newest workspace files
       and the last exchanges (:func:`~wqagent.core.session.build_session_context`);
    4. run the entry graph: ``intent_router`` sends conversation to ``chat_responder``
       (answered directly, no tools) and everything else to
       ``build_complex_workflow()`` - the whole planner/verifier pipeline;
    5. record the assistant's turn (``route`` from the final event, ``summary`` from
       its first sentence) and write ``session.json`` + ``SESSION_SUMMARY.md``;
    6. yield a closing ``session_saved`` event with the turn numbers and paths.

    The context reaches both sides: the entry nodes read it from disk (so "继续" can be
    understood), and the seeded system message puts it in front of every workflow node
    (so the planner knows what earlier turns did).

    Args:
        task: This turn's user input.
        session_workspace: Workspace whose session this is (its ``.wqagent``); also the
            run's workspace.
        workspace: Alias for *session_workspace*.
        session: An in-memory session to continue (skips the disk read); the same dict
            stays up to date for the next turn.
        save: Write the session files (``False`` keeps everything in memory).
        model: Chat model for every node; defaults to the nodes' lazy ``create_model()``.
        max_attempts: Plan/delegate/verify rounds before the run stops.
        max_loops: Tool rounds the planner may use per attempt.
        approval_mode: ``inline`` / ``auto`` / ``deny``; ``None`` keeps the environment.
        approval_handler: Consulted in ``inline`` mode.
        checkpoint_mode: ``light`` / ``strict`` / ``off``; ``None`` keeps the default.
        trace_mode: ``summary`` / ``full`` / ``off``; ``None`` keeps the default.
        entry_router: ``False`` skips the router and always runs the workflow.
        graph: A prebuilt workflow app; skips ``build_complex_workflow``.
        on_event: Optional ``(event_type, payload)`` callback mirroring the events.

    Events: everything :func:`stream_agent_events` yields, plus a final

        ``{"type": "session_saved", "session_id": str, "turn": int, "reply_turn": int,
           "turn_index": int, "route": "chat"|"workflow", "path": str,
           "summary_path": str, "turns": int}``
    """
    target = session_workspace if session_workspace is not None else workspace
    if target is None:
        raise ValueError(
            "pass session_workspace= (or workspace=) so the session has a home: the "
            "run's tools need a workspace too"
        )
    record = session if session is not None else load_or_create_session(target)
    turn = append_user_turn(record, task)
    if save:
        # persist the user's turn *now*: the entry nodes and the memory layer read the
        # session from disk, so the turn number they see must already be this turn's
        save_session(target, record)

    try:
        context = build_session_context(target, record)
    except Exception:  # noqa: BLE001 - context is a bonus, never a requirement
        context = ""

    runtime = create_runtime(
        target,
        approval_mode=approval_mode,
        approval_handler=approval_handler,
        checkpoint_mode=checkpoint_mode,
        trace_mode=trace_mode,
        session=record,
    )
    # Imported lazily for the same reason as in stream_agent_events.
    from wqagent.graph.state import DEFAULT_MAX_ATTEMPTS, new_graph_state

    inputs = new_graph_state(
        task, runtime, max_attempts=max_attempts or DEFAULT_MAX_ATTEMPTS
    )

    final: dict[str, Any] = {}
    try:
        for event in stream_agent_events(
            task,
            workspace=target or None,
            state=runtime,
            inputs=inputs,
            model=model,
            max_attempts=max_attempts,
            max_loops=max_loops,
            graph=graph,
            entry_router=entry_router,
            on_event=on_event,
        ):
            if event.get("type") == "final_answer":
                final = event
            yield event
    except BaseException:
        # a crashed or interrupted turn still belongs to the session: keep the user's
        # message (nothing else) so the conversation is not silently reset
        if save:
            save_session(target, record)
        raise

    route = str(final.get("route") or "workflow")
    content = str(final.get("content") or "")
    append_assistant_turn(
        record,
        turn=turn,
        route=route,
        content=content,
        summary=_summary_line(content),
    )
    if save:
        save_session(target, record)
    reply_turn = int(record.get("turn_index") or turn)
    yield _publish(
        on_event,
        {
            "type": "session_saved",
            "session_id": str(record.get("session_id") or ""),
            "turn": turn,
            "reply_turn": reply_turn,
            "turn_index": int(record.get("turn_index") or turn),
            "route": route,
            "path": str(session_path(target)),
            "summary_path": str(summary_path(target)),
            "turns": len(record.get("recent_turns") or []),
            "saved": bool(save),
        },
    )


class Agent:
    """Object wrapper around the same ReAct loop, returning an :class:`AgentResult`.

    Args:
        model: A LangChain chat model (``bind_tools`` is used when available).
        tools: ``BaseTool`` instances, usually
            :func:`wqagent.tools.registry.build_tools`.
        state: Runtime state the tools are bound to.
        system_prompt: Override :func:`default_system_prompt`.
        max_steps: Maximum model turns before the run is stopped.
        on_event: Optional ``(event_type, payload)`` callback; the event types are
            the ones documented in :func:`stream_agent_events`.
    """

    def __init__(
        self,
        model: Any,
        tools: Sequence[BaseTool],
        state: RuntimeState,
        *,
        system_prompt: str | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        on_event: EventCallback | None = None,
    ) -> None:
        self.state = state
        self.model = model
        self.tools: list[BaseTool] = list(tools)
        self.tools_by_name: dict[str, BaseTool] = {tool.name: tool for tool in self.tools}
        self.max_steps = max(1, int(max_steps))
        self.system_prompt = system_prompt or default_system_prompt(state, self.tools)
        self.on_event = on_event

    def stream(
        self, task: str, *, history: list[BaseMessage] | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yield ReAct events for *task* using this agent's model, tools and prompt."""
        messages: list[BaseMessage] = history if history is not None else []
        messages.append(SystemMessage(content=self.system_prompt))
        messages.append(HumanMessage(content=task))
        return react_loop(
            messages,
            model=self.model,
            tools=self.tools,
            max_loops=self.max_steps,
            on_event=self.on_event,
        )

    def run(self, task: str) -> AgentResult:
        """Execute *task* until the model stops calling tools or steps run out."""
        history: list[BaseMessage] = []
        records: list[ToolCallRecord] = []
        pending: dict[tuple[int, str], list[int]] = {}
        final: dict[str, Any] = {}

        for event in self.stream(task, history=history):
            kind = event.get("type")
            if kind == "tool_call":
                key = (int(event.get("loop") or 0), str(event.get("name") or ""))
                pending.setdefault(key, []).append(len(records))
                records.append(
                    ToolCallRecord(
                        step=key[0],
                        name=key[1],
                        args=dict(event.get("args") or {}),
                        output="",
                    )
                )
            elif kind == "tool_result":
                key = (int(event.get("loop") or 0), str(event.get("name") or ""))
                queue = pending.get(key) or []
                index = queue.pop(0) if queue else None
                result = dict(event.get("result") or {})
                if index is not None:
                    ok = bool(result.get("ok"))
                    records[index].ok = ok
                    records[index].output = (
                        str(result.get("result") or "")
                        if ok
                        else f"Error: {result.get('error')}"
                    )
            elif kind == "final_answer":
                final = dict(event)

        return AgentResult(
            text=str(final.get("content") or ""),
            steps=int(final.get("loop") or 0),
            messages=history,
            tool_calls=records,
            stopped_early=bool(final.get("stopped_early")),
        )
