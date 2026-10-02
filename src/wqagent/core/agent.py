"""The ReAct loop of WQ Agent.

One loop, two entry points:

* :func:`stream_agent_events` - a generator that yields one event dict per step
  (``ai_message`` / ``tool_call`` / ``tool_result`` / ``final_answer``). This is
  what the CLI consumes to print progress in real time.
* :class:`Agent` - a small object wrapper that drives the same loop and returns
  an :class:`AgentResult` for scripts and tests.

Both build on :func:`_react_loop`, so the workflow is defined exactly once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from wqagent.core.state import RuntimeState
from wqagent.prompts.stage1 import ACTOR_PROMPT
from wqagent.providers import create_model
from wqagent.tools import build_tools

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
    "message_text",
    "stream_agent_events",
]

#: Loop budget of :func:`stream_agent_events` (and of the ``wq`` default).
DEFAULT_MAX_LOOPS = 10

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


def _react_loop(
    task: str,
    *,
    state: RuntimeState,
    model: Any,
    tools: Sequence[Any],
    system_prompt: str,
    max_loops: int,
    history: list[BaseMessage] | None = None,
    on_event: EventCallback | None = None,
) -> Iterator[dict[str, Any]]:
    """The loop itself: think, act, observe - yielding one event per step."""
    messages: list[BaseMessage] = history if history is not None else []
    messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=task))

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
            result = execute_tool(name, args, tools_by_name)
            messages.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False, default=str),
                    tool_call_id=str(call.get("id") or f"call_{loop}_{index}"),
                )
            )
            yield _publish(
                on_event,
                {"type": "tool_result", "name": name, "result": result, "loop": loop},
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


def stream_agent_events(
    task: str,
    *,
    workspace: str | Path,
    max_loops: int = DEFAULT_MAX_LOOPS,
    model: Any | None = None,
    state: RuntimeState | None = None,
    tools: Sequence[Any] | None = None,
    system_prompt: str | None = None,
    history: list[BaseMessage] | None = None,
    on_event: EventCallback | None = None,
) -> Iterator[dict[str, Any]]:
    """Run *task* as a ReAct workflow and yield one dict per event.

    The loop builds ``SystemMessage(ACTOR_PROMPT) + HumanMessage(task)``, lets the
    tool-bound model think, runs every tool call it asks for, feeds the JSON
    result back as a ``ToolMessage``, and repeats until the model answers without
    tool calls or ``max_loops`` is reached.

    Events:
        ``{"type": "ai_message", "content": str, "tool_calls": [str], "loop": int}``
            One model turn. ``tool_calls`` lists the requested tool names.
        ``{"type": "tool_call", "name": str, "args": dict, "loop": int}``
            A tool is about to run.
        ``{"type": "tool_result", "name": str, "result": dict, "loop": int}``
            Its result: ``{"ok": True, "tool": ..., "result": str}`` or
            ``{"ok": False, "tool": ..., "error": str}``.
        ``{"type": "final_answer", "content": str, "loop": int, "stopped_early": bool}``
            Always the last event.

    Args:
        task: What the agent should do, in natural language.
        workspace: Workspace root; created automatically when missing.
        max_loops: Maximum model turns before the run is stopped.
        model: Chat model to use; defaults to :func:`~wqagent.providers.create_model`
            (``.env`` configuration).
        state: Reuse an existing :class:`RuntimeState` (CLI timeouts, output
            budget); takes precedence over ``workspace``.
        tools: Override the tool list; defaults to ``build_tools(state)``.
        system_prompt: Override :data:`ACTOR_PROMPT`.
        history: Optional list that receives the full message history.
        on_event: Optional ``(event_type, payload)`` callback mirroring the events.
    """
    runtime = state if state is not None else RuntimeState(workspace=workspace)
    chat_model = model if model is not None else create_model()
    tool_list = list(tools) if tools is not None else build_tools(runtime)
    yield from _react_loop(
        task,
        state=runtime,
        model=chat_model,
        tools=tool_list,
        system_prompt=system_prompt or ACTOR_PROMPT,
        max_loops=max_loops,
        history=history,
        on_event=on_event,
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
        return _react_loop(
            task,
            state=self.state,
            model=self.model,
            tools=self.tools,
            system_prompt=self.system_prompt,
            max_loops=self.max_steps,
            history=history,
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
