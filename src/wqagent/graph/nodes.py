"""LangGraph nodes of the WQ Agent workflow: plan/delegate -> verify -> route.

* :func:`planner_node` - the **supervisor**: plans through ``todo_write`` and
  delegates research/implementation to the specialist agents through the
  :class:`CallSearchAgentTool` / :class:`CallCodeAgentTool` handoff tools.
* :func:`_call_search_agent_tool` / :func:`_call_code_agent_tool` - run one
  handoff and record its state effects (notes, sources, todos, transcripts).
* :func:`verifier_node` - runs the verification commands, inspects the workspace
  with read-only tools and records a structured verdict.
* :func:`final_node` / :func:`verifier_route` - report the outcome, or go back to
  the planner for another attempt.
* :func:`actor_node` - the stage-1/2 ReAct actor. Kept for reference and manual
  wiring, but **not part of the stage-3 graph**: the code agent owns edits now.

Nodes take their model from :func:`get_model` (lazily ``create_model()`` from
``.env``); pass ``model=...`` to a single node, or call :func:`set_model` once to
pin one for the whole graph. The specialists reuse the planner's model unless the
node is given ``agent_model=...``.

Example wiring::

    graph = StateGraph(WQGraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("final", final_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "verifier")
    graph.add_conditional_edges("verifier", verifier_route, {"planner": "planner", "final": "final"})
    graph.add_edge("final", END)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from pydantic import BaseModel, Field

from wqagent.core.agent import (
    last_ai_text,
    message_text,
    react_loop,
    write_event,
)
from wqagent.core.state import RuntimeState
from wqagent.graph.memory import (
    COMPRESSION_FIELD_CHARS,
    COMPRESSION_ITEMS,
    COMPRESSION_SUMMARY_CHARS,
    DEFAULT_CONTEXT_TOKEN_LIMIT,
    ERROR_CHARS,
    HISTORY_SUMMARY_FILE,
    NOTEPAD_FILE,
    RESEARCH_CHARS,
    SUMMARY_CHARS,
    CompressionEvent,
    _short_text,
    _source_refs,
    _trim_handoffs,
    append_notepad,
    build_layered_memory,
    format_layered_memory_for_prompt,
    has_notepad_content,
    memory_event,
    read_todos,
    render_notepad_section,
    write_todos,
)
from wqagent.graph.state import (
    DEFAULT_MAX_ATTEMPTS,
    TODO_MARKS,
    TODO_STATUSES,
    AgentHandoff,
    SourceItem,
    TodoItem,
    VerificationCheck,
    VerificationResult,
    WQGraphState,
    make_todo,
    normalise_todos,
    render_todos,
)
from wqagent.prompts.entry import CHAT_RESPONDER_PROMPT, INTENT_ROUTER_PROMPT
from wqagent.prompts.stage1 import ACTOR_PROMPT
from wqagent.prompts.stage2 import FINAL_PROMPT
from wqagent.prompts.stage3 import PLANNER_PROMPT, VERIFIER_PROMPT
from wqagent.prompts.stage4 import CONTEXT_COMPRESSION_PROMPT
from wqagent.providers import create_model
from wqagent.tools.bash_tool import SHELL_NAME, BashTool, shell_guidance
from wqagent.tools.graph_tools import (
    TodoUpdateTool,
    TodoWriteTool,
    VerificationReportTool,
    apply_todo_updates,
)
from wqagent.tools.registry import build_read_only_tools, build_tools

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.tools import StructuredTool

__all__ = [
    "CallCodeAgentTool",
    "CallSearchAgentTool",
    "COMPRESSION_ITEMS",
    "COMPRESS_TRANSCRIPT_CHARS",
    "DEFAULT_ACTOR_LOOPS",
    "DEFAULT_CONTEXT_TOKEN_LIMIT",
    "DEFAULT_PLANNER_LOOPS",
    "DEFAULT_VERIFIER_LOOPS",
    "SOURCE_LIMIT",
    "VERIFY_OUTPUT_CHARS",
    "actor_node",
    "context_compressor_node",
    "context_compressor_route",
    "context_monitor_node",
    "context_monitor_route",
    "final_node",
    "format_final_answer",
    "get_model",
    "planner_node",
    "run_verification_commands",
    "set_model",
    "verifier_node",
    "verifier_route",
]

#: ReAct loops the supervisor planner may use per round.
DEFAULT_PLANNER_LOOPS = 10
#: ReAct loops the (pre-stage-3) actor may use per attempt.
DEFAULT_ACTOR_LOOPS = 10
#: Read-only inspection loops the verifier may use before reporting.
DEFAULT_VERIFIER_LOOPS = 6
#: stdout/stderr characters kept per verification command in the state.
VERIFY_OUTPUT_CHARS = 4_000
#: Transcript characters handed to the compressor in one request (head + tail).
COMPRESS_TRANSCRIPT_CHARS = 120_000
#: Items kept in the state's ``sources`` after a compression.
SOURCE_LIMIT = 20

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
    return normalise_todos(state.get("todos"))


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
def _forward(
    event: dict[str, Any],
    node: str,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    writer: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Hand one event to the node's callback and to the graph's stream.

    Both copies are tagged with ``node``: LangGraph's custom chunks are otherwise
    anonymous, and the consumer needs to know who produced them.
    """
    tagged = {**event, "node": node}
    if on_event is not None:
        try:
            on_event(tagged)
        except Exception:  # pragma: no cover - a broken callback must not kill the run
            pass
    write_event(writer, tagged)


# -- planner (supervisor) ----------------------------------------------------
class CallAgentInput(BaseModel):
    """Arguments of the specialist handoff tools."""

    instruction: str = Field(
        description=(
            "A self-contained instruction for the specialist: what to do, and what to "
            "return. It cannot see this conversation."
        )
    )


@dataclass
class _HandoffState:
    """State effects collected while the planner delegates to specialists."""

    plan_summary: str | None = None
    todos: list[TodoItem] | None = None
    acceptance_criteria: list[str] | None = None
    verification_commands: list[str] | None = None
    research_notes: str = ""
    sources: list[SourceItem] = field(default_factory=list)
    code_agent_summary: str = ""
    messages: list[BaseMessage] = field(default_factory=list)
    handoffs: list[AgentHandoff] = field(default_factory=list)


def _agent_state(state: Mapping[str, Any], collector: _HandoffState | None) -> dict[str, Any]:
    """The state view handed to a specialist: the plan as it stands right now."""
    view = dict(state)
    if collector is None:
        return view
    if collector.todos is not None:
        view["todos"] = [dict(todo) for todo in collector.todos]
    if collector.plan_summary is not None:
        view["plan_summary"] = collector.plan_summary
    if collector.research_notes:
        view["research_notes"] = collector.research_notes
    if collector.code_agent_summary:
        view["code_agent_summary"] = collector.code_agent_summary
    return view


def _agent_writer(
    writer: Callable[[dict[str, Any]], None] | None,
    agent: str,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> Callable[[dict[str, Any]], None]:
    """Tag the specialist's own events with the agent that produced them."""

    def emit(event: dict[str, Any]) -> None:
        tagged = {**event, "node": agent}
        if on_event is not None:
            try:
                on_event(tagged)
            except Exception:  # pragma: no cover - a broken callback
                pass
        write_event(writer, tagged)

    return emit


def _note(
    event: dict[str, Any],
    writer: Callable[[dict[str, Any]], None] | None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Emit a node-level event (e.g. a handoff) to both sinks."""
    if on_event is not None:
        try:
            on_event(event)
        except Exception:  # pragma: no cover - a broken callback
            pass
    write_event(writer, event)


def _merge_notes(existing: str, summary: str, sources: Sequence[Mapping[str, Any]]) -> str:
    """Append a research summary (with its source URLs) to the notes."""
    parts = [existing.strip()] if existing.strip() else []
    if summary.strip():
        parts.append(summary.strip())
    urls = [str(source.get("url") or "").strip() for source in sources]
    urls = [url for url in urls if url]
    if urls:
        parts.append("Sources:\n" + "\n".join(f"- {url}" for url in urls))
    return "\n\n".join(parts)


def _merge_sources(
    existing: Sequence[Mapping[str, Any]], found: Sequence[Mapping[str, Any]]
) -> list[SourceItem]:
    """Merge source lists, de-duplicating by url."""
    merged: list[SourceItem] = [dict(source) for source in existing if source.get("url")]
    known = {str(source.get("url")) for source in merged}
    for source in found:
        url = str(source.get("url") or "").strip()
        if not url or url in known:
            continue
        known.add(url)
        merged.append(dict(source))
    return merged


def _call_search_agent_tool(
    state: Mapping[str, Any],
    writer: Callable[[dict[str, Any]], None] | None,
    instruction: str,
    *,
    collector: _HandoffState | None = None,
    model: Any | None = None,
    tool: Any | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Delegate *instruction* to searchAgent and record the handoff.

    *state* is the view the specialist sees (see :func:`_agent_state`); the state
    effects are collected into *collector* so the planner node can return them as
    its update. The agent's own result dict is returned unchanged.
    """
    # Imported here on purpose: `wqagent.tools.graph_tools` pulls in
    # `wqagent.graph.state`, which executes this package's __init__, so a
    # module-level import would close a cycle back into graph_tools.
    from wqagent.agents.search_agent import run_search_agent

    instruction = str(instruction or "").strip()
    _note(
        {
            "type": "handoff",
            "node": "planner",
            "from": "planner",
            "to": "searchAgent",
            "instruction": instruction,
        },
        writer,
        on_event,
    )
    result = run_search_agent(
        state,
        instruction,
        writer=_agent_writer(writer, "searchAgent", on_event),
        model=model,
        tool=tool,
    )
    if collector is not None:
        sources = list(result.get("sources") or [])
        collector.handoffs.append(
            AgentHandoff(
                from_agent="planner",
                to_agent="searchAgent",
                instruction=instruction,
                result=str(result.get("summary") or ""),
            )
        )
        collector.research_notes = _merge_notes(
            collector.research_notes or str(state.get("research_notes") or ""),
            str(result.get("summary") or ""),
            sources,
        )
        collector.sources = _merge_sources(collector.sources, sources)
    return result


def _call_code_agent_tool(
    state: Mapping[str, Any],
    writer: Callable[[dict[str, Any]], None] | None,
    instruction: str,
    *,
    collector: _HandoffState | None = None,
    model: Any | None = None,
    tools: Sequence[Any] | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    persist: Callable[[Sequence[TodoItem]], None] | None = None,
) -> dict[str, Any]:
    """Delegate *instruction* to codeAgent and record the handoff.

    The specialist's todos, summary and transcript are collected into *collector*;
    its own result dict is returned unchanged.
    """
    # Local import for the same reason as in _call_search_agent_tool.
    from wqagent.agents.code_agent import run_code_agent

    instruction = str(instruction or "").strip()
    _note(
        {
            "type": "handoff",
            "node": "planner",
            "from": "planner",
            "to": "codeAgent",
            "instruction": instruction,
        },
        writer,
        on_event,
    )
    result = run_code_agent(
        state,
        instruction,
        writer=_agent_writer(writer, "codeAgent", on_event),
        model=model,
        tools=tools,
        persist=persist,
    )
    if collector is not None:
        collector.handoffs.append(
            AgentHandoff(
                from_agent="planner",
                to_agent="codeAgent",
                instruction=instruction,
                result=str(result.get("summary") or ""),
            )
        )
        todos = result.get("todos")
        if todos:
            collector.todos = normalise_todos(todos)
        summary = str(result.get("summary") or "").strip()
        if summary:
            collector.code_agent_summary = summary
        collector.messages.extend(result.get("messages") or [])
    return result


def _specialist_report(agent: str, result: Mapping[str, Any]) -> str:
    """The compact report the planner model sees as the tool result."""
    payload: dict[str, Any] = {
        "agent": agent,
        "ok": bool(result.get("ok")),
        "error": str(result.get("error") or ""),
        "summary": str(result.get("summary") or ""),
    }
    if agent == "searchAgent":
        payload["sources"] = [
            str(source.get("url") or "") for source in (result.get("sources") or [])
        ][:8]
    else:
        payload["todos"] = [
            {"id": todo.get("id"), "status": todo.get("status")}
            for todo in (result.get("todos") or [])
        ]
    return json.dumps(payload, ensure_ascii=False)


class CallSearchAgentTool:
    """Planner tool: hand a research instruction to searchAgent."""

    name: str = "call_search_agent"
    description: str = (
        "Delegate web/document research to the searchAgent specialist. Give it a "
        "self-contained question; it returns a research summary and source URLs, which are "
        "stored in the shared research notes."
    )
    args_schema: type[BaseModel] = CallAgentInput

    def __init__(
        self,
        state: Mapping[str, Any],
        *,
        writer: Callable[[dict[str, Any]], None] | None = None,
        collector: _HandoffState | None = None,
        model: Any | None = None,
        tool: Any | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._state = state
        self._writer = writer
        self._collector = collector
        self._model = model
        self._tool = tool
        self._on_event = on_event

    def run(self, instruction: str) -> str:
        result = _call_search_agent_tool(
            _agent_state(self._state, self._collector),
            self._writer,
            instruction,
            collector=self._collector,
            model=self._model,
            tool=self._tool,
            on_event=self._on_event,
        )
        return _specialist_report("searchAgent", result)

    def as_tool(self) -> "StructuredTool":
        """Wrap this implementation in a LangChain ``StructuredTool``."""
        from langchain_core.tools import StructuredTool

        return StructuredTool.from_function(
            func=self.run,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
        )


class CallCodeAgentTool:
    """Planner tool: hand an implementation instruction to codeAgent."""

    name: str = "call_code_agent"
    description: str = (
        "Delegate file/code implementation to the codeAgent specialist. Give it a "
        "self-contained instruction naming the files involved; it edits the workspace, keeps "
        "the todo list up to date and returns a summary of what changed."
    )
    args_schema: type[BaseModel] = CallAgentInput

    def __init__(
        self,
        state: Mapping[str, Any],
        *,
        writer: Callable[[dict[str, Any]], None] | None = None,
        collector: _HandoffState | None = None,
        model: Any | None = None,
        tools: Sequence[Any] | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        persist: Callable[[Sequence[TodoItem]], None] | None = None,
    ) -> None:
        self._state = state
        self._writer = writer
        self._collector = collector
        self._model = model
        self._tools = tools
        self._on_event = on_event
        self._persist = persist
        self._persist = persist

    def run(self, instruction: str) -> str:
        result = _call_code_agent_tool(
            _agent_state(self._state, self._collector),
            self._writer,
            instruction,
            collector=self._collector,
            model=self._model,
            tools=self._tools,
            on_event=self._on_event,
            persist=self._persist,
        )
        return _specialist_report("codeAgent", result)

    def as_tool(self) -> "StructuredTool":
        """Wrap this implementation in a LangChain ``StructuredTool``."""
        from langchain_core.tools import StructuredTool

        return StructuredTool.from_function(
            func=self.run,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
        )


def _todo_persister(
    runtime: RuntimeState, state: Mapping[str, Any]
) -> Callable[[Sequence[TodoItem]], None]:
    """A ``persist`` hook that mirrors the todo list into ``TODO.md``."""

    def persist(todos: Sequence[TodoItem]) -> None:
        write_todos(
            runtime,
            todos,
            task=str(state.get("task") or ""),
            plan_summary=str(state.get("plan_summary") or ""),
            acceptance_criteria=_text_list(state.get("acceptance_criteria")),
            verification_commands=_text_list(state.get("verification_commands")),
        )

    return persist


def _persist_plan(runtime: RuntimeState, state: Mapping[str, Any], **overrides: Any) -> None:
    """Mirror the plan into ``TODO.md`` after a node changed it (best effort)."""
    write_todos(
        runtime,
        overrides.get("todos", _todos(state)),
        task=str(state.get("task") or ""),
        plan_summary=str(overrides.get("plan_summary", state.get("plan_summary")) or ""),
        acceptance_criteria=_text_list(
            overrides.get("acceptance_criteria", state.get("acceptance_criteria"))
        ),
        verification_commands=_text_list(
            overrides.get("verification_commands", state.get("verification_commands"))
        ),
    )


def _resume_plan(
    runtime: RuntimeState, state: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Adopt the plan recorded in ``TODO.md`` when it belongs to this same task.

    A run that was killed mid-flight can simply be started again: the plan, the
    remaining todos and the verification commands are read back from the workspace
    instead of being replanned from scratch. Only an *unfinished* plan for the same
    task is adopted - a different task, or one whose todos are all completed, is
    ignored (and overwritten once the planner publishes its own plan).
    """
    if _todos(state):
        return None
    recorded = read_todos(runtime)
    if not recorded.get("exists") or not recorded.get("todos"):
        return None
    task = str(state.get("task") or "").strip()
    if not task or str(recorded.get("task") or "").strip() != task:
        return None
    todos = normalise_todos(recorded.get("todos"))
    if not todos or all(todo["status"] == "completed" for todo in todos):
        return None
    plan: dict[str, Any] = {"todos": todos}
    if recorded.get("plan_summary"):
        plan["plan_summary"] = str(recorded["plan_summary"])
    if recorded.get("acceptance_criteria"):
        plan["acceptance_criteria"] = _text_list(recorded["acceptance_criteria"])
    if recorded.get("verification_commands"):
        plan["verification_commands"] = _text_list(recorded["verification_commands"])
    return plan


def planner_node(
    state: WQGraphState,
    *,
    model: Any | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    writer: Callable[[dict[str, Any]], None] | None = None,
    max_loops: int = DEFAULT_PLANNER_LOOPS,
    agent_model: Any | None = None,
    code_tools: Sequence[Any] | None = None,
    search_tool: Any | None = None,
) -> dict[str, Any]:
    """Supervise the round: plan, delegate to the specialists, summarise.

    The planner cannot touch files or the web itself - it binds three tools
    (``todo_write``, ``call_search_agent``, ``call_code_agent``) and calls them in
    whatever order the task needs. Returns the plan, everything the specialists
    produced (todos, research notes, sources, transcripts, handoff log) and the
    supervisor summary in ``last_actor_summary``.
    """
    _runtime(state)  # fail fast if the graph state was not built properly
    runtime = _runtime(state)
    chat = model if model is not None else get_model()
    # a killed run leaves its plan in TODO.md; the same task picks it up again
    plan = _resume_plan(runtime, state) or {}
    working = {**dict(state), **plan} if plan else state
    memory = build_layered_memory(working, node="planner")
    _note(memory_event(memory, node="planner"), writer, on_event)
    # the specialists run on the planner's model unless one is pinned for them
    specialist_model = agent_model if agent_model is not None else chat
    collector = _HandoffState(
        plan_summary=str(working.get("plan_summary") or ""),
        todos=_todos(working),
        acceptance_criteria=_text_list(working.get("acceptance_criteria")),
        verification_commands=_text_list(working.get("verification_commands")),
        research_notes=str(state.get("research_notes") or ""),
        sources=[dict(source) for source in (state.get("sources") or [])],
        code_agent_summary=str(state.get("code_agent_summary") or ""),
    )
    tools: list[Any] = [
        TodoWriteTool().as_tool(),
        CallSearchAgentTool(
            working,
            writer=writer,
            collector=collector,
            model=specialist_model,
            tool=search_tool,
            on_event=on_event,
        ).as_tool(),
        CallCodeAgentTool(
            working,
            writer=writer,
            collector=collector,
            model=specialist_model,
            tools=code_tools,
            on_event=on_event,
            persist=_todo_persister(runtime, working),
        ).as_tool(),
    ]

    messages: list[BaseMessage] = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=_planner_input(working, memory, resumed=bool(plan))),
    ]

    def on_react_event(kind: str, payload: dict[str, Any]) -> None:
        if kind != "tool_call":
            return
        if str(payload.get("name") or "") != TodoWriteTool.name:
            return
        args = payload.get("args")
        published = args if isinstance(args, Mapping) else {}
        summary = _text(published.get("plan_summary"))
        if summary:
            collector.plan_summary = summary
        todos = _coerce_todos(published.get("todos"), fallback=collector.todos or ())
        if todos:
            collector.todos = todos
        criteria = _text_list(published.get("acceptance_criteria"))
        if criteria:
            collector.acceptance_criteria = criteria
        commands = _text_list(published.get("verification_commands"))
        if commands:
            collector.verification_commands = commands

    final_event: dict[str, Any] = {}
    for event in react_loop(
        messages, model=chat, tools=tools, max_loops=max_loops, on_event=on_react_event
    ):
        if event.get("type") == "final_answer":
            final_event = event
        _forward(event, "planner", on_event, writer)

    summary = (
        last_ai_text(messages)
        or str(final_event.get("content") or "").strip()
        or str(state.get("last_actor_summary") or "")
    )
    plan_summary = collector.plan_summary or ""
    plan_todos = collector.todos if collector.todos is not None else _todos(working)
    criteria = collector.acceptance_criteria or []
    commands_out = collector.verification_commands or []
    # the plan is the working state: keep TODO.md in the workspace in sync with it
    _persist_plan(
        runtime,
        working,
        todos=plan_todos,
        plan_summary=plan_summary,
        acceptance_criteria=criteria,
        verification_commands=commands_out,
    )
    return {
        "plan_summary": plan_summary,
        "todos": plan_todos,
        "acceptance_criteria": criteria,
        "verification_commands": commands_out,
        "messages": collector.messages,
        "last_actor_summary": summary,
        "research_notes": collector.research_notes,
        "sources": collector.sources,
        "agent_handoffs": [
            *[dict(handoff) for handoff in (state.get("agent_handoffs") or [])],
            *collector.handoffs,
        ],
        "code_agent_summary": collector.code_agent_summary,
        # the supervisor just planned/delegated, so verification is what comes next
        "context_next_node": "verifier",
        # what this node showed the model (later readers reuse it instead of rebuilding)
        "memory_snapshot": memory,
    }


def _planner_input(
    state: Mapping[str, Any], memory: Mapping[str, Any], *, resumed: bool = False
) -> str:
    """The planner's human message: the planning brief plus the layered memory."""
    brief = _planner_instruction(state)
    if resumed:
        brief = (
            "A previous run left an unfinished plan for this exact task in TODO.md; it is "
            "shown below and already loaded into the todos.\n\n" + brief
        )
    return _with_layered_memory(brief, memory)


def _with_layered_memory(text: str, memory: Mapping[str, Any] | None) -> str:
    """Append the formatted layered memory to a node's human message."""
    if not memory:
        return text
    return (
        f"{text}\n\n"
        "Layered memory (rules, current task state, durable notes and compressed "
        "history):\n"
        f"{format_layered_memory_for_prompt(memory)}"
    )


def _planner_instruction(state: Mapping[str, Any]) -> str:
    """What the supervisor is told before it starts delegating."""
    existing = _todos(state)
    last_error = str(state.get("last_error") or "").strip()
    lines = [f"Task:\n{str(state.get('task') or '').strip()}"]
    plan_summary = str(state.get("plan_summary") or "").strip()
    if plan_summary:
        lines.append(f"Approach so far:\n{plan_summary}")
    if existing:
        lines.append(f"Current plan:\n{render_todos(existing)}")
    criteria = _text_list(state.get("acceptance_criteria"))
    if criteria:
        lines.append("Acceptance criteria:\n" + "\n".join(f"- {item}" for item in criteria))
    commands = _text_list(state.get("verification_commands"))
    if commands:
        lines.append("Verification commands:\n" + "\n".join(f"- {command}" for command in commands))
    lines.append(
        f"Verification commands run in {SHELL_NAME} inside the workspace root. "
        f"{shell_guidance()}"
    )
    notes = str(state.get("research_notes") or "").strip()
    if notes:
        lines.append(f"Research notes already collected:\n{notes}")
    if last_error:
        lines.append(
            "The last verification FAILED with:\n"
            f"{last_error}\n\n"
            "Revise the plan (todo_write) and delegate only the missing fix."
        )
    else:
        lines.append("Publish the plan, delegate the work, then summarise.")
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
        last_ai_text(messages[first_new:])
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
            f"{render_todos(todos)}"
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
    on_event: Callable[[dict[str, Any]], None] | None = None,
    writer: Callable[[dict[str, Any]], None] | None = None,
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
    memory = build_layered_memory(state, node="verifier")
    _note(memory_event(memory, node="verifier"), writer, on_event)
    results = run_verification_commands(runtime, _text_list(state.get("verification_commands")))
    messages: list[BaseMessage] = [
        SystemMessage(content=VERIFIER_PROMPT),
        HumanMessage(content=_verifier_input(state, results, memory)),
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
    # verification can complete every todo: mirror that into TODO.md as well
    _persist_plan(runtime, state, todos=todos)

    return {
        "passed": passed,
        "attempts": attempts,
        "verification_results": results,
        "verification_checks": checks,
        "last_error": "" if passed else _compose_error(reason, instruction, results),
        "todos": todos,
        # what this node showed the model (later readers reuse it instead of rebuilding)
        "memory_snapshot": memory,
        # where a following context monitor should send the run if it must go on
        **({} if passed else {"context_next_node": "planner"}),
    }


def _verifier_input(
    state: Mapping[str, Any],
    results: Sequence[VerificationResult],
    memory: Mapping[str, Any],
) -> str:
    """The verifier's human message: the evidence plus the layered memory."""
    return _with_layered_memory(_verifier_instruction(state, results), memory)


def _verifier_instruction(
    state: Mapping[str, Any], results: Sequence[VerificationResult]
) -> str:
    lines = [f"Task:\n{str(state.get('task') or '').strip()}"]
    plan_summary = str(state.get("plan_summary") or "").strip()
    if plan_summary:
        lines.append(f"Plan summary:\n{plan_summary}")
    todos = _todos(state)
    if todos:
        lines.append(f"Todos as the actor left them:\n{render_todos(todos)}")
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
    implementer = str(state.get("code_agent_summary") or "").strip()
    if implementer:
        lines.append(f"The implementer (codeAgent) reports:\n{implementer}")
    if summary and summary != implementer:
        lines.append(f"The supervisor (planner) reports:\n{summary}")
    notes = str(state.get("research_notes") or "").strip()
    if notes:
        lines.append(f"Research notes gathered for this task:\n{notes}")
    lines.append(
        "Inspect the actual workspace with the read-only tools (do not trust the summaries "
        "above), then return the JSON verdict - as a `verification_report` tool call, or as "
        "plain JSON in your answer with the keys passed / reason / checks / "
        "recommended_next_instruction."
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


# -- context monitor ---------------------------------------------------------
def _memory_message(snapshot: Mapping[str, Any]) -> BaseMessage:
    """The layered-memory message injected into a node's input."""
    return SystemMessage(
        content="Layered memory:\n" + format_layered_memory_for_prompt(snapshot)
    )


def _memory_payload(state: Mapping[str, Any]) -> BaseMessage:
    """The layered-memory message that is counted as part of the context."""
    snapshot = state.get("memory_snapshot")
    if not isinstance(snapshot, Mapping) or not snapshot:
        snapshot = build_layered_memory(state, node="context_monitor")
    return _memory_message(snapshot)


def _context_text(messages: Sequence[Any]) -> str:
    """Everything the context is made of, as one string (for the fallback estimate)."""
    parts: list[str] = []
    for message in messages:
        text = message_text(message) if isinstance(message, BaseMessage) else str(message or "")
        if text:
            parts.append(text)
    return "\n".join(parts)


def _count_tokens(model: Any, messages: Sequence[Any]) -> int:
    """Tokens of ``messages``, or a ``len(text) // 4`` estimate when counting fails.

    The model's own counter is authoritative (it knows the provider's tokenizer);
    a model without ``get_num_tokens_from_messages`` - a fake in tests, or a
    provider that cannot count - must not break the monitor.
    """
    conversation = list(messages)
    counter = getattr(model, "get_num_tokens_from_messages", None)
    if callable(counter):
        try:
            counted = int(counter(conversation))
        except Exception:  # noqa: BLE001 - any counting failure falls back below
            counted = -1
        if counted >= 0:
            return counted
    return len(_context_text(conversation)) // 4


def context_monitor_node(
    state: WQGraphState,
    *,
    model: Any | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    writer: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Measure the context and decide whether it has to be compressed.

    1. Estimates the tokens of ``messages`` plus the layered-memory payload with
       ``model.get_num_tokens_from_messages(messages + [memory_payload])``, and
       falls back to ``len(text) // 4`` when the model cannot count.
    2. ``should_compress`` is ``token_count > context_token_limit`` (default
       :data:`~wqagent.graph.memory.DEFAULT_CONTEXT_TOKEN_LIMIT`, 400000).
    3. ``context_next_node`` is whatever the upstream node decided - the planner
       sets ``"verifier"``, a failed verifier sets ``"planner"`` - and defaults to
       ``"verifier"``.

    Returns ``context_token_count``, ``context_should_compress`` and
    ``context_next_node``.
    """
    _runtime(state)  # fail fast if the graph state was not built properly
    chat = model if model is not None else get_model()
    # reuse what the last node injected when it is still in the state
    memory = state.get("memory_snapshot")
    if not isinstance(memory, Mapping) or not memory:
        memory = build_layered_memory(state, node="context_monitor")
    # the memory payload is part of what is measured, so it is injected here too
    _note(memory_event(memory, node="context_monitor"), writer, on_event)
    messages = [*(state.get("messages") or []), _memory_message(memory)]
    token_count = _count_tokens(chat, messages)
    try:
        limit = int(state.get("context_token_limit") or DEFAULT_CONTEXT_TOKEN_LIMIT)
    except (TypeError, ValueError):  # a junk limit must not break the monitor
        limit = DEFAULT_CONTEXT_TOKEN_LIMIT
    next_node = str(state.get("context_next_node") or "").strip() or "verifier"
    last_after = _last_compressed_size(state)

    return {
        "context_token_count": token_count,
        "context_should_compress": token_count > limit
        and not _compression_would_be_pointless(last_after, limit),
        "context_next_node": next_node,
    }


def _last_compressed_size(state: Mapping[str, Any]) -> int | None:
    """The token count the previous compression achieved, if there was one."""
    events = state.get("compression_events")
    if not isinstance(events, Sequence) or not events:
        return None
    last = events[-1]
    if not isinstance(last, Mapping):
        return None
    try:
        return int(last.get("tokens_after"))
    except (TypeError, ValueError):
        return None


def _compression_would_be_pointless(last_after: int | None, limit: int) -> bool:
    """True when compressing again cannot get the context under the limit.

    A compression's result size is deterministic for a given state (the capped
    summary plus the memory payload), so once a compression has landed *above* the
    budget, repeating it can only reproduce that size. The monitor then lets the
    run continue instead of looping ``monitor -> compressor`` until LangGraph's
    recursion limit - a limit that low is a misconfiguration, not a task state.
    """
    return last_after is not None and last_after > limit


# -- routing -----------------------------------------------------------------
def context_monitor_route(state: WQGraphState) -> str:
    """Route after the context monitor: finish, compress, or continue.

    A passing verdict always wins, then a spent attempt budget stops the run (the
    same rule as :func:`verifier_route` - a failing task must never spin), then an
    over-budget context goes to the compressor, otherwise the run continues to the
    node the upstream one picked.
    """
    if state.get("passed"):
        return "final"
    attempts = int(state.get("attempts") or 0)
    max_attempts = int(state.get("max_attempts") or DEFAULT_MAX_ATTEMPTS)
    if attempts >= max_attempts:
        return "final"
    if state.get("context_should_compress"):
        return "context_compressor"
    return str(state.get("context_next_node") or "").strip() or "verifier"


def context_compressor_route(state: WQGraphState) -> str:
    """Route after compression: wherever the state says work continues."""
    return str(state.get("context_next_node") or "").strip() or "verifier"


# -- context compressor ------------------------------------------------------
def _transcript_text(messages: Sequence[Any], *, limit: int = COMPRESS_TRANSCRIPT_CHARS) -> str:
    """Render the conversation for the compressor, keeping head and tail.

    A context that is over budget can be far larger than one request may carry, so
    the middle is dropped: the head holds the task and the plan, the tail holds the
    latest failures and results. Everything in between is what the compressor is
    asked to summarise anyway.
    """
    lines: list[str] = []
    for message in messages:
        text = message_text(message) if isinstance(message, BaseMessage) else str(message or "")
        role = getattr(message, "type", "message")
        lines.append(f"[{role}] {text}".rstrip())
    transcript = "\n\n".join(lines)
    if limit <= 0 or len(transcript) <= limit:
        return transcript
    head = int(limit * 0.35)
    tail = limit - head
    dropped = len(transcript) - limit
    return (
        f"{transcript[:head]}\n\n... [{dropped} characters of transcript dropped "
        f"by context_compressor] ...\n\n{transcript[-tail:]}"
    )


def _compression_request(state: Mapping[str, Any], messages: Sequence[Any]) -> str:
    """The human message handed to the compressor: transcript + memory snapshot."""
    snapshot = state.get("memory_snapshot")
    if not isinstance(snapshot, Mapping) or not snapshot:
        snapshot = build_layered_memory(state, node="context_compressor")
    return (
        f"Task:\n{str(state.get('task') or '').strip()}\n\n"
        f"Messages to compress ({len(list(messages))}):\n"
        f"{_transcript_text(messages)}\n\n"
        "Layered memory:\n"
        f"{format_layered_memory_for_prompt(snapshot)}\n\n"
        "Return the compression JSON."
    )


#: Free-text fields of the compression JSON, and where each one is capped.
_COMPRESSION_TEXT_FIELDS = ("summary", "active_goal", "completed_work", "risks")
#: List fields of the compression JSON.
_COMPRESSION_LIST_FIELDS = (
    "open_todos",
    "important_files",
    "tool_findings",
    "sources",
    "next_steps",
)


def _compression_items(value: Any) -> list[str]:
    """Normalise a list field of the compression JSON.

    Models often stringify lists (``"important_files": "[]"`` or ``'["a.py"]'``),
    so a JSON-looking string is parsed and an empty result stays empty instead of
    showing up as a literal ``[]`` in the notepad.
    """
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            return [text]
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        if parsed is None or isinstance(parsed, Mapping):
            return []
        return [str(parsed).strip()]
    return _text_list(value)


def _compression_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise the compressor's JSON: strings, capped lists, never raising."""
    fields: dict[str, Any] = {}
    for name in _COMPRESSION_TEXT_FIELDS:
        limit = COMPRESSION_SUMMARY_CHARS if name == "summary" else COMPRESSION_FIELD_CHARS
        fields[name] = _short_text(_text(payload.get(name)), limit)
    for name in _COMPRESSION_LIST_FIELDS:
        fields[name] = _compression_items(payload.get(name))[:COMPRESSION_ITEMS]
    return fields


def _fallback_summary(state: Mapping[str, Any]) -> str:
    """A deterministic summary for when the compressor model is unusable.

    Losing detail is bad; leaving an over-budget context in place is worse, so the
    deterministic report of the state facts is used instead.
    """
    lines = [f"Task: {str(state.get('task') or '').strip()}"]
    plan_summary = str(state.get("plan_summary") or "").strip()
    if plan_summary:
        lines.append(f"Plan: {plan_summary}")
    todos = _todos(state)
    if todos:
        lines.append("Todos:\n" + render_todos(todos))
    criteria = _text_list(state.get("acceptance_criteria"))
    if criteria:
        lines.append("Acceptance criteria:\n" + "\n".join(f"- {item}" for item in criteria))
    commands = _text_list(state.get("verification_commands"))
    if commands:
        lines.append("Verification commands:\n" + "\n".join(f"- {item}" for item in commands))
    implementer = _text(state.get("code_agent_summary"))
    if implementer:
        lines.append(f"Implementer report: {implementer}")
    error = _text(state.get("last_error"))
    if error:
        lines.append(f"Last verification failure:\n{error}")
    notes = _text(state.get("research_notes"))
    if notes:
        lines.append(f"Research notes:\n{notes}")
    lines.append(
        f"History was compressed deterministically after {int(state.get('attempts') or 0)} "
        "attempt(s); no model summary was available."
    )
    return _short_text("\n\n".join(lines), COMPRESSION_SUMMARY_CHARS)


def _render_history_summary(
    fields: Mapping[str, Any], *, tokens_before: int, tokens_after: int, limit: int
) -> str:
    """The Markdown written to ``HISTORY_SUMMARY.md``."""
    lines = ["# History summary", "", _text(fields.get("summary"))]
    sections = (
        ("active_goal", "Active goal", False),
        ("completed_work", "Completed work", False),
        ("open_todos", "Open todos", True),
        ("important_files", "Important files", True),
        ("tool_findings", "Tool findings", True),
        ("sources", "Sources", True),
        ("next_steps", "Next steps", True),
        ("risks", "Risks", False),
    )
    for key, title, as_list in sections:
        value = fields.get(key)
        if as_list:
            items = _text_list(value)
            if not items:
                continue
            lines.extend(["", f"## {title}", *[f"- {item}" for item in items]])
        else:
            text = _text(value)
            if not text:
                continue
            lines.extend(["", f"## {title}", text])
    lines.extend(
        [
            "",
            "<!-- compressed by context_compressor_node:",
            f"     tokens {tokens_before} -> {tokens_after} (limit {limit}) -->",
            "",
        ]
    )
    return "\n".join(lines)


def _compression_count(state: Mapping[str, Any]) -> int:
    """How many compressions this run already recorded."""
    events = state.get("compression_events")
    return len(events) if isinstance(events, Sequence) else 0


def _looks_like_a_url(text: str) -> bool:
    """Cheap sanity filter for urls the compressor extracted from the transcript."""
    return "." in text and " " not in text and len(text) > 3


def _merge_source_refs(existing: Any, found: Any) -> list[dict[str, str]]:
    """Keep the structured sources (title + url) and add the urls the model kept."""
    refs = _source_refs(existing)
    known = {ref["url"] for ref in refs}
    for item in _text_list(found):
        if item in known or not _looks_like_a_url(item):
            continue
        known.add(item)
        refs.append({"title": "", "url": item})
    return refs[:SOURCE_LIMIT]


def _write_history_summary(runtime: RuntimeState, content: str) -> dict[str, Any]:
    """Write ``HISTORY_SUMMARY.md`` inside the workspace; never raises."""
    record: dict[str, Any] = {"ok": False, "path": HISTORY_SUMMARY_FILE, "error": ""}
    try:
        path = runtime.resolve(HISTORY_SUMMARY_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - persistence is best effort
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    record["ok"] = True
    return record


def context_compressor_node(
    state: WQGraphState,
    *,
    model: Any | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    writer: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Compress the transcript into one summary message plus compacted state.

    1. Asks the model (``CONTEXT_COMPRESSION_PROMPT``) for JSON with the resumable
       state: ``summary``, ``active_goal``, ``completed_work``, ``open_todos``,
       ``important_files``, ``tool_findings``, ``sources``, ``next_steps``,
       ``risks``. A model that fails or answers unusable JSON falls back to a
       deterministic summary of the state facts - an over-budget context must not
       survive just because the compressor could not run.
    2. Replaces the history with ``RemoveMessage(REMOVE_ALL_MESSAGES)`` followed by
       one ``AIMessage(content=summary)``.
    3. Persists the summary to ``HISTORY_SUMMARY.md`` (best effort).
    4. Caps every free-text field it writes back (``_short_text`` / ``_trim_handoffs``).
    5. Returns the new message list, the summary, the recounted tokens, the capped
       fields and the appended :class:`~wqagent.graph.memory.CompressionEvent`.
    """
    runtime = _runtime(state)
    chat = model if model is not None else get_model()
    current = list(state.get("messages") or [])
    memory = build_layered_memory(state, node="context_compressor")
    _note(memory_event(memory, node="context_compressor"), writer, on_event)
    payload = _memory_message(memory)
    tokens_before = _count_tokens(chat, [*current, payload])
    try:
        limit = int(state.get("context_token_limit") or DEFAULT_CONTEXT_TOKEN_LIMIT)
    except (TypeError, ValueError):
        limit = DEFAULT_CONTEXT_TOKEN_LIMIT

    fields: dict[str, Any] = {}
    ok = True
    error = ""
    try:
        response = chat.invoke(
            [
                SystemMessage(content=CONTEXT_COMPRESSION_PROMPT),
                HumanMessage(content=_compression_request(state, current)),
            ]
        )
        payload_fields = _json_object(message_text(response))
        fields = _compression_fields(payload_fields)
        if not fields.get("summary"):
            ok, error = False, "the compressor model returned no usable summary"
    except Exception as exc:  # noqa: BLE001 - a failed compressor still compresses
        ok, error = False, f"{type(exc).__name__}: {exc}"

    if not ok:
        fields = _compression_fields({"summary": _fallback_summary(state)})

    summary = _text(fields.get("summary"))
    new_messages = [RemoveMessage(id=REMOVE_ALL_MESSAGES), AIMessage(content=summary)]
    tokens_after = _count_tokens(chat, [AIMessage(content=summary), payload])
    messages_before = len(current)
    messages_after = 1  # the single summary message the transcript is replaced with

    persisted = _write_history_summary(
        runtime,
        _render_history_summary(
            fields, tokens_before=tokens_before, tokens_after=tokens_after, limit=limit
        ),
    )
    # the durable half of the compression lands in NOTEPAD.md - compressing the
    # context is exactly the moment the runtime distils what is worth keeping
    section = (
        render_notepad_section(
            fields,
            index=_compression_count(state) + 1,
            attempt=int(state.get("attempts") or 0),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )
        if has_notepad_content(fields)
        else ""
    )
    notepad = (
        append_notepad(runtime, section)
        if section
        else {
            "ok": False,
            "path": NOTEPAD_FILE,
            "error": "no durable fields in the compression",
            "chars": 0,
        }
    )
    event: CompressionEvent = {
        "node": "context_compressor",
        "attempt": int(state.get("attempts") or 0),
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "messages_before": messages_before,
        "messages_after": messages_after,
        "token_limit": limit,
        "summary": summary,
    }
    _forward(
        {
            "type": "context_compressed",
            "ok": ok,
            "error": error,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "messages_before": messages_before,
            "messages_after": messages_after,
            "token_limit": limit,
            "summary": summary,
            "persisted": persisted,
            "notepad": notepad,
            "history_path": HISTORY_SUMMARY_FILE,
            "notepad_path": NOTEPAD_FILE,
        },
        "context_compressor",
        on_event,
        writer,
    )

    return {
        "messages": new_messages,
        "context_summary": summary,
        "context_token_count": tokens_after,
        "context_should_compress": False,
        "context_next_node": str(state.get("context_next_node") or "").strip() or "verifier",
        # the bulky fields are capped to the budgets the memory layer uses
        "research_notes": _short_text(state.get("research_notes"), RESEARCH_CHARS),
        "agent_handoffs": _trim_handoffs(state.get("agent_handoffs")),
        "sources": _merge_source_refs(state.get("sources"), fields.get("sources")),
        "code_agent_summary": _short_text(state.get("code_agent_summary"), SUMMARY_CHARS),
        "history_summary": summary,
        "compression_events": [
            *[
                dict(item)
                for item in (state.get("compression_events") or [])
                if isinstance(item, Mapping)
            ],
            event,
        ],
        # derived data: the next reader rebuilds it from the compacted state
        "memory_snapshot": {},
    }


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
            mark = TODO_MARKS.get(str(todo.get("status")), TODO_MARKS["pending"])
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

    actor_summary = str(state.get("code_agent_summary") or "").strip() or str(
        state.get("last_actor_summary") or ""
    ).strip()
    if actor_summary:
        lines.append(f"**What the implementer reported:**\n{actor_summary}")

    sources = [
        source
        for source in (state.get("sources") or [])
        if isinstance(source, Mapping) and source.get("url")
    ]
    if sources:
        lines.append("**Sources:**")
        for source in sources[:10]:
            title = str(source.get("title") or "").strip()
            lines.append(
                f"- {title}: {source.get('url')}" if title else f"- {source.get('url')}"
            )

    handoffs = [
        handoff
        for handoff in (state.get("agent_handoffs") or [])
        if isinstance(handoff, Mapping)
    ]
    if handoffs:
        lines.append("**Delegations:**")
        for handoff in handoffs:
            lines.append(
                f"- {handoff.get('from_agent')} -> {handoff.get('to_agent')}: "
                f"{handoff.get('instruction')}"
            )

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
    on_event: Callable[[dict[str, Any]], None] | None = None,
    writer: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Write the user-facing ``final_answer``.

    A deterministic report is always built from the state; when *summarize* is
    true and a model is reachable, the model rewrites it (``FINAL_PROMPT``) into a
    concise answer - with the layered memory injected like every other node. Any
    model failure keeps the report, so this node never raises and ``final_answer``
    is never empty.
    """
    report = format_final_answer(state)
    if not summarize:
        return {"final_answer": report}
    memory: dict[str, Any] = {}
    try:
        memory = dict(build_layered_memory(state, node="final"))
    except Exception:  # noqa: BLE001 - memory must never break the answer
        memory = {}
    _note(memory_event(memory, node="final"), writer, on_event)
    try:
        chat = model if model is not None else get_model()
        response = chat.invoke(
            [
                SystemMessage(content=FINAL_PROMPT),
                HumanMessage(content=_final_instruction(report, memory)),
            ]
        )
        text = message_text(response).strip()
    except Exception:  # noqa: BLE001 - the deterministic report is always a valid answer
        text = ""
    return {"final_answer": text or report}


def _final_instruction(report: str, memory: Mapping[str, Any] | None = None) -> str:
    facts = (
        "Write the final answer for the user from these facts only.\n\n"
        f"--- facts ---\n{report}\n--- end of facts ---\n\n"
        "Short, no tool calls."
    )
    return _with_layered_memory(facts, memory)


# -- entry graph: intent routing and light chat ------------------------------
#: Confidence below which the router's answer is not trusted (it then routes to the
#: workflow, which can always fall back to asking/working).
INTENT_CONFIDENCE_THRESHOLD = 0.55
#: The routes the router may pick, and the default when it is unusable.
INTENT_ROUTES: tuple[str, ...] = ("chat", "workflow")
DEFAULT_INTENT_ROUTE = "workflow"


def _session_context(state: Mapping[str, Any]) -> str:
    """The bit of session context the entry nodes are allowed to use.

    Two sources, in this order: the on-disk conversation session
    (:func:`wqagent.core.session.build_session_context` - what happened in previous
    turns) and the current run's own state (task, compressed summary, plan). Workspace
    *facts* still belong to the workflow route, so nothing here claims to have read a
    file; the file list is only there to disambiguate follow-ups like "继续".
    """
    runtime = state.get("runtime")
    parts: list[str] = []
    workspace = getattr(runtime, "workspace", None)
    if workspace is not None:
        try:
            from wqagent.core.session import build_session_context

            on_disk = build_session_context(workspace).strip()
        except Exception:  # noqa: BLE001 - the session is context, never a requirement
            on_disk = ""
        if on_disk:
            parts.append(on_disk)
    session_id = str(getattr(runtime, "session_id", "") or state.get("session_id") or "")
    if session_id:
        parts.append(f"session: {session_id}")
    summary = _text(state.get("context_summary")) or _text(state.get("history_summary"))
    if summary:
        parts.append(f"earlier summary: {summary[:800]}")
    todos = _todos(state)
    if todos:
        parts.append("previous plan: " + "; ".join(f"{todo['id']}={todo['status']}" for todo in todos[:8]))
    messages = state.get("messages")
    if isinstance(messages, list):
        recent = [
            f"{getattr(message, 'type', '?')}: {message_text(message)[:300]}"
            for message in messages[-6:]
            if message_text(message).strip()
        ]
        if recent:
            parts.append("recent turns:\n" + "\n".join(recent))
    return "\n".join(parts)


def _entry_messages(system: str, instruction: str, state: Mapping[str, Any]) -> list[BaseMessage]:
    """System prompt + the session context block + the user's input."""
    context = _session_context(state)
    body = f"--- user input ---\n{_text(state.get('task')) or '(none)'}\n--- end of user input ---"
    if context:
        body = f"{body}\n\n--- session context ---\n{context}\n--- end of session context ---"
    return [SystemMessage(content=system), HumanMessage(content=f"{instruction}\n\n{body}")]


def intent_router_node(
    state: WQGraphState,
    *,
    model: Any | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    writer: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Decide whether the user's input is conversation (``chat``) or work.

    Asks the model for ``{"route", "reason", "confidence"}``. Anything unusable - bad
    JSON, an unknown route, a confidence below :data:`INTENT_CONFIDENCE_THRESHOLD` -
    falls back to ``workflow``, so a broken router can only ever cost one extra round
    of real work, never a wrongly skipped one.
    """
    route = DEFAULT_INTENT_ROUTE
    reason = "router unavailable; defaulting to workflow"
    confidence = 0.0
    try:
        chat = model if model is not None else get_model()
        response = chat.invoke(
            _entry_messages(
                INTENT_ROUTER_PROMPT,
                "Classify the user input above and answer with the JSON object only.",
                state,
            )
        )
        payload = _json_object(message_text(response))
        candidate = str(payload.get("route") or "").strip().lower()
        reason = _text(payload.get("reason")) or reason
        confidence = _confidence(payload.get("confidence"))
        if candidate not in INTENT_ROUTES:
            route = DEFAULT_INTENT_ROUTE
            reason = f"unusable route {candidate!r}; {reason}"
        elif confidence < INTENT_CONFIDENCE_THRESHOLD:
            route = DEFAULT_INTENT_ROUTE
            reason = (
                f"confidence {confidence:.2f} below {INTENT_CONFIDENCE_THRESHOLD}; "
                f"router said {candidate} ({reason})"
            )
        else:
            route = candidate
    except Exception as exc:  # noqa: BLE001 - routing must never break the run
        reason = f"router failed ({type(exc).__name__}: {exc}); defaulting to workflow"

    update = {
        "intent_route": route,
        "intent_reason": reason,
        "intent_confidence": confidence,
    }
    event = {"type": "intent", "node": "intent_router", **update}
    _note(event, writer, on_event)
    return update


def chat_responder_node(
    state: WQGraphState,
    *,
    model: Any | None = None,
    on_event: Callable[[str | dict[str, Any]], None] | None = None,
    writer: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Answer a conversational input directly - with no tools at all.

    Returns both ``chat_response`` and ``final_answer`` so the run closes exactly like
    a workflow run does.
    """
    try:
        chat = model if model is not None else get_model()
        response = chat.invoke(
            _entry_messages(
                CHAT_RESPONDER_PROMPT,
                "Reply to the user input above. No tool calls, no invented workspace facts.",
                state,
            )
        )
        text = message_text(response).strip()
    except Exception as exc:  # noqa: BLE001 - always answer something
        text = f"(chat unavailable: {type(exc).__name__}: {exc})"
    if not text:
        text = "I can chat, but I have nothing to add - ask me a question or give me a task."
    update = {"chat_response": text, "final_answer": text}
    _note({"type": "chat_response", "node": "chat_responder", **update}, writer, on_event)
    return update


def intent_route_fn(state: WQGraphState) -> str:
    """Entry-graph router: the node to run (``chat_responder``) or ``planner``."""
    return "chat_responder" if state.get("intent_route") == "chat" else "planner"


def _confidence(value: Any) -> float:
    """A 0.0-1.0 confidence from model output (bad values become 0.0)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:  # NaN
        return 0.0
    return min(1.0, max(0.0, number))
