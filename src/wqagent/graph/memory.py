"""Three-layer memory for the WQ Agent graph.

The runtime - not the agents - assembles memory. Agents never get a memory *write*
tool (see :data:`RULES_LAYER`); whatever they need to know is injected into their
prompt as one JSON snapshot built by :func:`build_layered_memory`:

1. **Rules layer** (:data:`RULES_LAYER`) - fixed, code-owned rules about scope,
   storage and what each file means. Never changes between runs.
2. **Working memory** (``working_memory``) - the live task state: task, plan,
   todos, criteria, commands, research notes, sources, recent handoffs, the
   implementer's/reviewer's summaries, the last error and the attempt counters.
   Every free-text field is capped, so one snapshot always fits a prompt.
3. **History summary store** (``history_summary_store``) - the durable files
   (``NOTEPAD.md``, ``HISTORY_SUMMARY.md``) plus the compressed ``context_summary``
   and the last few compression records.

Usage::

    from wqagent.graph.memory import build_layered_memory, format_layered_memory_for_prompt

    memory = build_layered_memory(state, node="planner")
    prompt_block = format_layered_memory_for_prompt(memory)

Both durable files are read through :meth:`RuntimeState.resolve`, so they can only
ever come from inside the workspace; a missing file is ``exists=False``, never an
exception.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Mapping, TypedDict

from wqagent.core.state import RuntimeState

__all__ = [
    "COMPRESSION_EVENT_LIMIT",
    "COMPRESSION_FIELD_CHARS",
    "COMPRESSION_ITEMS",
    "COMPRESSION_SUMMARY_CHARS",
    "CONTEXT_SUMMARY_CHARS",
    "CONTEXT_TOKEN_LIMIT_ENV",
    "DEFAULT_CONTEXT_TOKEN_LIMIT",
    "ERROR_CHARS",
    "HANDOFF_LIMIT",
    "HANDOFF_TEXT_CHARS",
    "HISTORY_CHARS",
    "HISTORY_SUMMARY_FILE",
    "NOTEPAD_CHARS",
    "NOTEPAD_FILE",
    "NOTEPAD_FILE_CHARS",
    "NOTEPAD_SECTIONS",
    "RESEARCH_CHARS",
    "SESSION_CHARS",
    "SESSION_FILE",
    "RULES_LAYER",
    "SUMMARY_CHARS",
    "TODO_FILE",
    "CompressionEvent",
    "HistorySummaryStore",
    "LayeredMemory",
    "RulesLayer",
    "WorkingMemory",
    "append_notepad",
    "build_layered_memory",
    "context_token_limit_from_env",
    "format_layered_memory_for_prompt",
    "has_notepad_content",
    "memory_digest",
    "memory_event",
    "read_history_summary",
    "read_notepad",
    "read_session",
    "read_todos",
    "render_notepad_section",
    "render_todo_plan",
    "write_todos",
]

#: Layer 1: the fixed rules. Code-owned, never written by a model.
RULES_LAYER: dict[str, Any] = {
    "scope": "workspace",
    "storage": "internal",
    "rules": [
        "Work inside the current workspace only.",
        "Use paths relative to the workspace; do not prefix paths with workspace/.",
        "Keep durable task context outside the raw messages transcript when possible.",
        "Treat TODO.md as working plan state, NOTEPAD.md as durable notes, and HISTORY_SUMMARY.md as compressed history.",
        "Do not expose memory write tools to agents; layered memory is assembled by the runtime.",
    ],
}

#: Durable notes file, relative to the workspace.
NOTEPAD_FILE = "NOTEPAD.md"
#: Compressed history file, relative to the workspace.
HISTORY_SUMMARY_FILE = "HISTORY_SUMMARY.md"
#: Display path of the conversation session (see :mod:`wqagent.core.session`).
SESSION_FILE = ".wqagent/session/session.json"
#: Working-plan file, relative to the workspace (a mirror of the state's todos).
TODO_FILE = "TODO.md"

#: Context budget (tokens) a snapshot is measured against by the context monitor.
DEFAULT_CONTEXT_TOKEN_LIMIT = 400_000
#: Environment variable that overrides the context budget (e.g. ``2000`` to force
#: compression while testing).
CONTEXT_TOKEN_LIMIT_ENV = "WQ_CONTEXT_TOKEN_LIMIT"


def context_token_limit_from_env(default: int = DEFAULT_CONTEXT_TOKEN_LIMIT) -> int:
    """The context budget from ``WQ_CONTEXT_TOKEN_LIMIT``, or *default*.

    A missing or unusable value falls back silently: this is a test/tuning knob,
    and a typo in an environment variable must not break a run.
    """
    raw = os.environ.get(CONTEXT_TOKEN_LIMIT_ENV, "").strip()
    if not raw:
        return default
    try:
        limit = int(float(raw))
    except (TypeError, ValueError, OverflowError):  # "nope", "1e9999"
        return default
    return limit if limit > 0 else default

#: Characters kept per free-text field of one snapshot.
NOTEPAD_CHARS = 1_800
HISTORY_CHARS = 2_200
RESEARCH_CHARS = 1_600
#: Characters of the conversation-session block kept in the working memory.
SESSION_CHARS = 1_600
SUMMARY_CHARS = 1_000
ERROR_CHARS = 1_400
CONTEXT_SUMMARY_CHARS = 1_600
#: Characters kept per handoff field (instructions/summaries can be very long).
HANDOFF_TEXT_CHARS = 400
#: Characters kept for the compressor's ``summary``, and its other text fields.
COMPRESSION_SUMMARY_CHARS = 4_000
#: Characters kept per field of the compression JSON (also caps notepad sections).
COMPRESSION_FIELD_CHARS = 800
#: Items kept per list field of a compression result.
COMPRESSION_ITEMS = 20
#: How many handoffs / compression records one snapshot keeps (newest last).
HANDOFF_LIMIT = 6
COMPRESSION_EVENT_LIMIT = 3
#: Hard cap on how much of a memory file is read (the snapshot caps it again).
MEMORY_READ_CHARS = 20_000
#: Sections kept in ``NOTEPAD.md``: durable notes may grow, but not without bound.
NOTEPAD_FILE_CHARS = 12_000
#: Title of the notepad file.
NOTEPAD_TITLE = "# Notepad"

#: Fallback session id: one CLI run is one process, so this is stable per run.
_DEFAULT_SESSION_ID = f"wq-{os.getpid():x}-{int(time.time()):x}"


class RulesLayer(TypedDict, total=False):
    """Layer 1: what memory is, where it lives and what the files mean."""

    scope: str
    storage: str
    rules: list[str]


class WorkingMemory(TypedDict, total=False):
    """Layer 2: the current task state, every free-text field capped."""

    node: str
    task: str
    session_id: str
    session_turn: int
    plan_summary: str
    todos: list[dict[str, Any]]
    acceptance_criteria: list[str]
    verification_commands: list[str]
    research_notes: str
    sources: list[dict[str, str]]
    agent_handoffs: list[dict[str, Any]]
    code_agent_summary: str
    verifier_summary: str
    last_error: str
    attempts: int
    max_attempts: int


class CompressionEvent(TypedDict, total=False):
    """One context-compression record, appended by the compression step."""

    node: str
    attempt: int
    tokens_before: int
    tokens_after: int
    messages_before: int
    messages_after: int
    token_limit: int
    summary: str


class HistorySummaryStore(TypedDict, total=False):
    """Layer 3: durable files plus the compressed history of this run."""

    history_path: str
    history_exists: bool
    history_summary: str
    notepad_path: str
    notepad_exists: bool
    notepad: str
    session_path: str
    session_exists: bool
    session: str
    context_summary: str
    compression_events: list[CompressionEvent]


class LayeredMemory(TypedDict, total=False):
    """The whole snapshot, as stored in ``WQGraphState['memory_snapshot']``."""

    rules: RulesLayer
    working_memory: WorkingMemory
    history_summary_store: HistorySummaryStore


# -- helpers -----------------------------------------------------------------
def _short_text(text: Any, limit: int) -> str:
    """Cap *text* at *limit* characters, marking the cut with ``...``."""
    if limit <= 0:
        return ""
    value = "" if text is None else str(text)
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "..."


def _trim_handoffs(
    handoffs: Any, limit: int = HANDOFF_LIMIT, *, text_limit: int = HANDOFF_TEXT_CHARS
) -> list[dict[str, Any]]:
    """Keep the most recent *limit* handoffs (newest last), as capped copies.

    Each handoff keeps its keys but its free-text fields (``instruction``,
    ``result``) are capped at *text_limit*: six full-length instructions would
    otherwise dominate the snapshot on their own.
    """
    if limit <= 0:
        return []
    items = [item for item in _as_sequence(handoffs) if isinstance(item, Mapping)]
    trimmed: list[dict[str, Any]] = []
    for item in items[-limit:]:
        record = dict(item)
        for field in ("instruction", "result"):
            if field in record:
                record[field] = _short_text(record[field], text_limit)
        trimmed.append(record)
    return trimmed


def _rules_layer() -> RulesLayer:
    """A copy of :data:`RULES_LAYER` - mutating a snapshot must not touch it."""
    return {
        "scope": str(RULES_LAYER["scope"]),
        "storage": str(RULES_LAYER["storage"]),
        "rules": [str(rule) for rule in RULES_LAYER["rules"]],
    }


def _as_sequence(value: Any) -> list[Any]:
    """Iterate a state value without ever raising (junk becomes ``[]``)."""
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return []
    if isinstance(value, Mapping):
        return [value]
    try:
        return list(value)
    except TypeError:
        return []


def _as_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _copy_items(value: Any) -> list[Any]:
    """Copy a list of dicts (todos) so the snapshot shares nothing with the state."""
    return [dict(item) if isinstance(item, Mapping) else item for item in _as_sequence(value)]


def _text_items(value: Any) -> list[str]:
    """A list of non-empty strings from whatever the state holds."""
    return [text for text in (str(item).strip() for item in _as_sequence(value)) if text]


def _source_refs(sources: Any) -> list[dict[str, str]]:
    """Keep only ``title`` and ``url`` of every source that has a url."""
    refs: list[dict[str, str]] = []
    for source in _as_sequence(sources):
        if not isinstance(source, Mapping):
            continue
        url = str(source.get("url") or "").strip()
        if not url:
            continue
        refs.append({"title": str(source.get("title") or "").strip(), "url": url})
    return refs


def _session_id(state: Mapping[str, Any], runtime: RuntimeState) -> str:
    """Session id from the state, else the runtime, else this process."""
    for candidate in (state.get("session_id"), getattr(runtime, "session_id", "")):
        text = str(candidate or "").strip()
        if text:
            return text
    return _DEFAULT_SESSION_ID


def _session_turn(state: Mapping[str, Any]) -> int:
    """The round the graph is in (``attempts + 1``) unless the state says otherwise."""
    explicit = state.get("session_turn")
    if not isinstance(explicit, bool) and isinstance(explicit, int):
        return explicit
    return _as_int(state.get("attempts"), 0) + 1


# -- the durable files -------------------------------------------------------
def _read_memory_file(runtime: RuntimeState, name: str) -> dict[str, Any]:
    """Read one memory file inside the workspace; never raises."""
    record: dict[str, Any] = {"path": name, "exists": False, "content": "", "error": ""}
    try:
        path = runtime.resolve(name)
    except Exception as exc:  # noqa: BLE001 - workspace errors are data here
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    try:
        if not path.is_file():
            return record
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            content = handle.read(MEMORY_READ_CHARS + 1)
    except OSError as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    record["exists"] = True
    if len(content) > MEMORY_READ_CHARS:
        content = content[:MEMORY_READ_CHARS] + "\n... [truncated]"
    record["content"] = content
    return record


def read_notepad(runtime: RuntimeState) -> dict[str, Any]:
    """Read ``NOTEPAD.md`` from the workspace.

    Returns ``{path, exists, content, error}`` - a missing file is
    ``exists=False`` with empty content, and any read problem lands in ``error``
    instead of raising (memory must never break a run).
    """
    return _read_memory_file(runtime, NOTEPAD_FILE)


def read_history_summary(runtime: RuntimeState) -> dict[str, Any]:
    """Read ``HISTORY_SUMMARY.md`` from the workspace (same contract as notepad)."""
    return _read_memory_file(runtime, HISTORY_SUMMARY_FILE)


def read_session(runtime: RuntimeState) -> dict[str, Any]:
    """The conversation session as a memory block (``{path, exists, content, error}``).

    Built from ``.wqagent/session`` via
    :func:`wqagent.core.session.build_session_context`, so every node - planner,
    verifier, both specialists - knows what earlier turns in this workspace were about.
    A missing or broken session is simply ``exists=False``.
    """
    record: dict[str, Any] = {
        "path": SESSION_FILE,
        "exists": False,
        "content": "",
        "error": "",
    }
    try:
        from wqagent.core.session import build_session_context, session_path

        path = session_path(runtime)
        record["path"] = str(path)
        if not path.is_file():
            return record
        content = build_session_context(runtime).strip()
    except Exception as exc:  # noqa: BLE001 - memory must never break a run
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    record["exists"] = bool(content)
    record["content"] = content
    return record


# -- the notepad (durable notes, written when the context is compressed) -----
#: The compression fields that belong in the notepad, and their section titles.
#: ``summary`` and ``open_todos`` are not here: the first is the compressed history
#: (``HISTORY_SUMMARY.md``) and the second is working plan state (``TODO.md``).
NOTEPAD_SECTIONS: tuple[tuple[str, str], ...] = (
    ("completed_work", "Completed work"),
    ("important_files", "Important files"),
    ("tool_findings", "Tool findings"),
    ("sources", "Sources"),
    ("next_steps", "Next steps"),
    ("risks", "Risks"),
)


def has_notepad_content(fields: Mapping[str, Any]) -> bool:
    """True when a compression result carries anything worth keeping durably."""
    if isinstance(fields.get("active_goal"), str) and str(fields["active_goal"]).strip():
        return True
    for key, _ in NOTEPAD_SECTIONS:
        value = fields.get(key)
        if isinstance(value, str):
            if value.strip():
                return True
        elif _as_sequence(value):  # lists and mappings with at least one item
            return True
    return False


def render_notepad_section(
    fields: Mapping[str, Any],
    *,
    index: int = 1,
    attempt: int = 0,
    tokens_before: int = 0,
    tokens_after: int = 0,
) -> str:
    """Render one durable-notes section from a compression result.

    ``summary`` is deliberately left out: it is the compressed *history* and lives
    in ``HISTORY_SUMMARY.md``. What stays here is the part worth keeping across
    runs - the goal, what got done, the files, findings, sources, next steps and
    risks.
    """
    goal = _short_text(fields.get("active_goal"), COMPRESSION_FIELD_CHARS)
    heading = f"## Compression #{index}"
    if goal:
        heading += f" - {goal}"
    lines = [heading, ""]
    lines.append(f"<!-- attempt {attempt}, tokens {tokens_before} -> {tokens_after} -->")
    for key, title in NOTEPAD_SECTIONS:
        value = fields.get(key)
        if isinstance(value, str):
            text = value.strip()
            if text:
                lines.extend(["", f"**{title}**", _short_text(text, COMPRESSION_FIELD_CHARS)])
            continue
        items = [str(item).strip() for item in _as_sequence(value) if str(item).strip()]
        if items:
            lines.extend(["", f"**{title}**"])
            lines.extend(f"- {_short_text(item, COMPRESSION_FIELD_CHARS)}" for item in items)
            continue
        # an empty list/dict is nothing to report; a scalar still is
        if value is not None and not isinstance(value, (list, tuple, set, Mapping)):
            text = _short_text(value, COMPRESSION_FIELD_CHARS)
            if text:
                lines.extend(["", f"**{title}**", text])
    return "\n".join(lines).rstrip() + "\n"


def append_notepad(runtime: RuntimeState, section: str) -> dict[str, Any]:
    """Append one section to ``NOTEPAD.md``, keeping the newest ones only.

    Returns ``{ok, path, error, chars}`` - a failed write is data, never an
    exception.
    """
    record: dict[str, Any] = {"ok": False, "path": NOTEPAD_FILE, "error": "", "chars": 0}
    if not str(section or "").strip():
        record["error"] = "nothing to write"
        return record
    existing = read_notepad(runtime)
    content = str(existing.get("content") or "").rstrip()
    if not content.startswith(NOTEPAD_TITLE):
        content = NOTEPAD_TITLE if not content else f"{NOTEPAD_TITLE}\n\n{content}"
    content = f"{content.rstrip()}\n\n{str(section).strip()}\n"
    content = _trim_notepad(content)
    try:
        path = runtime.resolve(NOTEPAD_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - persistence is best effort
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    record["ok"] = True
    record["chars"] = len(content)
    return record


def _trim_notepad(content: str, *, keep: int = NOTEPAD_FILE_CHARS) -> str:
    """Keep the title and the newest sections within *keep* characters."""
    if len(content) <= keep:
        return content
    blocks = content.split("\n## ")
    title, sections = blocks[0].rstrip(), blocks[1:]
    kept: list[str] = []
    total = 0
    for section in reversed(sections):
        rendered = f"## {section.strip()}\n"
        if kept and total + len(rendered) > keep:
            break
        kept.insert(0, rendered)
        total += len(rendered)
    if not kept:  # one huge section: cut it down rather than dropping it
        kept = [f"## {sections[-1].strip()}"[:keep] + "\n"]
    return f"{title}\n\n{''.join(kept)}"


# -- the working plan file ---------------------------------------------------
def render_todo_plan(
    todos: Any,
    *,
    task: str = "",
    plan_summary: str = "",
    acceptance_criteria: Any = None,
    verification_commands: Any = None,
) -> str:
    """Render the plan as ``TODO.md``: the task, the checklist and how it is verified."""
    lines = ["# Working plan", ""]
    if task.strip():
        lines.extend([f"**Task:** {task.strip()}", ""])
    if plan_summary.strip():
        lines.extend([plan_summary.strip(), ""])
    items = [item for item in _as_sequence(todos) if isinstance(item, Mapping)]
    if items:
        # the marks come from graph.state so the file and the prompts agree
        from wqagent.graph.state import TODO_MARKS

        for item in items:
            mark = TODO_MARKS.get(str(item.get("status") or "pending"), "[ ]")
            note = str(item.get("note") or "").strip()
            line = f"- {mark} {item.get('id')}: {item.get('content')}"
            lines.append(f"{line}  <!-- {note} -->" if note else line)
        lines.append("")
    criteria = _as_sequence(acceptance_criteria)
    if criteria:
        lines.extend(["## Acceptance criteria", ""])
        lines.extend(f"- {_short_text(item, 300)}" for item in criteria)
        lines.append("")
    commands = _as_sequence(verification_commands)
    if commands:
        lines.extend(["## Verification commands", ""])
        lines.extend(f"- `{_short_text(item, 300)}`" for item in commands)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_todos(
    runtime: RuntimeState,
    todos: Any,
    *,
    task: str = "",
    plan_summary: str = "",
    acceptance_criteria: Any = None,
    verification_commands: Any = None,
) -> dict[str, Any]:
    """Persist the working plan to ``TODO.md`` inside the workspace.

    Returns ``{ok, path, error}`` - a failed write is data, never an exception.
    """
    record: dict[str, Any] = {"ok": False, "path": TODO_FILE, "error": ""}
    content = render_todo_plan(
        todos,
        task=task,
        plan_summary=plan_summary,
        acceptance_criteria=acceptance_criteria,
        verification_commands=verification_commands,
    )
    try:
        path = runtime.resolve(TODO_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - persistence is best effort
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record
    record["ok"] = True
    return record


def read_todos(runtime: RuntimeState) -> dict[str, Any]:
    """Read the plan back from ``TODO.md``.

    Returns ``{path, exists, task, plan_summary, todos, acceptance_criteria,
    verification_commands, content, error}`` - a missing file is ``exists=False``
    with empty values, and unparsable content simply yields no todos.
    """
    record = _read_memory_file(runtime, TODO_FILE)
    plan = _parse_todo_plan(str(record.get("content") or ""))
    return {**record, **plan}


def _parse_todo_plan(content: str) -> dict[str, Any]:
    """Parse the Markdown written by :func:`render_todo_plan` (tolerant)."""
    plan: dict[str, Any] = {
        "task": "",
        "plan_summary": "",
        "todos": [],
        "acceptance_criteria": [],
        "verification_commands": [],
    }
    if not content.strip():
        return plan
    from wqagent.graph.state import TODO_MARKS

    by_mark = {mark: status for status, mark in TODO_MARKS.items()}
    section = ""
    summary: list[str] = []
    for raw in content.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("**Task:**"):
            plan["task"] = line[len("**Task:**") :].strip()
            continue
        if line.startswith("##"):
            section = line.lstrip("#").strip().lower()
            continue
        if line.startswith("#"):  # the document title, not a section
            continue
        if section == "acceptance criteria" and line.startswith("- "):
            plan["acceptance_criteria"].append(line[2:].strip())
            continue
        if section == "verification commands" and line.startswith("- "):
            plan["verification_commands"].append(line[2:].strip().strip("`"))
            continue
        if line.startswith("- ") and ": " in line:
            body = line[2:]
            note = ""
            if "<!--" in body:
                body, _, tail = body.partition("<!--")
                note = tail.replace("-->", "").strip()
            body = body.strip()
            # match the status marker by prefix: "[ ]" contains a space itself
            status = None
            rest = body
            for mark, candidate in by_mark.items():
                if body.startswith(mark):
                    status, rest = candidate, body[len(mark) :].strip()
                    break
            if status is None:
                continue
            identifier, _, text = rest.partition(": ")
            plan["todos"].append(
                {
                    "id": identifier.strip(),
                    "content": text.strip() or identifier.strip(),
                    "status": status,
                    "note": note,
                }
            )
            continue
        if not plan["todos"] and not section:
            summary.append(line)
    plan["plan_summary"] = " ".join(summary).strip()
    return plan


# -- assembly ----------------------------------------------------------------
def build_layered_memory(state: Mapping[str, Any], *, node: str = "graph") -> LayeredMemory:
    """Assemble the three memory layers for *node* from the graph state.

    Args:
        state: the graph state (or anything mapping-like) to snapshot.
        node: which node/agent the snapshot is for; recorded in
            ``working_memory['node']`` so a prompt can say who it was built for.

    Returns:
        ``{"rules": ..., "working_memory": ..., "history_summary_store": ...}``.

    Raises:
        ValueError: if the state carries no :class:`RuntimeState` (the workspace is
            what memory is scoped to, so there is nothing to read without it).
    """
    runtime = state.get("runtime")
    if not isinstance(runtime, RuntimeState):
        raise ValueError("layered memory needs a RuntimeState in state['runtime']")

    notepad = read_notepad(runtime)
    history = read_history_summary(runtime)
    session = read_session(runtime)
    # the state's own (compressed) history wins over the file on disk
    history_summary = str(state.get("history_summary") or history.get("content") or "")

    working_memory: WorkingMemory = {
        "node": str(node or "graph"),
        "task": str(state.get("task") or ""),
        "session_id": _session_id(state, runtime),
        "session_turn": _session_turn(state),
        "plan_summary": str(state.get("plan_summary") or ""),
        "todos": _copy_items(state.get("todos")),
        "acceptance_criteria": _text_items(state.get("acceptance_criteria")),
        "verification_commands": _text_items(state.get("verification_commands")),
        "research_notes": _short_text(state.get("research_notes"), RESEARCH_CHARS),
        "sources": _source_refs(state.get("sources")),
        "agent_handoffs": _trim_handoffs(state.get("agent_handoffs")),
        "code_agent_summary": _short_text(state.get("code_agent_summary"), SUMMARY_CHARS),
        "verifier_summary": _short_text(state.get("verifier_summary"), SUMMARY_CHARS),
        "last_error": _short_text(state.get("last_error"), ERROR_CHARS),
        "attempts": _as_int(state.get("attempts"), 0),
        # matches graph.state.DEFAULT_MAX_ATTEMPTS; `new_graph_state` always sets it
        "max_attempts": _as_int(state.get("max_attempts"), 3),
    }

    history_summary_store: HistorySummaryStore = {
        "history_path": HISTORY_SUMMARY_FILE,
        "history_exists": bool(history.get("exists", False)),
        "history_summary": _short_text(history_summary, HISTORY_CHARS),
        "notepad_path": NOTEPAD_FILE,
        "notepad_exists": bool(notepad.get("exists", False)),
        "notepad": _short_text(notepad.get("content", ""), NOTEPAD_CHARS),
        "session_path": SESSION_FILE,
        "session_exists": bool(session.get("exists", False)),
        "session": _short_text(session.get("content", ""), SESSION_CHARS),
        "context_summary": _short_text(state.get("context_summary"), CONTEXT_SUMMARY_CHARS),
        "compression_events": list(_as_sequence(state.get("compression_events")))[
            -COMPRESSION_EVENT_LIMIT:
        ]
        if COMPRESSION_EVENT_LIMIT > 0
        else [],
    }

    return {
        "rules": _rules_layer(),
        "working_memory": working_memory,
        "history_summary_store": history_summary_store,
    }


def format_layered_memory_for_prompt(memory: Mapping[str, Any] | None) -> str:
    """Render a snapshot as pretty JSON, ready to paste into a prompt."""
    if not memory:
        return "{}"
    try:
        return json.dumps(memory, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return json.dumps({"error": "memory snapshot is not serialisable"})


def memory_digest(memory: Mapping[str, Any] | None) -> str:
    """One-line summary of a snapshot (logs, CLI, compression events)."""
    if not memory:
        return "empty"
    working = memory.get("working_memory")
    store = memory.get("history_summary_store")
    working = working if isinstance(working, Mapping) else {}
    store = store if isinstance(store, Mapping) else {}
    parts = [
        f"{len(_as_sequence(working.get('todos')))} todos",
        f"{len(_as_sequence(working.get('sources')))} sources",
        f"{len(_as_sequence(working.get('agent_handoffs')))} handoffs",
        f"notepad={'yes' if store.get('notepad_exists') else 'no'}",
        f"history={'yes' if store.get('history_exists') else 'no'}",
    ]
    notes = len(str(working.get("research_notes") or ""))
    if notes:
        parts.append(f"notes={notes}ch")
    return ", ".join(parts)


def memory_event(memory: Mapping[str, Any] | None, *, node: str = "graph") -> dict[str, Any]:
    """The event a node emits when it injects the layered memory into a prompt.

    Carries the whole snapshot (so a consumer can see exactly what the agent was
    told) plus the size and a one-line digest for logs and the CLI.
    """
    snapshot = dict(memory) if isinstance(memory, Mapping) else {}
    return {
        "type": "memory",
        "node": str(node or "graph"),
        "digest": memory_digest(snapshot),
        "chars": len(format_layered_memory_for_prompt(snapshot)),
        "memory": snapshot,
    }
