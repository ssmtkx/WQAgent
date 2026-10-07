"""Execution tracing: what the run did, in what order, and how much of it.

Three modes (``RuntimeState.trace_mode`` / ``WQ_TRACE_MODE``):

``summary`` (default)
    ``trace.json`` (counters + a trimmed timeline) and ``timeline.md`` (a readable
    version of the same). No per-event log.
``full``
    The same, plus ``events.jsonl`` with one JSON object per streamed event.
``off``
    Nothing is written; :meth:`TraceRecorder.end` returns ``None``.

Everything lands in ``<workspace>/.wqagent/traces/<trace_id>``::

    .wqagent/traces/wq-20261003-094912-a1b2/
        trace.json     counters, node visits, timeline head/tail
        events.jsonl   full mode: every event, one per line
        timeline.md    human-readable timeline

Tracing never breaks a run: an unwritable directory degrades to an in-memory
recorder whose writes are skipped.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

__all__ = [
    "DEFAULT_TRACE_MODE",
    "TIMELINE_HEAD",
    "TIMELINE_TAIL",
    "TRACE_DIR",
    "TRACE_MODE_ENV",
    "VALID_TRACE_MODES",
    "TraceRecorder",
    "new_trace_id",
    "normalize_trace_mode",
    "trace_mode_from_env",
]

#: Directory (inside the workspace) that holds tracing output.
TRACE_DIR = ".wqagent"
#: Sub-directory of :data:`TRACE_DIR` that holds the traces.
TRACES_SUBDIR = "traces"
#: Environment variable that chooses the trace mode.
TRACE_MODE_ENV = "WQ_TRACE_MODE"
#: Modes a run may use.
VALID_TRACE_MODES: frozenset[str] = frozenset({"full", "summary", "off"})
#: Spellings accepted as another mode (``on`` is the friendly form of ``summary``).
TRACE_MODE_ALIASES: dict[str, str] = {"on": "summary", "true": "summary", "yes": "summary"}
#: Mode used when none was chosen (and for unusable values).
DEFAULT_TRACE_MODE = "summary"
#: Timeline entries kept verbatim at the start and the end of ``trace.json``.
TIMELINE_HEAD = 20
TIMELINE_TAIL = 80
#: Timeline entries kept in ``timeline.md`` (a readable digest, not the full log).
MARKDOWN_ENTRIES = 200
#: Characters kept per timeline detail.
DETAIL_CHARS = 200

#: Event types that count as milestones even in ``summary`` mode.
_MILESTONES = frozenset(
    {
        "run_start",
        "run_end",
        "handoff",
        "checkpoint_saved",
        "memory",
        "context_compressed",
        "final_answer",
        "node_end",
        "resume",
    }
)


def normalize_trace_mode(mode: str | None) -> str:
    """The mode to use: ``on`` means ``summary``; unknown values fall back to it."""
    text = str(mode or "").strip().lower()
    if text in VALID_TRACE_MODES:
        return text
    return TRACE_MODE_ALIASES.get(text, DEFAULT_TRACE_MODE)


def trace_mode_from_env(default: str = DEFAULT_TRACE_MODE) -> str:
    """The mode from ``WQ_TRACE_MODE``, or *default* (itself normalized)."""
    raw = os.environ.get(TRACE_MODE_ENV, "")
    return normalize_trace_mode(raw) if raw.strip() else normalize_trace_mode(default)


def new_trace_id() -> str:
    """A readable, filesystem-safe trace id (``wq-20261003-094912-a1b2``)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"wq-{stamp}-{uuid4().hex[:4]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _clock() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _json_default(value: Any) -> Any:
    """Encoder for event values that are not JSON: messages, dataclasses, sets."""
    to_dict = getattr(value, "model_dump", None) or getattr(value, "dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:  # noqa: BLE001 - fall back to str()
            return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(str(item) for item in value)
    return str(value)


def _short(value: Any, limit: int = DETAIL_CHARS) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _count(value: Any) -> int:
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if isinstance(value, Mapping):
        return len(value)
    return 0


class TraceRecorder:
    """Collect the counters and the timeline of one run."""

    def __init__(self, runtime: Any, task: str = "") -> None:
        self.workspace = Path(getattr(runtime, "workspace", "."))
        self.mode = normalize_trace_mode(getattr(runtime, "trace_mode", None))
        self.trace_id = str(getattr(runtime, "trace_id", "") or "") or new_trace_id()
        self.root = self.workspace / TRACE_DIR / TRACES_SUBDIR / self.trace_id
        self.task = str(task or "")
        # counters
        self.node_visits: dict[str, int] = {}
        self.tool_calls = 0
        self.failed_tool_calls = 0
        self.approval_count = 0
        self.checkpoint_count = 0
        self.handoff_count = 0
        # bookkeeping
        self.started_at = ""
        self.ended_at = ""
        self.status = "running"
        self.latest_node = ""
        self.resumed = False
        self.resume_event: dict[str, Any] = {}
        self.final_state: dict[str, Any] = {}
        self.timeline: list[dict[str, Any]] = []
        self._sequence = 0
        self._io_failed = False

    # -- paths / state -------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """False in ``off`` mode: nothing is recorded and nothing is written."""
        return self.mode != "off"

    @property
    def full(self) -> bool:
        return self.mode == "full"

    @property
    def trace_path(self) -> Path:
        return self.root / "trace.json"

    @property
    def events_path(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def timeline_path(self) -> Path:
        return self.root / "timeline.md"

    # -- recording -----------------------------------------------------------
    def start(
        self,
        inputs: Mapping[str, Any] | None,
        *,
        resumed: bool = False,
        resume_event: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Record the ``run_start`` event; returns it, or ``None`` when off."""
        if not self.enabled:
            return None
        state = _as_mapping(inputs)
        self.started_at = _now()
        self.resumed = bool(resumed)
        self.resume_event = dict(resume_event or {})
        if not self.task:
            self.task = str(state.get("task") or "")
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._io_failed = True
        event = {
            "type": "run_start",
            "trace_id": self.trace_id,
            "mode": self.mode,
            "task": self.task,
            "workspace": str(self.workspace),
            "resumed": self.resumed,
            "attempts": int(state.get("attempts") or 0),
            "max_attempts": int(state.get("max_attempts") or 0),
            "todos": _count(state.get("todos")),
            "messages": _count(state.get("messages")),
            "at": self.started_at,
        }
        if self.resumed:
            event["resume_status"] = str(self.resume_event.get("status") or "")
            event["restored_files"] = int(self.resume_event.get("restored_files") or 0)
        self._append(event, summary=f"run started{' (resumed)' if self.resumed else ''}")
        return event

    def record_custom_event(self, event: Mapping[str, Any] | None) -> None:
        """Record one streamed event and update the counters it feeds."""
        if not self.enabled:
            return
        data = dict(event or {})
        kind = str(data.get("type") or "")
        if kind == "tool_call":
            self.tool_calls += 1
        elif kind == "tool_result":
            result = data.get("result")
            mapping = _as_mapping(result)
            ok = mapping.get("ok", data.get("ok", True))
            if ok is False:
                self.failed_tool_calls += 1
            if self._needs_approval(data, mapping):
                self.approval_count += 1
        elif kind == "handoff":
            self.handoff_count += 1
        elif kind == "checkpoint_saved":
            self.checkpoint_count += 1
        self._append(data)

    def record_graph_update(self, event: Mapping[str, Any] | None) -> None:
        """Record one graph node update and count the visit."""
        if not self.enabled:
            return
        data = dict(event or {})
        node = str(data.get("node") or "")
        if node:
            self.node_visits[node] = self.node_visits.get(node, 0) + 1
        self._append(data)

    def snapshot(self, *, status: str = "running", latest_node: str = "") -> dict[str, Any] | None:
        """Write ``trace.json`` and ``timeline.md`` with what is known *so far*.

        Called at every node boundary: a run that is killed, or that stalls inside a
        model call for minutes, must still leave a usable trace on disk instead of an
        empty directory.
        """
        if not self.enabled:
            return None
        if status:
            self.status = str(status)
        if latest_node:
            self.latest_node = str(latest_node)
        return self._write_trace()

    def end(
        self,
        *,
        status: str,
        latest_node: str,
        final_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Close the trace: record ``run_end``, then write the final files."""
        if not self.enabled:
            return None
        self.ended_at = _now()
        self.status = str(status)
        self.latest_node = str(latest_node or "")
        self.final_state = dict(_as_mapping(final_state))
        self._append(
            {
                "type": "run_end",
                "status": self.status,
                "latest_node": self.latest_node,
            },
            summary=f"run ended: {self.status} (latest node: {self.latest_node or '-'})",
        )
        return self._write_trace()

    def _write_trace(self) -> dict[str, Any]:
        """Render and persist the current state of the trace."""
        state = self.final_state
        timeline = list(self.timeline)
        head = timeline[:TIMELINE_HEAD]
        tail = timeline[-TIMELINE_TAIL:] if len(timeline) > TIMELINE_HEAD else []
        kept = {item["seq"] for item in head} | {item["seq"] for item in tail}
        payload: dict[str, Any] = {
            "version": 1,
            "trace_id": self.trace_id,
            "mode": self.mode,
            "task": self.task,
            "workspace": str(self.workspace),
            "status": self.status,
            "latest_node": self.latest_node,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self._duration_ms(),
            "resumed": self.resumed,
            "node_visits": dict(self.node_visits),
            "tool_calls": self.tool_calls,
            "failed_tool_calls": self.failed_tool_calls,
            "approval_count": self.approval_count,
            "checkpoint_count": self.checkpoint_count,
            "handoff_count": self.handoff_count,
            "timeline_total": len(timeline),
            "timeline_head": head,
            "timeline_tail": tail,
            "timeline_omitted": len(timeline) - len(kept),
            "passed": bool(state.get("passed")),
            "attempts": int(state.get("attempts") or 0),
            "final_answer": _short(state.get("final_answer"), 2_000),
            "last_error": _short(state.get("last_error"), 1_000),
            "events_file": "events.jsonl" if self.full else "",
            "timeline_file": "timeline.md",
            "complete": self.ended_at != "",
        }
        self._write_json(self.trace_path, payload)
        self._write_text(self.timeline_path, self._render_markdown(payload))
        return payload

    # -- internals -----------------------------------------------------------
    def _append(self, event: Mapping[str, Any], *, summary: str = "") -> None:
        """Add one timeline entry, and log the raw event in ``full`` mode."""
        if not self.enabled:
            return
        self._sequence += 1
        kind = str(event.get("type") or "event")
        entry = {
            "seq": self._sequence,
            "at": _clock(),
            "type": kind,
            "node": str(event.get("node") or ""),
            "detail": _short(summary or self._summarize(event)),
        }
        self.timeline.append(entry)
        if self.full:
            self._append_jsonl(self.events_path, {"seq": self._sequence, **dict(event)})

    @staticmethod
    def _needs_approval(event: Mapping[str, Any], result: Mapping[str, Any]) -> bool:
        """True when a tool result came from a command that needed approval.

        The flag is carried by the rendered ``bash`` output (``requires_approval:
        True``); a structured flag on the result wins when a tool sets one.
        """
        for source in (result, event):
            if source.get("requires_approval") is True:
                return True
            if source.get("approval_note"):
                return True
        text = str(result.get("result") or "")
        return "requires_approval: True" in text

    @staticmethod
    def _summarize(event: Mapping[str, Any]) -> str:
        """One readable line for a streamed event."""
        kind = str(event.get("type") or "event")
        if kind == "tool_call":
            args = _as_mapping(event.get("args"))
            preview = ", ".join(f"{key}={_short(value, 60)}" for key, value in list(args.items())[:3])
            return f"{event.get('name')}({preview})"
        if kind == "tool_result":
            result = _as_mapping(event.get("result"))
            state = "ok" if result.get("ok", True) else "failed"
            detail = str(result.get("error") or result.get("result") or "")
            return f"{event.get('name')} {state}: {_short(detail, 120)}"
        if kind == "handoff":
            return (
                f"{event.get('from')} -> {event.get('to')}: "
                f"{_short(event.get('instruction'), 120)}"
            )
        if kind == "node_end":
            update = _as_mapping(event.get("update"))
            keys = ", ".join(list(update)[:6])
            return f"{event.get('node')} update: {keys or '(empty)'}"
        if kind == "memory":
            return f"memory[{event.get('node')}]: {event.get('digest')}"
        if kind == "context_compressed":
            written = []
            for record, name in (
                (_as_mapping(event.get("persisted")), event.get("history_path")),
                (_as_mapping(event.get("notepad")), event.get("notepad_path")),
            ):
                if record.get("ok") and name:
                    written.append(str(name))
            return (
                f"compressed {event.get('messages_before')} msgs/{event.get('tokens_before')} tokens"
                f" -> {event.get('messages_after')} msg/{event.get('tokens_after')} tokens"
                + (f" [{', '.join(written)}]" if written else "")
            )
        if kind == "checkpoint_saved":
            return (
                f"checkpoint {event.get('status')}/{event.get('latest_node')} "
                f"-> {event.get('files')} file(s)"
                + ("" if event.get("git_ok") else " (no git snapshot)")
            )
        if kind == "final_answer":
            return (
                f"final: passed={event.get('passed')} attempts={event.get('attempts')} "
                f"stopped_early={event.get('stopped_early')}"
            )
        if kind == "ai_message":
            return f"ai: {_short(event.get('content'), 120)}"
        if kind == "search_results":
            return f"{len(event.get('results') or [])} result(s) for {_short(event.get('query'), 60)!r}"
        if kind == "todo_update":
            return f"todos: {event.get('todos')}"
        return kind

    def _duration_ms(self) -> int:
        try:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(self.ended_at)
        except ValueError:
            return 0
        return max(0, int((end - start).total_seconds() * 1000))

    def _render_markdown(self, payload: Mapping[str, Any]) -> str:
        """The human-readable timeline (``timeline.md``)."""
        visits = payload.get("node_visits")
        visits = visits if isinstance(visits, Mapping) else {}
        node_line = ", ".join(f"{name} x{count}" for name, count in visits.items()) or "-"
        seconds = int(payload.get("duration_ms") or 0) / 1000
        lines = [
            f"# Trace {payload.get('trace_id')}",
            "",
            f"**Task:** {payload.get('task') or '(no task recorded)'}",
            f"**Status:** {payload.get('status')} (latest node: {payload.get('latest_node') or '-'})",
            f"**Started:** {payload.get('started_at')} | **Ended:** {payload.get('ended_at')} "
            f"| **Duration:** {seconds:.2f}s",
            f"**Nodes:** {node_line}",
            f"**Counters:** {payload.get('tool_calls', 0)} tool calls "
            f"({payload.get('failed_tool_calls', 0)} failed), "
            f"{payload.get('approval_count', 0)} approval(s), "
            f"{payload.get('checkpoint_count', 0)} checkpoint(s), "
            f"{payload.get('handoff_count', 0)} handoff(s)",
            "",
            "## Timeline",
            "",
            "| # | time | event | detail |",
            "| --- | --- | --- | --- |",
        ]
        head = payload.get("timeline_head")
        head = head if isinstance(head, Sequence) else []
        tail = payload.get("timeline_tail")
        tail = tail if isinstance(tail, Sequence) else []
        for entry in list(head) + list(tail):
            if not isinstance(entry, Mapping):
                continue
            lines.append(
                f"| {entry.get('seq')} | {entry.get('at')} | {entry.get('type')} "
                f"| {str(entry.get('detail') or '').replace('|', '/')} |"
            )
        omitted = int(payload.get("timeline_omitted") or 0)
        if omitted:
            lines.append(f"| | | | _... {omitted} event(s) omitted ..._ |")
        lines += [
            "",
            f"_Full event log: {'events.jsonl' if payload.get('events_file') else 'not recorded in this mode'}_",
            "",
        ]
        return "\n".join(lines)

    def _write_json(self, path: Path, payload: Any) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError):
            self._io_failed = True
            return False
        return True

    def _write_text(self, path: Path, text: str) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except OSError:
            self._io_failed = True
            return False
        return True

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n"
                )
        except (OSError, TypeError, ValueError):
            self._io_failed = True
            return False
        return True
