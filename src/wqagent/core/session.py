"""Conversation sessions: what the user asked before, and what came of it.

A session lives inside the workspace::

    <workspace>/.wqagent/session/
        session.json          the machine-readable record
        SESSION_SUMMARY.md    the same story for humans

One *turn* is one entry in ``recent_turns`` (a user message, then the assistant's
reply), and ``turn_index`` is the highest number handed out so far::

    {"session_id": "…", "turn_index": 3,
     "recent_turns": [
        {"turn": 1, "role": "user", "content": "…", "timestamp": "…"},
        {"turn": 2, "role": "assistant", "route": "workflow", "content": "…",
         "summary": "…", "timestamp": "…"},
        {"turn": 3, "role": "user", "content": "…", "timestamp": "…"}],
     "created_at": "…", "updated_at": "…"}

:func:`build_session_context` turns that into the short block the entry graph gets
(session id, the newest workspace files, the last few exchanges), bounded by
:data:`MAX_SESSION_CONTEXT` so it can never crowd out the actual prompt.

Nothing here raises: a corrupt file starts a fresh session, an unwritable directory
only sets ``last_error``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

__all__ = [
    "CONTEXT_FILES",
    "CONTEXT_TURNS",
    "MAX_RECENT_TURNS",
    "MAX_SESSION_CONTEXT",
    "MAX_TURN_CONTENT",
    "SESSION_FILE",
    "SESSION_ROOT",
    "SESSION_SUMMARY_FILE",
    "append_assistant_turn",
    "append_user_turn",
    "build_session_context",
    "load_or_create_session",
    "new_session",
    "record_assistant_turn",
    "render_session_summary",
    "save_session",
    "session_dir",
    "session_path",
    "summary_path",
    "workspace_files",
]

#: Directory (inside the workspace) that holds the session files.
SESSION_ROOT = ".wqagent/session"
#: The machine-readable session record.
SESSION_FILE = "session.json"
#: The human-readable session story.
SESSION_SUMMARY_FILE = "SESSION_SUMMARY.md"
#: Characters the session context block may use, in total.
MAX_SESSION_CONTEXT = 7000
#: Characters kept of any single turn's content.
MAX_TURN_CONTENT = 4000
#: Turns kept on disk (newest win); the context only ever shows :data:`CONTEXT_TURNS`.
MAX_RECENT_TURNS = 40
#: Turns shown in the context block.
CONTEXT_TURNS = 10
#: Workspace files listed in the context block (newest first).
CONTEXT_FILES = 30
#: Characters kept of a turn summary.
MAX_SUMMARY_CHARS = 400


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clip(text: Any, limit: int) -> str:
    value = text if isinstance(text, str) else ("" if text is None else str(text))
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def session_dir(workspace: Path | str | Any) -> Path:
    """``<workspace>/.wqagent/session`` (accepts a Path or a RuntimeState)."""
    root = getattr(workspace, "workspace", workspace)
    return Path(root) / SESSION_ROOT


def session_path(workspace: Path | str | Any) -> Path:
    """Path of ``session.json`` inside *workspace*."""
    return session_dir(workspace) / SESSION_FILE


def summary_path(workspace: Path | str | Any) -> Path:
    """Path of ``SESSION_SUMMARY.md`` inside *workspace*."""
    return session_dir(workspace) / SESSION_SUMMARY_FILE


def new_session() -> dict[str, Any]:
    """A fresh session record (nothing is written yet)."""
    stamp = _now()
    return {
        "session_id": str(uuid4()),
        "turn_index": 0,
        "recent_turns": [],
        "created_at": stamp,
        "updated_at": stamp,
    }


def _normalise(payload: Mapping[str, Any]) -> dict[str, Any]:
    """A usable session from whatever was on disk."""
    session = new_session()
    session_id = str(payload.get("session_id") or "").strip()
    if session_id:
        session["session_id"] = session_id
    for key in ("created_at", "updated_at"):
        text = str(payload.get(key) or "").strip()
        if text:
            session[key] = text
    turns: list[dict[str, Any]] = []
    raw_turns = payload.get("recent_turns")
    if isinstance(raw_turns, list):
        for item in raw_turns:
            if not isinstance(item, Mapping):
                continue
            entry: dict[str, Any] = {
                "turn": _as_turn(item.get("turn"), len(turns) + 1),
                "role": str(item.get("role") or "user"),
                "content": _clip(item.get("content"), MAX_TURN_CONTENT),
                "timestamp": str(item.get("timestamp") or ""),
            }
            if entry["role"] == "assistant":
                entry["route"] = str(item.get("route") or "workflow")
                entry["summary"] = _clip(item.get("summary"), MAX_SUMMARY_CHARS)
            turns.append(entry)
    session["recent_turns"] = turns[-MAX_RECENT_TURNS:]
    index = _as_turn(payload.get("turn_index"), 0)
    session["turn_index"] = max(index, max((turn["turn"] for turn in turns), default=0))
    return session


def _as_turn(value: Any, default: int) -> int:
    """A non-negative turn number from anything."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default


def load_or_create_session(workspace: Path | str | Any) -> dict[str, Any]:
    """Load ``session.json`` from *workspace*, or start a new session.

    A missing, unreadable or corrupt file yields a fresh session - this is called on
    every run, so it must never raise.
    """
    path = session_path(workspace)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return new_session()
    if not isinstance(payload, Mapping):
        return new_session()
    return _normalise(payload)


def append_user_turn(session: dict[str, Any], content: str) -> int:
    """Record what the user asked; returns the turn number it got."""
    turn = _as_turn(session.get("turn_index"), 0) + 1
    session["turn_index"] = turn
    session.setdefault("recent_turns", []).append(
        {
            "turn": turn,
            "role": "user",
            "content": _clip(content, MAX_TURN_CONTENT),
            "timestamp": _now(),
        }
    )
    _trim(session)
    return turn


def append_assistant_turn(
    session: dict[str, Any],
    *,
    turn: int,
    route: str,
    content: str,
    summary: str = "",
) -> None:
    """Record the assistant's reply to *turn* (``route`` is ``chat`` or ``workflow``).

    The reply gets its own turn number: the one after *turn* unless that number is
    already taken, in which case *turn* is honoured as given.
    """
    turns = session.setdefault("recent_turns", [])
    highest = max((_as_turn(item.get("turn"), 0) for item in turns), default=0)
    requested = _as_turn(turn, highest)
    entry_turn = requested if requested > highest else highest + 1
    turns.append(
        {
            "turn": entry_turn,
            "role": "assistant",
            "route": str(route or "workflow"),
            "content": _clip(content, MAX_TURN_CONTENT),
            "summary": _clip(summary, MAX_SUMMARY_CHARS),
            "timestamp": _now(),
        }
    )
    session["turn_index"] = max(_as_turn(session.get("turn_index"), 0), entry_turn)
    _trim(session)


def _trim(session: dict[str, Any]) -> None:
    """Keep the newest :data:`MAX_RECENT_TURNS` turns."""
    turns = session.get("recent_turns")
    if isinstance(turns, list) and len(turns) > MAX_RECENT_TURNS:
        session["recent_turns"] = turns[-MAX_RECENT_TURNS:]


def save_session(workspace: Path | str | Any, session: dict[str, Any]) -> dict[str, Any]:
    """Write ``session.json`` and ``SESSION_SUMMARY.md``; returns the session.

    Failures are recorded in ``last_error`` instead of raising: a session is
    bookkeeping, and losing it must never break a run.
    """
    session["updated_at"] = _now()
    session.pop("last_error", None)
    directory = session_dir(workspace)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        session_path(workspace).write_text(
            json.dumps(session, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        summary_path(workspace).write_text(
            render_session_summary(session), encoding="utf-8"
        )
    except (OSError, TypeError, ValueError) as exc:
        session["last_error"] = f"{type(exc).__name__}: {exc}"
    return session


def record_assistant_turn(
    workspace: Path | str | Any,
    session: dict[str, Any],
    *,
    turn: int,
    route: str,
    content: str,
    summary: str = "",
) -> dict[str, Any]:
    """Append the reply to *turn* and persist the session (append + save in one step).

    Returns the session, so callers can keep using the in-memory copy.
    """
    append_assistant_turn(
        session, turn=turn, route=route, content=content, summary=summary
    )
    return save_session(workspace, session)


def render_session_summary(session: Mapping[str, Any]) -> str:
    """The human-readable ``SESSION_SUMMARY.md``."""
    turns = session.get("recent_turns")
    turns = turns if isinstance(turns, list) else []
    lines = [
        "# Session",
        "",
        f"- session: `{session.get('session_id')}`",
        f"- turns recorded: {session.get('turn_index', 0)} ({len(turns)} kept)",
        f"- created: {session.get('created_at') or 'unknown'}",
        f"- updated: {session.get('updated_at') or 'unknown'}",
        "",
        "## Turns",
        "",
    ]
    if not turns:
        lines.append("(nothing recorded yet)")
    for item in turns:
        if not isinstance(item, Mapping):
            continue
        role = str(item.get("role") or "user")
        head = f"### {item.get('turn')}. {role}"
        if role == "assistant":
            head += f" ({item.get('route') or 'workflow'})"
        lines.append(head)
        lines.append("")
        content = str(item.get("content") or "").strip()
        if content:
            lines.append(content)
            lines.append("")
        summary = str(item.get("summary") or "").strip()
        if summary and role == "assistant":
            lines.append(f"_summary: {summary}_")
            lines.append("")
    if session.get("last_error"):
        lines += [f"_last save failed: {session['last_error']}_", ""]
    return "\n".join(lines)


def workspace_files(workspace: Path | str | Any, limit: int = CONTEXT_FILES) -> list[str]:
    """The newest *limit* files in the workspace, as relative POSIX paths.

    The agent's own ``.wqagent`` directory is skipped: it is bookkeeping, not work.
    """
    root = Path(getattr(workspace, "workspace", workspace))
    found: list[tuple[float, str]] = []
    try:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if relative == ".wqagent" or relative.startswith(".wqagent/"):
                continue
            try:
                found.append((path.stat().st_mtime, relative))
            except OSError:
                continue
    except OSError:
        return []
    found.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in found[: max(0, limit)]]


def build_session_context(
    workspace: Path | str | Any, session: Mapping[str, Any] | None = None
) -> str:
    """The session block for the intent router and the chat node.

    Contains the session id and turn number, the newest workspace files and the last
    few exchanges (user text plus, for assistant turns, the route and summary). The
    result never exceeds :data:`MAX_SESSION_CONTEXT` characters: old turns are dropped
    first, then the file list shrinks, and only then is the head of the block cut.
    """
    if session is None:
        session = load_or_create_session(workspace)
    turns = session.get("recent_turns")
    turns = [item for item in turns if isinstance(item, Mapping)] if isinstance(turns, list) else []
    files = workspace_files(workspace)
    head = [
        f"session: {session.get('session_id') or 'unknown'}",
        f"turn: {session.get('turn_index', 0)}",
    ]

    def render(chosen_turns: list[Mapping[str, Any]], chosen_files: list[str]) -> str:
        blocks = ["\n".join(head)]
        if chosen_files:
            blocks.append(
                "workspace files (newest first):\n"
                + "\n".join(f"- {name}" for name in chosen_files)
            )
        if chosen_turns:
            blocks.append(
                "recent turns (oldest first):\n"
                + "\n".join(_render_turn(turn) for turn in chosen_turns)
            )
        return "\n\n".join(blocks)

    selected = turns[-CONTEXT_TURNS:]
    text = render(selected, files)
    while len(text) > MAX_SESSION_CONTEXT and selected:
        selected = selected[1:]  # drop the oldest turn first
        text = render(selected, files)
    while len(text) > MAX_SESSION_CONTEXT and files:
        files = files[: max(0, len(files) // 2)]  # then shrink the file list
        text = render(selected, files)
    if len(text) > MAX_SESSION_CONTEXT:
        marker = "...[earlier session context omitted]...\n"
        keep = max(0, MAX_SESSION_CONTEXT - len(marker))
        text = marker + text[-keep:]
    return text


def _render_turn(turn: Mapping[str, Any]) -> str:
    """One line per turn: user text, or the assistant's route and summary."""
    role = str(turn.get("role") or "user")
    label = f"turn {turn.get('turn')} [{role}]"
    if role == "assistant":
        route = str(turn.get("route") or "workflow")
        summary = str(turn.get("summary") or "").strip()
        content = str(turn.get("content") or "").strip()
        detail = summary or _clip(content, MAX_SUMMARY_CHARS)
        return f"{label} ({route}): {detail}" if detail else f"{label} ({route})"
    return f"{label}: {str(turn.get('content') or '').strip()}"
