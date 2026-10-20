"""Unified diffs for file-changing tool calls.

The TUI shows what a ``file_write`` / ``file_edit`` actually changed. The tools
themselves only return a one-line summary (that is what the model needs), so the
diff is captured around the call instead:

    snapshot = capture(tool, args)      # read the file before the tool runs
    result = execute_tool(...)
    change = finish(snapshot)           # read it again and diff the two

Both reads are best-effort: a missing tool, an unreadable file or a binary blob
simply yields ``None``, so a diff can never break a turn.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "FILE_TOOLS",
    "FileChange",
    "Snapshot",
    "capture",
    "finish",
    "unified_diff",
]

#: Tools whose effect on the workspace is worth a diff.
FILE_TOOLS = frozenset({"file_write", "file_edit"})

#: Never read a file larger than this when snapshotting (bytes).
MAX_FILE_BYTES = 2 * 1024 * 1024
#: Lines kept in one rendered diff; the rest is summarised.
MAX_DIFF_LINES = 400
#: Context lines around each hunk.
CONTEXT_LINES = 3


@dataclass(slots=True)
class Snapshot:
    """What a file held just before a tool call touched it."""

    path: Path
    display: str
    text: str | None  # None means "did not exist"

    @property
    def existed(self) -> bool:
        return self.text is not None


@dataclass(slots=True)
class FileChange:
    """One file's before/after, as the UI needs it."""

    path: str
    added: int
    removed: int
    text: str
    truncated: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "added": self.added,
            "removed": self.removed,
            "text": self.text,
            "truncated": self.truncated,
        }


def _read(path: Path) -> str | None:
    """Read a UTF-8 text file, or ``None`` when it is absent, binary or too big."""
    try:
        if not path.is_file():
            return None
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data[:8192]:  # binary: never diff it
        return None
    return data.decode("utf-8", errors="replace")


def _state_of(tool: Any) -> Any:
    """The ``RuntimeState`` behind a tool, looking through LangChain's wrapper.

    ``build_tools()`` hands out ``StructuredTool`` objects, whose ``state`` lives on
    the bound method they wrap (``.func.__self__``); the plain tool objects keep it
    directly.
    """
    state = getattr(tool, "state", None)
    if state is not None:
        return state
    func = getattr(tool, "func", None)
    owner = getattr(func, "__self__", None)
    return getattr(owner, "state", None)


def capture(tool: Any, args: Mapping[str, Any]) -> Snapshot | None:
    """Snapshot the file *tool* is about to change; ``None`` for other tools."""
    if getattr(tool, "name", "") not in FILE_TOOLS:
        return None
    state = _state_of(tool)
    target = str(args.get("file_path") or "")
    if state is None or not target:
        return None
    try:
        path = state.resolve(target, kind="file")
        display = state.relative(path)
    except Exception:  # noqa: BLE001 - a diff must never break the call
        return None
    return Snapshot(path=path, display=display, text=_read(path))


def finish(snapshot: Snapshot | None) -> FileChange | None:
    """Diff the snapshot's file against what it holds now; ``None`` if unchanged."""
    if snapshot is None:
        return None
    after = _read(snapshot.path)
    before = snapshot.text
    if before == after:
        return None
    change = unified_diff(
        before if before is not None else "",
        after if after is not None else "",
        snapshot.display,
        existed=snapshot.existed and after is not None,
    )
    return change


def unified_diff(before: str, after: str, display: str, *, existed: bool = True) -> FileChange | None:
    """A unified diff of two texts, truncated to :data:`MAX_DIFF_LINES`."""
    if before == after:
        return None
    lines = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"a/{display}" if existed else "/dev/null",
            tofile=f"b/{display}",
            lineterm="",
            n=CONTEXT_LINES,
        )
    )
    added = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
    truncated = len(lines) > MAX_DIFF_LINES
    if truncated:
        kept = lines[:MAX_DIFF_LINES]
        kept.append(f"… ({len(lines) - MAX_DIFF_LINES} more diff lines)")
        lines = kept
    return FileChange(
        path=display,
        added=added,
        removed=removed,
        text="\n".join(lines),
        truncated=truncated,
    )
