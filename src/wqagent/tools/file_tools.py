"""File tools - read, write and targeted edits, always inside the workspace.

Each tool is a plain Python object with a :meth:`run` method (easy to unit
test) plus :meth:`as_tool`, which wraps it in a LangChain ``StructuredTool``
for ``model.bind_tools()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from wqagent.core.paths import WorkspacePathError
from wqagent.core.state import DEFAULT_READ_LINES, RuntimeState

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.tools import StructuredTool

__all__ = [
    "FileEditInput",
    "FileEditTool",
    "FileReadInput",
    "FileReadTool",
    "FileWriteInput",
    "FileWriteTool",
]

MAX_READ_LINES = 20_000
MAX_LINE_CHARS = 500


class FileReadInput(BaseModel):
    """Arguments accepted by :class:`FileReadTool`."""

    file_path: str = Field(
        description="File to read: absolute, or relative to the workspace root."
    )
    offset: int = Field(
        default=1, ge=1, description="1-based line number to start reading from."
    )
    limit: int = Field(
        default=DEFAULT_READ_LINES,
        ge=1,
        le=MAX_READ_LINES,
        description="Maximum number of lines to return.",
    )


class FileWriteInput(BaseModel):
    """Arguments accepted by :class:`FileWriteTool`."""

    file_path: str = Field(
        description="File to create or overwrite; relative to the workspace root."
    )
    content: str = Field(
        description="Full new content of the file. Any existing content is replaced."
    )


class FileEditInput(BaseModel):
    """Arguments accepted by :class:`FileEditTool`."""

    file_path: str = Field(description="File to edit; relative to the workspace root.")
    old_text: str = Field(
        description="Exact text to replace. Must match exactly once in the file."
    )
    new_text: str = Field(
        description="Replacement text; may be empty to delete the matched snippet."
    )


# -- helpers -----------------------------------------------------------------
def _read_text(path: Path) -> str:
    """Read a file as UTF-8 without newline translation."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _write_text(path: Path, text: str) -> int:
    """Write *text* verbatim (no newline translation); return the byte count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    return len(text.encode("utf-8"))


def _text_line_count(text: str) -> int:
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _file_line_count(path: Path) -> int:
    """Count lines by streaming the file in chunks."""
    newlines = 0
    last_byte = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            newlines += chunk.count(b"\n")
            last_byte = chunk[-1:]
    if last_byte and last_byte != b"\n":
        newlines += 1
    return newlines


def _match_lines(text: str, needle: str, limit: int = 10) -> list[int]:
    """1-based line numbers of the first *limit* non-overlapping matches."""
    lines: list[int] = []
    start = 0
    while len(lines) < limit:
        index = text.find(needle, start)
        if index < 0:
            break
        lines.append(text.count("\n", 0, index) + 1)
        start = index + max(len(needle), 1)
    return lines


def _with_crlf(text: str) -> str:
    """Rewrite lone LF as CRLF (used when a CRLF file is edited with LF text)."""
    return text.replace("\r\n", "\n").replace("\n", "\r\n")


# -- tools -------------------------------------------------------------------
class FileReadTool:
    """Read a UTF-8 text file, optionally a slice of lines, from the workspace."""

    name: str = "file_read"
    description: str = (
        "Read a UTF-8 text file from the workspace. Returns the lines with 1-based line "
        "numbers and reports how to continue when the file is longer than `limit`. "
        "Binary files are rejected."
    )
    args_schema: type[BaseModel] = FileReadInput

    def __init__(self, state: RuntimeState) -> None:
        self.state = state

    def as_tool(self) -> "StructuredTool":
        """Wrap this implementation in a LangChain ``StructuredTool``."""
        from langchain_core.tools import StructuredTool

        return StructuredTool.from_function(
            func=self.run,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
        )

    def run(
        self, file_path: str, offset: int = 1, limit: int = DEFAULT_READ_LINES
    ) -> str:
        """Return the requested lines of *file_path*, or an ``Error:`` message."""
        state = self.state
        try:
            path = state.resolve(file_path, must_exist=True, kind="file")
        except (WorkspacePathError, OSError) as exc:
            return f"Error: {exc}"
        display = state.relative(path)
        offset = max(1, int(offset))
        limit = max(1, int(limit))

        try:
            with path.open("rb") as handle:
                if b"\x00" in handle.read(8192):
                    return f"Error: {display} looks like a binary file; only UTF-8 text is supported."
            lines: list[str] = []
            more_lines = False
            with path.open(
                "r", encoding="utf-8", errors="replace", newline=""
            ) as handle:
                for number, raw in enumerate(handle, start=1):
                    if number < offset:
                        continue
                    if len(lines) >= limit:
                        more_lines = True
                        break
                    text = raw.rstrip("\n").rstrip("\r")
                    if len(text) > MAX_LINE_CHARS:
                        text = f"{text[:MAX_LINE_CHARS]} ...[line truncated]"
                    lines.append(text)
        except OSError as exc:
            return f"Error: cannot read {display}: {exc}"

        if not lines:
            total = _file_line_count(path)
            if total == 0:
                return f"{display}: file is empty (0 lines)."
            if offset > total:
                return (
                    f"Error: offset {offset} is past the end of {display} "
                    f"({total} lines). Use offset=1 to start over."
                )

        first = offset
        last = offset + len(lines) - 1
        header = f"{display}: showing lines {first}-{last}"
        if more_lines:
            header += f" (more lines follow; continue with offset={last + 1})"
        body = "\n".join(
            f"{number:>6}\t{text}"
            for number, text in zip(range(first, last + 1), lines)
        )
        return state.clip(f"{header}\n{body}", label=display)


class FileWriteTool:
    """Create or overwrite a file inside the workspace."""

    name: str = "file_write"
    description: str = (
        "Create a file or overwrite it completely. Missing parent directories are "
        "created. Content is written as UTF-8 without newline translation."
    )
    args_schema: type[BaseModel] = FileWriteInput

    def __init__(self, state: RuntimeState) -> None:
        self.state = state

    def as_tool(self) -> "StructuredTool":
        """Wrap this implementation in a LangChain ``StructuredTool``."""
        from langchain_core.tools import StructuredTool

        return StructuredTool.from_function(
            func=self.run,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
        )

    def run(self, file_path: str, content: str) -> str:
        """Write *content* to *file_path*, creating parents as needed."""
        state = self.state
        try:
            path = state.resolve(file_path, kind="file")
        except (WorkspacePathError, OSError) as exc:
            return f"Error: {exc}"
        if path.is_dir():
            return f"Error: {state.relative(path)} is a directory."
        existed = path.exists()
        try:
            written = _write_text(path, content)
        except UnicodeEncodeError as exc:
            return f"Error: content is not valid UTF-8 text ({exc})."
        except OSError as exc:
            return f"Error: cannot write {state.relative(path)}: {exc}"
        verb = "Overwrote" if existed else "Created"
        return (
            f"{verb} {state.relative(path)} "
            f"({written} bytes, {_text_line_count(content)} lines)."
        )


class FileEditTool:
    """Replace one uniquely matching snippet in an existing file."""

    name: str = "file_edit"
    description: str = (
        "Replace an exact snippet in a file. `old_text` must occur exactly once: zero "
        "matches or multiple matches are errors, so include enough surrounding context "
        "to make the match unique. Use file_write to create a file or replace it whole."
    )
    args_schema: type[BaseModel] = FileEditInput

    def __init__(self, state: RuntimeState) -> None:
        self.state = state

    def as_tool(self) -> "StructuredTool":
        """Wrap this implementation in a LangChain ``StructuredTool``."""
        from langchain_core.tools import StructuredTool

        return StructuredTool.from_function(
            func=self.run,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
        )

    def run(self, file_path: str, old_text: str, new_text: str) -> str:
        """Replace the single occurrence of *old_text* with *new_text*."""
        state = self.state
        try:
            path = state.resolve(file_path, must_exist=True, kind="file")
        except (WorkspacePathError, OSError) as exc:
            return f"Error: {exc}"
        display = state.relative(path)
        if not old_text:
            return (
                "Error: old_text must not be empty; "
                "use file_write to create a file or replace it whole."
            )
        if old_text == new_text:
            return "Error: old_text and new_text are identical; nothing to change."
        try:
            text = _read_text(path)
        except UnicodeDecodeError:
            return f"Error: {display} is not valid UTF-8 text; the file tools only handle UTF-8."
        except OSError as exc:
            return f"Error: cannot read {display}: {exc}"

        note = ""
        matches = text.count(old_text)
        if matches == 0 and "\r\n" in text and "\r\n" not in old_text:
            # The model sent LF text for a CRLF file: retry with the file's convention.
            crlf_old, crlf_new = _with_crlf(old_text), _with_crlf(new_text)
            if text.count(crlf_old) >= 1:
                old_text, new_text = crlf_old, crlf_new
                matches = text.count(old_text)
                note = " (matched CRLF line endings)"

        if matches == 0:
            return (
                f"Error: old_text was not found in {display}. Read the file and copy the "
                "snippet exactly, including indentation and line breaks."
            )
        if matches > 1:
            lines = _match_lines(text, old_text)
            shown = ", ".join(str(number) for number in lines)
            if matches > len(lines):
                shown += ", ..."
            return (
                f"Error: old_text matched {matches} times in {display} (lines {shown}). "
                "Include more surrounding context so that it matches exactly once."
            )

        index = text.find(old_text)
        line_number = text.count("\n", 0, index) + 1
        updated = text[:index] + new_text + text[index + len(old_text) :]
        try:
            written = _write_text(path, updated)
        except OSError as exc:
            return f"Error: cannot write {display}: {exc}"
        before = len(text.encode("utf-8"))
        return (
            f"Edited {display}{note}: replaced 1 occurrence at line {line_number} "
            f"({before} -> {written} bytes)."
        )
