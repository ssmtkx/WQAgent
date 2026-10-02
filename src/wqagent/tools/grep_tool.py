"""Regular-expression content search across the workspace.

Uses ripgrep when it is installed (fast, gitignore-aware) and falls back to a
built-in scanner otherwise. Set ``WQ_DISABLE_RIPGREP=1`` to force the built-in
scanner, which is also what the tests exercise.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from pydantic import BaseModel, Field

from wqagent.core.paths import WorkspacePathError
from wqagent.core.state import RuntimeState

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.tools import StructuredTool

__all__ = ["GrepInput", "GrepTool", "SKIPPED_DIRECTORIES"]

DEFAULT_HEAD_LIMIT = 100
MAX_HEAD_LIMIT = 1_000
MAX_LINE_CHARS = 500

#: Directories the built-in scanner never descends into.
SKIPPED_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".eggs",
        "build",
        "dist",
        "site-packages",
    }
)


class GrepInput(BaseModel):
    """Arguments accepted by :class:`GrepTool`."""

    pattern: str = Field(
        description="Python regular expression to search for, e.g. 'def main\\('."
    )
    path: str = Field(
        default=".",
        description="File or directory to search, relative to the workspace root.",
    )
    glob: str | None = Field(
        default=None,
        description="Filename filter, e.g. '*.py' or 'src/**/*.py'.",
    )
    head_limit: int = Field(
        default=DEFAULT_HEAD_LIMIT,
        ge=1,
        le=MAX_HEAD_LIMIT,
        description="Maximum number of matching lines to return.",
    )
    ignore_case: bool = Field(default=False, description="Match case-insensitively.")


def _glob_match(relative: str, name: str, pattern: str | None) -> bool:
    """Match a workspace-relative path (or bare filename) against *pattern*."""
    if not pattern:
        return True
    cleaned = pattern.replace("\\", "/").strip()
    if not cleaned:
        return True
    if cleaned.startswith("**/"):
        cleaned = cleaned[3:]
    if "/" in cleaned:
        return fnmatch.fnmatchcase(relative, cleaned)
    return fnmatch.fnmatchcase(name, cleaned)


class GrepTool:
    """Search file contents inside the workspace with a regular expression."""

    name: str = "grep"
    description: str = (
        "Search file contents inside the workspace with a regular expression and return "
        "'path:line: text' matches, truncated after `head_limit` lines. Binary files, "
        "hidden entries and common build directories are skipped. Searches one file when "
        "`path` points at a file, otherwise the whole subtree."
    )
    args_schema: type[BaseModel] = GrepInput

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
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        head_limit: int = DEFAULT_HEAD_LIMIT,
        ignore_case: bool = False,
    ) -> str:
        """Return matching lines (``path:line: text``) or an ``Error:`` message."""
        state = self.state
        if not pattern:
            return "Error: pattern must not be empty."
        try:
            target = state.resolve(path, must_exist=True)
        except (WorkspacePathError, OSError) as exc:
            return f"Error: {exc}"
        try:
            regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        except re.error as exc:
            return f"Error: invalid regular expression {pattern!r}: {exc}"

        head_limit = max(1, min(int(head_limit), MAX_HEAD_LIMIT))
        scope = state.relative(target)
        notes: list[str] = []
        matches: list[str] | None = None

        ripgrep = None if os.environ.get("WQ_DISABLE_RIPGREP") else shutil.which("rg")
        if ripgrep:
            try:
                matches, note = self._search_ripgrep(
                    ripgrep, target, scope, pattern, glob, head_limit, ignore_case
                )
                if note:
                    notes.append(note)
            except (OSError, subprocess.SubprocessError) as exc:
                notes.append(f"ripgrep failed ({exc}); used the built-in scanner")
                matches = None
        if matches is None:
            matches, builtin_notes = self._search_builtin(
                regex, target, glob, head_limit
            )
            notes.extend(builtin_notes)

        if not matches:
            suffix = f" ({'; '.join(notes)})" if notes else ""
            return f"No matches for {pattern!r} in {scope}{suffix}."
        header = (
            f"{len(matches)} match{'es' if len(matches) != 1 else ''} "
            f"for {pattern!r} in {scope}"
        )
        if len(matches) >= head_limit:
            header += f" (stopped at head_limit={head_limit})"
        body = "\n".join(matches)
        if notes:
            body += "\n" + "\n".join(f"[{note}]" for note in notes)
        return state.clip(f"{header}\n{body}", label="grep output")

    # -- backends ------------------------------------------------------------
    def _search_ripgrep(
        self,
        ripgrep: str,
        target: Path,
        scope: str,
        pattern: str,
        glob: str | None,
        head_limit: int,
        ignore_case: bool,
    ) -> tuple[list[str], str]:
        state = self.state
        argv = [
            ripgrep,
            "--line-number",
            "--no-heading",
            "--color=never",
            "--with-filename",
            "--max-columns",
            str(MAX_LINE_CHARS),
            "--max-columns-preview",
        ]
        if ignore_case:
            argv.append("--ignore-case")
        if glob:
            argv.extend(["--glob", glob])
        argv.extend(["--regexp", pattern, "--", scope])

        process = subprocess.Popen(
            argv,
            cwd=str(state.workspace),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        matches: list[str] = []
        truncated = False
        stderr = ""
        try:
            assert process.stdout is not None  # noqa: S101 - guaranteed by PIPE
            for raw_line in process.stdout:
                line = raw_line.rstrip("\r\n")
                if not line.strip():
                    continue
                matches.append(line)
                if len(matches) >= head_limit:
                    truncated = True
                    process.terminate()
                    break
            try:
                _, stderr = process.communicate(timeout=state.timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                _, stderr = process.communicate()
        finally:
            if process.poll() is None:
                process.kill()
        if not matches and process.returncode not in (0, 1):
            raise subprocess.SubprocessError(
                (stderr or "").strip() or f"ripgrep exit code {process.returncode}"
            )
        note = f"stopped at head_limit={head_limit}" if truncated else ""
        return matches, note

    def _search_builtin(
        self,
        regex: re.Pattern[str],
        target: Path,
        glob: str | None,
        head_limit: int,
    ) -> tuple[list[str], list[str]]:
        state = self.state
        matches: list[str] = []
        skipped_binary = 0
        skipped_large = 0
        skipped_unreadable = 0
        truncated = False

        for path in self._iter_files(target, glob):
            try:
                if path.stat().st_size > state.max_scan_bytes:
                    skipped_large += 1
                    continue
                with path.open("rb") as handle:
                    if b"\x00" in handle.read(8192):
                        skipped_binary += 1
                        continue
                relative = state.relative(path)
                with path.open(
                    "r", encoding="utf-8", errors="replace", newline=""
                ) as handle:
                    for number, raw in enumerate(handle, start=1):
                        text = raw.rstrip("\r\n")
                        if not regex.search(text):
                            continue
                        if len(text) > MAX_LINE_CHARS:
                            text = f"{text[:MAX_LINE_CHARS]}..."
                        matches.append(f"{relative}:{number}: {text}")
                        if len(matches) >= head_limit:
                            truncated = True
                            break
            except OSError:
                skipped_unreadable += 1
                continue
            if truncated:
                break

        notes: list[str] = []
        if truncated:
            notes.append(f"stopped at head_limit={head_limit}")
        if skipped_binary:
            notes.append(f"skipped {skipped_binary} binary file(s)")
        if skipped_large:
            notes.append(
                f"skipped {skipped_large} file(s) larger than "
                f"{state.max_scan_bytes // 1024} KiB"
            )
        if skipped_unreadable:
            notes.append(f"skipped {skipped_unreadable} unreadable file(s)")
        return matches, notes

    def _iter_files(self, target: Path, glob: str | None) -> Iterator[Path]:
        """Yield candidate files below *target* (or *target* itself)."""
        state = self.state
        if target.is_file():
            if _glob_match(target.name, target.name, glob):
                yield target
            return
        for dirpath, dirnames, filenames in os.walk(target):
            dirnames[:] = sorted(
                name
                for name in dirnames
                if name not in SKIPPED_DIRECTORIES and not name.startswith(".")
            )
            for name in sorted(filenames):
                if name.startswith("."):
                    continue
                path = Path(dirpath) / name
                if path.is_symlink() and not state.contains(path):
                    continue  # never follow links out of the workspace
                if _glob_match(state.relative(path), name, glob):
                    yield path
