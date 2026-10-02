"""Workspace path utilities.

Every filesystem-facing tool resolves its paths through this module, so a path
that escapes the workspace is rejected *before* any I/O happens. Symlinks are
resolved first, which also blocks the "symlink inside the workspace pointing
outside of it" trick.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "WorkspacePathError",
    "is_within_workspace",
    "relative_to_workspace",
    "resolve_in_workspace",
    "resolve_workspace",
]


class WorkspacePathError(ValueError):
    """Raised when a path is empty, malformed, or escapes the workspace."""


def _normalize(path: str | os.PathLike[str]) -> Path:
    """Absolute, symlink-resolved form of *path* (the path need not exist)."""
    candidate = Path(path)
    try:
        return candidate.resolve(strict=False)
    except OSError:  # pragma: no cover - exotic paths (unmapped drive, ...)
        return Path(os.path.abspath(os.path.normpath(str(candidate))))


def _key(path: str | os.PathLike[str]) -> str:
    """Comparison key: case-folded on Windows, separator-normalised everywhere."""
    return os.path.normcase(os.path.normpath(str(path))).rstrip("\\/")


def _is_same_or_child(root: Path, target: Path) -> bool:
    root_key = _key(root)
    target_key = _key(target)
    return target_key == root_key or target_key.startswith(root_key + os.sep)


def resolve_workspace(workspace: str | os.PathLike[str], *, create: bool = True) -> Path:
    """Return the absolute workspace directory.

    Relative paths resolve against the current working directory. When *create*
    is true (the default) the directory is created, including parents, so the
    CLI can point ``--workspace`` at a path that does not exist yet.
    """
    raw = str(workspace).strip()
    if not raw:
        raise WorkspacePathError("workspace path is empty")
    candidate = Path(os.path.expandvars(raw)).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    root = _normalize(candidate)
    if root.exists() and not root.is_dir():
        raise WorkspacePathError(f"workspace is not a directory: {root}")
    if create:
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise WorkspacePathError(f"cannot create workspace {root}: {exc}") from exc
    return root


def is_within_workspace(workspace: str | os.PathLike[str], target: str | os.PathLike[str]) -> bool:
    """True when *target* is the workspace itself or lives inside it."""
    return _is_same_or_child(_normalize(workspace), _normalize(target))


def resolve_in_workspace(
    workspace: str | os.PathLike[str],
    target: str | os.PathLike[str],
    *,
    must_exist: bool = False,
    kind: str | None = None,
) -> Path:
    """Resolve *target* inside *workspace* and refuse anything that escapes it.

    Args:
        workspace: Workspace root; *target* is joined to it when relative.
        target: Absolute or workspace-relative path.
        must_exist: Raise :class:`FileNotFoundError` when the path is missing.
        kind: ``"file"`` or ``"dir"`` - validated only for existing paths.

    Raises:
        WorkspacePathError: The path is empty, drive-relative, of the wrong
            kind, or resolves outside the workspace.
        FileNotFoundError: *must_exist* is true and the path does not exist.
    """
    root = _normalize(workspace)
    text = str(target).strip()
    if not text:
        raise WorkspacePathError("path is empty")
    if "\x00" in text:
        raise WorkspacePathError("path contains a NUL byte")
    raw = Path(os.path.expandvars(text)).expanduser()
    if raw.drive and not raw.is_absolute():
        raise WorkspacePathError(f"drive-relative paths are not supported: {target!r}")
    candidate = raw if raw.is_absolute() else root / raw
    resolved = _normalize(candidate)
    if not _is_same_or_child(root, resolved):
        raise WorkspacePathError(
            f"path escapes the workspace: {target!r} resolves to {resolved}, outside {root}"
        )
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"no such file or directory: {relative_to_workspace(root, resolved)}")
    if resolved.exists():
        if kind == "file" and not resolved.is_file():
            raise WorkspacePathError(f"not a file: {relative_to_workspace(root, resolved)}")
        if kind == "dir" and not resolved.is_dir():
            raise WorkspacePathError(f"not a directory: {relative_to_workspace(root, resolved)}")
    return resolved


def relative_to_workspace(
    workspace: str | os.PathLike[str], target: str | os.PathLike[str]
) -> str:
    """Display form of *target*: relative to the workspace when possible."""
    root = _normalize(workspace)
    resolved = _normalize(target)
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return str(resolved)
    return relative.as_posix() or "."
