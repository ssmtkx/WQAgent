"""Tests for the workspace path guard."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from wqagent.core.paths import (
    WorkspacePathError,
    is_within_workspace,
    relative_to_workspace,
    resolve_in_workspace,
    resolve_workspace,
)


def test_resolve_workspace_creates_missing_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "workspace"
    resolved = resolve_workspace(target)
    assert resolved == target.resolve()
    assert resolved.is_dir()


def test_resolve_workspace_rejects_a_file(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-dir"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(WorkspacePathError):
        resolve_workspace(file_path)


def test_relative_paths_are_joined_to_the_workspace(workspace: Path) -> None:
    resolved = resolve_in_workspace(workspace, "src/app.py")
    assert resolved == (workspace / "src" / "app.py").resolve()


def test_absolute_path_inside_the_workspace_is_allowed(workspace: Path) -> None:
    inside = workspace / "notes.txt"
    assert resolve_in_workspace(workspace, inside) == inside.resolve()


def test_parent_directory_escape_is_rejected(workspace: Path) -> None:
    with pytest.raises(WorkspacePathError, match="escapes the workspace"):
        resolve_in_workspace(workspace, "../outside.txt")
    with pytest.raises(WorkspacePathError, match="escapes the workspace"):
        resolve_in_workspace(workspace, "a/b/../../../outside.txt")


def test_absolute_path_outside_the_workspace_is_rejected(workspace: Path, tmp_path: Path) -> None:
    outside = tmp_path.parent / "definitely-outside.txt"
    with pytest.raises(WorkspacePathError, match="escapes the workspace"):
        resolve_in_workspace(workspace, outside)


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs privileges on Windows")
def test_symlink_escape_is_rejected(workspace: Path, tmp_path: Path) -> None:
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "secret.txt").write_text("secret", encoding="utf-8")
    link = workspace / "link"
    link.symlink_to(outside_dir, target_is_directory=True)
    with pytest.raises(WorkspacePathError, match="escapes the workspace"):
        resolve_in_workspace(workspace, "link/secret.txt")


def test_empty_and_nul_paths_are_rejected(workspace: Path) -> None:
    with pytest.raises(WorkspacePathError):
        resolve_in_workspace(workspace, "   ")
    with pytest.raises(WorkspacePathError):
        resolve_in_workspace(workspace, "bad\x00name")


def test_must_exist_and_kind_checks(workspace: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_in_workspace(workspace, "missing.txt", must_exist=True)
    directory = workspace / "pkg"
    directory.mkdir()
    with pytest.raises(WorkspacePathError, match="not a file"):
        resolve_in_workspace(workspace, "pkg", kind="file")
    file_path = workspace / "pkg" / "mod.py"
    file_path.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(WorkspacePathError, match="not a directory"):
        resolve_in_workspace(workspace, "pkg/mod.py", kind="dir")


def test_helpers(workspace: Path) -> None:
    assert is_within_workspace(workspace, workspace / "a" / "b.txt")
    assert not is_within_workspace(workspace, workspace.parent / "elsewhere")
    assert relative_to_workspace(workspace, workspace / "a" / "b.txt") == "a/b.txt"
    assert relative_to_workspace(workspace, workspace) == "."
