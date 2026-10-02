"""Shared fixtures: a throwaway workspace with a RuntimeState bound to it."""

from __future__ import annotations

from pathlib import Path

import pytest

from wqagent.core.state import RuntimeState


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """An empty, real workspace directory."""
    root = tmp_path / "workspace"
    root.mkdir()
    (root / ".git").mkdir()
    return root


@pytest.fixture()
def state(workspace: Path) -> RuntimeState:
    """RuntimeState pointing at the throwaway workspace."""
    return RuntimeState(workspace=workspace, timeout_seconds=20)


@pytest.fixture(autouse=True)
def _force_builtin_grep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the built-in scanner instead of ripgrep for deterministic tests."""
    monkeypatch.setenv("WQ_DISABLE_RIPGREP", "1")
