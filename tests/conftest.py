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


@pytest.fixture(autouse=True)
def _no_implicit_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the suite offline: a node that was not given a model must fail loudly.

    Nodes fall back to ``create_model()`` (the ``.env`` configuration). Every test
    is expected to pass ``model=`` or pin one with ``set_model()``; without this
    guard a forgotten argument silently turns into a real API call. Tests that
    build a model on purpose (e.g. against the fake HTTP server) patch
    ``wqagent.providers.create_model`` / the CLI's own reference instead, which
    this fixture does not touch.
    """

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError(
            "a node built a real model from .env; pass model=... or call set_model() "
            "in this test"
        )

    for target in (
        "wqagent.graph.nodes.create_model",
        "wqagent.agents.code_agent.create_model",
        "wqagent.agents.search_agent.create_model",
    ):
        monkeypatch.setattr(target, _boom)
