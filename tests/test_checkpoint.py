"""Tests for checkpointing and resume (``core/checkpoint.py``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from helpers import answer, runtime_for, tool_call
from wqagent.core.checkpoint import (
    CHECKPOINT_DIR,
    CHECKPOINT_MODE_ENV,
    DEFAULT_CHECKPOINT_MODE,
    MANIFEST_LIMIT,
    VALID_CHECKPOINT_MODES,
    CheckpointManager,
    build_recovery_markdown,
    checkpoint_mode_from_env,
    message_from_dict,
    message_to_dict,
    normalize_checkpoint_mode,
    resume_command,
    state_to_payload,
)
from wqagent.graph.state import make_todo, new_graph_state


def _state(tmp_path: Path, task: str = "write hello.py", **extra: Any) -> dict[str, Any]:
    """A graph state in ``tmp_path/ws`` carrying a four-message transcript."""
    runtime = runtime_for(tmp_path / "ws", checkpoint_mode=extra.pop("mode", "light"))
    state = new_graph_state(
        task, runtime, todos=[make_todo("t1", "write hello.py", status="completed")], **extra
    )
    state["messages"] = [
        HumanMessage(content="write hello.py"),
        tool_call("file_write", {"file_path": "hello.py"}, "c1"),
        ToolMessage(content="Created hello.py", tool_call_id="c1", name="file_write"),
        answer("done"),
    ]
    return state


def _save_or_skip(manager: CheckpointManager, state: dict[str, Any], **options: Any) -> dict[str, Any]:
    """Save *state*, skipping when git gave no snapshot for the restore tests."""
    saved = manager.save(state, **options)
    if not saved["git_ok"]:
        pytest.skip("git unavailable")  # pragma: no cover
    return saved


# -- modes -------------------------------------------------------------------
def test_valid_checkpoint_modes() -> None:
    assert VALID_CHECKPOINT_MODES == {"light", "strict", "off"}
    assert DEFAULT_CHECKPOINT_MODE == "light"


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "light"),
        ("", "light"),
        ("  ", "light"),
        ("nonsense", "light"),
        ("STRICT", "strict"),
        (" Off ", "off"),
        ("light", "light"),
    ],
)
def test_normalize_checkpoint_mode(value: Any, expected: str) -> None:
    assert normalize_checkpoint_mode(value) == expected


def test_checkpoint_mode_from_env(monkeypatch: Any) -> None:
    monkeypatch.delenv(CHECKPOINT_MODE_ENV, raising=False)
    assert checkpoint_mode_from_env() == "light"

    monkeypatch.setenv(CHECKPOINT_MODE_ENV, "strict")
    assert checkpoint_mode_from_env() == "strict"

    monkeypatch.setenv(CHECKPOINT_MODE_ENV, "junk")
    assert checkpoint_mode_from_env() == "light"


def test_runtime_state_picks_up_the_mode(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.delenv(CHECKPOINT_MODE_ENV, raising=False)
    assert runtime_for(tmp_path / "ws").checkpoint_mode == "light"

    monkeypatch.setenv(CHECKPOINT_MODE_ENV, "strict")
    assert runtime_for(tmp_path / "ws").checkpoint_mode == "strict"
    assert runtime_for(tmp_path / "ws", checkpoint_mode="off").checkpoint_mode == "off"
    assert runtime_for(tmp_path / "ws", checkpoint_mode="junk").checkpoint_mode == "light"


def test_manager_paths_and_enabled(tmp_path: Path) -> None:
    runtime = runtime_for(tmp_path / "ws", checkpoint_mode="light")
    manager = CheckpointManager(runtime, "write hello.py")

    assert manager.workspace == runtime.workspace
    assert manager.mode == "light"
    assert manager.enabled is True
    assert manager.root == runtime.workspace / CHECKPOINT_DIR / "checkpoints"
    assert manager.checkpoint_path == manager.root / "checkpoint.json"
    assert manager.recovery_path == manager.root / "RECOVERY.md"
    assert manager.state_path == manager.root / "state.json"
    assert manager.events_path == manager.root / "events.jsonl"

    off = CheckpointManager(runtime_for(tmp_path / "ws", checkpoint_mode="off"))
    assert off.enabled is False


def test_resume_command_is_stable(tmp_path: Path) -> None:
    assert resume_command(tmp_path) == f"wq --resume {tmp_path}"
    assert resume_command("demo") == "wq --resume demo"


# -- saving ------------------------------------------------------------------
def test_off_mode_writes_nothing(tmp_path: Path) -> None:
    state = _state(tmp_path, mode="off")
    manager = CheckpointManager(state["runtime"], "write hello.py")

    assert manager.save(state, status="running", latest_node="planner") is None

    assert not manager.root.exists()
    assert manager.append_event({"type": "node_end"}) is False


def test_light_mode_saves_metadata_recovery_and_snapshot(tmp_path: Path) -> None:
    state = _state(tmp_path)
    (tmp_path / "ws" / "hello.py").write_text("print('hi')\n", encoding="utf-8")
    manager = CheckpointManager(state["runtime"], "write hello.py")

    event = manager.save(state, status="running", latest_node="planner")

    assert event is not None
    assert event["type"] == "checkpoint_saved"
    assert event["mode"] == "light"
    assert event["strict"] is False
    assert event["status"] == "running" and event["latest_node"] == "planner"
    assert event["files"] == 1  # hello.py; the checkpoint dir is not part of the workspace
    assert event["resume_command"] == resume_command(tmp_path / "ws")

    payload = json.loads(manager.checkpoint_path.read_text(encoding="utf-8"))
    assert payload["task"] == "write hello.py"
    assert payload["mode"] == "light"
    assert payload["status"] == "running"
    assert payload["latest_node"] == "planner"
    assert payload["message_count"] == 4  # light mode keeps the count, not the messages
    assert payload["workspace_manifest"]["files"] == [{"path": "hello.py", "size": 13}]
    assert payload["state_file"] == "" and payload["events_file"] == ""
    # the state lives under its own key, so metadata can never leak into it
    state = payload["state"]
    assert state["task"] == "write hello.py"
    assert state["attempts"] == 0 and state["max_attempts"] == 3
    assert state["todos"] == [
        {"id": "t1", "content": "write hello.py", "status": "completed", "note": ""}
    ]
    assert "messages" not in state
    assert "status" not in state and "latest_node" not in state

    recovery = manager.recovery_path.read_text(encoding="utf-8")
    assert recovery.startswith("# Recovery")
    assert "write hello.py" in recovery
    assert "running" in recovery and "planner" in recovery
    assert "- [x] t1: write hello.py" in recovery
    assert "hello.py (13 bytes)" in recovery
    assert resume_command(tmp_path / "ws") in recovery
    assert not manager.state_path.exists() and not manager.events_path.exists()


def test_strict_mode_adds_state_and_event_log(tmp_path: Path) -> None:
    state = _state(tmp_path, mode="strict")
    manager = CheckpointManager(state["runtime"], "write hello.py")

    manager.append_event({"type": "node_end", "node": "planner", "update": {"attempts": 0}})
    event = manager.save(
        state, status="running", latest_node="verifier", event={"type": "node_end", "node": "verifier"}
    )

    assert event is not None and event["strict"] is True
    assert json.loads(manager.state_path.read_text(encoding="utf-8"))["task"] == "write hello.py"
    lines = manager.events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["node"] for line in lines] == ["planner", "verifier"]
    payload = json.loads(manager.checkpoint_path.read_text(encoding="utf-8"))
    assert payload["state_file"] == "state.json" and payload["events_file"] == "events.jsonl"


def test_manifest_skips_the_checkpoint_directory(tmp_path: Path) -> None:
    state = _state(tmp_path)
    workspace = tmp_path / "ws"
    (workspace / "src").mkdir(parents=True)
    (workspace / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
    manager = CheckpointManager(state["runtime"], "write hello.py")

    manager.save(state, status="running", latest_node="planner")
    manifest = manager.workspace_manifest()

    paths = [item["path"] for item in manifest["files"]]
    assert paths == ["src/app.py"]
    assert all(not path.startswith(CHECKPOINT_DIR) for path in paths)
    assert manifest["truncated"] is False


def test_manifest_truncates_a_huge_workspace(tmp_path: Path) -> None:
    state = _state(tmp_path)
    workspace = tmp_path / "ws"
    for index in range(MANIFEST_LIMIT + 5):
        (workspace / f"f{index:04d}.txt").write_text("x", encoding="utf-8")
    manager = CheckpointManager(state["runtime"], "write hello.py")

    manifest = manager.workspace_manifest()

    assert manifest["count"] == MANIFEST_LIMIT + 5
    assert manifest["truncated"] is True
    assert len(manifest["files"]) == MANIFEST_LIMIT


def test_git_snapshot_never_touches_the_workspace_repository(tmp_path: Path) -> None:
    """The snapshot lives in .wqagent/checkpoints/git, not in the workspace's .git."""
    state = _state(tmp_path)
    workspace = tmp_path / "ws"
    (workspace / "hello.py").write_text("print('hi')\n", encoding="utf-8")
    manager = CheckpointManager(state["runtime"], "write hello.py")

    git = manager.snapshot_workspace_git()

    if not git["ok"]:  # pragma: no cover - git is available in CI here
        pytest.skip(f"git unavailable: {git['error']}")
    assert manager.git_dir.is_dir()
    assert not (workspace / ".git").exists()
    assert git["files"] == 1
    # a second snapshot with no changes reuses the same commit
    again = manager.snapshot_workspace_git()
    assert again["ok"] is True and again["commit"] == git["commit"]
    # a change produces a new commit
    (workspace / "hello.py").write_text("print('changed')\n", encoding="utf-8")
    changed = manager.snapshot_workspace_git()
    assert changed["ok"] is True and changed["commit"] != git["commit"]


def test_git_failure_degrades_to_a_checkpoint_without_snapshot(
    tmp_path: Path, monkeypatch: Any
) -> None:
    state = _state(tmp_path)
    manager = CheckpointManager(state["runtime"], "write hello.py")
    monkeypatch.setattr(manager, "_ensure_git", lambda: (False, "git is not installed"))

    event = manager.save(state, status="running", latest_node="planner")

    assert event is not None and event["git_ok"] is False
    assert manager.checkpoint_path.is_file()  # the checkpoint is still written
    payload = json.loads(manager.checkpoint_path.read_text(encoding="utf-8"))
    assert payload["git"]["ok"] is False
    assert "no git snapshot" in manager.recovery_path.read_text(encoding="utf-8")


# -- recovery markdown -------------------------------------------------------
def test_build_recovery_markdown_from_a_sparse_payload() -> None:
    text = build_recovery_markdown({})

    assert text.startswith("# Recovery")
    assert "(no task recorded)" in text
    assert "wq --resume" in text
    assert "(the workspace was empty)" in text
    assert "no git snapshot" in text


def test_build_recovery_markdown_includes_everything(tmp_path: Path) -> None:
    payload = {
        "task": "ship it", "status": "failed", "latest_node": "verifier", "mode": "strict",
        "saved_at": "2026-01-01T00:00:00+00:00", "workspace": str(tmp_path),
        "attempts": 2, "max_attempts": 3, "resume_command": resume_command(tmp_path),
        "plan_summary": "write then verify",
        "todos": [{"id": "t1", "content": "write", "status": "in_progress"}],
        "last_error": "exit code 1", "context_summary": "compressed",
        "workspace_manifest": {"count": 1, "truncated": False, "files": [{"path": "a.py", "size": 3}]},
        "git": {"ok": True, "commit": "abc123", "files": 1},
        "state_file": "state.json", "events_file": "events.jsonl",
    }
    text = build_recovery_markdown(payload)

    for expected in (
        "**Task:** ship it", "**Status:** failed (latest node: verifier)", "**Attempts:** 2/3",
        "## Plan", "write then verify", "- [~] t1: write", "## Last error", "exit code 1",
        "## Compressed context", "## Workspace files", "- a.py (3 bytes)",
        "- git commit: `abc123`", "state.json", "events.jsonl", resume_command(tmp_path),
    ):
        assert expected in text, expected


# -- state serialisation -----------------------------------------------------
def test_message_round_trip_keeps_tool_calls() -> None:
    messages = [
        HumanMessage(content="do it"),
        tool_call("bash", {"command": "ls"}, "c1"),
        ToolMessage(content="ok", tool_call_id="c1", name="bash"),
        answer("done"),
    ]

    rebuilt = [message_from_dict(message_to_dict(message)) for message in messages]

    assert [message.type for message in rebuilt] == ["human", "ai", "tool", "ai"]
    assert rebuilt[1].tool_calls[0]["name"] == "bash"
    assert rebuilt[2].tool_call_id == "c1"
    assert rebuilt[3].content == "done"


def test_state_to_payload_skips_the_runtime(tmp_path: Path) -> None:
    state = _state(tmp_path)

    full = state_to_payload(state)
    light = state_to_payload(state, include_messages=False)

    assert "runtime" not in full and "runtime" not in light
    assert full["messages"][0]["type"] == "human"
    assert "messages" not in light and light["message_count"] == 4
    assert light["todos"] == full["todos"]
    json.dumps(full)  # serialisable


def test_state_to_payload_survives_odd_values(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state["weird"] = object()  # type: ignore[typeddict-unknown-key]

    payload = state_to_payload(state)

    assert isinstance(payload["weird"], str)
    json.dumps(payload)


# -- loading / resume --------------------------------------------------------
@pytest.mark.parametrize(
    "options, warning",
    [
        ({}, "no checkpoint.json"),
        ({"checkpoint_mode": "off"}, "checkpointing is off"),
    ],
)
def test_load_resume_inputs_without_a_usable_checkpoint(
    tmp_path: Path, options: dict[str, Any], warning: str
) -> None:
    inputs, event = CheckpointManager.load_resume_inputs(runtime_for(tmp_path / "ws", **options))

    assert inputs is None
    assert event["type"] == "resume" and event["ok"] is False
    assert any(warning in item for item in event["warnings"])


def test_resume_restores_the_workspace_and_the_state(tmp_path: Path) -> None:
    state = _state(tmp_path)
    workspace = tmp_path / "ws"
    (workspace / "hello.py").write_text("print('hi')\n", encoding="utf-8")
    manager = CheckpointManager(state["runtime"], "write hello.py")
    state["attempts"] = 1
    state["last_error"] = "exit code 1"
    _save_or_skip(manager, state, status="failed", latest_node="verifier")
    # the workspace changes after the checkpoint
    (workspace / "hello.py").write_text("broken\n", encoding="utf-8")
    (workspace / "extra.txt").write_text("added later\n", encoding="utf-8")

    runtime = runtime_for(tmp_path / "ws")  # a fresh runtime, as a new process would build
    inputs, event = CheckpointManager.load_resume_inputs(runtime, max_attempts=2)

    assert event["ok"] is True
    assert event["task"] == "write hello.py"
    assert event["status"] == "failed"
    assert event["latest_node"] == "verifier"
    assert event["restored_files"] == 1
    assert event["message_count"] == 0  # light mode: no transcript
    assert any("transcript was not checkpointed" in warning for warning in event["warnings"])
    assert any("differed from the snapshot" in warning for warning in event["warnings"])

    assert (workspace / "hello.py").read_text(encoding="utf-8") == "print('hi')\n"
    assert (workspace / "extra.txt").is_file()  # files added later are kept

    assert inputs is not None
    assert inputs["task"] == "write hello.py"
    assert inputs["attempts"] == 1
    assert inputs["last_error"] == "exit code 1"
    assert inputs["todos"][0]["status"] == "completed"
    assert inputs["max_attempts"] == 3  # the checkpoint's own value wins
    assert inputs["runtime"] is runtime


def test_strict_resume_restores_the_transcript(tmp_path: Path) -> None:
    state = _state(tmp_path, mode="strict")
    manager = CheckpointManager(state["runtime"], "write hello.py")
    _save_or_skip(manager, state, status="running", latest_node="planner")

    inputs, event = CheckpointManager.load_resume_inputs(
        runtime_for(tmp_path / "ws", checkpoint_mode="strict")
    )

    assert inputs is not None
    assert event["message_count"] == 4
    assert not any("transcript was not checkpointed" in warning for warning in event["warnings"])
    messages = inputs["messages"]
    assert [message.type for message in messages] == ["human", "ai", "tool", "ai"]
    assert messages[2].tool_call_id == "c1"


def test_resume_warns_when_the_task_differs(tmp_path: Path) -> None:
    state = _state(tmp_path, task="the original task")
    _save_or_skip(CheckpointManager(state["runtime"], "the original task"), state)

    inputs, event = CheckpointManager.load_resume_inputs(runtime_for(tmp_path / "ws"), task="a different task")

    assert inputs is not None and inputs["task"] == "the original task"
    assert any("different task" in warning for warning in event["warnings"])


def test_resume_does_not_leak_metadata_into_the_new_checkpoints(tmp_path: Path) -> None:
    """Regression: a resumed run must not inherit the old status / latest_node."""
    state = _state(tmp_path)
    manager = CheckpointManager(state["runtime"], "write hello.py")
    _save_or_skip(manager, state, status="running", latest_node="verifier")

    inputs, _ = CheckpointManager.load_resume_inputs(runtime_for(tmp_path / "ws"))

    assert inputs is not None
    for leaked in ("status", "latest_node", "mode", "saved_at", "workspace_manifest", "git"):
        assert leaked not in inputs, leaked

    manager = CheckpointManager(inputs["runtime"], str(inputs["task"]))
    event = manager.save(inputs, status="passed", latest_node="final")

    assert event is not None
    assert event["status"] == "passed" and event["latest_node"] == "final"
    payload = json.loads(manager.checkpoint_path.read_text(encoding="utf-8"))
    assert payload["status"] == "passed" and payload["latest_node"] == "final"
    assert payload["state"]["todos"][0]["status"] == "completed"


def test_load_resume_inputs_survives_a_corrupt_checkpoint(tmp_path: Path) -> None:
    runtime = runtime_for(tmp_path / "ws")
    manager = CheckpointManager(runtime)
    manager.root.mkdir(parents=True)
    manager.checkpoint_path.write_text("{not json", encoding="utf-8")

    inputs, event = CheckpointManager.load_resume_inputs(runtime)

    assert inputs is None and event["ok"] is False
