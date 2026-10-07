"""Tests for conversation sessions (``core/session.py``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from wqagent.core.session import (
    CONTEXT_FILES,
    CONTEXT_TURNS,
    MAX_RECENT_TURNS,
    MAX_SESSION_CONTEXT,
    MAX_TURN_CONTENT,
    SESSION_FILE,
    SESSION_ROOT,
    SESSION_SUMMARY_FILE,
    append_assistant_turn,
    append_user_turn,
    build_session_context,
    load_or_create_session,
    new_session,
    render_session_summary,
    save_session,
    session_dir,
    session_path,
    summary_path,
    workspace_files,
)


def _workspace(tmp_path: Path, *files: str) -> Path:
    root = tmp_path / "ws"
    root.mkdir(parents=True, exist_ok=True)
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {name}\n", encoding="utf-8")
    return root


# -- constants and shape -----------------------------------------------------
def test_constants() -> None:
    assert SESSION_ROOT == ".wqagent/session"
    assert SESSION_FILE == "session.json"
    assert SESSION_SUMMARY_FILE == "SESSION_SUMMARY.md"
    assert MAX_SESSION_CONTEXT == 7000
    assert MAX_TURN_CONTENT == 4000


def test_paths(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    assert session_dir(workspace) == workspace / ".wqagent" / "session"
    assert session_path(workspace).name == SESSION_FILE
    assert summary_path(workspace).name == SESSION_SUMMARY_FILE


def test_paths_accept_a_runtime_state(tmp_path: Path) -> None:
    from wqagent.core.state import RuntimeState

    workspace = _workspace(tmp_path)
    runtime = RuntimeState(workspace=workspace)

    assert session_path(runtime) == session_path(workspace)


def test_new_session_shape() -> None:
    session = new_session()

    assert set(session) == {"session_id", "turn_index", "recent_turns", "created_at", "updated_at"}
    assert session["turn_index"] == 0 and session["recent_turns"] == []
    assert session["session_id"] and session["created_at"]


# -- load / create -----------------------------------------------------------
def test_load_or_create_creates_a_new_session(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    session = load_or_create_session(workspace)

    assert session["turn_index"] == 0 and session["recent_turns"] == []
    assert session["session_id"]
    assert not session_path(workspace).exists()  # nothing is written until save


def test_load_or_create_reads_an_existing_session(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    save_session(workspace, {**new_session(), "session_id": "fixed", "turn_index": 7})

    session = load_or_create_session(workspace)

    assert session["session_id"] == "fixed"
    assert session["turn_index"] == 7


@pytest.mark.parametrize("payload", ["{not json", "[]", "null", '"text"'])
def test_load_or_create_survives_a_broken_file(tmp_path: Path, payload: str) -> None:
    workspace = _workspace(tmp_path)
    session_path(workspace).parent.mkdir(parents=True, exist_ok=True)
    session_path(workspace).write_text(payload, encoding="utf-8")

    session = load_or_create_session(workspace)

    assert session["turn_index"] == 0 and session["session_id"]


def test_load_or_create_repairs_missing_and_broken_fields(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    session_path(workspace).parent.mkdir(parents=True, exist_ok=True)
    session_path(workspace).write_text(
        json.dumps(
            {
                "recent_turns": [
                    "not an object",
                    {"turn": "2", "role": "user", "content": "x"},
                    {"turn": 3, "role": "assistant", "content": "y"},
                ],
                "turn_index": "junk",
            }
        ),
        encoding="utf-8",
    )

    session = load_or_create_session(workspace)

    assert [turn["turn"] for turn in session["recent_turns"]] == [2, 3]
    assert session["recent_turns"][1]["route"] == "workflow"  # defaulted
    assert session["turn_index"] == 3  # derived from the turns when the index is junk


# -- appending turns ---------------------------------------------------------
def test_append_user_turn_numbers_turns_and_clips_content(tmp_path: Path) -> None:
    session = new_session()

    first = append_user_turn(session, "hello")
    second = append_user_turn(session, "x" * (MAX_TURN_CONTENT + 500))

    assert (first, second) == (1, 2)
    assert session["turn_index"] == 2
    assert len(session["recent_turns"][1]["content"]) <= MAX_TURN_CONTENT
    assert session["recent_turns"][0]["role"] == "user"
    assert session["recent_turns"][0]["timestamp"]


def test_append_assistant_turn_uses_the_next_turn_number(tmp_path: Path) -> None:
    session = new_session()
    turn = append_user_turn(session, "写个 hello.py")

    append_assistant_turn(session, turn=turn, route="workflow", content="Done.", summary="wrote it")

    entry = session["recent_turns"][-1]
    assert entry == {
        "turn": 2,
        "role": "assistant",
        "route": "workflow",
        "content": "Done.",
        "summary": "wrote it",
        "timestamp": entry["timestamp"],
    }
    assert session["turn_index"] == 2
    # the next user turn continues after the assistant's number
    assert append_user_turn(session, "继续") == 3


def test_append_assistant_turn_honours_an_explicit_gap(tmp_path: Path) -> None:
    session = new_session()
    append_user_turn(session, "one")

    append_assistant_turn(session, turn=9, route="chat", content="hi")

    assert session["recent_turns"][-1]["turn"] == 9
    assert session["turn_index"] == 9


def test_append_assistant_turn_defaults_route_and_summary(tmp_path: Path) -> None:
    session = new_session()
    turn = append_user_turn(session, "hi")

    append_assistant_turn(session, turn=turn, route="", content="x" * (MAX_TURN_CONTENT + 10))

    entry = session["recent_turns"][-1]
    assert entry["route"] == "workflow"
    assert entry["summary"] == ""
    assert len(entry["content"]) <= MAX_TURN_CONTENT


def test_recent_turns_are_capped(tmp_path: Path) -> None:
    session = new_session()
    for index in range(MAX_RECENT_TURNS + 5):
        append_user_turn(session, f"turn {index}")

    assert len(session["recent_turns"]) == MAX_RECENT_TURNS
    assert session["recent_turns"][-1]["content"] == f"turn {MAX_RECENT_TURNS + 4}"


# -- saving ------------------------------------------------------------------
def test_save_session_writes_both_files(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    session = load_or_create_session(workspace)
    turn = append_user_turn(session, "帮我搭个 FastAPI 后台")
    append_assistant_turn(session, turn=turn, route="workflow", content="Done: app/main.py", summary="skeleton")

    saved = save_session(workspace, session)

    assert "last_error" not in saved
    payload = json.loads(session_path(workspace).read_text(encoding="utf-8"))
    assert payload["turn_index"] == 2
    assert [item["role"] for item in payload["recent_turns"]] == ["user", "assistant"]
    assert payload["updated_at"] >= payload["created_at"]

    summary = summary_path(workspace).read_text(encoding="utf-8")
    assert summary.startswith("# Session")
    assert payload["session_id"] in summary
    assert "### 1. user" in summary
    assert "### 2. assistant (workflow)" in summary
    assert "帮我搭个 FastAPI 后台" in summary
    assert "_summary: skeleton_" in summary


def test_render_session_summary_handles_an_empty_session() -> None:
    text = render_session_summary(new_session())

    assert "(nothing recorded yet)" in text


def test_save_session_reports_a_write_failure(tmp_path: Path, monkeypatch: Any) -> None:
    workspace = _workspace(tmp_path)
    session = new_session()

    def boom(self: Path, *args: Any, **kwargs: Any) -> int:
        raise OSError("disk is full")

    monkeypatch.setattr(Path, "write_text", boom)

    saved = save_session(workspace, session)

    assert "disk is full" in saved["last_error"]
    assert "last save failed" in render_session_summary(saved)


# -- workspace files ---------------------------------------------------------
def test_workspace_files_newest_first_and_without_the_agent_directory(tmp_path: Path) -> None:
    import os

    workspace = _workspace(tmp_path, "old.py", "src/new.py")
    (workspace / ".wqagent" / "traces").mkdir(parents=True, exist_ok=True)
    (workspace / ".wqagent" / "traces" / "trace.json").write_text("{}", encoding="utf-8")
    stamp = (workspace / "old.py").stat().st_mtime - 600  # make it clearly the oldest
    os.utime(workspace / "old.py", (stamp, stamp))

    files = workspace_files(workspace)

    assert "src/new.py" in files and "old.py" in files
    assert files[-1] == "old.py"  # newest first
    assert all(not name.startswith(".wqagent") for name in files)


def test_workspace_files_is_capped(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, *[f"f{index:03d}.txt" for index in range(CONTEXT_FILES + 10)])

    assert len(workspace_files(workspace)) == CONTEXT_FILES


def test_workspace_files_of_a_missing_directory(tmp_path: Path) -> None:
    assert workspace_files(tmp_path / "nope") == []


# -- the context block -------------------------------------------------------
def test_build_session_context_has_id_turn_files_and_turns(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, "app/main.py", "requirements.txt")
    session = load_or_create_session(workspace)
    turn = append_user_turn(session, "继续加个登录接口")
    append_assistant_turn(session, turn=turn, route="workflow", content="added", summary="login endpoint")

    text = build_session_context(workspace, session)

    assert session["session_id"] in text
    assert "turn: 2" in text
    assert "workspace files (newest first):" in text
    assert "- app/main.py" in text and "- requirements.txt" in text
    assert "recent turns (oldest first):" in text
    assert "turn 1 [user]: 继续加个登录接口" in text
    assert "turn 2 [assistant] (workflow): login endpoint" in text
    assert len(text) <= MAX_SESSION_CONTEXT


def test_build_session_context_loads_the_session_itself(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    session = load_or_create_session(workspace)
    append_user_turn(session, "earlier question")
    save_session(workspace, session)

    text = build_session_context(workspace)

    assert "earlier question" in text


def test_build_session_context_shows_at_most_ten_turns(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    session = new_session()
    for index in range(CONTEXT_TURNS + 5):
        append_user_turn(session, f"question {index:03d}")

    text = build_session_context(workspace, session)

    assert "question 001" not in text  # the oldest ones are dropped
    assert f"question {CONTEXT_TURNS + 4:03d}" in text


def test_build_session_context_stays_within_the_budget(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, *[f"file{index:03d}.py" for index in range(60)])
    session = new_session()
    for index in range(CONTEXT_TURNS + 6):
        turn = append_user_turn(session, f"a very long user message number {index} " + "x" * 3000)
        append_assistant_turn(
            session, turn=turn, route="workflow", content="y" * 3000, summary="z" * 300
        )

    text = build_session_context(workspace, session)

    assert len(text) <= MAX_SESSION_CONTEXT
    # the newest exchange survives: old turns are dropped before recent ones
    assert "message number 15" in text or "omitted" in text


def test_build_session_context_of_an_empty_workspace(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    text = build_session_context(workspace)

    assert "session:" in text and "turn: 0" in text
    assert "workspace files" not in text  # nothing to list
    assert "recent turns" not in text


# -- wiring ------------------------------------------------------------------
def test_entry_nodes_see_the_session_context(tmp_path: Path) -> None:
    """The router's prompt carries what happened in previous turns."""
    from helpers import ScriptedModel, answer

    from wqagent.core.state import RuntimeState
    from wqagent.graph.nodes import intent_router_node
    from wqagent.graph.state import new_graph_state

    workspace = _workspace(tmp_path, "app/main.py")
    session = load_or_create_session(workspace)
    turn = append_user_turn(session, "帮我搭个 FastAPI 后台")
    append_assistant_turn(session, turn=turn, route="workflow", content="done", summary="skeleton")
    save_session(workspace, session)
    runtime = RuntimeState(workspace=workspace, session_id=session["session_id"], session_turn=3)
    state = new_graph_state("继续", runtime)
    model = ScriptedModel([answer('{"route":"workflow","reason":"continuation","confidence":0.9}')])

    intent_router_node(state, model=model)

    instruction = model.seen[-1][1].content
    assert session["session_id"] in instruction
    assert "帮我搭个 FastAPI 后台" in instruction
    assert "app/main.py" in instruction  # the workspace file list
    assert "skeleton" in instruction  # the previous assistant summary
    assert "继续" in instruction


def test_create_runtime_carries_the_session(tmp_path: Path) -> None:
    from wqagent.core.state import create_runtime

    workspace = _workspace(tmp_path)
    session = load_or_create_session(workspace)
    append_user_turn(session, "first")

    runtime = create_runtime(workspace, session=session)

    assert runtime.session_id == session["session_id"]
    assert runtime.session_turn == 1


def test_new_graph_state_carries_the_session(tmp_path: Path) -> None:
    from wqagent.core.state import RuntimeState
    from wqagent.graph.state import new_graph_state

    workspace = _workspace(tmp_path)
    runtime = RuntimeState(workspace=workspace, session_id="abc", session_turn=4)

    state = new_graph_state("task", runtime)

    assert state["session_id"] == "abc" and state["session_turn"] == 4
