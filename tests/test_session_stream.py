"""Tests for :func:`wqagent.core.agent.stream_session_events` (multi-turn sessions)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from helpers import ScriptedModel, answer, passing_script
from wqagent.core.agent import _summary_line, stream_session_events
from wqagent.core.session import (
    append_user_turn,
    load_or_create_session,
    new_session,
    save_session,
    session_path,
    summary_path,
)
from wqagent.core.state import create_runtime
from wqagent.graph.memory import SESSION_CHARS


def _workspace(tmp_path: Path, *files: str) -> Path:
    root = tmp_path / "ws"
    root.mkdir(parents=True, exist_ok=True)
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {name}\n", encoding="utf-8")
    return root


def _route(route: str, confidence: float = 0.9) -> AIMessage:
    return answer(f'{{"route":"{route}","reason":"scripted","confidence":{confidence}}}')


def _chat_script(text: str = "你好！我是 WQ Agent。") -> list[Any]:
    return [_route("chat", 0.95), answer(text)]


def _workflow_script() -> list[Any]:
    return [_route("workflow")] + passing_script()


def _turns(workspace: Path) -> list[dict[str, Any]]:
    payload = json.loads(session_path(workspace).read_text(encoding="utf-8"))
    return payload["recent_turns"]


# -- one turn ----------------------------------------------------------------
def test_chat_turn_records_both_sides(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, "app/main.py")

    events = list(
        stream_session_events(
            "你好",
            session_workspace=workspace,
            model=ScriptedModel(_chat_script()),
            trace_mode="off",
        )
    )

    kinds = [event["type"] for event in events]
    assert kinds[-1] == "session_saved"
    assert "intent" in kinds and "chat_response" in kinds
    final = next(event for event in events if event["type"] == "final_answer")
    assert final["route"] == "chat" and final["content"] == "你好！我是 WQ Agent。"
    # no workflow node ran
    assert [event["node"] for event in events if event["type"] == "node_end"] == [
        "intent_router",
        "chat_responder",
    ]

    saved = events[-1]
    assert saved["turn"] == 1 and saved["reply_turn"] == 2
    assert saved["turn_index"] == 2 and saved["turns"] == 2
    assert saved["route"] == "chat" and saved["saved"] is True
    assert saved["path"] == str(session_path(workspace))
    assert saved["summary_path"] == str(summary_path(workspace))

    turns = _turns(workspace)
    assert [(turn["turn"], turn["role"]) for turn in turns] == [(1, "user"), (2, "assistant")]
    assert turns[0]["content"] == "你好"
    assert turns[1]["route"] == "chat"
    assert turns[1]["summary"] == "你好！我是 WQ Agent。"
    assert summary_path(workspace).is_file()


def test_workflow_turn_runs_the_pipeline(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    model = ScriptedModel(_workflow_script())

    events = list(
        stream_session_events(
            "写个 hello.py", session_workspace=workspace, model=model, trace_mode="off"
        )
    )

    nodes = [event["node"] for event in events if event["type"] == "node_end"]
    assert nodes == ["intent_router", "planner", "context_monitor", "verifier", "context_monitor", "final"]
    final = next(event for event in events if event["type"] == "final_answer")
    assert final["route"] == "workflow" and final["passed"] is True
    turns = _turns(workspace)
    assert turns[-1]["route"] == "workflow"
    assert turns[-1]["summary"]


def test_turns_continue_across_calls(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    first = list(
        stream_session_events(
            "你好", session_workspace=workspace, model=ScriptedModel(_chat_script()), trace_mode="off"
        )
    )
    second = list(
        stream_session_events(
            "写个 hello.py",
            session_workspace=workspace,
            model=ScriptedModel(_workflow_script()),
            trace_mode="off",
        )
    )

    assert first[-1]["session_id"] == second[-1]["session_id"]  # same session
    assert (first[-1]["turn"], first[-1]["reply_turn"]) == (1, 2)
    assert (second[-1]["turn"], second[-1]["reply_turn"]) == (3, 4)
    assert [(turn["turn"], turn["role"]) for turn in _turns(workspace)] == [
        (1, "user"),
        (2, "assistant"),
        (3, "user"),
        (4, "assistant"),
    ]
    assert json.loads(session_path(workspace).read_text(encoding="utf-8"))["turn_index"] == 4


# -- the context reaches both sides -----------------------------------------
def test_context_reaches_the_entry_router(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, "app/main.py")

    model = ScriptedModel(_chat_script())
    list(
        stream_session_events(
            "帮我看下项目结构", session_workspace=workspace, model=model, trace_mode="off"
        )
    )

    instruction = model.seen[0][1].content  # the router's own call
    assert "帮我看下项目结构" in instruction  # this very turn
    assert "app/main.py" in instruction  # the workspace file list
    assert "turn: 1" in instruction


def test_context_reaches_the_workflow_nodes(tmp_path: Path) -> None:
    """The memory layer carries the session, so the planner knows earlier turns."""
    workspace = _workspace(tmp_path, "app/main.py")
    # a first turn, recorded as if it had happened earlier
    session = load_or_create_session(workspace)
    turn = append_user_turn(session, "帮我搭个 FastAPI 后台")
    from wqagent.core.session import append_assistant_turn

    append_assistant_turn(session, turn=turn, route="workflow", content="done", summary="skeleton")
    save_session(workspace, session)

    model = ScriptedModel(_workflow_script())
    list(
        stream_session_events(
            "再帮我加个登录接口", session_workspace=workspace, model=model, trace_mode="off"
        )
    )

    # the planner's prompt for this turn carries the session block from the memory
    planner_calls = [
        messages
        for messages in model.seen
        if any("planner/supervisor" in str(getattr(item, "content", "")) for item in messages)
    ]
    assert planner_calls, "the planner should have run"
    blob = "\n".join(str(getattr(item, "content", "")) for item in planner_calls[0])
    assert "帮我搭个 FastAPI 后台" in blob  # the earlier user turn
    assert "skeleton" in blob  # the earlier assistant summary
    assert "app/main.py" in blob  # the file list
    assert "session" in blob.lower()


def test_seed_message_is_a_system_message(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    captured: dict[str, Any] = {}

    def on_event(kind: str, payload: dict[str, Any]) -> None:
        captured.setdefault(kind, payload)

    list(
        stream_session_events(
            "写个 hello.py",
            session_workspace=workspace,
            model=ScriptedModel(_workflow_script()),
            trace_mode="off",
            on_event=on_event,
        )
    )

    assert "intent" in captured and "final_answer" in captured


# -- options -----------------------------------------------------------------
def test_save_false_keeps_everything_in_memory(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    events = list(
        stream_session_events(
            "你好",
            session_workspace=workspace,
            model=ScriptedModel(_chat_script()),
            trace_mode="off",
            save=False,
        )
    )

    assert events[-1]["saved"] is False
    assert events[-1]["turn"] == 1 and events[-1]["reply_turn"] == 2
    assert not (workspace / ".wqagent" / "session").exists()


def test_session_can_be_handed_over(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    session = new_session()
    append_user_turn(session, "earlier turn")

    events = list(
        stream_session_events(
            "你好",
            session_workspace=workspace,
            session=session,
            model=ScriptedModel(_chat_script()),
            trace_mode="off",
        )
    )

    assert events[-1]["session_id"] == session["session_id"]
    assert events[-1]["turn"] == 2  # continues the handed-over session
    assert session["turn_index"] == 3
    assert "earlier turn" in json.loads(session_path(workspace).read_text(encoding="utf-8"))["recent_turns"][0]["content"]


def test_entry_router_can_be_skipped(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    model = ScriptedModel(passing_script())  # no router response: the planner goes first

    events = list(
        stream_session_events(
            "写个 hello.py",
            session_workspace=workspace,
            model=model,
            entry_router=False,
            trace_mode="off",
        )
    )

    nodes = [event["node"] for event in events if event["type"] == "node_end"]
    assert nodes[0] == "planner" and "intent_router" not in nodes
    assert events[-1]["route"] == "workflow"
    assert "planner" in model.seen[0][0].content.lower()


def test_workspace_is_required() -> None:
    with pytest.raises(ValueError, match="session_workspace="):
        list(stream_session_events("你好", model=ScriptedModel(_chat_script())))


def test_a_failed_turn_keeps_the_user_message(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    class Boom(ScriptedModel):
        def invoke(self, messages: Any) -> Any:
            raise RuntimeError("provider exploded")

    with pytest.raises(RuntimeError):
        list(
            stream_session_events(
                "do something",
                session_workspace=workspace,
                model=Boom([]),
                trace_mode="off",
            )
        )

    turns = _turns(workspace)
    assert [(turn["turn"], turn["role"]) for turn in turns] == [(1, "user")]
    assert turns[0]["content"] == "do something"


def test_checkpoint_and_trace_still_written(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)

    list(
        stream_session_events(
            "写个 hello.py",
            session_workspace=workspace,
            model=ScriptedModel(_workflow_script()),
            trace_mode="full",
        )
    )

    assert (workspace / ".wqagent" / "checkpoints" / "checkpoint.json").is_file()
    traces = list((workspace / ".wqagent" / "traces").iterdir())
    assert len(traces) == 1 and (traces[0] / "trace.json").is_file()
    # the session id is part of the recorded run
    payload = json.loads((traces[0] / "trace.json").read_text(encoding="utf-8"))
    assert payload["status"] == "passed"


# -- helpers -----------------------------------------------------------------
def test_summary_line() -> None:
    assert _summary_line("Done: created hello.py. Then it ran.") == "Done: created hello.py."
    assert _summary_line("完成。后面还有很多") == "完成。"
    assert _summary_line("") == ""
    assert _summary_line("   ") == ""
    assert len(_summary_line("x" * 500)) <= 200


def test_memory_carries_the_session(tmp_path: Path) -> None:
    """Every node gets the session through the layered memory's third layer."""
    from wqagent.core.session import append_assistant_turn
    from wqagent.graph.memory import build_layered_memory, read_session

    workspace = _workspace(tmp_path, "app/main.py")
    session = load_or_create_session(workspace)
    turn = append_user_turn(session, "帮我搭个 FastAPI 后台")
    append_assistant_turn(session, turn=turn, route="workflow", content="done", summary="skeleton")
    save_session(workspace, session)
    runtime = create_runtime(workspace, session=session)

    record = read_session(runtime)
    memory = build_layered_memory({"runtime": runtime, "task": "继续"}, node="planner")

    assert record["exists"] is True
    block = memory["history_summary_store"]
    assert block["session_exists"] is True
    assert "帮我搭个 FastAPI 后台" in block["session"]
    assert "skeleton" in block["session"]
    assert "app/main.py" in block["session"]
    assert len(block["session"]) <= SESSION_CHARS


def test_read_session_of_an_empty_workspace(tmp_path: Path) -> None:
    from wqagent.graph.memory import read_session

    workspace = _workspace(tmp_path)

    record = read_session(create_runtime(workspace))

    assert record["exists"] is False and record["content"] == "" and record["error"] == ""
