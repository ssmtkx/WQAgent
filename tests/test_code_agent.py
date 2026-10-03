"""Tests for the implementation specialist (``run_code_agent``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from wqagent.agents.code_agent import (
    DEFAULT_MAX_LOOPS,
    MemorySnapshot,
    build_code_messages,
    build_memory_snapshot,
    run_code_agent,
)
from wqagent.core.state import RuntimeState
from wqagent.graph import new_graph_state
from wqagent.graph.state import make_todo
from wqagent.prompts.stage3 import CODE_AGENT_PROMPT


class ScriptedModel:
    """Replays responses; records what was bound and what it received."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.bound: list[list[str]] = []
        self.seen: list[list[Any]] = []

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "ScriptedModel":
        self.bound.append([tool.name for tool in tools])
        return self

    def invoke(self, messages: list[Any]) -> Any:
        self.seen.append(list(messages))
        if not self.responses:
            raise AssertionError("ScriptedModel ran out of responses")
        return self.responses.pop(0)


class FailingModel(ScriptedModel):
    """Runs the first response, then fails - the agent must keep what it did."""

    def invoke(self, messages: list[Any]) -> Any:
        if not self.responses:
            raise RuntimeError("connection reset by peer")
        return super().invoke(messages)


def _tool_call(name: str, args: dict[str, Any], call_id: str = "c1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _todo_call(*updates: dict[str, Any], call_id: str = "c2") -> AIMessage:
    return _tool_call("todo_update", {"updates": list(updates)}, call_id)


@pytest.fixture()
def state(tmp_path: Path):
    return new_graph_state(
        "add a slugify helper",
        RuntimeState(workspace=tmp_path / "ws"),
        plan_summary="write utils.py, then test it",
        todos=[make_todo("t1", "write utils.py"), make_todo("t2", "run pytest")],
    )


# -- messages and memory -----------------------------------------------------
def test_messages_are_prompt_task_instruction_session_and_memory(state: Any) -> None:
    state["research_notes"] = "slugify examples: https://example.com/slugify"
    state["last_error"] = "pytest failed"

    messages = build_code_messages(state, "t2: run the tests")

    assert messages[0].content == CODE_AGENT_PROMPT
    human = messages[1].content
    assert "add a slugify helper" in human
    assert "t2: run the tests" in human
    assert "Session context:" in human
    assert "Plan progress - 0/2 todos completed" in human
    assert "[ ] t1: write utils.py" in human
    assert "write utils.py, then test it" in human
    assert "pytest failed" in human
    assert "Memory notes:" in human
    assert "https://example.com/slugify" in human


def test_memory_snapshot_layers(state: Any) -> None:
    state["research_notes"] = "found in docs"

    snapshot = build_memory_snapshot(state)

    assert snapshot.notes == "found in docs"
    assert "Plan progress" in snapshot.session and "Last verification error" not in snapshot.session
    assert snapshot.render().index("Session context:") < snapshot.render().index("Memory notes:")
    assert bool(snapshot) is True
    assert bool(MemorySnapshot()) is False


def test_memory_override_wins(state: Any) -> None:
    state["research_notes"] = "from state"

    explicit = build_memory_snapshot(state, MemorySnapshot(session="s", notes="explicit"))
    assert explicit.session == "s" and explicit.notes == "explicit"

    text = build_memory_snapshot(state, "just a string")
    assert text.notes == "just a string" and "Plan progress" in text.session


def test_messages_survive_a_missing_state_and_instruction() -> None:
    messages = build_code_messages(None, "")

    assert messages[0].content == CODE_AGENT_PROMPT
    assert "(no task given)" in messages[1].content


# -- the happy path ----------------------------------------------------------
def test_runs_tools_updates_todos_and_summarises(state: Any) -> None:
    workspace = state["runtime"].workspace
    model = ScriptedModel(
        [
            _todo_call({"id": "t1", "status": "in_progress"}),
            _tool_call(
                "file_write",
                {"file_path": "utils.py", "content": "def slugify(s):\n    return s\n"},
                "c3",
            ),
            _todo_call({"id": "t1", "status": "completed", "note": "written"}, call_id="c4"),
            AIMessage(content="Added utils.slugify and checked it imports."),
        ]
    )
    persisted: list[list[dict[str, Any]]] = []
    events: list[dict[str, Any]] = []

    result = run_code_agent(
        state,
        "t1: write utils.py",
        model=model,
        writer=events.append,
        persist=persisted.append,
    )

    assert result["ok"] is True and result["error"] == ""
    assert result["summary"] == "Added utils.slugify and checked it imports."
    assert (workspace / "utils.py").read_text(encoding="utf-8").startswith("def slugify")

    # the plan advanced exactly as the model asked
    assert [(todo["id"], todo["status"]) for todo in result["todos"]] == [
        ("t1", "completed"),
        ("t2", "pending"),
    ]
    assert result["todos"][0]["note"] == "written"
    # the state itself was not mutated
    assert state["todos"][0]["status"] == "pending"

    # persistence hook fired after every applied update, with the full list
    assert [len(snapshot) for snapshot in persisted] == [2, 2]
    assert persisted[-1][0]["status"] == "completed"

    # the workspace tools plus todo_update were bound
    assert model.bound == [
        ["file_read", "file_write", "file_edit", "grep", "bash", "todo_update"]
    ]

    # events: one tool_call + one tool_result each, with todo_update in between
    assert [event["type"] for event in events] == [
        "tool_call",
        "todo_update",
        "tool_result",
        "tool_call",
        "tool_result",
        "tool_call",
        "todo_update",
        "tool_result",
    ]
    assert result["tool_events"] == events
    assert events[1]["todos"][0]["status"] == "in_progress"
    assert events[0]["name"] == "todo_update"


def test_bad_todo_updates_are_reported_not_fatal(state: Any) -> None:
    model = ScriptedModel(
        [
            _todo_call({"id": "ghost", "status": "completed"}, {"id": "t1", "status": "nope"}),
            AIMessage(content="Tried to update the plan."),
        ]
    )

    result = run_code_agent(state, "t1", model=model)

    assert result["ok"] is True
    assert [todo["status"] for todo in result["todos"]] == ["pending", "pending"]
    update_event = next(e for e in result["tool_events"] if e["type"] == "todo_update")
    assert any("unknown todo id 'ghost'" in note for note in update_event["notes"])
    assert any("unknown status" in note for note in update_event["notes"])


# -- failure paths -----------------------------------------------------------
def test_missing_runtime_is_reported(state: Any) -> None:
    broken = {"task": "x", "todos": [make_todo("t1", "do it")]}

    result = run_code_agent(broken, "t1")

    assert result["ok"] is False
    assert "runtime" in result["error"]
    assert [todo["id"] for todo in result["todos"]] == ["t1"]
    assert result["tool_events"] == []
    assert result["messages"][0].content == CODE_AGENT_PROMPT


def test_model_failure_keeps_partial_progress(state: Any) -> None:
    workspace = state["runtime"].workspace
    model = FailingModel(
        [
            _tool_call("file_write", {"file_path": "half.py", "content": "print(1)\n"}),
        ]
    )

    result = run_code_agent(state, "write half.py", model=model)

    assert result["ok"] is False
    assert "connection reset by peer" in result["error"]
    assert (workspace / "half.py").is_file()  # the work already done is kept
    assert [event["type"] for event in result["tool_events"]] == ["tool_call", "tool_result"]
    assert result["messages"]  # the transcript is returned too


def test_max_loops_falls_back_to_a_digest(state: Any) -> None:
    model = ScriptedModel(
        [_tool_call("bash", {"command": "pytest -q"}, f"c{i}") for i in range(5)]
    )

    result = run_code_agent(state, "run the tests", model=model, max_loops=2)

    assert len([e for e in result["tool_events"] if e["type"] == "tool_call"]) == 2
    assert result["ok"] is True
    assert "Checks run: 2" in result["summary"]
    assert "0/2 todos completed" in result["summary"]


def test_agent_that_only_answers(state: Any) -> None:
    model = ScriptedModel([AIMessage(content="Nothing to do.")])

    result = run_code_agent(state, "look around", model=model)

    assert result["ok"] is True
    assert result["summary"] == "Nothing to do."
    assert result["tool_events"] == []


def test_tool_override_is_honoured(state: Any) -> None:
    class Stub:
        name = "stub"

        def as_tool(self) -> Any:
            from langchain_core.tools import StructuredTool

            return StructuredTool.from_function(
                func=lambda: "ok", name="stub", description="stub"
            )

    model = ScriptedModel([AIMessage(content="used the stub")])

    result = run_code_agent(state, "x", model=model, tools=[Stub().as_tool()])

    assert result["ok"] is True
    assert model.bound == [["stub"]]


def test_default_budget_is_ten() -> None:
    assert DEFAULT_MAX_LOOPS == 10
