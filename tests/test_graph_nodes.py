"""Tests for the graph nodes: planner, actor, verifier and the router."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from wqagent.core.state import RuntimeState
from wqagent.graph.nodes import (
    actor_node,
    get_model,
    planner_node,
    run_verification_commands,
    set_model,
    verifier_node,
    verifier_route,
)
from wqagent.graph.state import make_todo, new_graph_state
from wqagent.prompts.stage1 import ACTOR_PROMPT
from wqagent.prompts.stage2 import PLANNER_PROMPT, VERIFIER_PROMPT


class ScriptedModel:
    """Replays scripted responses; records what was bound and what it received."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.bound: list[tuple[list[str], Any]] = []
        self.seen: list[list[Any]] = []

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "ScriptedModel":
        self.bound.append(([tool.name for tool in tools], kwargs.get("tool_choice")))
        return self

    def invoke(self, messages: list[Any]) -> Any:
        self.seen.append(list(messages))
        if not self.responses:
            raise AssertionError("ScriptedModel ran out of scripted responses")
        return self.responses.pop(0)


class JsonModel:
    """Always answers with plain JSON content (no tool calls)."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.seen: list[list[Any]] = []

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "JsonModel":
        return self

    def invoke(self, messages: list[Any]) -> Any:
        self.seen.append(list(messages))
        return AIMessage(content=json.dumps(self.payload))


def _tool_call(name: str, args: dict[str, Any], call_id: str = "c1") -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}]
    )


def _state(tmp_path: Path, **extra: Any):
    return new_graph_state("write hello.py", RuntimeState(workspace=tmp_path / "ws"), **extra)


@pytest.fixture(autouse=True)
def _reset_module_model():
    """Keep the module-level model from leaking between tests."""
    set_model(None)
    yield
    set_model(None)


# -- planner -----------------------------------------------------------------
def test_planner_creates_the_plan(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            _tool_call(
                "todo_write",
                {
                    "plan_summary": "write the file, then run it",
                    "todos": [
                        {"id": "t1", "content": "write hello.py"},
                        {"id": "t2", "content": "run it", "note": "python hello.py"},
                    ],
                    "acceptance_criteria": ["hello.py prints Hello"],
                    "verification_commands": ["python hello.py"],
                },
            )
        ]
    )

    update = planner_node(_state(tmp_path), model=model)

    assert update["plan_summary"] == "write the file, then run it"
    assert [todo["id"] for todo in update["todos"]] == ["t1", "t2"]
    assert [todo["status"] for todo in update["todos"]] == ["pending", "pending"]
    assert update["todos"][1]["note"] == "python hello.py"
    assert update["acceptance_criteria"] == ["hello.py prints Hello"]
    assert update["verification_commands"] == ["python hello.py"]

    tools_bound, tool_choice = model.bound[0]
    assert tools_bound == ["todo_write"]
    assert tool_choice == "todo_write"  # forced structured output
    sent = model.seen[0]
    assert sent[0].content == PLANNER_PROMPT
    assert "write hello.py" in sent[1].content


def test_planner_revises_the_plan_after_a_failure(tmp_path: Path) -> None:
    state = _state(
        tmp_path,
        todos=[make_todo("t1", "write hello.py", status="completed"), make_todo("t2", "run it")],
        last_error="command failed: python hello.py (exit code 1)\n  NameError: name 'hello' is not defined",
    )
    model = ScriptedModel(
        [
            _tool_call(
                "todo_write",
                {
                    "plan_summary": "fix the NameError, then rerun",
                    "todos": [
                        {"id": "t1", "content": "write hello.py", "status": "completed"},
                        {"id": "t2", "content": "run it", "status": "pending"},
                        {"id": "t3", "content": "fix the NameError in hello.py"},
                    ],
                    "acceptance_criteria": ["hello.py prints Hello"],
                    "verification_commands": ["python hello.py"],
                },
            )
        ]
    )

    update = planner_node(state, model=model)

    prompt = model.seen[0][1].content
    assert "FAILED" in prompt
    assert "exit code 1" in prompt and "NameError" in prompt
    assert "[x] t1" in prompt  # the completed todo is shown as completed
    assert [todo["id"] for todo in update["todos"]] == ["t1", "t2", "t3"]
    assert update["todos"][0]["status"] == "completed"


def test_planner_keeps_an_existing_plan_without_calling_the_model(tmp_path: Path) -> None:
    state = _state(
        tmp_path,
        plan_summary="already planned",
        todos=[make_todo("t1", "do it")],
        acceptance_criteria=["it works"],
        verification_commands=["echo ok"],
    )
    model = ScriptedModel([])  # would raise if invoked

    update = planner_node(state, model=model)

    assert update["plan_summary"] == "already planned"
    assert update["todos"][0]["id"] == "t1"
    assert update["acceptance_criteria"] == ["it works"]
    assert model.seen == [] and model.bound == []


def test_planner_falls_back_to_json_content(tmp_path: Path) -> None:
    model = JsonModel(
        {
            "plan_summary": "json plan",
            "todos": [{"id": "t1", "content": "do it"}],
            "acceptance_criteria": ["ok"],
            "verification_commands": ["echo ok"],
        }
    )

    update = planner_node(_state(tmp_path), model=model)

    assert update["plan_summary"] == "json plan"
    assert update["todos"][0]["content"] == "do it"
    assert update["verification_commands"] == ["echo ok"]


def test_planner_keeps_the_previous_todos_when_the_model_answers_nothing(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path, todos=[make_todo("t1", "keep me")], last_error="boom")
    model = ScriptedModel([AIMessage(content="I have no idea")])

    update = planner_node(state, model=model)

    assert [todo["id"] for todo in update["todos"]] == ["t1"]


def test_planner_requires_a_runtime() -> None:
    with pytest.raises(ValueError, match="runtime"):
        planner_node({"task": "x"})  # type: ignore[arg-type]


# -- actor -------------------------------------------------------------------
def test_actor_executes_the_plan_and_records_todo_updates(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    state = new_graph_state(
        "write hello.py",
        RuntimeState(workspace=workspace),
        plan_summary="write, then run",
        todos=[make_todo("t1", "write hello.py"), make_todo("t2", "run it")],
        acceptance_criteria=["prints Hello"],
        verification_commands=["echo done"],
    )
    model = ScriptedModel(
        [
            _tool_call("file_write", {"file_path": "hello.py", "content": "print('Hello')\n"}),
            _tool_call(
                "todo_update",
                {"updates": [{"id": "t1", "status": "completed", "note": "written"}]},
            ),
            AIMessage(content="Wrote hello.py and ran it."),
        ]
    )
    events: list[dict[str, Any]] = []

    update = actor_node(state, model=model, on_event=events.append)

    assert (workspace / "hello.py").read_text(encoding="utf-8") == "print('Hello')\n"
    assert update["last_actor_summary"] == "Wrote hello.py and ran it."
    assert {todo["id"]: todo["status"] for todo in update["todos"]} == {
        "t1": "completed",
        "t2": "pending",
    }
    assert update["todos"][0]["note"] == "written"

    # only this node's contribution goes back into the graph transcript
    kinds = [message.type for message in update["messages"]]
    assert kinds[0] == "human" and "system" not in kinds
    assert any(isinstance(message, ToolMessage) for message in update["messages"])

    # the actor bound the five workspace tools plus todo_update
    names, _ = model.bound[0]
    assert names == ["file_read", "file_write", "file_edit", "grep", "bash", "todo_update"]
    assert model.seen[0][0].content == ACTOR_PROMPT

    # every ReAct event was forwarded
    assert [event["type"] for event in events][0] == "ai_message"
    assert [event["name"] for event in events if event["type"] == "tool_call"] == [
        "file_write",
        "todo_update",
    ]
    assert "execute the plan" in model.seen[0][2 - 1].content.lower()


def test_actor_replays_the_transcript_but_returns_only_the_delta(tmp_path: Path) -> None:
    state = _state(
        tmp_path,
        messages=[HumanMessage(content="first attempt"), AIMessage(content="did something")],
    )
    model = ScriptedModel([AIMessage(content="done now")])

    update = actor_node(state, model=model)

    sent = model.seen[0]
    assert sent[0].content == ACTOR_PROMPT
    assert [message.content for message in sent[1:3]] == ["first attempt", "did something"]
    assert [message.type for message in update["messages"]] == ["human", "ai"]
    assert update["messages"][-1].content == "done now"


def test_actor_reports_the_step_limit_in_the_summary(tmp_path: Path) -> None:
    model = ScriptedModel([_tool_call("bash", {"command": "echo tick"}) for _ in range(3)])

    update = actor_node(_state(tmp_path), model=model, max_loops=3)

    assert "Stopped after 3 steps" in update["last_actor_summary"]


def test_actor_ignores_updates_for_unknown_todos(tmp_path: Path) -> None:
    state = _state(tmp_path, todos=[make_todo("t1", "do it")])
    model = ScriptedModel(
        [
            _tool_call("todo_update", {"updates": [{"id": "ghost", "status": "completed"}]}),
            AIMessage(content="done"),
        ]
    )

    update = actor_node(state, model=model)

    assert [todo["id"] for todo in update["todos"]] == ["t1"]
    assert update["todos"][0]["status"] == "pending"
    assert "unknown todo id 'ghost'" in update["last_actor_summary"]


# -- verifier ----------------------------------------------------------------
def test_verifier_passes_and_marks_the_plan_completed(tmp_path: Path) -> None:
    state = _state(
        tmp_path,
        plan_summary="write, then run",
        todos=[make_todo("t1", "write hello.py"), make_todo("t2", "run it")],
        acceptance_criteria=["prints Hello"],
        verification_commands=["echo ok"],
    )
    model = ScriptedModel(
        [
            _tool_call("file_read", {"file_path": "hello.py"}),  # read-only inspection
            AIMessage(content="I read the file."),
            _tool_call(
                "verification_report",
                {
                    "passed": True,
                    "reason": "the file exists and the command succeeded",
                    "checks": [{"name": "prints Hello", "passed": True, "detail": "saw print"}],
                    "recommended_next_instruction": "",
                },
            ),
        ]
    )

    update = verifier_node(state, model=model)

    assert update["passed"] is True
    assert update["attempts"] == 1
    assert update["last_error"] == ""
    assert [todo["status"] for todo in update["todos"]] == ["completed", "completed"]
    assert update["verification_results"][0]["command"] == "echo ok"
    assert update["verification_results"][0]["ok"] is True
    assert update["verification_checks"][0]["name"] == "prints Hello"

    # the inspection agent only got read-only tools...
    assert model.bound[0][0] == ["file_read", "grep"]
    # ...and the verdict came from a forced verification_report call
    assert model.bound[-1] == (["verification_report"], "verification_report")
    assert model.seen[0][0].content == VERIFIER_PROMPT
    assert "exit_code=0" in model.seen[0][1].content


def test_verifier_overrules_a_pass_when_a_command_fails(tmp_path: Path) -> None:
    state = _state(tmp_path, todos=[make_todo("t1", "do it")], verification_commands=["exit 3"])
    model = ScriptedModel(
        [
            AIMessage(content="looks fine to me"),
            _tool_call(
                "verification_report",
                {
                    "passed": True,
                    "reason": "the actor said it is done",
                    "checks": [],
                    "recommended_next_instruction": "",
                },
            ),
        ]
    )

    update = verifier_node(state, model=model)

    assert update["passed"] is False
    assert "overruled" in update["last_error"]
    assert update["verification_results"][0]["exit_code"] == 3
    assert update["verification_results"][0]["ok"] is False
    assert [todo["status"] for todo in update["todos"]] == ["pending"]


def test_verifier_records_the_next_instruction_on_failure(tmp_path: Path) -> None:
    state = _state(tmp_path, verification_commands=["echo boom; exit 1"])
    model = ScriptedModel(
        [
            AIMessage(content="inspecting"),
            _tool_call(
                "verification_report",
                {
                    "passed": False,
                    "reason": "the command failed",
                    "checks": [{"name": "runs cleanly", "passed": False, "detail": "exit 1"}],
                    "recommended_next_instruction": "fix the import error in hello.py",
                },
            ),
        ]
    )

    update = verifier_node(state, model=model)

    assert update["passed"] is False
    assert "the command failed" in update["last_error"]
    assert "command failed: echo boom; exit 1 (exit code 1)" in update["last_error"]
    assert "Recommended next instruction: fix the import error in hello.py" in update["last_error"]
    assert update["verification_checks"][0]["passed"] is False
    assert update["attempts"] == 1


def test_verifier_fails_safely_without_a_verdict(tmp_path: Path) -> None:
    model = ScriptedModel(
        [AIMessage(content="not sure"), AIMessage(content="still not sure"), AIMessage(content="nope")]
    )

    update = verifier_node(_state(tmp_path), model=model)

    assert update["passed"] is False
    assert "no verdict" in update["last_error"]


def test_run_verification_commands_records_outcomes(tmp_path: Path) -> None:
    runtime = RuntimeState(workspace=tmp_path / "ws")

    results = run_verification_commands(
        runtime, ["echo hi", "exit 2", "   ", "rm -rf /"]
    )

    assert len(results) == 3  # blanks are skipped
    assert results[0]["ok"] is True and results[0]["exit_code"] == 0
    assert "hi" in results[0]["stdout"]
    assert results[1]["ok"] is False and results[1]["exit_code"] == 2
    assert results[2]["ok"] is False and "safety filter" in results[2]["stderr"]


# -- routing -----------------------------------------------------------------
def test_verifier_route() -> None:
    assert verifier_route({"passed": True, "attempts": 1, "max_attempts": 3}) == "final"
    assert verifier_route({"passed": False, "attempts": 1, "max_attempts": 3}) == "planner"
    assert verifier_route({"passed": False, "attempts": 3, "max_attempts": 3}) == "final"
    assert verifier_route({"passed": False, "attempts": 9, "max_attempts": 3}) == "final"
    assert verifier_route({"passed": False}) == "planner"  # attempts defaults to 0 < 3
    assert verifier_route({}) == "planner"


# -- model plumbing ----------------------------------------------------------
def test_get_model_is_lazy_and_cached(tmp_path: Path, monkeypatch: Any) -> None:
    created: list[str] = []
    model = ScriptedModel([AIMessage(content="ok")])

    def fake_create_model(**kwargs: Any) -> ScriptedModel:
        created.append("called")
        return model

    monkeypatch.setattr("wqagent.graph.nodes.create_model", fake_create_model)

    first = get_model()
    second = get_model()

    assert first is second is model
    assert created == ["called"]  # created once

    sentinel = object()
    set_model(sentinel)
    assert get_model() is sentinel


# -- the whole loop, wired into a real graph ---------------------------------
def test_graph_retries_after_a_failed_verification(tmp_path: Path) -> None:
    """planner -> actor -> verifier (fails) -> planner -> actor -> verifier (passes) -> END."""
    from langgraph.graph import END, START, StateGraph

    from wqagent.graph import WQGraphState

    workspace = tmp_path / "ws"
    runtime = RuntimeState(workspace=workspace)
    model = ScriptedModel(
        [
            # attempt 1: plan whose verification command fails
            _tool_call(
                "todo_write",
                {
                    "plan_summary": "write hello.py and run it",
                    "todos": [{"id": "t1", "content": "write hello.py"}],
                    "acceptance_criteria": ["hello.py runs cleanly"],
                    "verification_commands": ["exit 1"],
                },
            ),
            _tool_call("file_write", {"file_path": "hello.py", "content": "print('Hello')\n"}),
            AIMessage(content="Wrote hello.py."),
            AIMessage(content="Inspected the file, it looks right."),
            _tool_call(
                "verification_report",
                {
                    "passed": False,
                    "reason": "the verification command exited non-zero",
                    "checks": [
                        {"name": "hello.py runs cleanly", "passed": False, "detail": "exit 1"}
                    ],
                    "recommended_next_instruction": "re-run the program and make it exit 0",
                },
            ),
            # attempt 2: revised plan with a command that passes
            _tool_call(
                "todo_write",
                {
                    "plan_summary": "verify by actually running the program",
                    "todos": [{"id": "t1", "content": "write hello.py", "status": "completed"}],
                    "acceptance_criteria": ["hello.py runs cleanly"],
                    "verification_commands": ["echo ok"],
                },
            ),
            AIMessage(content="Re-ran the program, it works."),
            AIMessage(content="Verified the file and the command output."),
            _tool_call(
                "verification_report",
                {
                    "passed": True,
                    "reason": "the program prints Hello and the command succeeded",
                    "checks": [{"name": "hello.py runs cleanly", "passed": True, "detail": "exit 0"}],
                    "recommended_next_instruction": "",
                },
            ),
        ]
    )
    set_model(model)

    graph = StateGraph(WQGraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("actor", actor_node)
    graph.add_node("verifier", verifier_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "actor")
    graph.add_edge("actor", "verifier")
    graph.add_conditional_edges(
        "verifier", verifier_route, {"planner": "planner", "final": END}
    )

    result = graph.compile().invoke(new_graph_state("write hello.py", runtime, max_attempts=3))

    assert result["passed"] is True
    assert result["attempts"] == 2
    assert result["last_error"] == ""
    assert [todo["status"] for todo in result["todos"]] == ["completed"]
    # the last verification run replaced the failing one (last-value-wins channel)
    assert [check["ok"] for check in result["verification_results"]] == [True]
    assert result["verification_results"][0]["command"] == "echo ok"
    assert (workspace / "hello.py").read_text(encoding="utf-8") == "print('Hello')\n"
    # both attempts landed in the transcript, merged by add_messages
    assert [message.type for message in result["messages"]] == [
        "human",
        "ai",
        "tool",
        "ai",
        "human",
        "ai",
    ]
    # exactly the nine model calls the two attempts need - no extras
    assert len(model.seen) == 9
    assert model.responses == []


def test_graph_gives_up_after_max_attempts(tmp_path: Path) -> None:
    """A verifier that always fails must not loop forever."""
    from langgraph.graph import END, START, StateGraph

    from wqagent.graph import WQGraphState

    runtime = RuntimeState(workspace=tmp_path / "ws")
    responses: list[Any] = []
    for attempt in range(2):  # max_attempts=2 -> two planning rounds
        responses.extend(
            [
                _tool_call(
                    "todo_write",
                    {
                        "plan_summary": f"attempt {attempt + 1}",
                        "todos": [{"id": "t1", "content": "do the thing"}],
                        "acceptance_criteria": ["it works"],
                        "verification_commands": ["exit 1"],
                    },
                ),
                AIMessage(content=f"actor attempt {attempt + 1}"),
                AIMessage(content="inspected"),
                _tool_call(
                    "verification_report",
                    {
                        "passed": False,
                        "reason": "still failing",
                        "checks": [],
                        "recommended_next_instruction": "try again",
                    },
                ),
            ]
        )
    model = ScriptedModel(responses)
    set_model(model)

    graph = StateGraph(WQGraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("actor", actor_node)
    graph.add_node("verifier", verifier_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "actor")
    graph.add_edge("actor", "verifier")
    graph.add_conditional_edges(
        "verifier", verifier_route, {"planner": "planner", "final": END}
    )

    result = graph.compile().invoke(new_graph_state("do the thing", runtime, max_attempts=2))

    assert result["passed"] is False
    assert result["attempts"] == 2  # stopped exactly at the budget
    assert "still failing" in result["last_error"]
    assert len(model.seen) == 8
    assert model.responses == []
