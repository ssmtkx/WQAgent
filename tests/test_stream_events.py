"""Tests for the agent entry point ``stream_agent_events`` (graph-driven)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from wqagent.core.agent import DEFAULT_MAX_LOOPS, execute_tool, stream_agent_events
from wqagent.core.state import RuntimeState
from wqagent.graph import build_workflow, new_graph_state
from wqagent.graph.state import make_todo


class ScriptedModel:
    """Replays scripted responses; records what was bound and received."""

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
            raise AssertionError("ScriptedModel ran out of scripted responses")
        return self.responses.pop(0)


def _tool_call(name: str, args: dict[str, Any], call_id: str = "c1") -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}]
    )


def _plan(**overrides: Any) -> AIMessage:
    plan: dict[str, Any] = {
        "plan_summary": "write hello.py then run it",
        "todos": [{"id": "t1", "content": "write hello.py"}],
        "acceptance_criteria": ["hello.py prints Hello"],
        "verification_commands": ["echo ok"],
    }
    plan.update(overrides)
    return _tool_call("todo_write", plan)


def _report(passed: bool = True, **overrides: Any) -> AIMessage:
    report: dict[str, Any] = {
        "passed": passed,
        "reason": "checked everything",
        "checks": [{"name": "hello.py prints Hello", "passed": passed, "detail": "saw it"}],
        "recommended_next_instruction": "" if passed else "fix hello.py",
    }
    report.update(overrides)
    return _tool_call("verification_report", report)


def _passing_script() -> list[Any]:
    return [
        _plan(),
        _tool_call("file_write", {"file_path": "hello.py", "content": "print('Hello')\n"}),
        AIMessage(content="Wrote hello.py."),
        AIMessage(content="Inspected the workspace."),
        _report(True),
        AIMessage(content="Done: hello.py works."),
    ]


def _kinds(events: list[dict[str, Any]]) -> list[str]:
    return [str(event["type"]) for event in events]


# -- the unified event stream ------------------------------------------------
def test_streams_the_whole_workflow(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    model = ScriptedModel(_passing_script())

    events = list(stream_agent_events("write hello.py", workspace=workspace, model=model))

    assert events[-1]["type"] == "final_answer"
    # the actor's own round marker must not surface as the run's final answer
    assert _kinds(events).count("final_answer") == 1
    assert [event["node"] for event in events if event["type"] == "node_end"] == [
        "planner",
        "actor",
        "verifier",
        "final",
    ]
    # tool events are tagged with the node that produced them
    tool_calls = [event for event in events if event["type"] == "tool_call"]
    assert {event["node"] for event in tool_calls} == {"actor"}
    assert [event["name"] for event in tool_calls] == ["file_write"]
    assert {event["node"] for event in events if event["type"] == "tool_result"} == {"actor"}

    final = events[-1]
    assert final["passed"] is True
    assert final["attempts"] == 1
    assert final["stopped_early"] is False
    assert final["content"] == "Done: hello.py works."
    assert (workspace / "hello.py").read_text(encoding="utf-8") == "print('Hello')\n"


def test_node_end_events_carry_each_nodes_update(tmp_path: Path) -> None:
    model = ScriptedModel(_passing_script())

    events = list(stream_agent_events("write hello.py", workspace=tmp_path / "ws", model=model))
    updates = {
        event["node"]: event["update"] for event in events if event["type"] == "node_end"
    }

    assert updates["planner"]["plan_summary"] == "write hello.py then run it"
    assert [todo["id"] for todo in updates["planner"]["todos"]] == ["t1"]
    assert updates["actor"]["last_actor_summary"] == "Wrote hello.py."
    assert updates["verifier"]["passed"] is True
    assert updates["verifier"]["verification_results"][0]["command"] == "echo ok"
    assert updates["final"]["final_answer"] == "Done: hello.py works."


def test_failing_verification_closes_with_a_failed_verdict(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            _plan(verification_commands=["exit 1"]),
            AIMessage(content="Tried."),
            AIMessage(content="Inspected."),
            _report(False),
            AIMessage(content="Not done: exit 1 still fails."),
        ]
    )

    events = list(
        stream_agent_events("do it", workspace=tmp_path / "ws", model=model, max_attempts=1)
    )

    final = events[-1]
    assert final["passed"] is False
    assert final["stopped_early"] is True
    assert final["attempts"] == 1
    assert final["content"] == "Not done: exit 1 still fails."
    verifier = next(
        event for event in events if event["type"] == "node_end" and event["node"] == "verifier"
    )
    assert verifier["update"]["passed"] is False
    assert "exit 1" in verifier["update"]["last_error"]


def test_retry_replans_before_the_second_attempt(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            _plan(verification_commands=["exit 1"]),
            AIMessage(content="First attempt."),
            AIMessage(content="Inspected."),
            _report(False),
            _plan(plan_summary="second try", verification_commands=["echo ok"]),
            AIMessage(content="Second attempt."),
            AIMessage(content="Inspected again."),
            _report(True),
            AIMessage(content="Done on the second attempt."),
        ]
    )

    events = list(
        stream_agent_events("do it", workspace=tmp_path / "ws", model=model, max_attempts=3)
    )

    nodes = [event["node"] for event in events if event["type"] == "node_end"]
    assert nodes == [
        "planner",
        "actor",
        "verifier",
        "planner",
        "actor",
        "verifier",
        "final",
    ]
    assert events[-1]["passed"] is True
    assert events[-1]["attempts"] == 2


def test_on_event_mirrors_the_stream(tmp_path: Path) -> None:
    seen: list[tuple[str, dict[str, Any]]] = []
    model = ScriptedModel(_passing_script())

    events = list(
        stream_agent_events(
            "write hello.py",
            workspace=tmp_path / "ws",
            model=model,
            on_event=lambda kind, payload: seen.append((kind, payload)),
        )
    )

    assert [kind for kind, _ in seen] == _kinds(events)
    assert seen[-1][0] == "final_answer"
    assert any(kind == "node_end" for kind, _ in seen)


def test_workspace_is_created_from_the_runtime(tmp_path: Path) -> None:
    workspace = tmp_path / "fresh" / "nested" / "ws"
    model = ScriptedModel(_passing_script())

    list(stream_agent_events("write hello.py", workspace=workspace, model=model))

    assert workspace.is_dir()


def test_state_override_wins_over_workspace(tmp_path: Path) -> None:
    state = RuntimeState(workspace=tmp_path / "from-state")
    model = ScriptedModel(_passing_script())

    list(
        stream_agent_events(
            "write hello.py", workspace=tmp_path / "ignored", state=state, model=model
        )
    )

    assert (tmp_path / "from-state" / "hello.py").exists()
    assert not (tmp_path / "ignored").exists()


def test_prebuilt_inputs_skip_planning(tmp_path: Path) -> None:
    state = RuntimeState(workspace=tmp_path / "ws")
    inputs = new_graph_state("do it", state, max_attempts=1)
    inputs["todos"] = [make_todo("t1", "do it")]
    model = ScriptedModel(
        [AIMessage(content="did it"), AIMessage(content="inspected"), _report(True), AIMessage(content="All good.")]
    )

    events = list(
        stream_agent_events("do it", workspace=tmp_path / "ignored", inputs=inputs, model=model)
    )

    assert events[-1]["passed"] is True
    # the planner kept the plan as-is: no model call for it
    assert all("todo_write" not in names for names in model.bound)
    assert state.workspace.is_dir()


def test_prebuilt_graph_is_reused(tmp_path: Path) -> None:
    model = ScriptedModel(_passing_script())
    app = build_workflow(model=model, actor_loops=4)

    events = list(stream_agent_events("write hello.py", workspace=tmp_path / "ws", graph=app))

    assert events[-1]["passed"] is True
    assert ["file_read", "file_write", "file_edit", "grep", "bash", "todo_update"] in model.bound


def test_actor_loop_budget_is_honoured(tmp_path: Path) -> None:
    script: list[Any] = [_plan()]
    script.extend(_tool_call("bash", {"command": "echo tick"}) for _ in range(4))
    script.extend([AIMessage(content="inspected"), _report(False), AIMessage(content="Not done.")])
    model = ScriptedModel(script)

    events = list(
        stream_agent_events(
            "do it", workspace=tmp_path / "ws", model=model, max_attempts=1, max_loops=3
        )
    )

    actor = next(
        event["update"]
        for event in events
        if event["type"] == "node_end" and event["node"] == "actor"
    )
    assert "Stopped after 3 steps" in actor["last_actor_summary"]
    assert events[-1]["passed"] is False


def test_missing_workspace_is_an_error() -> None:
    with pytest.raises(ValueError, match="workspace="):
        list(stream_agent_events("do it", model=ScriptedModel([])))


# -- engine level ------------------------------------------------------------
def test_default_loop_budget_is_ten() -> None:
    assert DEFAULT_MAX_LOOPS == 10


def test_execute_tool_contract() -> None:
    class Echo:
        name = "echo"

        def invoke(self, args: dict[str, Any]) -> str:
            return "Error: nope" if args.get("fail") else f"echo {args.get('text', '')}"

    tools = {"echo": Echo()}
    assert execute_tool("echo", {"text": "hi"}, tools) == {
        "ok": True,
        "tool": "echo",
        "result": "echo hi",
    }
    assert execute_tool("echo", {"fail": True}, tools) == {
        "ok": False,
        "tool": "echo",
        "error": "nope",
    }
    unknown = execute_tool("missing", {}, tools)
    assert unknown["ok"] is False
    assert "unknown tool 'missing'" in unknown["error"]
    assert json.dumps(unknown)  # always serialisable for the ToolMessage
