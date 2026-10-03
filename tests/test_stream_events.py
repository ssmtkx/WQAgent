"""Tests for the agent entry point ``stream_agent_events`` (graph-driven)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from helpers import (
    ScriptedModel,
    answer,
    code_call,
    passing_script,
    plan_call,
    report_call,
    tool_call,
)

from wqagent.core.agent import DEFAULT_MAX_LOOPS, execute_tool, stream_agent_events
from wqagent.core.state import RuntimeState
from wqagent.graph import build_workflow, new_graph_state
from wqagent.graph.state import make_todo


def _kinds(events: list[dict[str, Any]]) -> list[str]:
    return [str(event["type"]) for event in events]


def _by_type(events: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [event for event in events if event["type"] == kind]


# -- the unified event stream ------------------------------------------------
def test_streams_the_whole_workflow(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    model = ScriptedModel(passing_script(final_text="Done: hello.py works."))

    events = list(stream_agent_events("write hello.py", workspace=workspace, model=model))

    assert events[-1]["type"] == "final_answer"
    # only the `final` node closes the run - exactly one final_answer
    assert _kinds(events).count("final_answer") == 1

    # the graph nodes reported themselves, in order, and no actor is involved;
    # every hop between planning and verification goes through the context monitor
    assert [event["node"] for event in _by_type(events, "node_end")] == [
        "planner",
        "context_monitor",
        "verifier",
        "context_monitor",
        "final",
    ]

    # the delegation is streamed as a handoff, and the specialist's work is tagged
    assert [(event["from"], event["to"]) for event in _by_type(events, "handoff")] == [
        ("planner", "codeAgent")
    ]
    assert {event["node"] for event in _by_type(events, "tool_call")} == {
        "planner",
        "codeAgent",
    }
    assert [
        event["name"]
        for event in _by_type(events, "tool_call")
        if event["node"] == "codeAgent"
    ] == ["file_write"]

    final = events[-1]
    assert final["passed"] is True
    assert final["attempts"] == 1
    assert final["stopped_early"] is False
    assert final["content"] == "Done: hello.py works."
    assert (workspace / "hello.py").read_text(encoding="utf-8") == "print('Hello')\n"


def test_node_end_events_carry_each_nodes_update(tmp_path: Path) -> None:
    model = ScriptedModel(passing_script())

    events = list(stream_agent_events("write hello.py", workspace=tmp_path / "ws", model=model))
    updates = {
        event["node"]: event["update"] for event in events if event["type"] == "node_end"
    }

    assert updates["planner"]["plan_summary"] == "do the thing"
    assert [todo["id"] for todo in updates["planner"]["todos"]] == ["t1"]
    assert [h["to_agent"] for h in updates["planner"]["agent_handoffs"]] == ["codeAgent"]
    assert updates["planner"]["code_agent_summary"] == "Wrote hello.py."
    assert updates["verifier"]["passed"] is True
    assert updates["verifier"]["verification_results"][0]["command"] == "echo ok"
    assert updates["final"]["final_answer"].startswith("Done:")


def test_research_handoff_lands_in_the_state(tmp_path: Path) -> None:
    from wqagent.tools.web_search_tool import WebSearchTool

    class FakeSearchTool(WebSearchTool):
        def __init__(self) -> None:
            super().__init__(api_key="fake")

        def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
            return {
                "ok": True,
                "query": query,
                "answer": "0.115 is current.",
                "results": [
                    {"title": "FastAPI", "url": "https://fastapi.tiangolo.com/", "content": "d", "score": 0.9}
                ],
                "error": "",
            }

    model = ScriptedModel(
        [
            plan_call(summary="research then build"),
            tool_call("call_search_agent", {"instruction": "which version?"}, "c2"),
            tool_call("web_search", {"query": "which version?"}, "s1"),
            answer("0.115 is current."),
            answer("Researched it."),
            answer("Inspected."),
            report_call(True),
            answer("Done."),
        ]
    )

    # the search tool is injected into the planner node directly
    from functools import partial

    from langgraph.graph import END, START, StateGraph

    from wqagent.graph import WQGraphState, final_node, planner_node, verifier_node, verifier_route

    graph = StateGraph(WQGraphState)
    graph.add_node("planner", partial(planner_node, model=model, search_tool=FakeSearchTool()))
    graph.add_node("verifier", partial(verifier_node, model=model))
    graph.add_node("final", partial(final_node, model=model))
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "verifier")
    graph.add_conditional_edges(
        "verifier", verifier_route, {"final": "final", "planner": "planner"}
    )
    graph.add_edge("final", END)

    events = list(
        stream_agent_events("pin fastapi", workspace=tmp_path / "ws", graph=graph.compile())
    )

    planner_update = next(
        event["update"] for event in events if event["type"] == "node_end" and event["node"] == "planner"
    )
    assert "0.115 is current." in planner_update["research_notes"]
    assert [source["url"] for source in planner_update["sources"]] == [
        "https://fastapi.tiangolo.com/"
    ]
    assert [(event["node"], event["name"]) for event in _by_type(events, "tool_call")] == [
        ("planner", "todo_write"),
        ("planner", "call_search_agent"),
        ("searchAgent", "web_search"),
    ]


def test_failing_verification_closes_with_a_failed_verdict(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            plan_call(commands=["exit 1"]),
            answer("Planned it."),
            answer("Inspected."),
            report_call(False),
            answer("Not done: exit 1 still fails."),
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
            # round 1: plan + delegate, verification fails
            plan_call(commands=["exit 1"]),
            code_call("first attempt"),
            answer("Wrote it."),
            answer("Delegated."),
            answer("Inspected."),
            report_call(False),
            # round 2: revised plan, verification passes
            plan_call(summary="second try", commands=["echo ok"]),
            code_call("the missing fix"),
            answer("Fixed it."),
            answer("Delegated the fix."),
            answer("Inspected again."),
            report_call(True),
            # final
            answer("Done on the second attempt."),
        ]
    )

    events = list(
        stream_agent_events("do it", workspace=tmp_path / "ws", model=model, max_attempts=3)
    )

    assert [event["node"] for event in events if event["type"] == "node_end"] == [
        "planner",
        "context_monitor",
        "verifier",
        "context_monitor",
        "planner",
        "context_monitor",
        "verifier",
        "context_monitor",
        "final",
    ]
    assert [event["to"] for event in _by_type(events, "handoff")] == ["codeAgent", "codeAgent"]
    assert events[-1]["passed"] is True
    assert events[-1]["attempts"] == 2


def test_on_event_mirrors_the_stream(tmp_path: Path) -> None:
    seen: list[tuple[str, dict[str, Any]]] = []
    model = ScriptedModel(passing_script())

    events = list(
        stream_agent_events(
            "write hello.py",
            workspace=tmp_path / "ws",
            model=model,
            on_event=lambda kind, payload: seen.append((kind, payload)),
        )
    )

    assert [kind for kind, _ in seen] == _kinds(events)
    # the payload is the event without its "type" key (that is the first argument)
    assert [dict(payload, type=kind) for kind, payload in seen] == events
    assert seen[-1][0] == "final_answer"
    assert any(kind == "handoff" for kind, _ in seen)
    assert any(kind == "node_end" for kind, _ in seen)


def test_workspace_is_created_from_the_runtime(tmp_path: Path) -> None:
    workspace = tmp_path / "fresh" / "nested" / "ws"
    model = ScriptedModel(passing_script())

    list(stream_agent_events("write hello.py", workspace=workspace, model=model))

    assert workspace.is_dir()


def test_state_override_wins_over_workspace(tmp_path: Path) -> None:
    state = RuntimeState(workspace=tmp_path / "from-state")
    model = ScriptedModel(passing_script())

    list(
        stream_agent_events(
            "write hello.py", workspace=tmp_path / "ignored", state=state, model=model
        )
    )

    assert (tmp_path / "from-state" / "hello.py").exists()
    assert not (tmp_path / "ignored").exists()


def test_prebuilt_inputs_are_used(tmp_path: Path) -> None:
    state = RuntimeState(workspace=tmp_path / "ws")
    inputs = new_graph_state("do it", state, max_attempts=1)
    inputs["todos"] = [make_todo("t1", "prebuilt step")]
    inputs["plan_summary"] = "prebuilt plan"
    model = ScriptedModel(
        [
            answer("Nothing more to plan."),
            answer("Inspected."),
            report_call(True),
            answer("All good."),
        ]
    )

    events = list(
        stream_agent_events("do it", workspace=tmp_path / "ignored", inputs=inputs, model=model)
    )

    assert events[-1]["passed"] is True
    # the prebuilt plan was handed to the planner instead of being regenerated
    planner_prompt = model.seen[0][1].content
    assert "prebuilt plan" in planner_prompt
    assert "[ ] t1: prebuilt step" in planner_prompt
    assert state.workspace.is_dir()


def test_prebuilt_graph_is_reused(tmp_path: Path) -> None:
    model = ScriptedModel(passing_script())
    app = build_workflow(model=model, planner_loops=4)

    events = list(stream_agent_events("write hello.py", workspace=tmp_path / "ws", graph=app))

    assert events[-1]["passed"] is True
    assert model.bound[0] == ["todo_write", "call_search_agent", "call_code_agent"]
    assert ["file_read", "grep"] in model.bound  # the verifier's read-only tools


def test_planner_loop_budget_is_honoured(tmp_path: Path) -> None:
    script: list[Any] = [
        tool_call("todo_write", {"plan_summary": f"round {i}"}, f"c{i}") for i in range(4)
    ]
    script.extend([answer("Inspected."), report_call(False), answer("Not done.")])
    model = ScriptedModel(script)

    events = list(
        stream_agent_events(
            "do it", workspace=tmp_path / "ws", model=model, max_attempts=1, max_loops=3
        )
    )

    planner = next(
        event["update"]
        for event in events
        if event["type"] == "node_end" and event["node"] == "planner"
    )
    assert "Stopped after 3 steps" in planner["last_actor_summary"]
    assert events[-1]["passed"] is False


def test_missing_workspace_is_an_error() -> None:
    with pytest.raises(ValueError, match="workspace="):
        list(stream_agent_events("do it", model=ScriptedModel([])))


def test_complex_workflow_streams_through_the_monitor(tmp_path: Path) -> None:
    """The context-managed graph works through the same entry point."""
    from wqagent.graph import build_complex_workflow

    model = ScriptedModel(passing_script())
    app = build_complex_workflow(model=model)

    events = list(stream_agent_events("write hello.py", workspace=tmp_path / "ws", graph=app))

    assert [event["node"] for event in _by_type(events, "node_end")] == [
        "planner",
        "context_monitor",
        "verifier",
        "context_monitor",
        "final",
    ]
    monitor = next(
        event["update"]
        for event in events
        if event["type"] == "node_end" and event["node"] == "context_monitor"
    )
    assert monitor["context_token_count"] > 0
    assert monitor["context_should_compress"] is False
    assert events[-1]["passed"] is True
    assert (tmp_path / "ws" / "hello.py").is_file()


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
