"""Tests for the compiled workflow (``wqagent.graph.workflow``) and ``final_node``."""

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
)

from wqagent.core.state import RuntimeState
from wqagent.graph import (
    COMPLEX_NODE_NAMES,
    COMPRESSOR_ROUTE_TARGETS,
    CONTEXT_ROUTE_TARGETS,
    NODE_NAMES,
    ROUTE_TARGETS,
    WQGraphState,
    build_complex_workflow,
    build_workflow,
    final_node,
    format_final_answer,
    new_graph_state,
    set_model,
)
from wqagent.graph.state import make_todo
from wqagent.prompts import stage1, stage2, stage3
from wqagent.prompts.stage4 import CONTEXT_COMPRESSION_PROMPT

#: A compression payload the compressor model can answer with.
COMPRESSION_PAYLOAD: dict[str, Any] = {
    "summary": "compressed",
    "active_goal": "finish hello.py",
    "completed_work": "wrote hello.py",
    "open_todos": ["t1: rerun"],
    "important_files": ["hello.py"],
    "tool_findings": ["python 3.12"],
    "sources": ["https://docs.test/python"],
    "next_steps": ["run it"],
    "risks": "none",
}


class ScriptedModel:
    """Replays scripted responses; records what it was asked."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.seen: list[list[Any]] = []

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "ScriptedModel":
        return self

    def invoke(self, messages: list[Any]) -> Any:
        self.seen.append(list(messages))
        if not self.responses:
            raise AssertionError("ScriptedModel ran out of scripted responses")
        return self.responses.pop(0)


class BrokenModel:
    """Every call fails - the final node must still produce an answer."""

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "BrokenModel":
        return self

    def invoke(self, messages: list[Any]) -> Any:
        raise RuntimeError("model offline")


def _tool_call(name: str, args: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": "c1", "type": "tool_call"}]
    )


@pytest.fixture(autouse=True)
def _reset_module_model():
    set_model(None)
    yield
    set_model(None)


# -- graph shape -------------------------------------------------------------
def test_workflow_has_the_expected_nodes_and_edges() -> None:
    app = build_workflow()
    drawing = app.get_graph()

    node_ids = set(drawing.nodes)
    for name in NODE_NAMES:
        assert name in node_ids, name
    assert {"__start__", "__end__"} <= node_ids
    assert "actor" not in node_ids  # stage 3: the planner delegates instead

    edges = {(edge.source, edge.target) for edge in drawing.edges}
    assert ("__start__", "planner") in edges
    assert ("planner", "verifier") in edges
    assert ("final", "__end__") in edges
    assert ("planner", "actor") not in edges

    # the verifier branches: one edge per route target
    verifier_targets = {target for source, target in edges if source == "verifier"}
    assert verifier_targets == set(ROUTE_TARGETS.values()) == {"final", "planner"}
    assert set(NODE_NAMES) == {"planner", "verifier", "final"}


def test_every_node_is_the_function_from_nodes_module() -> None:
    from wqagent.graph import nodes as node_module

    drawing = build_workflow().get_graph()

    for name in NODE_NAMES:
        data = drawing.nodes[name].data
        wired = getattr(data, "func", data)
        assert wired is getattr(node_module, f"{name}_node"), name


def test_build_workflow_binds_model_and_loop_budgets() -> None:
    from functools import partial

    from wqagent.graph import nodes as node_module

    sentinel = object()
    specialist = object()
    drawing = build_workflow(
        model=sentinel, planner_loops=3, agent_model=specialist
    ).get_graph()

    planner = getattr(drawing.nodes["planner"].data, "func", drawing.nodes["planner"].data)
    assert isinstance(planner, partial)
    assert planner.func is node_module.planner_node
    assert planner.keywords == {
        "model": sentinel,
        "max_loops": 3,
        "agent_model": specialist,
    }

    verifier = getattr(drawing.nodes["verifier"].data, "func", drawing.nodes["verifier"].data)
    assert isinstance(verifier, partial)
    assert verifier.keywords == {"model": sentinel}


# -- final_node --------------------------------------------------------------
def _finished_state(tmp_path: Path, **extra: Any):
    state = new_graph_state("write hello.py", RuntimeState(workspace=tmp_path / "ws"), **extra)
    return state


def test_final_node_reports_success(tmp_path: Path) -> None:
    state = _finished_state(
        tmp_path,
        plan_summary="write, then run",
        todos=[make_todo("t1", "write hello.py", status="completed", note="verified")],
        acceptance_criteria=["prints Hello"],
        verification_commands=["python hello.py"],
        verification_results=[
            {"command": "python hello.py", "ok": True, "exit_code": 0, "stdout": "Hello", "stderr": ""}
        ],
        verification_checks=[{"name": "prints Hello", "passed": True, "detail": "stdout was Hello"}],
        last_actor_summary="Wrote hello.py and ran it.",
    )
    state["passed"] = True
    state["attempts"] = 1

    update = final_node(state, summarize=False)
    answer = update["final_answer"]

    assert "**Status:** completed" in answer
    assert "attempts: 1/3" in answer
    assert "write, then run" in answer
    assert "[x] t1: write hello.py — verified" in answer
    assert "PASS — prints Hello (stdout was Hello)" in answer
    assert "`python hello.py` -> exit code 0 (ok)" in answer
    assert "Wrote hello.py and ran it." in answer
    assert "not completed" not in answer


def test_final_node_reports_failure_with_the_remaining_work(tmp_path: Path) -> None:
    state = _finished_state(
        tmp_path,
        plan_summary="write, then run",
        todos=[
            make_todo("t1", "write hello.py", status="completed"),
            make_todo("t2", "run it", status="pending"),
        ],
        acceptance_criteria=["prints Hello"],
        verification_commands=["python hello.py"],
        verification_results=[
            {"command": "python hello.py", "ok": False, "exit_code": 1, "stdout": "", "stderr": "NameError"}
        ],
        last_error="command failed: python hello.py (exit code 1)",
    )
    state["passed"] = False
    state["attempts"] = 3
    state["max_attempts"] = 3

    answer = final_node(state, summarize=False)["final_answer"]

    assert "**Status:** not completed" in answer
    assert "attempts: 3/3" in answer
    assert "**Why it is not done:**" in answer
    assert "command failed: python hello.py (exit code 1)" in answer
    assert "**Remaining work:**" in answer
    assert "- t2: run it" in answer
    assert "- t1:" not in answer.split("**Remaining work:**")[1]


def test_format_final_answer_never_returns_empty() -> None:
    answer = format_final_answer({})
    assert "**Task:** (no task recorded)" in answer
    assert "**Status:** not completed" in answer


def test_final_node_uses_the_model_summary(tmp_path: Path) -> None:
    state = _finished_state(tmp_path, todos=[make_todo("t1", "do it", status="completed")])
    model = ScriptedModel([AIMessage(content="Done: hello.py works.")])

    update = final_node(state, model=model)

    assert update["final_answer"] == "Done: hello.py works."
    sent = model.seen[0]
    assert sent[0].content == stage2.FINAL_PROMPT
    assert "--- facts ---" in sent[1].content
    assert "**Status:**" in sent[1].content  # the report is handed over as facts


def test_final_node_falls_back_when_the_model_fails(tmp_path: Path) -> None:
    state = _finished_state(tmp_path, todos=[make_todo("t1", "do it", status="completed")])
    state["passed"] = True

    update = final_node(state, model=BrokenModel())

    assert update["final_answer"] == format_final_answer(state)
    assert "**Status:** completed" in update["final_answer"]


def test_final_node_ignores_an_empty_model_answer(tmp_path: Path) -> None:
    state = _finished_state(tmp_path)
    model = ScriptedModel([AIMessage(content="   ")])

    update = final_node(state, model=model)

    assert update["final_answer"] == format_final_answer(state)


# -- prompts -----------------------------------------------------------------
def test_all_four_prompts_are_available_from_stage2() -> None:
    for name in ("PLANNER_PROMPT", "ACTOR_PROMPT", "VERIFIER_PROMPT", "FINAL_PROMPT"):
        text = getattr(stage2, name)
        assert isinstance(text, str) and len(text) > 100, name
        assert name in stage2.__all__, name
    # ACTOR_PROMPT still comes from stage 1, the planner/verifier prompts from stage 3
    assert stage2.ACTOR_PROMPT is stage1.ACTOR_PROMPT
    assert stage2.PLANNER_PROMPT is stage3.PLANNER_PROMPT
    assert stage2.VERIFIER_PROMPT is stage3.VERIFIER_PROMPT


def test_planner_prompt_is_the_stage3_supervisor_prompt() -> None:
    assert "planner/supervisor" in stage3.PLANNER_PROMPT
    for tool in ("TodoWriteTool", "CallSearchAgentTool", "CallCodeAgentTool"):
        assert tool in stage3.PLANNER_PROMPT
    assert "Always call TodoWriteTool before delegating new work" in stage3.PLANNER_PROMPT


def test_prompts_describe_their_json_and_tools() -> None:
    assert "You are verifier" in stage3.VERIFIER_PROMPT
    assert "Return only JSON" in stage3.VERIFIER_PROMPT
    for key in ("passed", "reason", "checks", "recommended_next_instruction"):
        assert key in stage3.VERIFIER_PROMPT
    assert "must not modify files" in stage3.VERIFIER_PROMPT
    assert "tool call" in stage2.FINAL_PROMPT or "No tool calls" in stage2.FINAL_PROMPT
    # the specialist prompts name their own tools
    assert "WebSearchTool" in stage3.SEARCH_AGENT_PROMPT
    assert "TodoUpdateTool" in stage3.CODE_AGENT_PROMPT


# -- the whole workflow ------------------------------------------------------
def test_workflow_passes_on_the_first_attempt(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    model = ScriptedModel(passing_script(final_text="Done: hello.py prints Hello."))
    set_model(model)

    result = build_workflow().invoke(
        new_graph_state("write hello.py", RuntimeState(workspace=workspace))
    )

    assert result["passed"] is True
    assert result["attempts"] == 1
    assert result["final_answer"] == "Done: hello.py prints Hello."
    assert [todo["status"] for todo in result["todos"]] == ["completed"]
    assert (workspace / "hello.py").read_text(encoding="utf-8") == "print('Hello')\n"
    # the delegation is recorded, and the implementer's report reached the state
    assert [handoff["to_agent"] for handoff in result["agent_handoffs"]] == ["codeAgent"]
    assert result["code_agent_summary"] == "Wrote hello.py."
    # planner x3, codeAgent x2, verifier x2, final x1
    assert len(model.seen) == 8
    assert model.responses == []


def test_workflow_writes_a_failure_answer_when_the_budget_runs_out(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    model = ScriptedModel(
        [
            plan_call(
                summary="try to make the test pass",
                todos=[{"id": "t1", "content": "fix the failing test"}],
                criteria=["pytest passes"],
                commands=["exit 1"],
            ),
            code_call("fix the failing test"),
            answer("Tried to fix it."),
            answer("Delegated the fix to codeAgent."),
            answer("The test still fails."),
            report_call(
                False,
                reason="the test command still exits 1",
                checks=[{"name": "pytest passes", "passed": False, "detail": "exit 1"}],
                recommended_next_instruction="fix the assertion in test_x.py",
            ),
            answer("Not done: `exit 1` still fails; next step is to fix the assertion."),
        ]
    )
    set_model(model)

    result = build_workflow().invoke(
        new_graph_state("fix the failing test", RuntimeState(workspace=workspace), max_attempts=1)
    )

    assert result["passed"] is False
    assert result["attempts"] == 1  # budget spent, so the run went straight to final
    assert "Not done" in result["final_answer"]
    assert result["final_answer"]  # never empty
    assert "fix the assertion in test_x.py" in result["last_error"]
    assert len(model.seen) == 7  # planner x3, codeAgent x2, verifier x2... plus final
    assert model.responses == []


# -- the complex workflow (with context management) --------------------------
def _complex_run(tmp_path: Path, model: Any, *, max_attempts: int = 3, **extra: Any):
    """Run the complex graph once; return (final state, node order)."""
    state = new_graph_state(
        "write hello.py", RuntimeState(workspace=tmp_path / "ws"), max_attempts=max_attempts, **extra
    )
    app = build_complex_workflow(model=model)
    order: list[str] = []
    result: dict[str, Any] = dict(state)
    for mode, chunk in app.stream(state, stream_mode=["updates", "values"]):
        if mode == "updates":
            order.extend(chunk)
        else:
            result = chunk
    return result, order


def test_complex_workflow_has_the_expected_nodes_and_edges() -> None:
    drawing = build_complex_workflow().get_graph()

    node_ids = set(drawing.nodes)
    for name in COMPLEX_NODE_NAMES:
        assert name in node_ids, name
    assert set(COMPLEX_NODE_NAMES) == {
        "planner",
        "context_monitor",
        "context_compressor",
        "verifier",
        "final",
    }

    edges = {(edge.source, edge.target) for edge in drawing.edges}
    assert ("__start__", "planner") in edges
    assert ("planner", "context_monitor") in edges
    assert ("verifier", "context_monitor") in edges  # a verification also passes the monitor
    assert ("final", "__end__") in edges
    # nothing bypasses the monitor any more
    assert ("planner", "verifier") not in edges
    assert ("verifier", "final") not in edges

    monitor_targets = {target for source, target in edges if source == "context_monitor"}
    assert monitor_targets == set(CONTEXT_ROUTE_TARGETS.values())
    compressor_targets = {target for source, target in edges if source == "context_compressor"}
    assert compressor_targets == set(COMPRESSOR_ROUTE_TARGETS.values())


def test_complex_workflow_binds_model_and_loop_budgets() -> None:
    from functools import partial

    from wqagent.graph import nodes as node_module

    sentinel = object()
    drawing = build_complex_workflow(
        model=sentinel, planner_loops=5, verifier_loops=2
    ).get_graph()

    planner = getattr(drawing.nodes["planner"].data, "func", drawing.nodes["planner"].data)
    assert isinstance(planner, partial)
    assert planner.keywords == {"model": sentinel, "max_loops": 5}
    verifier = getattr(drawing.nodes["verifier"].data, "func", drawing.nodes["verifier"].data)
    assert verifier.keywords == {"model": sentinel, "max_loops": 2}
    for name in ("context_monitor", "context_compressor", "final"):
        wired = getattr(drawing.nodes[name].data, "func", drawing.nodes[name].data)
        assert isinstance(wired, partial)
        assert wired.func is getattr(node_module, f"{name}_node")
        assert wired.keywords == {"model": sentinel}


def test_complex_workflow_passes_without_compressing(tmp_path: Path) -> None:
    model = ScriptedModel(passing_script(final_text="Done: hello.py works."))

    result, order = _complex_run(tmp_path, model)

    assert result["passed"] is True
    assert result["attempts"] == 1
    assert result["final_answer"] == "Done: hello.py works."
    assert order == ["planner", "context_monitor", "verifier", "context_monitor", "final"]
    # the default budget is far away, so nothing was compressed
    assert result["compression_events"] == []
    assert result["context_should_compress"] is False
    assert not (tmp_path / "ws" / "HISTORY_SUMMARY.md").exists()
    assert len(model.seen) == 8  # the monitor counts tokens, it does not call the model


def test_complex_workflow_compresses_an_over_budget_context(tmp_path: Path) -> None:
    summary = "compressed: hello.py is written"
    model = ScriptedModel(
        [
            plan_call(),
            answer("Planned it."),
            # the compressor asks the model for the nine-key JSON
            AIMessage(content=json.dumps({**COMPRESSION_PAYLOAD, "summary": summary})),
            answer("Inspected."),
            report_call(True),
            answer("Done after compression."),
        ]
    )

    result, order = _complex_run(tmp_path, model, context_token_limit=1)

    assert order == [
        "planner",
        "context_monitor",
        "context_compressor",
        "verifier",
        "context_monitor",
        "final",
    ]
    assert result["passed"] is True
    assert result["context_summary"] == summary
    assert result["history_summary"] == summary
    assert [event["summary"] for event in result["compression_events"]] == [summary]
    # the transcript really was replaced by the one summary message
    assert [message.type for message in result["messages"]] == ["ai"]
    assert result["messages"][0].content == summary
    # ...and the compressor was asked exactly once (limit=1 is unreachable)
    assert len(model.seen) == 6
    assert (tmp_path / "ws" / "HISTORY_SUMMARY.md").read_text(encoding="utf-8").startswith(
        "# History summary"
    )


def test_complex_workflow_stops_at_the_attempt_budget(tmp_path: Path) -> None:
    """A failing task must end in `final`, not spin through the monitor."""
    model = ScriptedModel(
        [
            plan_call(commands=["exit 1"]),
            answer("Planned it."),
            answer("Inspected."),
            report_call(False, recommended_next_instruction="fix it"),
            answer("Not done: exit 1 still fails."),
        ]
    )

    result, order = _complex_run(tmp_path, model, max_attempts=1)

    assert result["passed"] is False
    assert result["attempts"] == 1
    assert result["final_answer"] == "Not done: exit 1 still fails."
    assert order == ["planner", "context_monitor", "verifier", "context_monitor", "final"]
    assert len(model.seen) == 5  # no replanning, no runaway loop


def test_complex_workflow_replans_after_a_failed_verification(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            plan_call(commands=["exit 1"]),
            answer("Planned it."),
            answer("Inspected."),
            report_call(False),
            plan_call(summary="second try", commands=["echo ok"]),
            answer("Replanned."),
            answer("Inspected again."),
            report_call(True),
            answer("Done on the second attempt."),
        ]
    )

    result, order = _complex_run(tmp_path, model, max_attempts=2)

    assert result["passed"] is True
    assert result["attempts"] == 2
    assert order == [
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
    assert len(model.seen) == 9


def test_workflow_answers_even_without_a_working_model(tmp_path: Path) -> None:
    """The final node must not need the model: its report is always available."""
    set_model(BrokenModel())

    # only the final node is exercised here - the other nodes need a model
    state = new_graph_state("do something", RuntimeState(workspace=tmp_path / "ws"))
    state["todos"] = [make_todo("t1", "do it")]
    state["passed"] = False
    state["attempts"] = 1
    state["last_error"] = "boom"

    answer = final_node(state)["final_answer"]

    assert "**Status:** not completed" in answer
    assert "boom" in answer
    json.dumps(answer)  # plain text, nothing exotic
