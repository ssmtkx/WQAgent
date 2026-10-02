"""Tests for the compiled workflow (``wqagent.graph.workflow``) and ``final_node``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from wqagent.core.state import RuntimeState
from wqagent.graph import (
    NODE_NAMES,
    ROUTE_TARGETS,
    WQGraphState,
    build_workflow,
    final_node,
    format_final_answer,
    new_graph_state,
    set_model,
)
from wqagent.graph.state import make_todo
from wqagent.prompts import stage1, stage2


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

    edges = {(edge.source, edge.target) for edge in drawing.edges}
    assert ("__start__", "planner") in edges
    assert ("planner", "actor") in edges
    assert ("actor", "verifier") in edges
    assert ("final", "__end__") in edges

    # the verifier branches: one edge per route target
    verifier_targets = {target for source, target in edges if source == "verifier"}
    assert verifier_targets == set(ROUTE_TARGETS.values()) == {"final", "planner"}
    assert set(NODE_NAMES) == {"planner", "actor", "verifier", "final"}


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
    drawing = build_workflow(model=sentinel, actor_loops=3).get_graph()

    actor = getattr(drawing.nodes["actor"].data, "func", drawing.nodes["actor"].data)
    assert isinstance(actor, partial)
    assert actor.func is node_module.actor_node
    assert actor.keywords == {"model": sentinel, "max_loops": 3}

    planner = getattr(drawing.nodes["planner"].data, "func", drawing.nodes["planner"].data)
    assert isinstance(planner, partial)
    assert planner.keywords == {"model": sentinel}


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
    # ACTOR_PROMPT still comes from stage 1 - same object, single source of truth
    assert stage2.ACTOR_PROMPT is stage1.ACTOR_PROMPT


def test_planner_and_verifier_prompts_describe_their_json() -> None:
    assert "todo_write" in stage2.PLANNER_PROMPT
    for key in ("plan_summary", "todos", "acceptance_criteria", "verification_commands"):
        assert key in stage2.PLANNER_PROMPT
    assert "verification_report" in stage2.VERIFIER_PROMPT
    for key in ("passed", "reason", "checks", "recommended_next_instruction"):
        assert key in stage2.VERIFIER_PROMPT
    assert "tool call" in stage2.FINAL_PROMPT or "No tool calls" in stage2.FINAL_PROMPT


# -- the whole workflow ------------------------------------------------------
def test_workflow_passes_on_the_first_attempt(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    model = ScriptedModel(
        [
            _tool_call(
                "todo_write",
                {
                    "plan_summary": "write hello.py then run it",
                    "todos": [{"id": "t1", "content": "write hello.py"}],
                    "acceptance_criteria": ["hello.py runs cleanly"],
                    "verification_commands": ["echo ok"],
                },
            ),
            _tool_call("file_write", {"file_path": "hello.py", "content": "print('Hello')\n"}),
            AIMessage(content="Wrote hello.py."),
            AIMessage(content="Read the file, it looks right."),
            _tool_call(
                "verification_report",
                {
                    "passed": True,
                    "reason": "the file exists and the command succeeded",
                    "checks": [{"name": "hello.py runs cleanly", "passed": True, "detail": "exit 0"}],
                    "recommended_next_instruction": "",
                },
            ),
            AIMessage(content="Done: hello.py prints Hello and `echo ok` passed."),
        ]
    )
    set_model(model)

    result = build_workflow().invoke(
        new_graph_state("write hello.py", RuntimeState(workspace=workspace))
    )

    assert result["passed"] is True
    assert result["attempts"] == 1
    assert result["final_answer"] == "Done: hello.py prints Hello and `echo ok` passed."
    assert [todo["status"] for todo in result["todos"]] == ["completed"]
    assert (workspace / "hello.py").read_text(encoding="utf-8") == "print('Hello')\n"
    assert len(model.seen) == 6  # planner, actor x2, verifier x2, final
    assert model.responses == []


def test_workflow_writes_a_failure_answer_when_the_budget_runs_out(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    model = ScriptedModel(
        [
            _tool_call(
                "todo_write",
                {
                    "plan_summary": "try to make the test pass",
                    "todos": [{"id": "t1", "content": "fix the failing test"}],
                    "acceptance_criteria": ["pytest passes"],
                    "verification_commands": ["exit 1"],
                },
            ),
            AIMessage(content="Tried to fix it."),
            AIMessage(content="The test still fails."),
            _tool_call(
                "verification_report",
                {
                    "passed": False,
                    "reason": "the test command still exits 1",
                    "checks": [{"name": "pytest passes", "passed": False, "detail": "exit 1"}],
                    "recommended_next_instruction": "fix the assertion in test_x.py",
                },
            ),
            AIMessage(content="Not done: `exit 1` still fails; next step is to fix the assertion."),
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
    assert len(model.seen) == 5  # planner, actor, verifier x2, final
    assert model.responses == []


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
