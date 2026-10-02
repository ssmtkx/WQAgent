"""Tests for the LangGraph shared state (``wqagent.graph.state``)."""

from __future__ import annotations

import typing
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from wqagent.core.state import RuntimeState
from wqagent.graph import (
    DEFAULT_MAX_ATTEMPTS,
    TODO_STATUSES,
    TodoItem,
    VerificationResult,
    WQGraphState,
    make_todo,
    new_graph_state,
    todo_progress,
)

EXPECTED_KEYS = {
    "task",
    "runtime",
    "messages",
    "plan_summary",
    "todos",
    "acceptance_criteria",
    "verification_commands",
    "verification_results",
    "verification_checks",
    "passed",
    "attempts",
    "max_attempts",
    "last_actor_summary",
    "last_error",
    "final_answer",
}


# -- schema ------------------------------------------------------------------
def test_state_has_exactly_the_expected_keys() -> None:
    assert set(WQGraphState.__annotations__) == EXPECTED_KEYS


def test_every_key_is_optional() -> None:
    """`total=False`: nodes may return partial updates and start from a partial state."""
    assert set(WQGraphState.__optional_keys__) == EXPECTED_KEYS
    assert WQGraphState.__required_keys__ == frozenset()


def test_messages_uses_the_add_messages_reducer() -> None:
    hints = typing.get_type_hints(WQGraphState, include_extras=True)
    annotation = hints["messages"]
    assert typing.get_origin(annotation) is typing.Annotated
    base, *metadata = typing.get_args(annotation)
    assert base == list[BaseMessage]
    assert metadata == [add_messages]
    # every other channel is last-value-wins (no reducer metadata)
    annotated = [key for key, value in hints.items() if typing.get_origin(value) is typing.Annotated]
    assert annotated == ["messages"]


def test_typed_dicts_are_plain_dicts_at_runtime() -> None:
    todo = TodoItem(id="t1", content="write the parser", status="pending", note="")
    assert todo == {"id": "t1", "content": "write the parser", "status": "pending", "note": ""}
    result = VerificationResult(
        command="pytest -q", ok=True, exit_code=0, stdout="83 passed", stderr=""
    )
    assert isinstance(result, dict)
    assert result["ok"] is True


# -- reducer semantics -------------------------------------------------------
def test_add_messages_merges_instead_of_replacing() -> None:
    history = [HumanMessage(content="hi", id="m1")]
    merged = add_messages(history, [AIMessage(content="a", id="m2"), AIMessage(content="b", id="m3")])
    assert [message.content for message in merged] == ["hi", "a", "b"]
    assert history == [HumanMessage(content="hi", id="m1")]  # the input list is not mutated


def test_add_messages_updates_by_id() -> None:
    updated = add_messages([AIMessage(content="old", id="x")], [AIMessage(content="new", id="x")])
    assert len(updated) == 1
    assert updated[0].content == "new"


# -- helpers -----------------------------------------------------------------
def test_make_todo_validates_the_status() -> None:
    for status in TODO_STATUSES:
        assert make_todo("t1", "do it", status=status)["status"] == status
    with pytest.raises(ValueError, match="unknown todo status"):
        make_todo("t1", "do it", status="finished")


def test_todo_progress() -> None:
    todos = [
        make_todo("1", "a", status="completed"),
        make_todo("2", "b", status="completed"),
        make_todo("3", "c", status="in_progress"),
        make_todo("4", "d", status="blocked"),
    ]
    assert todo_progress(todos) == "2/4 todos completed (1 blocked)"
    assert todo_progress([]) == "0/0 todos completed"


def test_new_graph_state_defaults(tmp_path: Path) -> None:
    runtime = RuntimeState(workspace=tmp_path / "ws")
    state = new_graph_state("fix the bug", runtime)

    assert state["task"] == "fix the bug"
    assert state["runtime"] is runtime
    assert state["messages"] == []
    assert state["todos"] == [] and state["acceptance_criteria"] == []
    assert state["verification_commands"] == [] and state["verification_results"] == []
    assert state["passed"] is False
    assert state["attempts"] == 0
    assert state["max_attempts"] == DEFAULT_MAX_ATTEMPTS
    assert state["final_answer"] == ""


def test_new_graph_state_overrides(tmp_path: Path) -> None:
    runtime = RuntimeState(workspace=tmp_path / "ws")
    state = new_graph_state(
        "fix the bug",
        runtime,
        max_attempts=0,  # clamped to at least 1
        acceptance_criteria=["pytest passes"],
        verification_commands=["pytest -q"],
    )
    assert state["max_attempts"] == 1
    assert state["acceptance_criteria"] == ["pytest passes"]
    assert state["verification_commands"] == ["pytest -q"]


def test_new_graph_state_rejects_a_bad_runtime(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="runtime must be a RuntimeState"):
        new_graph_state("task", str(tmp_path))  # type: ignore[arg-type]


# -- through a real LangGraph graph ------------------------------------------
def _build_graph() -> typing.Any:
    """plan -> act -> verify, each node appending to `messages`."""

    def plan(state: WQGraphState) -> WQGraphState:
        return {
            "plan_summary": "two steps",
            "todos": [make_todo("t1", "write code"), make_todo("t2", "run tests")],
            "messages": [HumanMessage(content="plan")],
        }

    def act(state: WQGraphState) -> WQGraphState:
        assert state["plan_summary"] == "two steps"  # previous keys are still there
        return {"messages": [AIMessage(content="acted")], "attempts": state.get("attempts", 0) + 1}

    def verify(state: WQGraphState) -> WQGraphState:
        return {
            "messages": [AIMessage(content="verified")],
            "verification_results": [
                VerificationResult(
                    command="pytest -q", ok=True, exit_code=0, stdout="ok", stderr=""
                )
            ],
            "passed": True,
        }

    graph = StateGraph(WQGraphState)
    graph.add_node("plan", plan)
    graph.add_node("act", act)
    graph.add_node("verify", verify)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "act")
    graph.add_edge("act", "verify")
    graph.add_edge("verify", END)
    return graph.compile()


def test_graph_merges_messages_across_nodes(tmp_path: Path) -> None:
    runtime = RuntimeState(workspace=tmp_path / "ws")
    state = new_graph_state("do the thing", runtime)

    result = _build_graph().invoke(state)

    # messages from three different nodes accumulated in order
    assert [message.content for message in result["messages"]] == ["plan", "acted", "verified"]
    assert [message.type for message in result["messages"]] == ["human", "ai", "ai"]
    # last-value-wins keys
    assert result["attempts"] == 1
    assert result["passed"] is True
    assert result["verification_results"][0]["command"] == "pytest -q"
    # keys no node touched kept their initial values
    assert result["task"] == "do the thing"
    assert result["final_answer"] == ""
    assert isinstance(result["runtime"], RuntimeState)
    assert result["runtime"].workspace == runtime.workspace


def test_graph_accepts_a_partial_state(tmp_path: Path) -> None:
    """total=False means the graph can start from just the keys you have."""

    def act(state: WQGraphState) -> WQGraphState:
        return {"messages": [AIMessage(content="hello")], "passed": True}

    graph = StateGraph(WQGraphState)
    graph.add_node("act", act)
    graph.add_edge(START, "act")
    graph.add_edge("act", END)
    app = graph.compile()

    result = app.invoke({"messages": []})

    assert result["passed"] is True
    assert [message.content for message in result["messages"]] == ["hello"]


def test_todo_list_survives_a_node_update(tmp_path: Path) -> None:
    """`todos` has no reducer, so a node replaces the whole list - and can read it."""
    seen: list[list[dict[str, object]]] = []

    def plan(state: WQGraphState) -> WQGraphState:
        return {"todos": [make_todo("t1", "a"), make_todo("t2", "b")]}

    def finish(state: WQGraphState) -> WQGraphState:
        todos = list(state["todos"])
        seen.append([dict(todo) for todo in todos])  # snapshot the values we received
        todos[0] = make_todo("t1", "a", status="completed", note="done")
        return {"todos": todos}

    graph = StateGraph(WQGraphState)
    graph.add_node("plan", plan)
    graph.add_node("finish", finish)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "finish")
    graph.add_edge("finish", END)

    result = graph.compile().invoke({"messages": []})

    assert seen[0][0]["status"] == "pending"
    assert [todo["status"] for todo in result["todos"]] == ["completed", "pending"]
    assert todo_progress(result["todos"]) == "1/2 todos completed"


def test_no_reducer_channels_are_passed_by_reference() -> None:
    """Documents the semantics: only `messages` is merged/copied by a reducer.

    Channels without a reducer hand the *same* object to the next node, so a node
    that mutates a list in place would leak that change even without returning it.
    Nodes should build and return a new list instead.
    """
    observed: list[str] = []

    def plan(state: WQGraphState) -> WQGraphState:
        return {"todos": [make_todo("t1", "a")]}

    def sneaky(state: WQGraphState) -> WQGraphState:
        state["todos"][0]["status"] = "completed"  # in-place, no return value for `todos`
        return {}

    def check(state: WQGraphState) -> WQGraphState:
        observed.append(state["todos"][0]["status"])
        return {}

    graph = StateGraph(WQGraphState)
    graph.add_node("plan", plan)
    graph.add_node("sneaky", sneaky)
    graph.add_node("check", check)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "sneaky")
    graph.add_edge("sneaky", "check")
    graph.add_edge("check", END)

    result = graph.compile().invoke({"messages": []})

    assert observed == ["completed"]  # the in-place edit was visible downstream
    assert result["todos"][0]["status"] == "completed"
