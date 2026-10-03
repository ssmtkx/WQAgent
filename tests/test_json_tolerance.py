"""Tolerance tests: broken structured output from the model must never crash a node.

The planner and verifier ask for JSON through a forced tool call, but models
misbehave: fenced blocks, prose around the object, trailing commas, truncated
output, numbers where lists belong, a single object instead of a list, unknown
status values - and sometimes no JSON at all. Every one of those has to degrade
gracefully (fall back to the previous plan / a failed verdict), never raise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from helpers import ScriptedModel, tool_call

from wqagent.core.state import RuntimeState
from wqagent.graph.nodes import (
    _as_items,
    _balanced_object,
    _coerce_checks,
    _coerce_todos,
    _flag,
    _json_object,
    _request_structured,
    _text,
    _text_list,
    planner_node,
    verifier_node,
)
from wqagent.graph.state import TODO_STATUSES, make_todo, new_graph_state
from wqagent.tools.graph_tools import TodoWriteTool, parse_todo_updates

PLAN = {
    "plan_summary": "write it, then run it",
    "todos": [
        {"id": "t1", "content": "write hello.py"},
        {"id": "t2", "content": "run it", "note": "python hello.py"},
    ],
    "acceptance_criteria": ["prints Hello"],
    "verification_commands": ["python hello.py"],
}


class ContentModel:
    """Always answers with raw content - never a tool call."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "ContentModel":
        return self

    def invoke(self, messages: list[Any]) -> Any:
        self.calls += 1
        return AIMessage(content=self.content)


class NoToolChoiceModel(ContentModel):
    """A provider that rejects ``tool_choice`` outright."""

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "ContentModel":
        if "tool_choice" in kwargs:
            raise TypeError("this model does not support tool_choice")
        return self


class _BrokenBound:
    def __init__(self, owner: "BrokenToolCallModel") -> None:
        self.owner = owner

    def invoke(self, messages: list[Any]) -> Any:
        raise RuntimeError("400 unknown field: tool_choice")


class BrokenToolCallModel(ContentModel):
    """Forced tool calling fails at request time; plain invocation still works."""

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> _BrokenBound:
        return _BrokenBound(self)


class HostileModel:
    """Every call fails."""

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "HostileModel":
        raise RuntimeError("no tools here")

    def invoke(self, messages: list[Any]) -> Any:
        raise RuntimeError("model offline")


@pytest.fixture()
def revision_state(tmp_path: Path):
    """A state whose planner is revising a previous plan (so it calls the model)."""
    return new_graph_state(
        "do the thing",
        RuntimeState(workspace=tmp_path / "ws"),
        todos=[make_todo("t0", "old plan")],
        last_error="command failed: exit 1",
    )


# -- brace/JSON extraction ---------------------------------------------------
@pytest.mark.parametrize(
    "name, text, expected",
    [
        ("plain", '{"a": 1}', {"a": 1}),
        ("fenced-json", '```json\n{"a": 1}\n```', {"a": 1}),
        ("fenced-bare", '```\n{"a": 1}\n```', {"a": 1}),
        ("prose-around", 'Sure!\n{"a": 1}\nHope that helps.', {"a": 1}),
        ("nested", '{"a": {"b": [1, 2, {"c": 3}]}}', {"a": {"b": [1, 2, {"c": 3}]}}),
        ("braces-in-string", '{"a": "use { and } freely"}', {"a": "use { and } freely"}),
        ("escaped-quote", r'{"a": "say \"hi\""}', {"a": 'say "hi"'}),
        ("array-wrapped", '[{"a": 1}]', {"a": 1}),
        ("recovers-after-broken", '{"bad": } then {"a": 1}', {"a": 1}),
        ("unicode", '{"a": "中文 ✓"}', {"a": "中文 ✓"}),
    ],
)
def test_json_object_parses_recoverable_text(name: str, text: str, expected: dict) -> None:
    assert _json_object(text) == expected


@pytest.mark.parametrize(
    "name, text",
    [
        ("empty", ""),
        ("whitespace", "   \n "),
        ("prose-only", "I could not produce a plan."),
        ("truncated", '{"a": 1, "b": [1, 2'),
        ("trailing-comma", '{"a": 1,}'),
        ("single-quotes", "{'a': 1}"),
        ("json-null", "null"),
        ("array-of-scalars", "[1, 2, 3]"),
        ("unterminated-string", '{"a": "oops}'),
        ("only-open-brace", "{"),
    ],
)
def test_json_object_returns_empty_instead_of_raising(name: str, text: str) -> None:
    assert _json_object(text) == {}


def test_balanced_object_edges() -> None:
    assert _balanced_object("no braces") is None
    assert _balanced_object("{}") == "{}"
    assert _balanced_object('{"a": "}"}') == '{"a": "}"}'
    assert _balanced_object('x {"a": {"b": 1}} y') == '{"a": {"b": 1}}'
    assert _balanced_object('{"a": 1') is None
    assert _balanced_object(r'{"a": "\""}') == r'{"a": "\""}'


# -- element coercion --------------------------------------------------------
@pytest.mark.parametrize(
    "value, expected",
    [
        (None, []),
        (5, []),
        (True, []),
        ("string", []),
        ({"a": 1}, [{"a": 1}]),
        ([1, 2], [1, 2]),
        ((1,), [1]),
        ({"todos": []}, [{"todos": []}]),
    ],
)
def test_as_items_never_raises(value: Any, expected: list[Any]) -> None:
    assert _as_items(value) == expected
    assert _as_items(object()) == []  # not iterable at all


@pytest.mark.parametrize(
    "value, expected",
    [("  spaced  ", "spaced"), ("", ""), (42, ""), (None, ""), (True, ""), ({"a": 1}, ""), (["a"], "")],
)
def test_text_only_accepts_strings(value: Any, expected: str) -> None:
    assert _text(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, []),
        ("", []),
        ("  ", []),
        ("single", ["single"]),
        (["a", "b"], ["a", "b"]),
        (("a",), ["a"]),
        (["a", "", "  ", None], ["a"]),
        (7, []),
        (True, []),
        ({"a": 1}, []),
        ([1, 2], ["1", "2"]),
        ([{"nested": 1}, "keep"], ["keep"]),
    ],
)
def test_text_list_never_raises(value: Any, expected: list[str]) -> None:
    assert _text_list(value) == expected


def test_coerce_todos_tolerates_garbage() -> None:
    fallback = [make_todo("t0", "old")]

    assert _coerce_todos(None, fallback=fallback)[0]["id"] == "t0"
    assert _coerce_todos("not a list", fallback=fallback)[0]["id"] == "t0"
    assert _coerce_todos([], fallback=fallback)[0]["id"] == "t0"
    assert _coerce_todos([None, 42, "text", {"content": ""}], fallback=fallback)[0]["id"] == "t0"

    # a single object instead of an array
    single = _coerce_todos({"id": "t1", "content": "only one"})
    assert [todo["id"] for todo in single] == ["t1"]

    # unknown status is normalised, missing id is generated
    todos = _coerce_todos([{"content": "no id", "status": "DONE"}, {"id": "t2", "content": "ok"}])
    assert [(todo["id"], todo["status"]) for todo in todos] == [("t1", "pending"), ("t2", "pending")]
    assert all(todo["status"] in TODO_STATUSES for todo in todos)


def test_coerce_checks_tolerates_garbage() -> None:
    assert _coerce_checks(None) == []
    assert _coerce_checks("nope") == []
    assert _coerce_checks(5) == []  # not iterable at all
    assert _coerce_checks([None, 1, {"detail": "no name"}, {"name": ""}]) == []
    single = _coerce_checks({"name": "it works", "passed": "yes"})
    assert single == [{"name": "it works", "passed": True, "detail": ""}]


@pytest.mark.parametrize(
    "value, expected",
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        (None, False),
        ("", False),
        ("true", True),
        ("YES", True),
        ("passed", True),
        ("false", False),  # a truthy string must not become a passing verdict
        ("no", False),
        ("maybe", False),
        ({"unexpected": "object"}, False),
    ],
)
def test_flag_reads_verdicts_strictly(value: Any, expected: bool) -> None:
    assert _flag(value) is expected


def test_parse_todo_updates_tolerates_garbage() -> None:
    updates, problems = parse_todo_updates(None)
    assert updates == [] and problems == []

    updates, problems = parse_todo_updates([{"id": "t1", "status": "completed"}])
    assert [update.id for update in updates] == ["t1"] and problems == []

    # a single object instead of a list
    updates, _ = parse_todo_updates({"id": "t1", "status": "completed"})
    assert [update.id for update in updates] == ["t1"]

    # scalars and strings are junk, not crashes
    assert parse_todo_updates(5) == ([], [])
    assert parse_todo_updates("nope") == ([], [])

    updates, problems = parse_todo_updates([{"id": "t1", "status": "done"}, "junk", {"status": "completed"}])
    assert updates == []
    assert len(problems) == 3  # reported, not raised


# -- the structured-output request chain -------------------------------------
def test_request_structured_prefers_the_tool_call() -> None:
    class ToolModel(ContentModel):
        def invoke(self, messages: list[Any]) -> Any:
            self.calls += 1
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "todo_write",
                        "args": {"plan_summary": "from the tool call"},
                        "id": "c1",
                        "type": "tool_call",
                    }
                ],
            )

    args = _request_structured(ToolModel("ignored"), TodoWriteTool(), [])

    assert args == {"plan_summary": "from the tool call"}


def test_request_structured_ignores_an_unrelated_tool_call() -> None:
    class WrongToolModel(ContentModel):
        def invoke(self, messages: list[Any]) -> Any:
            self.calls += 1
            return AIMessage(
                content='{"plan_summary": "from content"}',
                tool_calls=[
                    {"name": "file_write", "args": {"file_path": "x"}, "id": "c1", "type": "tool_call"}
                ],
            )

    args = _request_structured(WrongToolModel(""), TodoWriteTool(), [])

    assert args == {"plan_summary": "from content"}


def test_request_structured_survives_a_provider_without_tool_choice() -> None:
    model = NoToolChoiceModel(json.dumps(PLAN))

    args = _request_structured(model, TodoWriteTool(), [])

    assert args["plan_summary"] == PLAN["plan_summary"]


def test_request_structured_recovers_when_forced_calling_fails() -> None:
    model = BrokenToolCallModel(json.dumps(PLAN))

    args = _request_structured(model, TodoWriteTool(), [])

    assert [todo["id"] for todo in args["todos"]] == ["t1", "t2"]


def test_request_structured_returns_empty_when_everything_fails() -> None:
    assert _request_structured(HostileModel(), TodoWriteTool(), []) == {}


# -- planner end to end ------------------------------------------------------
# The supervisor planner reads its plan from `todo_write` *tool calls*, so a model
# that answers with prose/JSON content cannot corrupt the plan - the previous one
# is kept. Tolerance now lives in the tool-argument path.
JUNK_CONTENT = [
    ("empty", ""),
    ("prose-only", "I am not sure what the plan should be."),
    ("fenced-json", "```json\n" + json.dumps(PLAN) + "\n```"),
    ("truncated", '{"plan_summary": "half", "todos": [{"id": "t1", "cont'),
    ("trailing-comma", '{"plan_summary": "x", "todos": [{"id": "t1", "content": "c",}],}'),
    ("single-quotes", "{'plan_summary': 'x', 'todos': []}"),
    ("json-null", "null"),
    ("wrong-types", '{"plan_summary": 42, "todos": "not a list"}'),
    ("deep-garbage", '{"todos": {"nested": {"weird": [1, {"x": null}]}}}'),
    ("array-wrapped", json.dumps([PLAN])),
]

PLANNER_KEYS = {
    "plan_summary",
    "todos",
    "acceptance_criteria",
    "verification_commands",
    "messages",
    "last_actor_summary",
    "research_notes",
    "sources",
    "agent_handoffs",
    "code_agent_summary",
}


@pytest.mark.parametrize("name, content", JUNK_CONTENT, ids=[n for n, _ in JUNK_CONTENT])
def test_planner_keeps_the_previous_plan_when_no_tool_is_called(
    name: str, content: str, revision_state: Any
) -> None:
    update = planner_node(revision_state, model=ContentModel(content))

    # the node always returns a complete, usable update ...
    assert PLANNER_KEYS <= set(update)
    assert isinstance(update["plan_summary"], str)
    assert isinstance(update["acceptance_criteria"], list)
    assert isinstance(update["verification_commands"], list)
    # ... and the plan survives untouched
    assert [todo["id"] for todo in update["todos"]] == ["t0"]
    assert all(todo["status"] in TODO_STATUSES for todo in update["todos"])
    assert update["agent_handoffs"] == []


GARBAGE_PLAN_ARGS: list[tuple[str, Any]] = [
    ("missing-todos", {"plan_summary": "only a summary"}),
    ("todos-not-a-list", {"plan_summary": "x", "todos": "not a list"}),
    ("todos-of-junk", {"plan_summary": "x", "todos": [None, 42, "", {"id": "", "content": ""}]}),
    (
        "wrong-types",
        {"plan_summary": 42, "todos": "nope", "acceptance_criteria": 7, "verification_commands": True},
    ),
    ("deep-garbage", {"todos": {"nested": {"weird": [1, {"x": None}]}}}),
    ("empty-object", {}),
]


@pytest.mark.parametrize(
    "name, args", GARBAGE_PLAN_ARGS, ids=[n for n, _ in GARBAGE_PLAN_ARGS]
)
def test_planner_survives_garbage_todo_write_arguments(
    name: str, args: Any, revision_state: Any
) -> None:
    model = ScriptedModel([tool_call("todo_write", args), AIMessage(content="done")])

    update = planner_node(revision_state, model=model)

    assert PLANNER_KEYS <= set(update)
    assert isinstance(update["plan_summary"], str)
    assert isinstance(update["acceptance_criteria"], list)
    assert isinstance(update["verification_commands"], list)
    assert update["todos"], "the planner must always return a usable plan"
    assert all(todo["status"] in TODO_STATUSES for todo in update["todos"])
    assert all(isinstance(todo["id"], str) and todo["id"] for todo in update["todos"])


def test_planner_uses_partial_tool_arguments(revision_state: Any) -> None:
    """A plan with only some usable fields keeps the rest from the previous state."""
    model = ScriptedModel(
        [
            tool_call("todo_write", {"todos": [{"id": "t1", "content": "only todos survived"}]}),
            AIMessage(content="planned"),
        ]
    )

    update = planner_node(revision_state, model=model)

    assert [todo["id"] for todo in update["todos"]] == ["t1"]
    assert update["todos"][0]["status"] == "pending"  # defaults, not a crash
    assert update["acceptance_criteria"] == []
    assert update["verification_commands"] == []
    assert update["plan_summary"] == ""  # nothing usable -> state fallback


def test_planner_tolerates_type_and_status_garbage(revision_state: Any) -> None:
    model = ScriptedModel(
        [
            tool_call(
                "todo_write",
                {
                    "plan_summary": 42,
                    "todos": [
                        None,
                        42,
                        {"id": "t1", "content": "keep me", "status": "DONE"},
                        {"content": ""},
                    ],
                    "acceptance_criteria": 7,
                    "verification_commands": {"not": "a list"},
                },
            ),
            AIMessage(content="planned"),
        ]
    )

    update = planner_node(revision_state, model=model)

    assert update["plan_summary"] == ""  # 42 is not a usable summary -> state fallback
    assert [(todo["id"], todo["status"]) for todo in update["todos"]] == [("t1", "pending")]
    assert update["acceptance_criteria"] == []
    assert update["verification_commands"] == []


def test_planner_ignores_unparsable_tool_calls(tmp_path: Path) -> None:
    """A tool call whose arguments the provider could not parse cannot corrupt the plan."""
    broken = AIMessage(
        content="",
        invalid_tool_calls=[
            {
                "name": "todo_write",
                "args": '{"plan_summary": "half", "todos": [',
                "id": "c1",
                "error": "unparsable arguments",
                "type": "invalid_tool_call",
            }
        ],
    )
    state = new_graph_state(
        "do the thing",
        RuntimeState(workspace=tmp_path / "ws"),
        todos=[make_todo("t0", "old plan")],
    )

    update = planner_node(state, model=ScriptedModel([broken]))

    assert [todo["id"] for todo in update["todos"]] == ["t0"]  # untouched, no crash
    assert update["agent_handoffs"] == []


# -- verifier end to end -----------------------------------------------------
@pytest.mark.parametrize(
    "name, content",
    [
        ("prose", "I think it passed."),
        ("truncated", '{"passed": true, "reason": "looks'),
        ("truthy-string", '{"passed": "false", "reason": "I am not sure"}'),
        ("junk-types", '{"passed": "maybe", "checks": 5}'),
        ("json-null", "null"),
        ("empty", ""),
    ],
)
def test_verifier_treats_unusable_verdicts_as_failures(
    name: str, content: str, tmp_path: Path
) -> None:
    state = new_graph_state(
        "do the thing",
        RuntimeState(workspace=tmp_path / "ws"),
        todos=[make_todo("t1", "do it")],
        verification_commands=["echo ok"],
    )

    update = verifier_node(state, model=ContentModel(content))

    assert update["passed"] is False
    assert update["attempts"] == 1
    assert update["last_error"]  # always explains itself
    assert update["verification_results"][0]["ok"] is True  # commands still ran
    assert update["todos"][0]["status"] == "pending"  # not marked done on failure


@pytest.mark.parametrize(
    "name, content",
    [
        ("checks-not-a-list", '{"passed": true, "reason": "fine", "checks": 5}'),
        ("junk-checks", '{"passed": true, "reason": "fine", "checks": ["junk", {"name": ""}, null]}'),
    ],
)
def test_verifier_survives_junk_checks_on_a_pass(
    name: str, content: str, tmp_path: Path
) -> None:
    state = new_graph_state(
        "do the thing",
        RuntimeState(workspace=tmp_path / "ws"),
        todos=[make_todo("t1", "do it")],
        verification_commands=["echo ok"],
    )

    update = verifier_node(state, model=ContentModel(content))

    assert update["passed"] is True
    assert update["verification_checks"] == []
    assert update["last_error"] == ""
    assert update["todos"][0]["status"] == "completed"
