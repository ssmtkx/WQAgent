"""Tolerance tests: broken structured output from the model must never crash a node.

Models misbehave: fenced blocks, prose around the object, trailing commas, truncated
output, junk types, a single object instead of a list, unknown status values - and
sometimes no JSON at all. Every shape has to degrade gracefully (fall back to the
previous plan / a failed verdict), never raise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from helpers import ScriptedModel, graph_state, tool_call

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
from wqagent.graph.state import TODO_STATUSES, make_todo
from wqagent.tools.graph_tools import TodoWriteTool, parse_todo_updates

PLAN = {
    "plan_summary": "write it, then run it",
    "todos": [{"id": "t1", "content": "write hello.py"},
              {"id": "t2", "content": "run it", "note": "python hello.py"}],
    "acceptance_criteria": ["prints Hello"],
    "verification_commands": ["python hello.py"],
}


class ContentModel:
    """Always answers with raw content - never a tool call."""

    def __init__(self, content: str) -> None:
        self.content = content

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "ContentModel":
        return self

    def invoke(self, messages: list[Any]) -> Any:
        return AIMessage(content=self.content)


class NoToolChoiceModel(ContentModel):
    """A provider that rejects ``tool_choice`` outright."""

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "ContentModel":
        if "tool_choice" in kwargs:
            raise TypeError("this model does not support tool_choice")
        return self


class _BrokenBound:
    """A bound model whose request fails the way a provider would reject it."""

    def invoke(self, messages: list[Any]) -> Any:
        raise RuntimeError("400 unknown field: tool_choice")


class BrokenToolCallModel(ContentModel):
    """Forced tool calling fails at request time; plain invocation still works."""

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> _BrokenBound:
        return _BrokenBound()


class HostileModel:
    """Every call fails."""

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "HostileModel":
        raise RuntimeError("no tools here")

    def invoke(self, messages: list[Any]) -> Any:
        raise RuntimeError("model offline")


@pytest.fixture()
def revision_state(tmp_path: Path) -> Any:
    """A state whose planner is revising a previous plan (so it calls the model)."""
    return graph_state("do the thing", root=tmp_path / "ws",
                       todos=[make_todo("t0", "old plan")], last_error="command failed: exit 1")


@pytest.fixture()
def verifier_state(tmp_path: Path) -> Any:
    """A state with one open todo and one verification command that passes."""
    return graph_state("do the thing", root=tmp_path / "ws",
                       todos=[make_todo("t1", "do it")], verification_commands=["echo ok"])


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


@pytest.mark.parametrize(
    "garbage",
    [None, "not a list", [], [None, 42, "text", {"content": ""}]],
    ids=["none", "string", "empty-list", "list-of-junk"],
)
def test_coerce_todos_falls_back_to_the_previous_plan(garbage: Any) -> None:
    assert _coerce_todos(garbage, fallback=[make_todo("t0", "old")])[0]["id"] == "t0"


def test_coerce_todos_normalises_ids_and_statuses() -> None:
    # a single object instead of an array
    single = _coerce_todos({"id": "t1", "content": "only one"})
    assert [todo["id"] for todo in single] == ["t1"]

    # unknown status is normalised, missing id is generated
    todos = _coerce_todos([{"content": "no id", "status": "DONE"}, {"id": "t2", "content": "ok"}])
    assert [(todo["id"], todo["status"]) for todo in todos] == [("t1", "pending"), ("t2", "pending")]
    assert all(todo["status"] in TODO_STATUSES for todo in todos)


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, []),
        ("nope", []),
        (5, []),  # not iterable at all
        ([None, 1, {"detail": "no name"}, {"name": ""}], []),
        ({"name": "it works", "passed": "yes"},
         [{"name": "it works", "passed": True, "detail": ""}]),
    ],
)
def test_coerce_checks_tolerates_garbage(value: Any, expected: list[dict[str, Any]]) -> None:
    assert _coerce_checks(value) == expected


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


@pytest.mark.parametrize("garbage", [None, 5, "nope"], ids=["none", "scalar", "string"])
def test_parse_todo_updates_ignores_junk(garbage: Any) -> None:
    assert parse_todo_updates(garbage) == ([], [])


def test_parse_todo_updates_accepts_a_list_or_a_single_object() -> None:
    updates, problems = parse_todo_updates([{"id": "t1", "status": "completed"}])
    assert [update.id for update in updates] == ["t1"] and problems == []

    # a single object instead of a list
    updates, _ = parse_todo_updates({"id": "t1", "status": "completed"})
    assert [update.id for update in updates] == ["t1"]


def test_parse_todo_updates_reports_problems_instead_of_raising() -> None:
    updates, problems = parse_todo_updates(
        [{"id": "t1", "status": "done"}, "junk", {"status": "completed"}]
    )
    assert updates == []
    assert len(problems) == 3  # reported, not raised


# -- the structured-output request chain -------------------------------------
@pytest.mark.parametrize(
    "message, expected",
    [
        pytest.param(
            tool_call("todo_write", {"plan_summary": "from the tool call"}),
            {"plan_summary": "from the tool call"},
            id="matching-tool-call-wins",
        ),
        pytest.param(
            AIMessage(content='{"plan_summary": "from content"}',
                      tool_calls=[{"name": "file_write", "args": {"file_path": "x"},
                                   "id": "c1", "type": "tool_call"}]),
            {"plan_summary": "from content"},
            id="unrelated-tool-call-ignored",
        ),
    ],
)
def test_request_structured_prefers_the_matching_tool_call(
    message: AIMessage, expected: dict[str, Any]
) -> None:
    assert _request_structured(ScriptedModel([message]), TodoWriteTool(), []) == expected


@pytest.mark.parametrize(
    "model",
    [NoToolChoiceModel(json.dumps(PLAN)), BrokenToolCallModel(json.dumps(PLAN))],
    ids=["no-tool-choice-support", "forced-call-fails"],
)
def test_request_structured_recovers_without_forced_calling(model: ContentModel) -> None:
    args = _request_structured(model, TodoWriteTool(), [])

    assert args["plan_summary"] == PLAN["plan_summary"]
    assert [todo["id"] for todo in args["todos"]] == ["t1", "t2"]


def test_request_structured_returns_empty_when_everything_fails() -> None:
    assert _request_structured(HostileModel(), TodoWriteTool(), []) == {}


# -- planner end to end ------------------------------------------------------
# The supervisor planner reads its plan from `todo_write` *tool calls*, so a model
# that answers with prose/JSON content cannot corrupt the plan - the previous one
# is kept. Tolerance now lives in the tool-argument path.
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

GARBAGE_PLAN_ARGS: list[tuple[str, Any]] = [
    ("missing-todos", {"plan_summary": "only a summary"}),
    ("todos-not-a-list", {"plan_summary": "x", "todos": "not a list"}),
    ("todos-of-junk", {"plan_summary": "x", "todos": [None, 42, "", {"id": "", "content": ""}]}),
    ("wrong-types", {"plan_summary": 42, "todos": "nope", "acceptance_criteria": 7,
                     "verification_commands": True}),
    ("deep-garbage", {"todos": {"nested": {"weird": [1, {"x": None}]}}}),
    ("empty-object", {}),
]


def _assert_usable_plan(update: dict[str, Any]) -> None:
    """The planner always returns a complete, typed update, whatever the model said."""
    assert PLANNER_KEYS <= set(update)
    assert isinstance(update["plan_summary"], str)
    assert isinstance(update["acceptance_criteria"], list)
    assert isinstance(update["verification_commands"], list)


@pytest.mark.parametrize("name, content", JUNK_CONTENT, ids=[n for n, _ in JUNK_CONTENT])
def test_planner_keeps_the_previous_plan_when_no_tool_is_called(
    name: str, content: str, revision_state: Any
) -> None:
    update = planner_node(revision_state, model=ContentModel(content))

    _assert_usable_plan(update)
    # ... and the plan survives untouched
    assert [todo["id"] for todo in update["todos"]] == ["t0"]
    assert all(todo["status"] in TODO_STATUSES for todo in update["todos"])
    assert update["agent_handoffs"] == []


@pytest.mark.parametrize("name, args", GARBAGE_PLAN_ARGS, ids=[n for n, _ in GARBAGE_PLAN_ARGS])
def test_planner_survives_garbage_todo_write_arguments(
    name: str, args: Any, revision_state: Any
) -> None:
    model = ScriptedModel([tool_call("todo_write", args), AIMessage(content="done")])

    update = planner_node(revision_state, model=model)

    _assert_usable_plan(update)
    assert update["todos"], "the planner must always return a usable plan"
    assert all(todo["status"] in TODO_STATUSES for todo in update["todos"])
    assert all(isinstance(todo["id"], str) and todo["id"] for todo in update["todos"])


@pytest.mark.parametrize(
    "args, expected_ids",
    [
        pytest.param(
            {"todos": [{"id": "t1", "content": "only todos survived"}]},
            ["t1"],
            id="partial-arguments",
        ),
        pytest.param(
            {"plan_summary": 42,
             "todos": [None, 42, {"id": "t1", "content": "keep me", "status": "DONE"},
                       {"content": ""}],
             "acceptance_criteria": 7,
             "verification_commands": {"not": "a list"}},
            ["t1"],
            id="type-and-status-garbage",
        ),
    ],
)
def test_planner_normalises_partial_or_garbage_arguments(
    args: Any, expected_ids: list[str], revision_state: Any
) -> None:
    """Usable fields win, junk is dropped and defaults fill the rest - never a crash."""
    model = ScriptedModel([tool_call("todo_write", args), AIMessage(content="planned")])

    update = planner_node(revision_state, model=model)

    assert [todo["id"] for todo in update["todos"]] == expected_ids
    assert [todo["status"] for todo in update["todos"]] == ["pending"]  # defaults, not a crash
    assert update["acceptance_criteria"] == []
    assert update["verification_commands"] == []
    assert update["plan_summary"] == ""  # nothing usable -> state fallback


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
    state = graph_state("do the thing", root=tmp_path / "ws", todos=[make_todo("t0", "old plan")])

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
    name: str, content: str, verifier_state: Any
) -> None:
    update = verifier_node(verifier_state, model=ContentModel(content))

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
    name: str, content: str, verifier_state: Any
) -> None:
    update = verifier_node(verifier_state, model=ContentModel(content))

    assert update["passed"] is True
    assert update["verification_checks"] == []
    assert update["last_error"] == ""
    assert update["todos"][0]["status"] == "completed"
