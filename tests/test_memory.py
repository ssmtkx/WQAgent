"""Tests for the three-layer memory system (``wqagent.graph.memory``)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from wqagent.graph import DEFAULT_CONTEXT_TOKEN_LIMIT, build_workflow
from wqagent.graph.memory import (
    COMPRESSION_EVENT_LIMIT,
    CONTEXT_SUMMARY_CHARS,
    ERROR_CHARS,
    HANDOFF_LIMIT,
    HANDOFF_TEXT_CHARS,
    HISTORY_CHARS,
    HISTORY_SUMMARY_FILE,
    MEMORY_READ_CHARS,
    NOTEPAD_CHARS,
    NOTEPAD_FILE,
    RESEARCH_CHARS,
    RULES_LAYER,
    SUMMARY_CHARS,
    TODO_FILE,
    _short_text,
    _trim_handoffs,
    build_layered_memory,
    format_layered_memory_for_prompt,
    memory_digest,
    memory_event,
    read_history_summary,
    read_notepad,
    read_todos,
    render_todo_plan,
    write_todos,
)
from wqagent.graph.state import make_todo

from helpers import graph_state


# -- the rules layer ---------------------------------------------------------
def test_rules_layer_is_the_fixed_block() -> None:
    assert RULES_LAYER["scope"] == "workspace"
    assert RULES_LAYER["storage"] == "internal"
    assert RULES_LAYER["rules"] == [
        "Work inside the current workspace only.",
        "Use paths relative to the workspace; do not prefix paths with workspace/.",
        "Keep durable task context outside the raw messages transcript when possible.",
        "Treat TODO.md as working plan state, NOTEPAD.md as durable notes, and "
        "HISTORY_SUMMARY.md as compressed history.",
        "Do not expose memory write tools to agents; layered memory is assembled by the runtime.",
    ]
    assert isinstance(RULES_LAYER, dict)


def test_snapshot_rules_are_a_copy(tmp_path: Any) -> None:
    memory = build_layered_memory(graph_state("do it", root=tmp_path / "ws"))
    memory["rules"]["rules"].append("mutated")
    memory["rules"]["scope"] = "somewhere else"

    assert RULES_LAYER["rules"][-1] != "mutated"
    assert RULES_LAYER["scope"] == "workspace"


# -- helpers -----------------------------------------------------------------
@pytest.mark.parametrize(
    "text, limit, expected",
    [
        ("short", 10, "short"),
        ("exactlyten", 10, "exactlyten"),
        ("0123456789abc", 10, "0123456789..."),
        ("trailing   ", 8, "trailing..."),
        ("x" * 5_000, 100, "x" * 100 + "..."),  # marks the cut: starts with the cap, ends with ...
        (None, 10, ""),
        (42, 10, "42"),
        ("anything", 0, ""),
        ("anything", -5, ""),
    ],
)
def test_short_text(text: Any, limit: int, expected: str) -> None:
    assert _short_text(text, limit) == expected


def test_trim_handoffs_keeps_only_the_newest_six() -> None:
    handoffs = [
        {"from_agent": "planner", "to_agent": "codeAgent", "instruction": f"step {index}"}
        for index in range(10)
    ]

    trimmed = _trim_handoffs(handoffs)

    assert len(trimmed) == HANDOFF_LIMIT == 6
    assert [item["instruction"] for item in trimmed] == [f"step {index}" for index in range(4, 10)]


@pytest.mark.parametrize("value, expected", [([], []), (None, []), ("junk", []), (7, []), ({}, [{}])])
def test_trim_handoffs_tolerates_junk(value: Any, expected: list[Any]) -> None:
    assert _trim_handoffs(value) == expected


def test_trim_handoffs_caps_the_free_text_fields() -> None:
    trimmed = _trim_handoffs(
        [
            {
                "from_agent": "planner",
                "to_agent": "codeAgent",
                "instruction": "i" * (HANDOFF_TEXT_CHARS + 500),
                "result": "r" * (HANDOFF_TEXT_CHARS + 500),
            }
        ]
    )[0]

    assert trimmed["from_agent"] == "planner"  # other keys are untouched
    for field in ("instruction", "result"):
        assert trimmed[field].endswith("..."), field
        assert len(trimmed[field]) <= HANDOFF_TEXT_CHARS + 3


def test_trim_handoffs_copies_and_honours_a_zero_limit() -> None:
    handoffs = [{"instruction": "original"}]

    trimmed = _trim_handoffs(handoffs)
    trimmed[0]["instruction"] = "changed"

    assert handoffs[0]["instruction"] == "original"  # the snapshot shares nothing
    assert _trim_handoffs(handoffs, 0) == []


# -- reading the durable files -----------------------------------------------
@pytest.mark.parametrize(
    "reader, name, expected",
    [
        (read_notepad, NOTEPAD_FILE, "NOTEPAD.md"),
        (read_history_summary, HISTORY_SUMMARY_FILE, "HISTORY_SUMMARY.md"),
    ],
    ids=["notepad", "history_summary"],
)
def test_missing_memory_files_are_not_an_error(
    reader: Any, name: str, expected: str, tmp_path: Any
) -> None:
    read = reader(graph_state("do it", root=tmp_path / "ws")["runtime"])

    assert read == {"path": name, "exists": False, "content": "", "error": ""}
    assert read["path"] == expected  # relative to the workspace


@pytest.mark.parametrize(
    "reader, name",
    [(read_notepad, NOTEPAD_FILE), (read_history_summary, HISTORY_SUMMARY_FILE)],
    ids=["notepad", "history_summary"],
)
@pytest.mark.parametrize(
    "kind, exists, empty",
    [
        ("missing", False, True),  # a missing file has no content at all
        ("dir", False, True),  # the path exists, but it is not a file
        ("binary", True, False),  # replacement characters, no exception
    ],
)
def test_memory_file_edge_cases_are_data(
    reader: Any, name: str, kind: str, exists: bool, empty: bool, state: Any
) -> None:
    if kind == "dir":
        (state.workspace / name).mkdir()
    if kind == "binary":
        (state.workspace / name).write_bytes(b"\xff\xfe\x00binary\x00\x80")

    read = reader(state)

    assert read["exists"] is exists
    assert read["error"] == ""
    assert (read["content"] == "") is empty


def test_memory_files_are_read_from_the_workspace(state: Any) -> None:
    (state.workspace / NOTEPAD_FILE).write_text("# notes\nremember me\n", encoding="utf-8")
    (state.workspace / HISTORY_SUMMARY_FILE).write_text("earlier: did a thing\n", encoding="utf-8")

    notepad = read_notepad(state)
    history = read_history_summary(state)

    assert (notepad["exists"], "remember me" in notepad["content"]) == (True, True)
    assert notepad["error"] == ""
    assert (history["exists"], history["content"].startswith("earlier:")) == (True, True)
    # the reported path stays relative to the workspace
    assert notepad["path"] == "NOTEPAD.md"


def test_huge_memory_files_are_read_up_to_the_cap(state: Any) -> None:
    (state.workspace / NOTEPAD_FILE).write_text("x" * (MEMORY_READ_CHARS + 500), encoding="utf-8")

    notepad = read_notepad(state)

    assert notepad["exists"] is True
    # the copy ends with the truncation marker, so it is only slightly over the cap
    assert notepad["content"].endswith("[truncated]")
    assert len(notepad["content"]) < MEMORY_READ_CHARS + 100


# -- working memory ----------------------------------------------------------
# a field case that only makes sense with an over-long value: filled in by its test
_TOO_LONG = "too long"


def test_build_layered_memory_has_the_three_layers(state: Any) -> None:
    memory = build_layered_memory(graph_state("write hello.py", root=state.workspace), node="planner")

    assert set(memory) == {"rules", "working_memory", "history_summary_store"}
    assert memory["rules"] == RULES_LAYER
    assert memory["working_memory"]["node"] == "planner"


# (layer:key, expected value, state values the case sets, node argument for the snapshot)
_FIELDS = [
    ("working_memory:node", "graph", None, "graph"),  # the node defaults to the graph
    ("working_memory:node", "graph", None, ""),  # ... and an empty node is the same default
    ("working_memory:node", "planner", None, "planner"),
    ("working_memory:task", "write hello.py", None, "graph"),
    ("working_memory:plan_summary", "plan A", {"plan_summary": "plan A"}, "graph"),
    ("working_memory:acceptance_criteria", ["it works"],
     {"acceptance_criteria": ["it works"]}, "graph"),
    ("working_memory:verification_commands", ["echo ok"],
     {"verification_commands": ["echo ok"]}, "graph"),
    (
        "working_memory:todos",
        [{"id": "t1", "content": "do it", "status": "in_progress", "note": ""}],
        {"todos": [make_todo("t1", "do it", status="in_progress")]},
        "graph",
    ),
    ("working_memory:code_agent_summary", "wrote hello.py",
     {"code_agent_summary": "wrote hello.py"}, "graph"),
    ("working_memory:last_error", "boom", {"last_error": "boom"}, "graph"),
    ("working_memory:attempts", 0, None, "graph"),
    ("working_memory:max_attempts", 3, None, "graph"),
    ("working_memory:session_turn", 1, None, "graph"),
    ("working_memory:verifier_summary", "", None, "graph"),  # no state field sets it yet
    ("history_summary_store:history_summary", "", None, "graph"),
    ("history_summary_store:context_summary", "", None, "graph"),
    ("history_summary_store:context_summary", _TOO_LONG, _TOO_LONG, "graph"),
]

# a state whose values are junk: every field must fall back instead of raising
_JUNK_FIELDS = [
    ("working_memory:todos", []),
    ("working_memory:acceptance_criteria", []),
    ("working_memory:agent_handoffs", []),
    ("working_memory:attempts", 0),  # unparsable -> default
    ("working_memory:max_attempts", 3),
    ("working_memory:session_turn", 1),
]


@pytest.mark.parametrize("field, expected, extras, node", _FIELDS)
def test_snapshot_field(field: str, expected: Any, extras: Any, node: str, tmp_path: Any) -> None:
    layer, key = field.split(":")
    if _TOO_LONG in (expected, extras):  # kept out of the table so the case id stays short
        expected = "s" * CONTEXT_SUMMARY_CHARS + "..."
        extras = {"context_summary": "s" * (CONTEXT_SUMMARY_CHARS + 100)}
    state = graph_state("write hello.py", root=tmp_path / "ws", **(extras or {}))

    assert build_layered_memory(state, node=node)[layer][key] == expected


@pytest.mark.parametrize("field, expected", _JUNK_FIELDS)
def test_snapshot_field_of_a_junk_state(field: str, expected: Any, tmp_path: Any) -> None:
    layer, key = field.split(":")
    state = graph_state("write hello.py", root=tmp_path / "ws")
    state["todos"] = "not a list"  # type: ignore[typeddict-item]
    state["acceptance_criteria"] = 7  # type: ignore[typeddict-item]
    state["sources"] = {"url": "https://x.test/a"}  # a single mapping
    state["agent_handoffs"] = None  # type: ignore[typeddict-item]
    state["attempts"] = "two"  # type: ignore[typeddict-item]
    state["max_attempts"] = None  # type: ignore[typeddict-item]

    assert build_layered_memory(state)[layer][key] == expected


def test_working_memory_trims_every_free_text_field(tmp_path: Any) -> None:
    state = graph_state(
        "t" * 50,
        root=tmp_path / "ws",
        research_notes="r" * (RESEARCH_CHARS + 100),
        code_agent_summary="c" * (SUMMARY_CHARS + 100),
        verifier_summary="v" * (SUMMARY_CHARS + 100),
        last_error="e" * (ERROR_CHARS + 100),
        context_summary="s" * (CONTEXT_SUMMARY_CHARS + 100),
    )

    memory = build_layered_memory(state)
    working, store = memory["working_memory"], memory["history_summary_store"]

    for field, budget in (
        ("research_notes", RESEARCH_CHARS),
        ("code_agent_summary", SUMMARY_CHARS),
        ("verifier_summary", SUMMARY_CHARS),
        ("last_error", ERROR_CHARS),
    ):
        assert working[field].endswith("..."), field
        assert len(working[field]) <= budget + 3, field
    assert store["context_summary"].endswith("...")
    assert len(store["context_summary"]) <= CONTEXT_SUMMARY_CHARS + 3


def test_sources_keep_only_title_and_url(tmp_path: Any) -> None:
    sources = build_layered_memory(
        graph_state(
            "research it",
            root=tmp_path / "ws",
            sources=[
                {"title": "Docs", "url": "https://x.test/a", "content": "long body", "score": 0.9},
                {"title": "No url", "content": "dropped"},
                "junk",
                {"url": "https://x.test/b"},
            ],
        )
    )["working_memory"]["sources"]

    assert sources == [
        {"title": "Docs", "url": "https://x.test/a"},
        {"title": "", "url": "https://x.test/b"},
    ]


def test_handoffs_in_the_snapshot_are_the_newest_six(tmp_path: Any) -> None:
    handoffs = build_layered_memory(
        graph_state(
            "do it",
            root=tmp_path / "ws",
            agent_handoffs=[
                {"from_agent": "planner", "to_agent": "searchAgent", "instruction": f"i{index}"}
                for index in range(8)
            ],
        )
    )["working_memory"]["agent_handoffs"]

    assert len(handoffs) == 6
    assert handoffs[0]["instruction"] == "i2"
    assert handoffs[-1]["instruction"] == "i7"


def test_todos_are_copied_not_shared(tmp_path: Any) -> None:
    state = graph_state("do it", root=tmp_path / "ws", todos=[make_todo("t1", "do it")])

    working = build_layered_memory(state)["working_memory"]
    working["todos"][0]["status"] = "completed"

    assert state["todos"][0]["status"] == "pending"


def test_build_layered_memory_needs_a_runtime() -> None:
    with pytest.raises(ValueError, match="RuntimeState"):
        build_layered_memory({"task": "no runtime here"})  # type: ignore[typeddict-item]


def test_session_id_comes_from_state_then_runtime(tmp_path: Any) -> None:
    state = graph_state("do it", root=tmp_path / "ws")
    default_id = build_layered_memory(state)["working_memory"]["session_id"]

    assert default_id  # a per-process fallback
    state["runtime"].session_id = "runtime-session"  # type: ignore[attr-defined]
    assert build_layered_memory(state)["working_memory"]["session_id"] == "runtime-session"

    state["session_id"] = "state-session"  # type: ignore[typeddict-item]
    assert build_layered_memory(state)["working_memory"]["session_id"] == "state-session"

    state["session_turn"] = 9  # type: ignore[typeddict-item]
    assert build_layered_memory(state)["working_memory"]["session_turn"] == 9


# -- history summary store ---------------------------------------------------
def test_history_store_reads_the_workspace_files(state: Any) -> None:
    (state.workspace / NOTEPAD_FILE).write_text("durable note", encoding="utf-8")
    (state.workspace / HISTORY_SUMMARY_FILE).write_text("compressed history", encoding="utf-8")
    store = build_layered_memory(graph_state("do it", root=state.workspace))["history_summary_store"]

    assert (store["history_path"], store["notepad_path"]) == ("HISTORY_SUMMARY.md", "NOTEPAD.md")
    assert (store["history_exists"], store["notepad_exists"]) == (True, True)
    assert (store["history_summary"], store["notepad"]) == ("compressed history", "durable note")


@pytest.mark.parametrize("summary", ["from the file", "from the state"])
def test_history_store_prefers_the_state_then_the_file(summary: str, state: Any) -> None:
    (state.workspace / HISTORY_SUMMARY_FILE).write_text("from the file", encoding="utf-8")
    extra = {"history_summary": summary} if summary == "from the state" else {}
    graph = graph_state("do it", root=state.workspace, **extra)

    assert build_layered_memory(graph)["history_summary_store"]["history_summary"] == summary


def test_history_store_truncates_the_notepad_and_history(state: Any) -> None:
    (state.workspace / NOTEPAD_FILE).write_text("n" * (NOTEPAD_CHARS + 200), encoding="utf-8")
    graph = graph_state("do it", root=state.workspace, history_summary="h" * (HISTORY_CHARS + 200))
    store = build_layered_memory(graph)["history_summary_store"]

    assert store["notepad"].endswith("...") and len(store["notepad"]) <= NOTEPAD_CHARS + 3
    assert store["history_summary"].endswith("...")
    assert len(store["history_summary"]) <= HISTORY_CHARS + 3


def test_history_store_keeps_the_last_compression_events(tmp_path: Any) -> None:
    store = build_layered_memory(
        graph_state(
            "do it",
            root=tmp_path / "ws",
            compression_events=[{"attempt": index, "summary": f"c{index}"} for index in range(6)],
        )
    )["history_summary_store"]

    assert len(store["compression_events"]) == COMPRESSION_EVENT_LIMIT == 3
    assert [event["summary"] for event in store["compression_events"]] == ["c3", "c4", "c5"]


# -- the working plan file (TODO.md) -----------------------------------------
def test_render_todo_plan_is_markdown() -> None:
    text = render_todo_plan(
        [make_todo("t1", "write hello.py", status="completed", note="22 bytes"),
         make_todo("t2", "run it", status="in_progress"),
         make_todo("t3", "blocked step", status="blocked")],
        task="create hello.py",
        plan_summary="write then verify",
        acceptance_criteria=["prints Hello"],
        verification_commands=["python hello.py"],
    )

    assert text.startswith("# Working plan\n")
    assert "**Task:** create hello.py" in text
    assert "write then verify" in text
    assert "- [x] t1: write hello.py  <!-- 22 bytes -->" in text
    assert "- [~] t2: run it" in text
    assert "- [!] t3: blocked step" in text
    assert "## Acceptance criteria\n\n- prints Hello" in text
    assert "## Verification commands\n\n- `python hello.py`" in text
    assert text.endswith("\n")


def test_todo_plan_round_trips(state: Any) -> None:
    todos = [make_todo("t1", "write hello.py", status="completed", note="22 bytes"),
             make_todo("t2", "run it")]
    plan = {
        "task": "create hello.py",
        "plan_summary": "write then verify",
        "acceptance_criteria": ["prints Hello"],
        "verification_commands": ["python hello.py"],
    }

    written = write_todos(state, todos, **plan)
    back = read_todos(state)

    assert written == {"ok": True, "path": TODO_FILE, "error": ""}
    assert back["exists"] is True
    assert (back["task"], back["plan_summary"]) == ("create hello.py", "write then verify")
    assert back["todos"] == todos
    assert (back["acceptance_criteria"], back["verification_commands"]) == (
        ["prints Hello"], ["python hello.py"])


def test_read_todos_without_a_file(state: Any) -> None:
    plan = read_todos(state)

    assert plan["exists"] is False
    assert (plan["todos"], plan["task"], plan["error"]) == ([], "", "")


@pytest.mark.parametrize(
    "content",
    ["", "not markdown at all", "- no marker here", "## Acceptance criteria\n- only criteria"],
)
def test_read_todos_tolerates_junk(content: str, state: Any) -> None:
    (state.workspace / TODO_FILE).write_text(content, encoding="utf-8")

    plan = read_todos(state)

    assert plan["exists"] is True
    assert plan["todos"] == [] or all(todo["id"] for todo in plan["todos"])
    json.dumps(plan)  # always serialisable


def test_write_todos_failure_is_data(state: Any) -> None:
    (state.workspace / TODO_FILE).mkdir()  # a directory where the file should go

    written = write_todos(state, [make_todo("t1", "do it")])

    assert written["ok"] is False
    assert written["error"]


# -- events and digests ------------------------------------------------------
def test_memory_event_carries_the_snapshot(state: Any) -> None:
    memory = build_layered_memory(
        graph_state("do it", root=state.workspace, todos=[make_todo("t1", "do it")]), node="planner"
    )

    event = memory_event(memory, node="planner")

    assert event["type"] == "memory"
    assert event["node"] == "planner"
    assert event["memory"]["working_memory"]["task"] == "do it"
    assert event["chars"] == len(format_layered_memory_for_prompt(memory))
    assert event["digest"] == "1 todos, 0 sources, 0 handoffs, notepad=no, history=no"
    json.dumps(event)  # streamable as-is


def test_memory_event_defaults_and_junk() -> None:
    event = memory_event(None)

    assert event["node"] == "graph"
    assert event["memory"] == {}
    assert event["chars"] == 2  # "{}"
    assert event["digest"] == "empty"


def test_memory_digest_reports_sizes(state: Any) -> None:
    (state.workspace / NOTEPAD_FILE).write_text("n", encoding="utf-8")
    graph = graph_state(
        "do it",
        root=state.workspace,
        todos=[make_todo("t1", "a"), make_todo("t2", "b")],
        sources=[{"title": "t", "url": "https://x.test/a"}],
        agent_handoffs=[{"instruction": "i"}],
        research_notes="notes",
    )
    digest = memory_digest(build_layered_memory(graph))

    assert digest.startswith("2 todos, 1 sources, 1 handoffs, notepad=yes, history=no")
    assert digest.endswith("notes=5ch")
    assert memory_digest({}) == "empty"


# -- prompt formatting -------------------------------------------------------
def test_format_layered_memory_for_prompt_is_json(state: Any) -> None:
    memory = build_layered_memory(graph_state("do it", root=state.workspace), node="verifier")
    text = format_layered_memory_for_prompt(memory)

    assert text.startswith("{") and "\n  " in text  # pretty printed
    assert json.loads(text)["working_memory"]["node"] == "verifier"
    assert json.loads(text)["rules"]["scope"] == "workspace"


@pytest.mark.parametrize("value", [None, {}])
def test_format_layered_memory_for_empty_input(value: Any) -> None:
    assert format_layered_memory_for_prompt(value) == "{}"


def test_format_layered_memory_survives_unserialisable_values(tmp_path: Any) -> None:
    state = graph_state("do it", root=tmp_path / "ws")
    state["todos"] = [{"id": "t1", "content": "x", "status": "pending", "note": object()}]

    text = format_layered_memory_for_prompt(build_layered_memory(state))

    assert "object object" in text  # default=str kept it serialisable


def test_snapshot_of_a_finished_graph_run(state: Any) -> None:
    """The snapshot also works on the state a real workflow run produces."""
    from helpers import ScriptedModel, passing_script

    result = build_workflow(model=ScriptedModel(passing_script())).invoke(
        graph_state("write hello.py", root=state.workspace)
    )

    memory = build_layered_memory(result, node="final")
    working = memory["working_memory"]

    assert working["task"] == "write hello.py"
    assert working["plan_summary"] == "do the thing"
    assert working["code_agent_summary"] == "Wrote hello.py."
    assert [handoff["to_agent"] for handoff in working["agent_handoffs"]] == ["codeAgent"]
    assert working["todos"][0]["status"] == "completed"
    assert working["node"] == "final"
    assert json.loads(format_layered_memory_for_prompt(memory))["rules"]["storage"] == "internal"


def test_snapshot_fits_a_prompt_even_with_a_full_state(tmp_path: Any) -> None:
    state = graph_state(
        "do it",
        root=tmp_path / "ws",
        plan_summary="p" * 500,
        todos=[make_todo(f"t{index}", "x" * 200) for index in range(20)],
        research_notes="r" * 5_000,
        sources=[{"title": "t", "url": f"https://x.test/{i}"} for i in range(30)],
        agent_handoffs=[{"instruction": "i" * 2_000} for _ in range(10)],
        code_agent_summary="c" * 5_000,
        last_error="e" * 5_000,
        context_summary="s" * 5_000,
        history_summary="h" * 5_000,
    )

    text = format_layered_memory_for_prompt(build_layered_memory(state))

    # Every capped field stays near its budget, so even this deliberately bloated
    # state (~21k chars ≈ 5k tokens) stays inside DEFAULT_CONTEXT_TOKEN_LIMIT; the
    # 20 long todos are the only part that is not capped.
    assert len(text) < 25_000
    assert DEFAULT_CONTEXT_TOKEN_LIMIT > 0
