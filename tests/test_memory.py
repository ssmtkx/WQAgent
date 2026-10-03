"""Tests for the three-layer memory system (``wqagent.graph.memory``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from wqagent.core.state import RuntimeState
from wqagent.graph.memory import (
    COMPRESSION_EVENT_LIMIT,
    CONTEXT_SUMMARY_CHARS,
    DEFAULT_CONTEXT_TOKEN_LIMIT,
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
from wqagent.graph.state import make_todo, new_graph_state


def _runtime(tmp_path: Path, **kwargs: Any) -> RuntimeState:
    return RuntimeState(workspace=tmp_path / "ws", **kwargs)


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


def test_snapshot_rules_are_a_copy(tmp_path: Path) -> None:
    state = new_graph_state("do it", _runtime(tmp_path))

    memory = build_layered_memory(state)
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
        (None, 10, ""),
        (42, 10, "42"),
        ("anything", 0, ""),
        ("anything", -5, ""),
    ],
)
def test_short_text(text: Any, limit: int, expected: str) -> None:
    assert _short_text(text, limit) == expected


def test_short_text_marks_the_cut() -> None:
    capped = _short_text("x" * 5_000, 100)

    assert capped.endswith("...")
    assert len(capped) <= 103
    assert capped.startswith("x" * 100)


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
    handoffs = [
        {
            "from_agent": "planner",
            "to_agent": "codeAgent",
            "instruction": "i" * (HANDOFF_TEXT_CHARS + 500),
            "result": "r" * (HANDOFF_TEXT_CHARS + 500),
        }
    ]

    trimmed = _trim_handoffs(handoffs)

    assert trimmed[0]["from_agent"] == "planner"  # other keys are untouched
    for field in ("instruction", "result"):
        assert trimmed[0][field].endswith("..."), field
        assert len(trimmed[0][field]) <= HANDOFF_TEXT_CHARS + 3


def test_trim_handoffs_copies_and_honours_a_zero_limit() -> None:
    handoffs = [{"instruction": "original"}]

    trimmed = _trim_handoffs(handoffs)
    trimmed[0]["instruction"] = "changed"

    assert handoffs[0]["instruction"] == "original"  # the snapshot shares nothing
    assert _trim_handoffs(handoffs, 0) == []


# -- reading the durable files -----------------------------------------------
def test_missing_memory_files_are_not_an_error(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    notepad = read_notepad(runtime)
    history = read_history_summary(runtime)

    assert notepad == {"path": NOTEPAD_FILE, "exists": False, "content": "", "error": ""}
    assert history == {"path": HISTORY_SUMMARY_FILE, "exists": False, "content": "", "error": ""}


def test_memory_files_are_read_from_the_workspace(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.workspace.mkdir(parents=True, exist_ok=True)
    (runtime.workspace / NOTEPAD_FILE).write_text("# notes\nremember me\n", encoding="utf-8")
    (runtime.workspace / HISTORY_SUMMARY_FILE).write_text("earlier: did a thing\n", encoding="utf-8")

    notepad = read_notepad(runtime)
    history = read_history_summary(runtime)

    assert notepad["exists"] is True
    assert "remember me" in notepad["content"]
    assert notepad["error"] == ""
    assert history["exists"] is True
    assert history["content"].startswith("earlier:")
    # the reported path stays relative to the workspace
    assert notepad["path"] == "NOTEPAD.md"


def test_a_directory_named_like_the_notepad_is_not_a_file(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    (runtime.workspace / NOTEPAD_FILE).mkdir(parents=True)

    notepad = read_notepad(runtime)

    assert notepad["exists"] is False
    assert notepad["content"] == ""


def test_huge_memory_files_are_read_up_to_the_cap(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.workspace.mkdir(parents=True, exist_ok=True)
    (runtime.workspace / NOTEPAD_FILE).write_text("x" * (MEMORY_READ_CHARS + 500), encoding="utf-8")

    notepad = read_notepad(runtime)

    assert notepad["exists"] is True
    assert notepad["content"].endswith("[truncated]")
    assert len(notepad["content"]) < MEMORY_READ_CHARS + 100


def test_binary_memory_file_does_not_raise(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.workspace.mkdir(parents=True, exist_ok=True)
    (runtime.workspace / NOTEPAD_FILE).write_bytes(b"\xff\xfe\x00binary\x00\x80")

    notepad = read_notepad(runtime)

    assert notepad["exists"] is True  # replacement characters, no exception
    assert notepad["error"] == ""


# -- working memory ----------------------------------------------------------
def test_build_layered_memory_has_the_three_layers(tmp_path: Path) -> None:
    state = new_graph_state("write hello.py", _runtime(tmp_path))

    memory = build_layered_memory(state, node="planner")

    assert set(memory) == {"rules", "working_memory", "history_summary_store"}
    assert memory["rules"] == RULES_LAYER
    assert memory["working_memory"]["node"] == "planner"
    assert memory["working_memory"]["task"] == "write hello.py"
    assert memory["working_memory"]["attempts"] == 0
    assert memory["working_memory"]["max_attempts"] == 3
    assert memory["working_memory"]["session_turn"] == 1


def test_node_defaults_to_graph(tmp_path: Path) -> None:
    state = new_graph_state("do it", _runtime(tmp_path))

    assert build_layered_memory(state)["working_memory"]["node"] == "graph"
    assert build_layered_memory(state, node="")["working_memory"]["node"] == "graph"


def test_working_memory_carries_the_task_state(tmp_path: Path) -> None:
    state = new_graph_state(
        "ship it",
        _runtime(tmp_path),
        plan_summary="plan A",
        todos=[make_todo("t1", "do it", status="in_progress")],
        acceptance_criteria=["it works"],
        verification_commands=["echo ok"],
        attempts=2,
        code_agent_summary="wrote hello.py",
        last_error="boom",
    )

    working = build_layered_memory(state)["working_memory"]

    assert working["plan_summary"] == "plan A"
    assert working["todos"] == [{"id": "t1", "content": "do it", "status": "in_progress", "note": ""}]
    assert working["acceptance_criteria"] == ["it works"]
    assert working["verification_commands"] == ["echo ok"]
    assert working["code_agent_summary"] == "wrote hello.py"
    assert working["last_error"] == "boom"
    assert working["session_turn"] == 3  # attempts + 1
    assert working["verifier_summary"] == ""  # no state field sets it yet


def test_working_memory_trims_every_free_text_field(tmp_path: Path) -> None:
    state = new_graph_state(
        "t" * 50,
        _runtime(tmp_path),
        research_notes="r" * (RESEARCH_CHARS + 100),
        code_agent_summary="c" * (SUMMARY_CHARS + 100),
        verifier_summary="v" * (SUMMARY_CHARS + 100),
        last_error="e" * (ERROR_CHARS + 100),
        context_summary="s" * (CONTEXT_SUMMARY_CHARS + 100),
    )

    memory = build_layered_memory(state)
    working = memory["working_memory"]
    store = memory["history_summary_store"]

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


def test_sources_keep_only_title_and_url(tmp_path: Path) -> None:
    state = new_graph_state(
        "research it",
        _runtime(tmp_path),
        sources=[
            {"title": "Docs", "url": "https://x.test/a", "content": "long body", "score": 0.9},
            {"title": "No url", "content": "dropped"},
            "junk",
            {"url": "https://x.test/b"},
        ],
    )

    sources = build_layered_memory(state)["working_memory"]["sources"]

    assert sources == [
        {"title": "Docs", "url": "https://x.test/a"},
        {"title": "", "url": "https://x.test/b"},
    ]


def test_handoffs_in_the_snapshot_are_the_newest_six(tmp_path: Path) -> None:
    state = new_graph_state(
        "do it",
        _runtime(tmp_path),
        agent_handoffs=[
            {"from_agent": "planner", "to_agent": "searchAgent", "instruction": f"i{index}"}
            for index in range(8)
        ],
    )

    handoffs = build_layered_memory(state)["working_memory"]["agent_handoffs"]

    assert len(handoffs) == 6
    assert handoffs[0]["instruction"] == "i2"
    assert handoffs[-1]["instruction"] == "i7"


def test_todos_are_copied_not_shared(tmp_path: Path) -> None:
    todo = make_todo("t1", "do it")
    state = new_graph_state("do it", _runtime(tmp_path), todos=[todo])

    working = build_layered_memory(state)["working_memory"]
    working["todos"][0]["status"] = "completed"

    assert state["todos"][0]["status"] == "pending"


def test_junk_state_values_do_not_break_the_snapshot(tmp_path: Path) -> None:
    state = new_graph_state("do it", _runtime(tmp_path))
    state["todos"] = "not a list"  # type: ignore[typeddict-item]
    state["acceptance_criteria"] = 7  # type: ignore[typeddict-item]
    state["sources"] = {"url": "https://x.test/a"}  # a single mapping
    state["agent_handoffs"] = None  # type: ignore[typeddict-item]
    state["attempts"] = "two"  # type: ignore[typeddict-item]
    state["max_attempts"] = None  # type: ignore[typeddict-item]

    working = build_layered_memory(state)["working_memory"]

    assert working["todos"] == []
    assert working["acceptance_criteria"] == []
    assert working["sources"] == [{"title": "", "url": "https://x.test/a"}]
    assert working["agent_handoffs"] == []
    assert working["attempts"] == 0  # unparsable -> default
    assert working["max_attempts"] == 3
    assert working["session_turn"] == 1


def test_build_layered_memory_needs_a_runtime() -> None:
    with pytest.raises(ValueError, match="RuntimeState"):
        build_layered_memory({"task": "no runtime here"})


def test_session_id_comes_from_state_then_runtime(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    state = new_graph_state("do it", runtime)

    default_id = build_layered_memory(state)["working_memory"]["session_id"]
    assert default_id  # a per-process fallback

    runtime.session_id = "runtime-session"  # type: ignore[attr-defined]
    assert build_layered_memory(state)["working_memory"]["session_id"] == "runtime-session"

    state["session_id"] = "state-session"  # type: ignore[typeddict-item]
    assert build_layered_memory(state)["working_memory"]["session_id"] == "state-session"

    state["session_turn"] = 9  # type: ignore[typeddict-item]
    assert build_layered_memory(state)["working_memory"]["session_turn"] == 9


# -- history summary store ---------------------------------------------------
def test_history_store_reads_the_workspace_files(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    state = new_graph_state("do it", runtime)
    runtime.workspace.mkdir(parents=True, exist_ok=True)
    (runtime.workspace / NOTEPAD_FILE).write_text("durable note", encoding="utf-8")
    (runtime.workspace / HISTORY_SUMMARY_FILE).write_text("compressed history", encoding="utf-8")

    store = build_layered_memory(state)["history_summary_store"]

    assert store["history_path"] == "HISTORY_SUMMARY.md"
    assert store["notepad_path"] == "NOTEPAD.md"
    assert store["history_exists"] is True and store["notepad_exists"] is True
    assert store["history_summary"] == "compressed history"
    assert store["notepad"] == "durable note"


def test_history_store_prefers_the_state_then_the_file(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.workspace.mkdir(parents=True, exist_ok=True)
    (runtime.workspace / HISTORY_SUMMARY_FILE).write_text("from the file", encoding="utf-8")
    state = new_graph_state("do it", runtime)

    assert build_layered_memory(state)["history_summary_store"]["history_summary"] == "from the file"

    state["history_summary"] = "from the state"
    assert build_layered_memory(state)["history_summary_store"]["history_summary"] == "from the state"


def test_history_store_truncates_the_notepad_and_history(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    state = new_graph_state("do it", runtime)
    runtime.workspace.mkdir(parents=True, exist_ok=True)
    (runtime.workspace / NOTEPAD_FILE).write_text("n" * (NOTEPAD_CHARS + 200), encoding="utf-8")
    state["history_summary"] = "h" * (HISTORY_CHARS + 200)

    store = build_layered_memory(state)["history_summary_store"]

    assert store["notepad"].endswith("...") and len(store["notepad"]) <= NOTEPAD_CHARS + 3
    assert store["history_summary"].endswith("...")
    assert len(store["history_summary"]) <= HISTORY_CHARS + 3


def test_history_store_keeps_the_last_compression_events(tmp_path: Path) -> None:
    state = new_graph_state(
        "do it",
        _runtime(tmp_path),
        compression_events=[
            {"node": "planner", "attempt": index, "summary": f"c{index}"} for index in range(6)
        ],
    )

    store = build_layered_memory(state)["history_summary_store"]

    assert len(store["compression_events"]) == COMPRESSION_EVENT_LIMIT == 3
    assert [event["summary"] for event in store["compression_events"]] == ["c3", "c4", "c5"]


# -- the working plan file (TODO.md) -----------------------------------------
def test_render_todo_plan_is_markdown(tmp_path: Path) -> None:
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


def test_todo_plan_round_trips(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    todos = [
        make_todo("t1", "write hello.py", status="completed", note="22 bytes"),
        make_todo("t2", "run it"),
    ]

    written = write_todos(
        runtime,
        todos,
        task="create hello.py",
        plan_summary="write then verify",
        acceptance_criteria=["prints Hello"],
        verification_commands=["python hello.py"],
    )
    back = read_todos(runtime)

    assert written == {"ok": True, "path": TODO_FILE, "error": ""}
    assert back["exists"] is True
    assert back["task"] == "create hello.py"
    assert back["plan_summary"] == "write then verify"
    assert back["todos"] == todos
    assert back["acceptance_criteria"] == ["prints Hello"]
    assert back["verification_commands"] == ["python hello.py"]


def test_read_todos_without_a_file(tmp_path: Path) -> None:
    plan = read_todos(_runtime(tmp_path))

    assert plan["exists"] is False
    assert plan["todos"] == [] and plan["task"] == ""
    assert plan["error"] == ""


@pytest.mark.parametrize(
    "content",
    ["", "not markdown at all", "- no marker here", "## Acceptance criteria\n- only criteria"],
)
def test_read_todos_tolerates_junk(content: str, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.workspace.mkdir(parents=True, exist_ok=True)
    (runtime.workspace / TODO_FILE).write_text(content, encoding="utf-8")

    plan = read_todos(runtime)

    assert plan["exists"] is True
    assert plan["todos"] == [] or all(todo["id"] for todo in plan["todos"])
    json.dumps(plan)  # always serialisable


def test_write_todos_failure_is_data(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.workspace.mkdir(parents=True, exist_ok=True)
    (runtime.workspace / TODO_FILE).mkdir()  # a directory where the file should go

    written = write_todos(runtime, [make_todo("t1", "do it")])

    assert written["ok"] is False
    assert written["error"]


# -- events and digests ------------------------------------------------------
def test_memory_event_carries_the_snapshot(tmp_path: Path) -> None:
    state = new_graph_state("do it", _runtime(tmp_path), todos=[make_todo("t1", "do it")])
    memory = build_layered_memory(state, node="planner")

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


def test_memory_digest_reports_sizes(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    state = new_graph_state(
        "do it",
        runtime,
        todos=[make_todo("t1", "a"), make_todo("t2", "b")],
        sources=[{"title": "t", "url": "https://x.test/a"}],
        agent_handoffs=[{"instruction": "i"}],
        research_notes="notes",
    )
    runtime.workspace.mkdir(parents=True, exist_ok=True)
    (runtime.workspace / NOTEPAD_FILE).write_text("n", encoding="utf-8")

    digest = memory_digest(build_layered_memory(state))

    assert digest.startswith("2 todos, 1 sources, 1 handoffs, notepad=yes, history=no")
    assert digest.endswith("notes=5ch")
    assert memory_digest({}) == "empty"


# -- prompt formatting -------------------------------------------------------
def test_format_layered_memory_for_prompt_is_json(tmp_path: Path) -> None:
    state = new_graph_state("do it", _runtime(tmp_path))
    memory = build_layered_memory(state, node="verifier")

    text = format_layered_memory_for_prompt(memory)

    assert text.startswith("{") and "\n  " in text  # pretty printed
    assert json.loads(text)["working_memory"]["node"] == "verifier"
    assert json.loads(text)["rules"]["scope"] == "workspace"


@pytest.mark.parametrize("value", [None, {}])
def test_format_layered_memory_for_empty_input(value: Any) -> None:
    assert format_layered_memory_for_prompt(value) == "{}"


def test_format_layered_memory_survives_unserialisable_values(tmp_path: Path) -> None:
    state = new_graph_state("do it", _runtime(tmp_path))
    state["todos"] = [{"id": "t1", "content": "x", "status": "pending", "note": object()}]

    text = format_layered_memory_for_prompt(build_layered_memory(state))

    assert "object object" in text  # default=str kept it serialisable


def test_snapshot_of_a_finished_graph_run(tmp_path: Path) -> None:
    """The snapshot also works on the state a real workflow run produces."""
    from helpers import ScriptedModel, passing_script

    from wqagent.graph import build_workflow, new_graph_state

    runtime = _runtime(tmp_path)
    result = build_workflow(model=ScriptedModel(passing_script())).invoke(
        new_graph_state("write hello.py", runtime)
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


def test_snapshot_fits_a_prompt_even_with_a_full_state(tmp_path: Path) -> None:
    state = new_graph_state(
        "do it",
        _runtime(tmp_path),
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
