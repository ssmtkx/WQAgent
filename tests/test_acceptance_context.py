"""Acceptance tests for the context-management work.

Four criteria, one section each:

1. ``WQ_CONTEXT_TOKEN_LIMIT=2000`` reliably triggers compression and the task can
   still continue afterwards.
2. ``HISTORY_SUMMARY.md`` lands on disk and its content is readable.
3. The three-layer memory is injected into every node and the CLI shows it.
4. Compression shows a before/after comparison of tokens *and* messages.
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from rich.console import Console

import wqagent.cli.app as cli_module
from helpers import (
    ScriptedModel,
    answer,
    graph_state,
    passing_script,
    plan_call,
    report_call,
)

from wqagent.core.agent import stream_agent_events
from wqagent.graph.memory import (
    CONTEXT_TOKEN_LIMIT_ENV,
    DEFAULT_CONTEXT_TOKEN_LIMIT,
    read_history_summary,
    read_notepad,
)

#: The compression JSON the scripted compressor model answers with.
COMPRESSION_PAYLOAD: dict[str, Any] = {
    "summary": "compressed: hello.py written and verified",
    "active_goal": "finish hello.py",
    "completed_work": "hello.py written (22 bytes)",
    "open_todos": ["t1: verify"],
    "important_files": ["hello.py"],
    "tool_findings": ["python 3.12 is on PATH"],
    "sources": ["https://docs.test/python"],
    "next_steps": ["run python hello.py"],
    "risks": "none",
}


def _big_context() -> str:
    """Text long enough to be over the 2000-token budget with the fallback count."""
    return "transcript line\n" * 900  # ~14k chars -> ~3600 tokens by len // 4


def _run(state: dict[str, Any], model: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the complex graph through the agent entry point, collecting events."""
    events = list(
        stream_agent_events(state["task"], workspace=state["runtime"].workspace, model=model, inputs=state)
    )
    return events


def _final_state(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Rebuild the final state from the streamed node updates."""
    state: dict[str, Any] = {}
    for event in events:
        if event["type"] == "node_end":
            state.update(event["update"])
    state["final_answer"] = events[-1].get("content", "")
    state["passed"] = events[-1].get("passed", False)
    return state


def _state(tmp_path: Path, task: str = "write hello.py") -> dict[str, Any]:
    """A fresh graph state in this test's workspace."""
    return graph_state(task, root=tmp_path / "ws")


def _compression_model(final: str = "Done.") -> ScriptedModel:
    """Plan, compress (scripted JSON), inspect, verify, finish."""
    return ScriptedModel(
        [
            plan_call(),
            answer("Planned it."),
            AIMessage(content=json.dumps(COMPRESSION_PAYLOAD)),
            answer("Inspected."),
            report_call(True),
            answer(final),
        ]
    )


def _run_compression(
    tmp_path: Path,
    monkeypatch: Any,
    *,
    final: str = "Done.",
    messages: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the 2000-token-budget scenario and return ``(events, state)``."""
    monkeypatch.setenv(CONTEXT_TOKEN_LIMIT_ENV, "2000")
    state = _state(tmp_path)
    state["messages"] = [HumanMessage(content=_big_context())]
    if messages > 1:
        state["messages"] += [AIMessage(content="still working"), HumanMessage(content="and more")]
    return _run(state, _compression_model(final)), state


def _capture_cli(monkeypatch: Any) -> StringIO:
    """Point the CLI renderer at a string buffer (no terminal, no colour)."""
    stream = StringIO()
    monkeypatch.setattr(
        cli_module, "console", Console(file=stream, width=200, force_terminal=False)
    )
    monkeypatch.setattr(cli_module, "_can_encode", lambda text: False)
    return stream


# -- 1. WQ_CONTEXT_TOKEN_LIMIT=2000 triggers compression, the task continues ---
def test_env_var_sets_the_context_budget(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv(CONTEXT_TOKEN_LIMIT_ENV, "2000")

    state = _state(tmp_path, "do it")

    assert state["context_token_limit"] == 2000


@pytest.mark.parametrize("raw", ["", "  ", "nope", "0", "-5", "1e9999"])
def test_env_var_junk_keeps_the_default(raw: str, tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv(CONTEXT_TOKEN_LIMIT_ENV, raw)

    state = _state(tmp_path, "do it")

    assert state["context_token_limit"] == DEFAULT_CONTEXT_TOKEN_LIMIT


def test_compression_triggers_and_the_task_continues(tmp_path: Path, monkeypatch: Any) -> None:
    """AC1: with a 2000-token budget the run compresses, then keeps working."""
    events, state = _run_compression(
        tmp_path, monkeypatch, final="Done after compressing the context.", messages=3
    )
    order = [event["node"] for event in events if event["type"] == "node_end"]
    update = {
        event["node"]: event["update"] for event in events if event["type"] == "node_end"
    }

    # the budget came from the environment and was actually exceeded...
    monitors = [
        event["update"]
        for event in events
        if event["type"] == "node_end" and event["node"] == "context_monitor"
    ]
    assert monitors[0]["context_should_compress"] is True
    assert monitors[0]["context_token_count"] > 2000
    # ...and after compressing, the context fits the budget again
    assert state["context_token_limit"] == 2000  # the budget the node measured against
    assert monitors[-1]["context_should_compress"] is False
    assert monitors[-1]["context_token_count"] < monitors[0]["context_token_count"]

    # ...the run compressed, then carried on to verification and final
    assert order[0] == "planner"
    assert "context_compressor" in order
    assert order.index("context_compressor") < order.index("verifier")
    assert order[-1] == "final"
    assert len(update["context_compressor"]["compression_events"]) == 1
    assert events[-1]["type"] == "final_answer"
    assert events[-1]["passed"] is True
    assert events[-1]["content"] == "Done after compressing the context."
    # the task really continued: verification ran after the compression
    assert update["verifier"]["verification_results"]


def test_compression_does_not_trigger_on_a_small_context(tmp_path: Path, monkeypatch: Any) -> None:
    """The same run without the environment variable never compresses."""
    monkeypatch.delenv(CONTEXT_TOKEN_LIMIT_ENV, raising=False)
    model = ScriptedModel(passing_script())

    events = _run(_state(tmp_path), model)

    order = [event["node"] for event in events if event["type"] == "node_end"]
    assert "context_compressor" not in order
    assert events[-1]["passed"] is True


# -- 2. HISTORY_SUMMARY.md lands on disk and is readable ---------------------
def test_history_summary_is_written_and_readable(tmp_path: Path, monkeypatch: Any) -> None:
    workspace = tmp_path / "ws"
    events, state = _run_compression(tmp_path, monkeypatch)

    path = workspace / "HISTORY_SUMMARY.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")  # decodes cleanly -> readable UTF-8
    assert text.startswith("# History summary")
    assert COMPRESSION_PAYLOAD["summary"] in text
    for heading in (
        "## Active goal",
        "## Completed work",
        "## Open todos",
        "## Important files",
        "## Tool findings",
        "## Sources",
        "## Next steps",
        "## Risks",
    ):
        assert heading in text, heading
    assert "compressed by context_compressor_node" in text
    assert "tokens" in text and "limit" in text

    # the same content is readable through the memory API
    recorded = read_history_summary(state["runtime"])
    assert recorded["exists"] is True
    assert recorded["error"] == ""
    assert recorded["content"] == text
    # the nodes that ran *after* the compression see it in their snapshot (the
    # compressor's own snapshot is taken before it writes, so it reports the
    # pre-compression state)
    verifier_snapshot = next(
        event for event in events if event["type"] == "memory" and event["node"] == "verifier"
    )
    store = verifier_snapshot["memory"]["history_summary_store"]
    assert store["history_exists"] is True
    assert store["history_summary"] in text
    assert store["notepad_exists"] is True
    # and the durable notes were distilled at the same moment
    assert read_notepad(state["runtime"])["exists"] is True


# -- 3. the memory is injected everywhere and the CLI shows it ---------------
def test_every_node_gets_the_layered_memory(tmp_path: Path, monkeypatch: Any) -> None:
    events, _ = _run_compression(tmp_path, monkeypatch)

    injected = [event["node"] for event in events if event["type"] == "memory"]
    assert set(injected) == {
        "planner",
        "context_monitor",
        "context_compressor",
        "verifier",
        "final",
    }
    for event in events:
        if event["type"] != "memory":
            continue
        assert event["memory"]["rules"]["scope"] == "workspace"
        assert event["memory"]["working_memory"]["task"] == "write hello.py"
        assert event["digest"] and event["chars"] > 0
        json.dumps(event["memory"])  # serialisable snapshot


def test_every_model_call_sees_the_memory(tmp_path: Path, monkeypatch: Any) -> None:
    """Nodes with a model call carry the snapshot in their human message."""
    monkeypatch.delenv(CONTEXT_TOKEN_LIMIT_ENV, raising=False)
    model = ScriptedModel(passing_script())

    events = _run(_state(tmp_path), model)

    # the planner's brief, the verifier's evidence and the final facts all include it
    planner_prompt = model.seen[0][1].content
    verifier_prompt = next(
        messages[1].content for messages in model.seen if "Verification command results" in messages[1].content
    )
    final_prompt = next(
        messages[1].content for messages in model.seen if "--- facts ---" in messages[1].content
    )
    for prompt in (planner_prompt, verifier_prompt, final_prompt):
        assert "Layered memory" in prompt
        assert '"rules"' in prompt
        assert '"working_memory"' in prompt


def test_cli_shows_the_memory_snapshot(tmp_path: Path, monkeypatch: Any) -> None:
    """AC3: the terminal shows the snapshot (digest always, full JSON in -v)."""
    stream = _capture_cli(monkeypatch)
    event = {
        "type": "memory",
        "node": "planner",
        "digest": "1 todos, 0 sources, 0 handoffs, notepad=no, history=no",
        "chars": 1314,
        "memory": {
            "rules": {"scope": "workspace", "storage": "internal", "rules": ["Work inside …"]},
            "working_memory": {"node": "planner", "task": "write hello.py"},
            "history_summary_store": {},
        },
    }

    cli_module._NodeRenderer(verbose=False).render(event)
    quiet = stream.getvalue()
    assert "memory[planner]: 1 todos, 0 sources, 0 handoffs, notepad=no, history=no (1314 chars)" in quiet
    assert '"working_memory"' not in quiet  # the full snapshot is not dumped by default

    stream.seek(0)
    stream.truncate()
    cli_module._NodeRenderer(verbose=True).render(event)
    verbose = stream.getvalue()
    assert '"task": "write hello.py"' in verbose  # the snapshot itself is shown
    assert '"storage": "internal"' in verbose


# -- 4. before/after comparison of tokens and messages -----------------------
def test_compression_event_compares_tokens_and_messages(tmp_path: Path, monkeypatch: Any) -> None:
    events, _ = _run_compression(tmp_path, monkeypatch, messages=3)
    event = next(item for item in events if item["type"] == "context_compressed")

    assert event["messages_before"] == 3
    assert event["messages_after"] == 1
    assert event["tokens_before"] > event["tokens_after"]
    assert event["token_limit"] == 2000
    assert event["messages_before"] >= 1
    # the same numbers are recorded in the state's compression log
    compressor = next(
        item["update"]
        for item in events
        if item["type"] == "node_end" and item["node"] == "context_compressor"
    )
    recorded = compressor["compression_events"][-1]
    assert recorded["messages_before"] == event["messages_before"]
    assert recorded["messages_after"] == event["messages_after"]
    assert recorded["tokens_before"] == event["tokens_before"]
    assert recorded["tokens_after"] == event["tokens_after"]


def test_cli_renders_the_before_after_comparison(tmp_path: Path, monkeypatch: Any) -> None:
    stream = _capture_cli(monkeypatch)
    renderer = cli_module._NodeRenderer(verbose=False)

    renderer.render(
        {
            "type": "context_compressed",
            "node": "context_compressor",
            "ok": True,
            "tokens_before": 2394,
            "tokens_after": 405,
            "messages_before": 13,
            "messages_after": 1,
            "persisted": {"ok": True},
            "notepad": {"ok": True},
            "history_path": "HISTORY_SUMMARY.md",
            "notepad_path": "NOTEPAD.md",
        }
    )

    text = " ".join(stream.getvalue().split())
    assert "compressed 13 msgs/2394 tokens -> 1 msg/405 tokens" in text
    assert "HISTORY_SUMMARY.md" in text and "NOTEPAD.md" in text
