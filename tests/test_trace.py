"""Tests for execution tracing (``core/trace.py``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from wqagent.core.state import RuntimeState
from wqagent.core.trace import (
    DEFAULT_TRACE_MODE,
    TIMELINE_HEAD,
    TIMELINE_TAIL,
    TRACE_DIR,
    TRACE_MODE_ENV,
    VALID_TRACE_MODES,
    TraceRecorder,
    new_trace_id,
    normalize_trace_mode,
    trace_mode_from_env,
)


def _runtime(tmp_path: Path, **kwargs: Any) -> RuntimeState:
    return RuntimeState(workspace=tmp_path / "ws", **kwargs)


def _inputs(**extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task": "write hello.py",
        "attempts": 0,
        "max_attempts": 3,
        "todos": [{"id": "t1", "content": "write"}],
        "messages": [],
    }
    payload.update(extra)
    return payload


# -- modes and ids -----------------------------------------------------------
def test_valid_trace_modes() -> None:
    assert VALID_TRACE_MODES == {"full", "summary", "off"}
    assert DEFAULT_TRACE_MODE == "summary"


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "summary"),
        ("", "summary"),
        ("  ", "summary"),
        ("nonsense", "summary"),
        ("FULL", "full"),
        (" Off ", "off"),
        ("summary", "summary"),
    ],
)
def test_normalize_trace_mode(value: Any, expected: str) -> None:
    assert normalize_trace_mode(value) == expected


def test_trace_mode_from_env(monkeypatch: Any) -> None:
    monkeypatch.delenv(TRACE_MODE_ENV, raising=False)
    assert trace_mode_from_env() == "summary"

    monkeypatch.setenv(TRACE_MODE_ENV, "full")
    assert trace_mode_from_env() == "full"

    monkeypatch.setenv(TRACE_MODE_ENV, "junk")
    assert trace_mode_from_env() == "summary"


def test_new_trace_id_is_readable_and_unique() -> None:
    ids = {new_trace_id() for _ in range(20)}

    assert len(ids) == 20
    for value in ids:
        assert value.startswith("wq-")
        assert len(value.split("-")) == 4
        assert all(part.isalnum() for part in value.split("-"))


def test_runtime_state_generates_a_trace_id(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.delenv(TRACE_MODE_ENV, raising=False)
    first = _runtime(tmp_path)
    second = _runtime(tmp_path)

    assert first.trace_mode == "summary"
    assert first.trace_id.startswith("wq-")
    assert first.trace_id != second.trace_id
    assert _runtime(tmp_path, trace_id="fixed-id").trace_id == "fixed-id"
    assert _runtime(tmp_path, trace_mode="off").trace_mode == "off"
    assert _runtime(tmp_path, trace_mode="junk").trace_mode == "summary"


def test_recorder_paths_and_enabled(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, trace_mode="full", trace_id="wq-test")
    recorder = TraceRecorder(runtime, "write hello.py")

    assert recorder.workspace == runtime.workspace
    assert recorder.mode == "full" and recorder.enabled is True and recorder.full is True
    assert recorder.root == runtime.workspace / TRACE_DIR / "traces" / "wq-test"
    assert recorder.trace_path == recorder.root / "trace.json"
    assert recorder.events_path == recorder.root / "events.jsonl"
    assert recorder.timeline_path == recorder.root / "timeline.md"

    off = TraceRecorder(_runtime(tmp_path, trace_mode="off"), "t")
    assert off.enabled is False and off.full is False
    assert off.start(_inputs()) is None
    assert off.end(status="passed", latest_node="final") is None


def test_recorder_uses_the_runtime_default_id(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    recorder = TraceRecorder(runtime, "write hello.py")

    assert recorder.trace_id == runtime.trace_id


# -- start / counters --------------------------------------------------------
def test_start_records_run_start(tmp_path: Path) -> None:
    recorder = TraceRecorder(_runtime(tmp_path, trace_mode="full", trace_id="wq-a"), "write hello.py")

    event = recorder.start(_inputs(attempts=1), resumed=True, resume_event={"status": "running", "restored_files": 2})

    assert event is not None
    assert event["type"] == "run_start"
    assert event["trace_id"] == "wq-a" and event["mode"] == "full"
    assert event["resumed"] is True and event["resume_status"] == "running"
    assert event["restored_files"] == 2
    assert event["todos"] == 1 and event["attempts"] == 1
    lines = recorder.events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["type"] == "run_start"


def test_counters_follow_the_event_stream(tmp_path: Path) -> None:
    recorder = TraceRecorder(_runtime(tmp_path, trace_mode="full"), "task")
    recorder.start(_inputs())

    recorder.record_custom_event({"type": "tool_call", "name": "bash"})
    recorder.record_custom_event(
        {"type": "tool_result", "name": "bash", "result": {"ok": True, "result": "fine"}}
    )
    recorder.record_custom_event(
        {"type": "tool_result", "name": "bash", "result": {"ok": False, "error": "boom"}}
    )
    recorder.record_custom_event(
        {
            "type": "tool_result",
            "name": "bash",
            "result": {"ok": True, "result": "$ pip install x\nexit_code: 0\nrequires_approval: True (auto-approved)"},
        }
    )
    recorder.record_custom_event({"type": "handoff", "from": "planner", "to": "codeAgent"})
    recorder.record_custom_event({"type": "checkpoint_saved", "status": "running", "latest_node": "planner"})
    recorder.record_graph_update({"type": "node_end", "node": "planner", "update": {}})
    recorder.record_graph_update({"type": "node_end", "node": "verifier", "update": {}})
    recorder.record_graph_update({"type": "node_end", "node": "planner", "update": {}})

    assert recorder.tool_calls == 1
    assert recorder.failed_tool_calls == 1
    assert recorder.approval_count == 1
    assert recorder.handoff_count == 1
    assert recorder.checkpoint_count == 1
    assert recorder.node_visits == {"planner": 2, "verifier": 1}


def test_approval_count_from_a_structured_flag(tmp_path: Path) -> None:
    recorder = TraceRecorder(_runtime(tmp_path), "task")

    recorder.record_custom_event(
        {"type": "tool_result", "name": "bash", "result": {"ok": True, "requires_approval": True}}
    )

    assert recorder.approval_count == 1


def test_recording_ignored_when_off(tmp_path: Path) -> None:
    recorder = TraceRecorder(_runtime(tmp_path, trace_mode="off"), "task")

    recorder.record_custom_event({"type": "tool_call", "name": "bash"})
    recorder.record_graph_update({"type": "node_end", "node": "planner"})

    assert recorder.tool_calls == 0 and recorder.node_visits == {}
    assert recorder.timeline == []


# -- end: trace.json / timeline.md ------------------------------------------
def test_end_writes_trace_json_and_timeline(tmp_path: Path) -> None:
    recorder = TraceRecorder(
        _runtime(tmp_path, trace_mode="full", trace_id="wq-b"), "write hello.py"
    )
    recorder.start(_inputs())
    recorder.record_custom_event({"type": "tool_call", "name": "file_write", "args": {"file_path": "hello.py"}})
    recorder.record_graph_update({"type": "node_end", "node": "planner", "update": {"plan_summary": "x"}})

    payload = recorder.end(
        status="passed",
        latest_node="final",
        final_state={"passed": True, "attempts": 1, "final_answer": "done"},
    )

    assert payload is not None
    for key in (
        "trace_id",
        "task",
        "status",
        "started_at",
        "ended_at",
        "duration_ms",
        "node_visits",
        "tool_calls",
        "failed_tool_calls",
        "approval_count",
        "checkpoint_count",
        "handoff_count",
        "timeline_head",
        "timeline_tail",
        "timeline_omitted",
    ):
        assert key in payload, key
    assert payload["trace_id"] == "wq-b"
    assert payload["status"] == "passed" and payload["latest_node"] == "final"
    assert payload["node_visits"] == {"planner": 1}
    assert payload["tool_calls"] == 1
    assert payload["passed"] is True and payload["attempts"] == 1
    assert payload["timeline_total"] == 4  # start, tool_call, node_end, run_end
    assert payload["timeline_omitted"] == 0
    assert [item["type"] for item in payload["timeline_head"]][-1] == "run_end"
    assert payload["duration_ms"] >= 0

    on_disk = json.loads(recorder.trace_path.read_text(encoding="utf-8"))
    assert on_disk == payload

    markdown = recorder.timeline_path.read_text(encoding="utf-8")
    assert markdown.startswith("# Trace wq-b")
    assert "**Task:** write hello.py" in markdown
    assert "**Status:** passed (latest node: final)" in markdown
    assert "**Counters:** 1 tool calls (0 failed)" in markdown
    assert "| run_start |" in markdown and "| run_end |" in markdown
    assert "Full event log: events.jsonl" in markdown


def test_timeline_head_tail_and_omitted(tmp_path: Path) -> None:
    recorder = TraceRecorder(_runtime(tmp_path), "task")
    recorder.start(_inputs())
    for index in range(200):
        recorder.record_custom_event({"type": "tool_call", "name": f"tool{index}"})

    payload = recorder.end(status="running", latest_node="planner")

    assert payload is not None
    total = payload["timeline_total"]
    assert total == 202  # start + 200 + run_end
    assert len(payload["timeline_head"]) == TIMELINE_HEAD
    assert len(payload["timeline_tail"]) == TIMELINE_TAIL
    assert payload["timeline_omitted"] == total - TIMELINE_HEAD - TIMELINE_TAIL
    assert payload["timeline_head"][0]["type"] == "run_start"
    assert payload["timeline_tail"][-1]["type"] == "run_end"
    # the two windows do not overlap and keep the outer parts of the run
    head_seq = {item["seq"] for item in payload["timeline_head"]}
    tail_seq = {item["seq"] for item in payload["timeline_tail"]}
    assert not head_seq & tail_seq
    assert min(tail_seq) > max(head_seq)

    markdown = recorder.timeline_path.read_text(encoding="utf-8")
    assert "event(s) omitted" in markdown


def test_summary_mode_logs_no_events_file(tmp_path: Path) -> None:
    recorder = TraceRecorder(_runtime(tmp_path, trace_mode="summary"), "task")
    recorder.start(_inputs())
    recorder.record_custom_event({"type": "tool_call", "name": "bash"})

    payload = recorder.end(status="running", latest_node="planner")

    assert payload is not None and payload["events_file"] == ""
    assert recorder.trace_path.is_file() and recorder.timeline_path.is_file()
    assert not recorder.events_path.exists()
    assert "not recorded in this mode" in recorder.timeline_path.read_text(encoding="utf-8")


def test_full_mode_logs_every_event(tmp_path: Path) -> None:
    recorder = TraceRecorder(_runtime(tmp_path, trace_mode="full"), "task")
    recorder.start(_inputs())
    recorder.record_custom_event({"type": "tool_call", "name": "bash", "args": {"command": "ls"}})
    recorder.record_graph_update({"type": "node_end", "node": "planner", "update": {"a": 1}})
    payload = recorder.end(status="running", latest_node="planner")

    assert payload is not None and payload["events_file"] == "events.jsonl"
    lines = [json.loads(line) for line in recorder.events_path.read_text(encoding="utf-8").splitlines()]
    # the log is closed with its own run_end entry
    assert [item["type"] for item in lines] == ["run_start", "tool_call", "node_end", "run_end"]
    assert [item["seq"] for item in lines] == [1, 2, 3, 4]
    assert lines[1]["args"] == {"command": "ls"}


def test_end_without_start_still_works(tmp_path: Path) -> None:
    recorder = TraceRecorder(_runtime(tmp_path), "task")

    payload = recorder.end(status="failed", latest_node="final")

    assert payload is not None
    assert payload["started_at"] == "" and payload["duration_ms"] == 0
    assert recorder.trace_path.is_file()


def test_unwritable_trace_directory_degrades(tmp_path: Path, monkeypatch: Any) -> None:
    recorder = TraceRecorder(_runtime(tmp_path), "task")
    monkeypatch.setattr(
        Path, "mkdir", lambda self, *args, **kwargs: (_ for _ in ()).throw(OSError("denied"))
    )

    recorder.start(_inputs())  # must not raise
    recorder.record_custom_event({"type": "tool_call", "name": "bash"})
    payload = recorder.end(status="running", latest_node="planner")  # must not raise

    assert payload is not None and recorder._io_failed is True
    assert recorder.tool_calls == 1  # counters still work in memory


def test_timeline_summaries_are_readable(tmp_path: Path) -> None:
    recorder = TraceRecorder(_runtime(tmp_path), "task")
    recorder.record_custom_event({"type": "tool_call", "name": "bash", "args": {"command": "pytest -q"}})
    recorder.record_custom_event(
        {"type": "tool_result", "name": "bash", "result": {"ok": False, "error": "exit 1"}}
    )
    recorder.record_custom_event({"type": "handoff", "from": "planner", "to": "codeAgent", "instruction": "write it"})
    recorder.record_custom_event({"type": "memory", "node": "planner", "digest": "1 todos"})
    recorder.record_custom_event(
        {
            "type": "context_compressed",
            "messages_before": 13,
            "tokens_before": 2394,
            "messages_after": 1,
            "tokens_after": 405,
            "persisted": {"ok": True},
            "history_path": "HISTORY_SUMMARY.md",
        }
    )
    recorder.record_custom_event({"type": "checkpoint_saved", "status": "running", "latest_node": "planner", "files": 2, "git_ok": False})
    recorder.record_graph_update({"type": "node_end", "node": "verifier", "update": {"passed": True}})

    details = [entry["detail"] for entry in recorder.timeline]

    assert 'bash(command=pytest -q)' in details[0]
    assert "failed: exit 1" in details[1]
    assert "planner -> codeAgent: write it" in details[2]
    assert "memory[planner]: 1 todos" in details[3]
    assert "compressed 13 msgs/2394 tokens -> 1 msg/405 tokens [HISTORY_SUMMARY.md]" in details[4]
    assert "checkpoint running/planner -> 2 file(s) (no git snapshot)" in details[5]
    assert "verifier update: passed" in details[6]
    assert all(entry["at"] for entry in recorder.timeline)


# -- integration -------------------------------------------------------------
def test_a_real_run_writes_a_trace(tmp_path: Path) -> None:
    from helpers import ScriptedModel, passing_script

    from wqagent.core.agent import stream_agent_events
    from wqagent.graph.state import new_graph_state

    runtime = _runtime(tmp_path, trace_id="wq-run")
    state = new_graph_state("write hello.py", runtime)

    events = list(
        stream_agent_events(
            "write hello.py",
            workspace=runtime.workspace,
            model=ScriptedModel(passing_script()),
            inputs=state,
        )
    )

    trace_events = [event for event in events if event["type"] == "trace_saved"]
    assert len(trace_events) == 1
    summary = trace_events[0]
    assert summary["trace_id"] == "wq-run"
    assert summary["status"] == "passed"
    assert summary["tool_calls"] == len([e for e in events if e["type"] == "tool_call"])
    assert summary["handoff_count"] == len([e for e in events if e["type"] == "handoff"])
    assert summary["checkpoint_count"] == len([e for e in events if e["type"] == "checkpoint_saved"])
    assert summary["node_visits"]["planner"] == 1

    root = runtime.workspace / TRACE_DIR / "traces" / "wq-run"
    payload = json.loads((root / "trace.json").read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["task"] == "write hello.py"
    assert payload["node_visits"] == {"planner": 1, "context_monitor": 2, "verifier": 1, "final": 1}
    assert (root / "timeline.md").is_file()
    # every counter covers the whole run, including the closing checkpoint
    assert payload["checkpoint_count"] == summary["checkpoint_count"] >= 1
    assert payload["timeline_tail"][-1]["type"] == "run_end"
