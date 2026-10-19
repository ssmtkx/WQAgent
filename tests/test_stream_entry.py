"""Tests for the run entry point's knobs and bookkeeping (``core/agent.py``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from helpers import ScriptedModel, answer, graph_state, passing_script, report_call
from wqagent.core import agent as agent_module
from wqagent.core.agent import _custom_event_needs_checkpoint, stream_agent_events
from wqagent.core.checkpoint import CheckpointManager
from wqagent.core.state import create_runtime


# -- create_runtime ----------------------------------------------------------
def test_create_runtime_defaults(tmp_path: Path, monkeypatch: Any) -> None:
    for name in ("WQ_APPROVAL_MODE", "WQ_CHECKPOINT_MODE", "WQ_TRACE_MODE"):
        monkeypatch.delenv(name, raising=False)

    state = create_runtime(tmp_path / "ws")

    assert state.workspace == (tmp_path / "ws").resolve()
    assert state.approval_mode == "inline"
    assert state.checkpoint_mode == "light"
    assert state.trace_mode == "summary"  # "on" is the friendly spelling
    assert state.trace_id.startswith("wq-")
    assert state.resume_from == ""


def _handler(request: Any) -> Any:
    """A stand-in approval handler (identity is what the tests check)."""
    from wqagent.core.approval import ApprovalDecision

    return ApprovalDecision(approved=True)


def test_create_runtime_honours_the_modes(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.delenv("WQ_TRACE_MODE", raising=False)

    state = create_runtime(
        tmp_path / "ws",
        approval_mode="deny",
        approval_handler=_handler,
        checkpoint_mode="strict",
        trace_mode="full",
        timeout_seconds=45,
    )

    assert state.approval_mode == "deny" and state.approval_handler is _handler
    assert state.checkpoint_mode == "strict" and state.trace_mode == "full"
    assert state.timeout_seconds == 45


def test_create_runtime_none_defers_to_the_environment(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("WQ_APPROVAL_MODE", "auto")
    monkeypatch.setenv("WQ_CHECKPOINT_MODE", "off")
    monkeypatch.setenv("WQ_TRACE_MODE", "full")

    state = create_runtime(tmp_path / "ws", approval_mode=None, checkpoint_mode=None, trace_mode=None)

    assert state.approval_mode == "auto"
    assert state.checkpoint_mode == "off"
    assert state.trace_mode == "full"


def test_create_runtime_resume_from_becomes_the_workspace(tmp_path: Path) -> None:
    target = tmp_path / "old"

    state = create_runtime(tmp_path / "new", resume_from=target)

    assert state.workspace == target.resolve()
    assert state.resume_from == str(target.resolve())


def test_trace_mode_on_is_summary(tmp_path: Path) -> None:
    from wqagent.core.trace import normalize_trace_mode

    assert normalize_trace_mode("on") == "summary"
    assert normalize_trace_mode("ON") == "summary"
    assert normalize_trace_mode("full") == "full"
    assert normalize_trace_mode("off") == "off"


# -- the checkpoint-on-custom-event rule -------------------------------------
@pytest.mark.parametrize(
    "kind, expected",
    [
        ("handoff", True),
        ("context_compressed", True),
        ("tool_call", False),
        ("tool_result", False),
        ("ai_message", False),
        ("memory", False),
        ("", False),
    ],
)
def test_custom_event_needs_checkpoint(kind: str, expected: bool) -> None:
    assert _custom_event_needs_checkpoint({"type": kind}) is expected


# -- the stream itself -------------------------------------------------------
def _read(path: Path) -> dict[str, Any]:
    """Parse the JSON file at *path*."""
    return json.loads(path.read_text(encoding="utf-8"))


def _traces(workspace: Path) -> list[Path]:
    """The run's trace directories."""
    return list((workspace / ".wqagent" / "traces").iterdir())


def _trace(workspace: Path) -> dict[str, Any]:
    """The run's parsed ``trace.json``."""
    return _read(_traces(workspace)[0] / "trace.json")


def _checkpoint(workspace: Path) -> dict[str, Any]:
    """The run's parsed ``checkpoint.json``."""
    return _read(workspace / ".wqagent" / "checkpoints" / "checkpoint.json")


def _kinds(events: list[dict[str, Any]]) -> list[str]:
    return [str(event["type"]) for event in events]


def _by_type(events: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [event for event in events if event["type"] == kind]


def _run(
    tmp_path: Path, model: ScriptedModel | None = None, **kwargs: Any
) -> tuple[list[dict[str, Any]], Path]:
    """Stream one passing run in ``tmp_path/ws`` (*model* defaults to a passing script)."""
    workspace = kwargs.setdefault("workspace", tmp_path / "ws")
    events = list(stream_agent_events("say hi", model=model or ScriptedModel(passing_script()), **kwargs))
    return events, workspace


def test_stream_creates_the_runtime_from_the_mode_arguments(tmp_path: Path) -> None:
    events, workspace = _run(tmp_path, checkpoint_mode="off", trace_mode="off")

    assert events[-1]["type"] == "final_answer" and events[-1]["passed"] is True
    hidden = workspace / ".wqagent"
    # both off: no checkpoints and no traces (the child temp dir may exist)
    assert not (hidden / "checkpoints").exists()
    assert not (hidden / "traces").exists()
    assert _by_type(events, "trace_saved") == []
    assert _by_type(events, "checkpoint_saved") == []


def test_stream_writes_the_opening_checkpoint_first(tmp_path: Path) -> None:
    events, workspace = _run(tmp_path, trace_mode="off")

    checkpoints = _by_type(events, "checkpoint_saved")
    assert checkpoints, "at least the opening checkpoint"
    assert checkpoints[0]["status"] == "started"
    assert checkpoints[0]["latest_node"] == "start"
    # it is published before the first node ends
    kinds = _kinds(events)
    assert kinds.index("checkpoint_saved") < kinds.index("node_end")
    assert checkpoints[-1]["status"] == "passed" and checkpoints[-1]["latest_node"] == "final"

    assert _checkpoint(workspace)["status"] == "passed"


def test_stream_checkpoints_on_handoff_and_compression(tmp_path: Path) -> None:
    """A delegation is worth a checkpoint even before the node ends."""
    events, _ = _run(tmp_path, trace_mode="off")

    kinds = _kinds(events)
    handoff_at, node_end_at = kinds.index("handoff"), kinds.index("node_end")
    saves = [index for index, kind in enumerate(kinds) if kind == "checkpoint_saved"]
    # the checkpoint caused by the handoff comes right after it, before the node_end
    assert any(handoff_at < index < node_end_at for index in saves), (handoff_at, saves, node_end_at)
    assert len(saves) >= 3


def test_stream_keeps_the_flat_event_contract(tmp_path: Path) -> None:
    """Events stay flat: callers switch on `type`, they are not wrapped."""
    events, _ = _run(tmp_path)

    kinds = {event["type"] for event in events}
    assert {"tool_call", "tool_result", "handoff", "node_end", "final_answer"} <= kinds
    assert not kinds & {"custom_event", "graph_event"}
    assert events[-1]["type"] == "final_answer"


def test_stream_records_every_event_in_the_trace(tmp_path: Path) -> None:
    events, workspace = _run(tmp_path, trace_mode="full")

    traces = _traces(workspace)
    assert len(traces) == 1
    payload = _trace(workspace)
    assert payload["status"] == "passed"
    # every yielded event is recorded except `trace_saved` itself, plus the two
    # trace-only markers (`run_start` and `run_end`)
    assert payload["timeline_total"] == len(events) - 1 + 2
    logged = [
        json.loads(line)
        for line in (traces[0] / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(isinstance(line, dict) and line.get("type") for line in logged)
    # node updates are logged without the transcript
    node_lines = [line for line in logged if line["type"] == "node_end"]
    assert node_lines and all("messages" not in line["update"] for line in node_lines)


@pytest.mark.parametrize(
    "error, status, match",
    [
        (RuntimeError("provider exploded"), "failed", "provider exploded"),
        (KeyboardInterrupt(), "interrupted", None),
    ],
    ids=["crash", "interrupt"],
)
def test_stream_closes_the_trail_when_the_run_aborts(
    tmp_path: Path, error: Exception, status: str, match: str | None
) -> None:
    """A crash or an interrupt still closes the checkpoint and the trace."""

    class Aborted(ScriptedModel):
        def invoke(self, messages: Any) -> Any:
            raise error

    with pytest.raises(type(error), match=match):
        _run(tmp_path, Aborted(passing_script()))

    workspace = tmp_path / "ws"
    assert _checkpoint(workspace)["status"] == status
    traces = _traces(workspace)
    assert len(traces) == 1
    trace = _read(traces[0] / "trace.json")
    assert trace["status"] == status
    assert trace["timeline_total"] >= 2


def test_a_model_timeout_keeps_the_trail_and_raises(tmp_path: Path) -> None:
    """A provider timeout must surface, and must leave trace + checkpoint behind."""

    class TimedOut(ScriptedModel):
        def invoke(self, messages: Any) -> Any:
            if self.seen:
                raise TimeoutError("Request timed out.")
            return super().invoke(messages)

    with pytest.raises(TimeoutError):
        _run(tmp_path, TimedOut(passing_script()), trace_mode="full")

    workspace = tmp_path / "ws"
    payload = _trace(workspace)
    assert payload["status"] == "failed"
    assert payload["complete"] is True
    assert payload["latest_node"] == "start"
    assert _checkpoint(workspace)["status"] == "failed"


def test_trace_is_on_disk_after_the_first_node(tmp_path: Path) -> None:
    """A run that never finishes (killed, or stuck in a model call) still leaves a trace."""
    workspace = tmp_path / "ws"
    generator = stream_agent_events(
        "say hi", workspace=workspace, model=ScriptedModel(passing_script()), trace_mode="full"
    )

    for event in generator:  # stop right after the first node ends
        if event["type"] == "node_end":
            break
    generator.close()  # the run is abandoned here

    traces = _traces(workspace)
    assert len(traces) == 1
    payload = _trace(workspace)
    assert payload["status"] == "running"
    assert payload["latest_node"] == "planner"
    assert payload["complete"] is False  # it was a mid-run snapshot
    assert payload["timeline_total"] >= 1
    assert (traces[0] / "timeline.md").is_file()


def test_trace_snapshot_is_updated_at_every_node(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    snapshots: list[tuple[str, int]] = []
    for event in stream_agent_events(
        "say hi", workspace=workspace, model=ScriptedModel(passing_script())
    ):
        if event["type"] == "node_end":
            payload = _trace(workspace)
            latest = str(payload["latest_node"])
            snapshots.append((latest, int(payload["node_visits"].get(latest, 0))))

    assert [node for node, _ in snapshots] == [
        "planner",
        "context_monitor",
        "verifier",
        "context_monitor",
        "final",
    ]
    # the closing write marks the trace complete
    assert _trace(workspace)["complete"] is True


def test_every_run_writes_the_trace_before_the_answer(tmp_path: Path) -> None:
    """`trace_saved` (and the files) come before the closing event."""
    events, workspace = _run(tmp_path)

    kinds = _kinds(events)
    assert kinds.index("trace_saved") < kinds.index("final_answer")
    assert _trace(workspace)["complete"] is True


def test_stream_resume_workspace_loads_and_continues(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    state = graph_state(root=workspace)
    (workspace / "hello.py").write_text("print('hi')\n", encoding="utf-8")
    saved = CheckpointManager(state["runtime"], "write hello.py").save(
        state, status="running", latest_node="verifier"
    )
    assert saved is not None and saved["git_ok"], saved
    (workspace / "hello.py").write_text("scribbled\n", encoding="utf-8")

    # a script that delegates nothing, so the restored snapshot is what remains
    # (spare answers on purpose: the verifier's loop needs a couple of turns)
    quiet_script = [
        answer("The plan is already done."),
        answer("Inspected the workspace."),
        report_call(True),
        answer("Done."),
        answer("Done."),
    ]
    events = list(
        stream_agent_events(
            "",
            resume_workspace=workspace,
            model=ScriptedModel(quiet_script),
        )
    )

    assert events[-1]["type"] == "final_answer" and events[-1]["passed"] is True
    assert (workspace / "hello.py").read_text(encoding="utf-8") == "print('hi')\n"
    # the trace knows this run continued a checkpoint
    trace = _trace(workspace)
    assert trace["task"] == "write hello.py" and trace["resumed"] is True
    start = trace["timeline_head"][0]
    assert start["type"] == "run_start" and "resumed" in start["detail"]


def test_stream_resume_workspace_without_a_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(LookupError, match="no checkpoint to resume"):
        list(
            stream_agent_events(
                "say hi",
                resume_workspace=tmp_path / "empty",
                model=ScriptedModel(passing_script()),
            )
        )


def test_stream_still_requires_a_place_to_work() -> None:
    with pytest.raises(ValueError, match="pass workspace=, state=, inputs= or resume_workspace="):
        list(stream_agent_events("say hi", model=ScriptedModel(passing_script())))


def test_stream_reuses_a_given_state_and_its_modes(tmp_path: Path) -> None:
    """state= wins: the mode arguments must not silently override it."""
    state = create_runtime(tmp_path / "ws", checkpoint_mode="off", trace_mode="off")

    events, _ = _run(
        tmp_path,
        workspace=tmp_path / "elsewhere",
        state=state,
        checkpoint_mode="strict",
        trace_mode="full",
    )

    assert events[-1]["passed"] is True
    assert not (tmp_path / "ws" / ".wqagent" / "checkpoints").exists()
    assert not (tmp_path / "ws" / ".wqagent" / "traces").exists()
    assert not (tmp_path / "elsewhere").exists()


def test_agent_module_exports_the_helper() -> None:
    assert agent_module.CHECKPOINT_CUSTOM_EVENTS == {"handoff", "context_compressed"}
    assert agent_module.DEFAULT_MAX_LOOPS == 10
