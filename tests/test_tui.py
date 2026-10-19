"""Tests for the Textual TUI (``cli/tui``): the approval gate, modal and app wiring."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("textual", reason="the TUI needs the optional 'textual' dependency")

from textual.widgets import Input, Static  # noqa: E402

from wqagent.cli.tui import TEXTUAL_MISSING_HINT, run_tui, textual_available  # noqa: E402
from wqagent.cli.tui.app import AgentEventMessage, WQAgentTuiApp, format_event  # noqa: E402
from wqagent.cli.tui.approval import (  # noqa: E402
    ApprovalGate,
    ApprovalModal,
    TuiApprovalHandler,
)
from wqagent.core.approval import ApprovalRequest, new_approval_request  # noqa: E402


def _request(command: str = "pip install fastapi") -> ApprovalRequest:
    return new_approval_request(command, "Python package installation")


def _scripted(events: list[dict[str, Any]]) -> Any:
    """A runner that replays *events*."""

    def runner(task: str) -> Any:
        for event in events:
            yield {"task": task, **event}

    return runner


def _text(widget: Any) -> str:
    """What a ``Static`` currently shows (Textual 8 spells it ``render()``)."""
    return str(widget.render())


def _stream(app: WQAgentTuiApp) -> str:
    return "\n".join(_text(widget) for widget in app.query("#stream Static"))


def _run(
    scenario: Callable[[WQAgentTuiApp, Any], Awaitable[None]],
    *,
    runner: Any = None,
    **options: Any,
) -> None:
    """The boilerplate every TUI test shares: one app, one headless pilot, one event loop."""

    async def main() -> None:
        app = WQAgentTuiApp(runner=runner, **options)
        async with app.run_test() as pilot:
            await scenario(app, pilot)

    asyncio.run(main())


async def _settle(app: WQAgentTuiApp, pilot: Any, limit: int = 60) -> None:
    """Pause until the worker thread has finished the turn."""
    for _ in range(limit):
        await pilot.pause()
        if not app._busy:
            return


async def _answer_approval(app: WQAgentTuiApp, pilot: Any, key: str) -> Any:
    """Ask for approval on a worker thread, answer the modal with *key*, return the decision."""
    handler = TuiApprovalHandler(app, timeout=10)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(handler, _request())
        for _ in range(60):  # wait for the modal to appear
            await pilot.pause()
            if isinstance(app.screen, ApprovalModal):
                break
        assert isinstance(app.screen, ApprovalModal)
        assert not future.done(), "the tool thread must be blocked while asking"

        await pilot.press(key)
        decision = future.result(timeout=10)
        await pilot.pause()
    return decision


# -- the gate ----------------------------------------------------------------
def test_gate_resolve_then_wait() -> None:
    gate = ApprovalGate(_request(), timeout=5)

    assert gate.resolved is False and gate.decision is None
    decision = gate.resolve(True, "looks fine")

    assert decision.approved is True and decision.reason == "looks fine"
    assert gate.resolved is True
    assert gate.wait(0.1).approved is True  # an already-resolved gate returns at once


def test_gate_first_answer_wins() -> None:
    gate = ApprovalGate(_request())

    gate.resolve(True)
    gate.resolve(False, "changed my mind")

    assert gate.wait(0.1).approved is True


def test_gate_wait_blocks_until_resolved() -> None:
    gate = ApprovalGate(_request(), timeout=10)
    result: list[Any] = []

    worker = threading.Thread(target=lambda: result.append(gate.wait()), daemon=True)
    worker.start()
    time.sleep(0.15)
    assert result == []  # still waiting

    gate.resolve(False, "denied")
    worker.join(timeout=5)

    assert result and result[0].approved is False and result[0].reason == "denied"


def test_gate_timeout_denies() -> None:
    gate = ApprovalGate(_request(), timeout=0.2)

    started = time.monotonic()
    decision = gate.wait()

    assert decision.approved is False
    assert "no answer within 0.2s" in decision.reason
    assert time.monotonic() - started < 3


def test_gate_timeout_zero_waits_until_resolved() -> None:
    gate = ApprovalGate(_request(), timeout=0)
    worker = threading.Thread(target=gate.wait, daemon=True)
    worker.start()
    time.sleep(0.1)
    assert gate.resolved is False  # 0 means "no timeout"

    gate.resolve(True)
    worker.join(timeout=2)
    assert gate.wait().approved is True


def test_handler_without_an_app_denies() -> None:
    class Broken:
        def post_message(self, message: Any) -> None:
            raise RuntimeError("no event loop")

    decision = TuiApprovalHandler(Broken())(_request())

    assert decision.approved is False
    assert "could not ask" in decision.reason


# -- the modal ---------------------------------------------------------------
def test_modal_shows_the_request_and_approves_with_y() -> None:
    async def scenario(app: WQAgentTuiApp, pilot: Any) -> None:
        gate = ApprovalGate(_request(), timeout=5)
        results: list[Any] = []
        app.push_screen(ApprovalModal(gate, workspace="D:/demo"), results.append)
        await pilot.pause()
        assert "pip install fastapi" in _text(app.screen.query_one("#approval-command"))
        rows = [_text(widget) for widget in app.screen.query(".approval-row")]
        assert any("BashTool" in row for row in rows)  # the tool name
        assert any("Python package installation" in row for row in rows)  # the risk
        assert any("D:/demo" in row for row in rows)  # the workspace

        await pilot.press("y")
        await pilot.pause()

        assert results == [True]

    _run(scenario, workspace=".", runner=_scripted([]))


@pytest.mark.parametrize(
    ("interaction", "target", "expected"),
    [
        ("press", "y", True),
        ("press", "n", False),
        ("press", "escape", False),
        ("press", "enter", True),
        ("click", "#approval-approve", True),
        ("click", "#approval-deny", False),
    ],
)
def test_modal_answers_by_key_and_by_button(interaction: str, target: str, expected: bool) -> None:
    async def scenario(app: WQAgentTuiApp, pilot: Any) -> None:
        gate = ApprovalGate(_request(), timeout=5)
        results: list[Any] = []
        app.push_screen(ApprovalModal(gate), results.append)
        await pilot.pause()

        await (pilot.press(target) if interaction == "press" else pilot.click(target))
        await pilot.pause()

        assert results == [expected], target

    _run(scenario, workspace=".", runner=_scripted([]))


# -- the app -----------------------------------------------------------------
def test_app_renders_the_layout(tmp_path: Path) -> None:
    async def scenario(app: WQAgentTuiApp, pilot: Any) -> None:
        await pilot.pause()
        assert app.query_one("#status", Static)
        assert app.query_one("#plan", Static)
        assert app.query_one("#stream")
        assert app.query_one("#prompt", Input)
        assert "new" in _text(app.query_one("#status", Static))  # no session yet

    _run(scenario, workspace=tmp_path / "ws", runner=_scripted([]))


def test_app_turn_updates_stream_plan_and_status(tmp_path: Path) -> None:
    events = [
        {"type": "intent", "intent_route": "workflow", "intent_confidence": 0.9,
         "intent_reason": "needs files"},
        {"type": "tool_call", "node": "codeAgent", "name": "file_write",
         "args": {"file_path": "app.py"}},
        {"type": "tool_result", "node": "codeAgent", "name": "file_write",
         "result": {"ok": True, "result": "Created app.py"}},
        {"type": "handoff", "from": "planner", "to": "codeAgent", "instruction": "write app.py"},
        {"type": "checkpoint_saved", "status": "running", "latest_node": "planner", "files": 2,
         "git_commit": "abcdef1234"},
        {"type": "todo_update", "node": "planner",
         "todos": [{"id": "t1", "status": "completed"}, {"id": "t2", "status": "in_progress"}]},
        {"type": "final_answer", "content": "Done: app.py created.", "passed": True,
         "route": "workflow"},
        {"type": "session_saved", "session_id": "8f3c1d2e4b5a", "turn": 1, "reply_turn": 2,
         "turn_index": 2, "route": "workflow", "saved": True},
    ]

    async def scenario(app: WQAgentTuiApp, pilot: Any) -> None:
        app.query_one("#prompt", Input).value = "build the app"
        await pilot.press("enter")
        await _settle(app, pilot)

        stream = _stream(app)
        assert "build the app" in stream  # the user's turn
        assert "file_write" in stream and "app.py" in stream  # the tool call
        assert "Created app.py" in stream  # the tool result
        assert "handoff · planner" in stream and "codeAgent" in stream
        assert "checkpoint · running/planner" in stream
        assert "Done: app.py created." in stream  # the final answer

        plan = _text(app.query_one("#plan", Static))
        assert "t1" in plan and "t2" in plan
        assert app.turn == 2 and app.session_id == "8f3c1d2e4b5a"
        status = _text(app.query_one("#status", Static))
        assert "8f3c1d2e" in status and "turn 2" in status

    _run(scenario, workspace=tmp_path / "ws", runner=_scripted(events))


def test_app_reports_a_failing_turn(tmp_path: Path) -> None:
    def runner(task: str) -> Any:
        raise RuntimeError("provider exploded")
        yield {}  # pragma: no cover - makes this a generator

    async def scenario(app: WQAgentTuiApp, pilot: Any) -> None:
        app.query_one("#prompt", Input).value = "do it"
        await pilot.press("enter")
        await _settle(app, pilot, limit=30)
        stream = _stream(app)
        assert "turn failed" in stream and "provider exploded" in stream

    _run(scenario, workspace=tmp_path / "ws", runner=runner)


def test_app_ignores_a_second_turn_while_busy(tmp_path: Path) -> None:
    release = threading.Event()

    def runner(task: str) -> Any:
        yield {"type": "intent", "intent_route": "workflow"}
        release.wait(timeout=10)

    async def scenario(app: WQAgentTuiApp, pilot: Any) -> None:
        prompt = app.query_one("#prompt", Input)
        prompt.value = "first"
        await pilot.press("enter")
        await pilot.pause()
        prompt.value = "second"
        await pilot.press("enter")
        await pilot.pause()
        assert "still working" in _stream(app)
        release.set()
        await _settle(app, pilot, limit=30)

    _run(scenario, workspace=tmp_path / "ws", runner=runner)


def test_app_approval_flow_approves(tmp_path: Path) -> None:
    """The tool thread waits on the gate; pressing Y in the modal releases it."""

    async def scenario(app: WQAgentTuiApp, pilot: Any) -> None:
        decision = await _answer_approval(app, pilot, "y")

        assert decision.approved is True
        stream = _stream(app)
        assert "approval required" in stream and "pip install fastapi" in stream
        assert "approved" in stream

    _run(scenario, workspace=tmp_path / "ws", runner=_scripted([]))


def test_app_approval_flow_denies_with_escape(tmp_path: Path) -> None:
    async def scenario(app: WQAgentTuiApp, pilot: Any) -> None:
        decision = await _answer_approval(app, pilot, "escape")

        assert decision.approved is False
        assert "denied" in decision.reason
        assert "denied" in _stream(app)

    _run(scenario, workspace=tmp_path / "ws", runner=_scripted([]))


def test_app_clear_stream(tmp_path: Path) -> None:
    async def scenario(app: WQAgentTuiApp, pilot: Any) -> None:
        app._append("hello", "event-meta")
        await pilot.pause()
        assert app.query("#stream Static")
        app.action_clear_stream()
        await pilot.pause()
        assert len(app.query("#stream Static")) == 0

    _run(scenario, workspace=tmp_path / "ws", runner=_scripted([]))


# -- the default runner ------------------------------------------------------
def test_default_runner_uses_stream_session_events(tmp_path: Path, monkeypatch: Any) -> None:
    """Without a scripted runner the app runs a real session turn (with a fake model)."""
    captured: dict[str, Any] = {}

    def fake_stream(task: str, **kwargs: Any) -> Any:
        captured["task"] = task
        captured.update(kwargs)
        yield {"type": "final_answer", "content": "ok", "passed": True, "route": "chat"}
        yield {"type": "session_saved", "session_id": "abc", "turn": 1, "reply_turn": 2,
               "saved": True}

    monkeypatch.setattr("wqagent.core.agent.stream_session_events", fake_stream)

    async def scenario(app: WQAgentTuiApp, pilot: Any) -> None:
        app.query_one("#prompt", Input).value = "你好"
        await pilot.press("enter")
        await _settle(app, pilot, limit=40)

    _run(
        scenario,
        workspace=tmp_path / "ws",
        model="fake-model",
        approval_mode="auto",
        trace_mode="off",
        entry_router=False,
    )

    assert captured["task"] == "你好"
    assert captured["session_workspace"] == str(tmp_path / "ws")
    assert captured["model"] == "fake-model"
    assert captured["approval_mode"] == "auto"
    assert captured["entry_router"] is False
    assert isinstance(captured["approval_handler"], TuiApprovalHandler)
    assert captured["trace_mode"] == "off"


def test_run_tui_prints_the_logo_before_the_app(monkeypatch: Any) -> None:
    """The rich logo runs at startup, right before the UI takes over the screen."""
    import sys
    import types

    import wqagent.cli.tui as tui_package

    calls: list[str] = []

    class FakeApp:
        def __init__(self, **kwargs: Any) -> None:
            calls.append("app")
            self.kwargs = kwargs

        def run(self) -> None:
            calls.append("run")

    fake_module = types.ModuleType("wqagent.cli.tui.app")
    fake_module.WQAgentTuiApp = FakeApp  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wqagent.cli.tui.app", fake_module)
    monkeypatch.setattr("wqagent.cli.tui.logo.print_logo", lambda *a, **k: calls.append("logo"))

    code = tui_package.run_tui(workspace=".", approval_mode="auto")

    assert code == 0
    assert calls == ["logo", "app", "run"]


def test_run_tui_reports_a_missing_textual(monkeypatch: Any) -> None:
    """Without textual the launcher explains the extra instead of crashing."""
    import sys

    import wqagent.cli.tui as tui_package

    monkeypatch.setitem(sys.modules, "wqagent.cli.tui.app", None)  # import -> ImportError

    code = tui_package.run_tui(workspace=".")

    assert code == 2
    assert "textual" in TEXTUAL_MISSING_HINT and "wqagent[tui]" in TEXTUAL_MISSING_HINT


def test_textual_is_available_in_this_environment() -> None:
    assert textual_available()


# -- rendering helpers -------------------------------------------------------
def test_format_event_lines() -> None:
    tool = format_event({"type": "tool_call", "name": "bash", "args": {"command": "ls"}})[0]
    assert "bash" in tool and "command=ls" in tool

    assert format_event({"type": "turn_finished"})[0] == ""
    assert "route · chat" in format_event(
        {"type": "intent", "intent_route": "chat", "intent_confidence": 0.8}
    )[0]
    assert "codeAgent" in format_event(
        {"type": "tool_call", "node": "codeAgent", "name": "file_write", "args": {"file_path": "a.py"}}
    )[0]
    assert "checkpoint" in format_event(
        {"type": "checkpoint_saved", "status": "running", "latest_node": "planner"}
    )[0]
    handoff = format_event(
        {"type": "handoff", "from": "planner", "to": "codeAgent", "instruction": "write it"}
    )
    assert handoff[1] == "event-handoff" and "planner" in handoff[0]
    failed = format_event(
        {"type": "tool_result", "name": "bash", "result": {"ok": False, "error": "exit 1"}}
    )
    assert failed[1] == "event-error" and "exit 1" in failed[0]


def test_format_event_renders_todo_chips_instead_of_a_repr() -> None:
    line, css = format_event(
        {
            "type": "todo_update",
            "todos": [
                {"id": "scaffold", "status": "completed"},
                {"id": "auth", "status": "in_progress"},
            ],
        }
    )

    assert css == "event-plan"
    assert "scaffold" in line and "auth" in line
    assert "{'id'" not in line  # no Python repr in the stream


def test_event_message_copies_the_event() -> None:
    payload = {"type": "tool_call", "name": "bash"}
    message = AgentEventMessage(payload)
    payload["name"] = "changed"

    assert message.event["name"] == "bash"
