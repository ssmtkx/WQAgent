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

from textual.screen import ModalScreen  # noqa: E402
from textual.widgets import Input, Static  # noqa: E402

from wqagent.cli.tui import TEXTUAL_MISSING_HINT, run_tui, textual_available  # noqa: E402
from wqagent.cli.tui.app import AgentEventMessage, WQAgentTuiApp, format_event  # noqa: E402
from wqagent.cli.tui.diff_view import DiffBlock  # noqa: E402
from wqagent.cli.tui.approval import (  # noqa: E402
    ApprovalBar,
    ApprovalGate,
    ApprovalRequestedMessage,
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


def _plan(app: WQAgentTuiApp) -> str:
    """The plan pane is a stack of one Static per todo row."""
    return "\n".join(_text(widget) for widget in app.query("#plan Static"))


def _run(
    scenario: Callable[[WQAgentTuiApp, Any], Awaitable[None]],
    *,
    runner: Any = None,
    size: tuple[int, int] | None = None,
    **options: Any,
) -> None:
    """The boilerplate every TUI test shares: one app, one headless pilot, one event loop."""

    async def main() -> None:
        app = WQAgentTuiApp(runner=runner, **options)
        async with app.run_test(size=size) as pilot:
            await scenario(app, pilot)

    asyncio.run(main())


async def _settle(app: WQAgentTuiApp, pilot: Any, limit: int = 60) -> None:
    """Pause until the worker thread has finished the turn."""
    for _ in range(limit):
        await pilot.pause()
        if not app._busy:
            return


async def _answer_approval(
    app: WQAgentTuiApp, pilot: Any, key: str, *, command: str = "pip install fastapi"
) -> Any:
    """Ask for approval on a worker thread, answer the inline bar, return the decision."""
    handler = TuiApprovalHandler(app, timeout=10)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(handler, _request(command))
        for _ in range(60):  # wait for the bar to appear
            await pilot.pause()
            if app._active_gate is not None:
                break
        assert app._active_gate is not None, "the approval bar never appeared"
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


# -- the inline approval bar -------------------------------------------------
def test_approval_bar_is_inline_and_shows_the_request() -> None:
    """The prompt sits above the composer, in the activity pane - never over it."""

    async def scenario(app: WQAgentTuiApp, pilot: Any) -> None:
        gate = ApprovalGate(_request(), timeout=5)
        app.post_message(ApprovalRequestedMessage(gate))
        for _ in range(60):
            await pilot.pause()
            if app.query(ApprovalBar):
                break

        bar = app.query_one(ApprovalBar)
        assert bar.parent is app.query_one("#activitypane")  # inline, not a modal
        assert not isinstance(app.screen, ModalScreen)  # the screen is still the app

        title = _text(app.query_one("#approval-title", Static))
        assert "approval required" in title and "BashTool" in title
        body = _text(app.query_one("#approval-command", Static))
        assert "pip install fastapi" in body  # the command
        assert "Python package installation" in body  # the risk
        hint = _text(app.query_one("#approval-hint", Static))
        assert "approve" in hint and "deny" in hint and "esc" in hint

        # nothing gets typed into the composer while a gate waits
        assert app.query_one("#prompt", Input).disabled is True

        await pilot.press("y")
        await pilot.pause()

        assert not app.query(ApprovalBar)  # the bar is gone again
        assert app.query_one("#prompt", Input).disabled is False
        assert gate.decision is not None and gate.decision.approved is True

    _run(scenario, workspace=".", runner=_scripted([]))


@pytest.mark.parametrize(("key", "expected"), [("y", True), ("n", False), ("escape", False)])
def test_approval_bar_answers_by_key(key: str, expected: bool) -> None:
    async def scenario(app: WQAgentTuiApp, pilot: Any) -> None:
        decision = await _answer_approval(app, pilot, key)

        assert decision.approved is expected
        assert ("approved" if expected else "denied") in _stream(app)

    _run(scenario, workspace=".", runner=_scripted([]))


def test_second_approval_waits_for_the_first() -> None:
    """Two gates at once: one bar, a "more waiting" hint, then the second request."""

    async def scenario(app: WQAgentTuiApp, pilot: Any) -> None:
        handler = TuiApprovalHandler(app, timeout=10)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(handler, _request("pip install fastapi"))
            second = pool.submit(handler, _request("rm -rf build"))
            for _ in range(60):
                await pilot.pause()
                if app.query(ApprovalBar):
                    break

            assert len(app.query(ApprovalBar)) == 1
            assert "pip install fastapi" in _text(app.query_one("#approval-command", Static))
            assert "1 more waiting" in _text(app.query_one("#approval-title", Static))

            await pilot.press("y")
            assert first.result(timeout=10).approved is True

            for _ in range(60):
                await pilot.pause()
                if "rm -rf build" in _text(app.query_one("#approval-command", Static)):
                    break
            assert "rm -rf build" in _text(app.query_one("#approval-command", Static))

            await pilot.press("n")
            assert second.result(timeout=10).approved is False

    _run(scenario, workspace=".", runner=_scripted([]))


# -- the diff view -----------------------------------------------------------
DIFF_EVENT = {
    "type": "tool_result",
    "name": "file_write",
    "result": {"ok": True, "result": "Created app/main.py (412 lines)"},
    "diff": {
        "path": "app/main.py",
        "added": 2,
        "removed": 1,
        "text": (
            "--- a/app/main.py\n+++ b/app/main.py\n@@ -1,2 +1,3 @@\n-old\n+new\n+extra\n"
        ),
        "truncated": False,
    },
}


def test_diff_block_lands_in_the_stream_and_toggles(tmp_path: Path) -> None:
    async def scenario(app: WQAgentTuiApp, pilot: Any) -> None:
        app.post_message(AgentEventMessage(dict(DIFF_EVENT)))
        for _ in range(40):
            await pilot.pause()
            if app.query(DiffBlock):
                break

        block = app.query_one(DiffBlock)
        assert block.parent is app.query_one("#stream")  # inline in the log
        header = _text(block.query_one("#diff-header", Static))
        assert "file_write" in header and "app/main.py" in header
        assert "+2" in header and "−1" in header  # the counts are the summary

        body = block.query_one("#diff-body", Static)
        assert body.display is False  # collapsed by default

        app.action_toggle_diff()
        await pilot.pause()
        assert body.display is True
        assert "+new" in _text(body) and "-old" in _text(body)

        app.action_toggle_diff()
        await pilot.pause()
        assert body.display is False

    _run(scenario, workspace=tmp_path / "ws", runner=_scripted([]))


def test_tool_result_without_a_diff_stays_a_plain_line(tmp_path: Path) -> None:
    async def scenario(app: WQAgentTuiApp, pilot: Any) -> None:
        app.post_message(
            AgentEventMessage(
                {"type": "tool_result", "name": "file_read", "result": {"ok": True, "result": "ok"}}
            )
        )
        await pilot.pause()
        assert not app.query(DiffBlock)
        assert "file_read" in _stream(app)

    _run(scenario, workspace=tmp_path / "ws", runner=_scripted([]))


# -- streaming replies -------------------------------------------------------
def test_tokens_stream_into_one_line_then_settle(tmp_path: Path) -> None:
    async def scenario(app: WQAgentTuiApp, pilot: Any) -> None:
        for piece in ("Hel", "lo", " there"):
            app.post_message(AgentEventMessage({"type": "token", "text": piece}))
        await pilot.pause()
        app._flush_tokens()  # what the 50 ms timer does
        await pilot.pause()

        live = app._live
        assert live is not None
        assert _text(live.query_one("#live-gutter", Static)) == "»  chat :"
        assert "Hello there" in _text(live.body)  # the text wraps beside the gutter

        app.post_message(
            AgentEventMessage({"type": "chat_response", "chat_response": "Hello there"})
        )
        for _ in range(20):
            await pilot.pause()

        assert app._live is None  # the line settled
        lives = app.query("#stream LiveReply")
        assert len(lives) == 1  # one block, not a duplicate
        assert "Hello there" in _text(lives[0].body)

    _run(scenario, workspace=tmp_path / "ws", runner=_scripted([]))


def test_the_chat_route_does_not_print_the_reply_twice(tmp_path: Path) -> None:
    """The chat route reports the same text as chat_response and final_answer."""

    async def scenario(app: WQAgentTuiApp, pilot: Any) -> None:
        app.post_message(AgentEventMessage({"type": "token", "text": "hi"}))
        await pilot.pause()
        app.post_message(AgentEventMessage({"type": "chat_response", "chat_response": "hi"}))
        app.post_message(
            AgentEventMessage(
                {"type": "final_answer", "content": "hi", "passed": True, "route": "chat"}
            )
        )
        for _ in range(20):
            await pilot.pause()

        assert "answer :" not in _stream(app)
        assert len(app.query("#stream LiveReply")) == 1

    _run(scenario, workspace=tmp_path / "ws", runner=_scripted([]))


# -- the app -----------------------------------------------------------------
def test_app_renders_the_layout(tmp_path: Path) -> None:
    async def scenario(app: WQAgentTuiApp, pilot: Any) -> None:
        await pilot.pause()
        assert app.query_one("#status", Static)
        assert app.query_one("#plan")
        assert app.query_one("#stream")
        assert app.query_one("#prompt", Input)
        assert "new" in _text(app.query_one("#status", Static))  # no session yet

    _run(scenario, workspace=tmp_path / "ws", runner=_scripted([]))


def test_layout_has_real_geometry(tmp_path: Path) -> None:
    """Regression: the panes must actually occupy rows (a blank screen used to pass)."""

    async def scenario(app: WQAgentTuiApp, pilot: Any) -> None:
        await pilot.pause()
        stream = app.query_one("#stream")
        assert stream.content_size.height > 10, "the activity log has no visible rows"
        assert stream.region.width > 40, "the activity log is squeezed to nothing"
        assert app.query_one("#planpane").display is True
        assert app.query_one("#planpane").region.width == 38
        inputbar = app.query_one("#inputbar").region
        assert inputbar.y + inputbar.height <= app.size.height, "the composer is off screen"

    _run(scenario, size=(120, 34), workspace=tmp_path / "ws", runner=_scripted([]))


def test_narrow_terminal_folds_the_plan_sidebar(tmp_path: Path) -> None:
    """Below NARROW_WIDTH the sidebar yields its columns to the activity log."""

    async def scenario(app: WQAgentTuiApp, pilot: Any) -> None:
        await pilot.pause()
        assert app.query_one("#planpane").display is False
        assert app.query_one("#activitypane").region.width == app.size.width
        assert "plan ▸" in str(app.query_one("#keys", Static).render())

        await pilot.press("ctrl+b")  # the user may insist on seeing the plan
        await pilot.pause()
        assert app.query_one("#planpane").display is True
        assert "plan ▾" in str(app.query_one("#keys", Static).render())

    _run(scenario, size=(84, 26), workspace=tmp_path / "ws", runner=_scripted([]))


def test_toggle_plan_folds_and_restores_it(tmp_path: Path) -> None:
    async def scenario(app: WQAgentTuiApp, pilot: Any) -> None:
        await pilot.pause()
        assert app.query_one("#planpane").display is True

        await pilot.press("ctrl+b")
        await pilot.pause()
        assert app.query_one("#planpane").display is False
        assert app.query_one("#activitypane").region.width == app.size.width

        await pilot.press("ctrl+b")
        await pilot.pause()
        assert app.query_one("#planpane").display is True

    _run(scenario, size=(120, 34), workspace=tmp_path / "ws", runner=_scripted([]))


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
        assert "handoff" in stream and "planner" in stream and "codeAgent" in stream
        assert "checkpoint" in stream and "running/planner" in stream
        assert "Done: app.py created." in stream  # the final answer

        plan = _plan(app)
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
    assert "bash" in tool and "ls" in tool  # the target, not key=value noise

    assert format_event({"type": "turn_finished"})[0] == ""
    intent = format_event(
        {"type": "intent", "intent_route": "chat", "intent_confidence": 0.8}
    )[0]
    assert "intent" in intent and "chat" in intent and "0.80" in intent
    written = format_event(
        {"type": "tool_call", "node": "codeAgent", "name": "file_write", "args": {"file_path": "a.py"}}
    )[0]
    assert "file_write" in written and "a.py" in written
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


def test_format_event_never_uses_emoji() -> None:
    """The design is explicit: text markers (✓ ✕ ↳ › » ●), never emoji."""
    samples = [
        {"type": "tool_result", "name": "f", "result": {"ok": True, "result": "x"}},
        {"type": "tool_result", "name": "f", "result": {"ok": False, "error": "x"}},
        {"type": "handoff", "from": "a", "to": "b"},
        {"type": "tool_call", "name": "bash", "args": {"command": "ls"}},
    ]
    for event in samples:
        line = format_event(event)[0]
        assert not any(ord(char) > 0x2190 and char not in "↳›»→●✓✕…·" for char in line), line


def test_format_event_summarises_todo_updates_instead_of_a_repr() -> None:
    line, css = format_event(
        {
            "type": "todo_update",
            "todos": [
                {"id": "scaffold", "content": "scaffold the app", "status": "completed"},
                {"id": "auth", "content": "add auth", "status": "in_progress"},
            ],
        }
    )

    assert css == "event-plan"
    assert "1 / 2" in line and "add auth" in line  # progress + the active item
    assert "{'id'" not in line  # no Python repr in the stream


def test_event_message_copies_the_event() -> None:
    payload = {"type": "tool_call", "name": "bash"}
    message = AgentEventMessage(payload)
    payload["name"] = "changed"

    assert message.event["name"] == "bash"
