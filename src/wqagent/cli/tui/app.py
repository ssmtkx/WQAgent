"""The WQ Agent TUI, built with `Textual <https://textual.textualize.io>`_.

    ┌─────────────────────────────────────────────┐
    │ 🐺 WQ Agent    session: 8f3c…  turn 2       │  header + status bar
    ├─────────────────────────────────────────────┤
    │ [Plan] t1 ✅  t2 🔄  t3 ⬜                   │  plan panel
    │                                             │
    │ 🔧 file_write → app.py                      │  event stream (scrollable)
    │ 📝 notepad: "Created FastAPI app"           │
    │ 🔄 handoff: planner → codeAgent             │
    ├─────────────────────────────────────────────┤
    │ 💬 Input …                                  │  one Input per session turn
    └─────────────────────────────────────────────┘

The agent itself runs in a **background thread** (one thread per turn), and every event
it yields is posted to the UI thread as an :class:`AgentEventMessage`; the UI then
updates the plan panel, the event stream, the status bar or the approval modal. Each
submitted line is one :func:`~wqagent.core.agent.stream_session_events` turn, so the
conversation continues exactly like the ``wq`` CLI does.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Input, Static

from wqagent.cli.tui.approval import (
    ApprovalModal,
    ApprovalRequestedMessage,
    TuiApprovalHandler,
)

__all__ = ["AgentEventMessage", "WQAgentTuiApp", "format_event"]

#: Event kinds that should not clutter the stream (they are bookkeeping).
QUIET_EVENTS = frozenset({"node_end", "memory", "ai_message", "trace_saved"})


def format_event(event: Mapping[str, Any]) -> tuple[str, str]:
    """One stream line for an agent event: ``(text, css_class)``."""
    kind = str(event.get("type") or "event")
    node = str(event.get("node") or "")
    where = f"{node} · " if node else ""
    if kind == "user":
        return f"💬 {event.get('text') or ''}", "event-user"
    if kind == "tool_call":
        return f"🔧 {where}{event.get('name')} → {_args(event.get('args'))}", "event-tool"
    if kind == "tool_result":
        result = event.get("result") if isinstance(event.get("result"), Mapping) else {}
        ok = result.get("ok", True)
        detail = str(result.get("error") or result.get("result") or "")
        mark = "✔" if ok else "✖"
        return f"{mark} {where}{event.get('name')}: {_short(detail, 160)}", (
            "event-ok" if ok else "event-error"
        )
    if kind == "handoff":
        return (
            f"🔄 handoff · {event.get('from')} → {event.get('to')}: "
            f"{_short(event.get('instruction'), 120)}",
            "event-handoff",
        )
    if kind == "intent":
        confidence = float(event.get("intent_confidence") or 0.0)
        return (
            f"🧭 route · {event.get('intent_route')} [{confidence:.2f}] "
            f"{_short(event.get('intent_reason'), 100)}",
            "event-intent",
        )
    if kind == "chat_response":
        return f"💬 {_short(event.get('chat_response'), 160)}", "event-chat"
    if kind == "checkpoint_saved":
        git = str(event.get("git_commit") or "")[:8]
        return (
            f"💾 checkpoint · {event.get('status')}/{event.get('latest_node')} "
            f"({event.get('files')} files{', git ' + git if git else ''})",
            "event-meta",
        )
    if kind == "context_compressed":
        return (
            f"🗜 compressed · {event.get('messages_before')} msgs/"
            f"{event.get('tokens_before')} tokens → {event.get('tokens_after')} tokens",
            "event-meta",
        )
    if kind == "todo_update":
        return f"📝 plan · {_plan_chips(event.get('todos'))}", "event-plan"
    if kind == "plan_snapshot":
        return f"📋 plan · {_short(event.get('plan_summary'), 120)}", "event-plan"
    if kind == "search_results":
        return (
            f"🔍 {len(event.get('results') or [])} result(s) for {event.get('query')!r}",
            "event-tool",
        )
    if kind == "session_saved":
        return (
            f"🧵 session · turn {event.get('turn')} → {event.get('reply_turn')} "
            f"({event.get('route')}) saved={event.get('saved')}",
            "event-meta",
        )
    if kind == "turn_failed":
        return f"✖ turn failed: {_short(event.get('error'), 200)}", "event-error"
    if kind == "turn_finished":
        return "", "event-meta"
    return f"· {kind}: {_short(_json(event), 160)}", "event-meta"


def _plan_chips(todos: Any) -> str:
    """``✅ a  🔄 b`` for a todo list, instead of its Python repr."""
    if not isinstance(todos, (list, tuple)) or not todos:
        return _short(todos, 120)
    marks = {"completed": "✅", "in_progress": "🔄", "blocked": "⛔", "pending": "⬜"}
    chips = []
    for todo in todos:
        if isinstance(todo, Mapping):
            glyph = marks.get(str(todo.get("status") or "pending"), "⬜")
            chips.append(f"{glyph} {todo.get('id')}")
        else:
            chips.append(str(todo))
    return "   ".join(chips)


def _args(value: Any) -> str:
    if isinstance(value, Mapping):
        return ", ".join(f"{key}={_short(item, 60)}" for key, item in list(value.items())[:3])
    return _short(value, 80)


def _json(value: Any) -> str:
    import json

    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):  # pragma: no cover - default=str covers this
        return str(value)


def _short(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


class AgentEventMessage(Message):
    """One agent event, delivered on the UI thread."""

    def __init__(self, event: Mapping[str, Any]) -> None:
        self.event = dict(event)
        super().__init__()


class WQAgentTuiApp(App[None]):
    """The interactive front end: talk to the agent, watch what it does."""

    CSS = """
    /* ---- palette (matches the startup logo: slate + cyan/violet) ---------- */
    Screen {
        layout: vertical;
        background: #0b1220;
        color: #e2e8f0;
    }

    /* ---- brand bar ------------------------------------------------------- */
    #brandbar {
        height: 3;
        padding: 0 1;
        background: #0f172a;
        border-bottom: solid #1e293b;
    }
    #brand {
        width: auto;
        height: 3;
        content-align: left middle;
        text-style: bold;
        color: #22d3ee;
    }
    #status {
        width: 1fr;
        height: 3;
        content-align: right middle;
        color: #64748b;
    }

    /* ---- cards ----------------------------------------------------------- */
    #plan {
        height: auto;
        max-height: 7;
        margin: 1 1 0 1;
        padding: 0 1;
        background: #111c33;
        border: round #334155;
        border-title-color: #a78bfa;
        border-title-style: bold;
        color: #cbd5e1;
    }
    #plan-progress { color: #a78bfa; }
    .plan-chip { color: #cbd5e1; }

    #stream {
        height: 1fr;
        margin: 1;
        padding: 0 1;
        background: #0d1526;
        border: round #334155;
        border-title-color: #22d3ee;
        border-title-style: bold;
        scrollbar-size-vertical: 1;
        scrollbar-color: #334155;
        scrollbar-background: #0d1526;
    }
    #stream Static { padding: 0 0 0 1; }

    /* ---- event styles ---------------------------------------------------- */
    .event-user { color: #f8fafc; text-style: bold; }
    .event-tool { color: #38bdf8; }
    .event-ok { color: #34d399; }
    .event-error { color: #f87171; }
    .event-handoff { color: #c084fc; }
    .event-intent { color: #22d3ee; }
    .event-chat { color: #a7f3d0; }
    .event-plan { color: #fbbf24; }
    .event-meta { color: #64748b; }
    .event-final {
        color: #34d399;
        text-style: bold;
        background: #0f2a22;
        padding: 0 1;
    }
    .event-rule { color: #1e293b; }

    /* ---- input ----------------------------------------------------------- */
    #inputbar {
        height: 3;
        margin: 0 1 1 1;
        background: #111c33;
        border: round #334155;
    }
    #inputbar:focus-within { border: round #22d3ee; }
    #prompt-prefix {
        width: 3;
        content-align: center middle;
        color: #22d3ee;
        text-style: bold;
    }
    #prompt {
        border: none;
        background: #111c33;
        color: #e2e8f0;
    }
    #prompt:focus { border: none; }
    #hints {
        height: 1;
        padding: 0 2;
        color: #475569;
    }

    /* ---- approval modal -------------------------------------------------- */
    #approval-dialog {
        width: 84;
        height: auto;
        padding: 1 2;
        background: #111c33;
        border: thick #fbbf24;
        border-title-color: #fbbf24;
        border-title-style: bold;
        align: center middle;
    }
    #approval-title { text-style: bold; color: #fbbf24; padding: 0 0 1 0; }
    .approval-row { color: #94a3b8; }
    #approval-command {
        margin: 1 0;
        padding: 0 1;
        background: #0b1220;
        color: #f8fafc;
    }
    #approval-buttons { height: auto; align: center middle; padding: 1 0 0 0; }
    #approval-buttons Button { margin: 0 1; min-width: 16; }
    #approval-hint { color: #64748b; }
    """

    TITLE = "🐺 WQ Agent"
    SUB_TITLE = "workspace-scoped coding agent"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear_stream", "Clear stream"),
    ]

    #: Spinner frames for the "working" indicator in the brand bar.
    SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    #: Lines kept in the activity stream (the oldest are dropped).
    STREAM_LIMIT = 400
    #: Separator drawn between turns.
    TURN_RULE = "╌" * 46

    def __init__(
        self,
        *,
        workspace: str | Path | None = None,
        model: Any | None = None,
        runner: Callable[[str], Iterator[dict[str, Any]]] | None = None,
        approval_mode: str | None = None,
        approval_handler: Any | None = None,
        entry_router: bool = True,
        checkpoint_mode: str | None = None,
        trace_mode: str | None = None,
        max_attempts: int | None = None,
        max_loops: int | None = None,
        **options: Any,
    ) -> None:
        super().__init__()
        self.workspace = str(workspace or Path("workspace"))
        self.model = model
        self._runner = runner
        self.approval_mode = approval_mode
        self.approval_handler = approval_handler
        self.entry_router = entry_router
        self.checkpoint_mode = checkpoint_mode
        self.trace_mode = trace_mode
        self.max_attempts = max_attempts
        self.max_loops = max_loops
        self.options = options
        self.session: dict[str, Any] | None = None
        self.session_id = ""
        self.turn = 0
        self.plan: list[dict[str, Any]] = []
        self._busy = False
        self._pending_approvals = 0
        self._spin = 0
        self._spinner_timer: Any = None

    # -- layout --------------------------------------------------------------
    def compose(self) -> ComposeResult:
        with Horizontal(id="brandbar"):
            yield Static(f"🐺 WQ Agent", id="brand")
            yield Static(id="status")
        with Vertical():
            plan = Static(id="plan")
            plan.border_title = "Plan"
            plan.display = False  # nothing to show until the planner speaks
            yield plan
            stream = VerticalScroll(id="stream")
            stream.border_title = "Activity"
            with stream:
                yield Static(
                    "Ready — describe a task or ask a question.",
                    classes="event-meta",
                )
        with Horizontal(id="inputbar"):
            yield Static("›", id="prompt-prefix")
            yield Input(placeholder="ask WQ Agent…", id="prompt")
        yield Static(
            "Enter send   ·   Ctrl+L clear   ·   Ctrl+C quit",
            id="hints",
        )

    def on_mount(self) -> None:
        self._refresh_status()
        self.query_one("#prompt", Input).focus()

    # -- the turn ------------------------------------------------------------
    def on_input_submitted(self, event: Input.Submitted) -> None:
        task = str(event.value or "").strip()
        event.input.value = ""
        if not task:
            return
        if self._busy:
            self._append("⚠ still working on the previous turn - wait for it to finish", "event-error")
            return
        self._start_turn(task)

    def _start_turn(self, task: str) -> None:
        """One session turn, on its own thread."""
        self._busy = True
        if self.plan:  # separate turns visually once something has happened
            self._turn_rule()
        self._append(f"💬 {task}", "event-user")
        self._start_spinner()
        self._refresh_status()
        thread = threading.Thread(
            target=self._run_turn, args=(task,), name="wq-tui-turn", daemon=True
        )
        thread.start()

    def _run_turn(self, task: str) -> None:
        try:
            for event in self._runner_for(task):
                self.post_message(AgentEventMessage(event))
        except Exception as exc:  # noqa: BLE001 - the UI must stay usable
            self.post_message(
                AgentEventMessage({"type": "turn_failed", "error": f"{type(exc).__name__}: {exc}"})
            )
        finally:
            self.post_message(AgentEventMessage({"type": "turn_finished"}))

    def _runner_for(self, task: str) -> Iterator[dict[str, Any]]:
        if self._runner is not None:
            return self._runner(task)
        from wqagent.core.agent import stream_session_events

        return stream_session_events(
            task,
            session_workspace=self.workspace,
            model=self.model,
            session=self.session,
            approval_mode=self.approval_mode,
            approval_handler=self.approval_handler or TuiApprovalHandler(self),
            entry_router=self.entry_router,
            checkpoint_mode=self.checkpoint_mode,
            trace_mode=self.trace_mode,
            max_attempts=self.max_attempts,
            max_loops=self.max_loops,
            **self.options,
        )

    # -- events --------------------------------------------------------------
    def on_agent_event_message(self, message: AgentEventMessage) -> None:
        event = message.event
        kind = str(event.get("type") or "")
        if kind == "turn_finished":
            self._busy = False
            self._stop_spinner()
            self._refresh_status()
            return
        if kind == "final_answer":
            self._append(f"📝 {_short(event.get('content'), 400)}", "event-final")
            self._refresh_status(passed=bool(event.get("passed")), route=str(event.get("route") or ""))
            return
        if kind == "session_saved":
            self.session_id = str(event.get("session_id") or "")
            self.turn = int(event.get("turn_index") or self.turn)
            self._refresh_status(route=str(event.get("route") or ""))
            self._append(format_event(event)[0], "event-meta")
            return
        if self._updates_plan(event):
            self._update_plan(event)
            if kind == "todo_update":
                self._append(format_event(event)[0], "event-plan")
            return
        if kind in QUIET_EVENTS:
            return
        text, css = format_event(event)
        if text:
            self._append(text, css)

    @staticmethod
    def _updates_plan(event: Mapping[str, Any]) -> bool:
        """True when the event carries the plan (a snapshot, a todo update or a node)."""
        if str(event.get("type") or "") in {"plan_snapshot", "todo_update"}:
            return True
        update = event.get("update")
        return isinstance(update, Mapping) and bool(update.get("todos"))

    def _update_plan(self, event: Mapping[str, Any]) -> None:
        raw = event.get("todos")
        if raw is None and isinstance(event.get("update"), Mapping):
            raw = event["update"].get("todos")
        if isinstance(raw, list):
            self.plan = [item for item in raw if isinstance(item, Mapping)]
        summary = str(event.get("plan_summary") or "")
        if not summary and isinstance(event.get("update"), Mapping):
            summary = str(event["update"].get("plan_summary") or "")
        panel = self.query_one("#plan", Static)
        panel.update(self._render_plan(summary))
        panel.display = True  # the card only exists once there is a plan

    def _render_plan(self, summary: str = "") -> str:
        """The plan card: a progress line, then one chip per todo."""
        if not self.plan and not summary:
            return "(no plan yet)"
        done = sum(1 for todo in self.plan if str(todo.get("status")) == "completed")
        total = len(self.plan)
        bar = self._progress_bar(done, total)
        header = f"{bar}  {done}/{total} done" if total else bar
        if summary:
            header += f"  ·  {_short(summary, 60)}"
        marks = {
            "completed": ("✅", "dim #64748b"),
            "in_progress": ("🔄", "bold #fbbf24"),
            "blocked": ("⛔", "bold #f87171"),
            "pending": ("⬜", "#cbd5e1"),
        }
        chips = []
        for todo in self.plan:
            glyph, _ = marks.get(str(todo.get("status") or "pending"), ("⬜", ""))
            chips.append(f"{glyph} {todo.get('id')}")
        lines = [header]
        if chips:
            lines.append("   ".join(chips))
        return "\n".join(lines)

    @staticmethod
    def _progress_bar(done: int, total: int, width: int = 10) -> str:
        if total <= 0:
            return "▱" * width
        filled = max(0, min(width, round(width * done / total)))
        return "▰" * filled + "▱" * (width - filled)

    def _append(self, text: str, css: str = "event-meta") -> None:
        """Add one line to the (scrolling) activity stream."""
        if not str(text or "").strip():
            return
        stream = self.query_one("#stream", VerticalScroll)
        stream.mount(Static(text, classes=css))
        children = list(stream.children)
        for stale in children[: max(0, len(children) - self.STREAM_LIMIT)]:
            stale.remove()
        stream.scroll_end(animate=False)

    def _turn_rule(self) -> None:
        """A dim separator, drawn between turns."""
        self._append(self.TURN_RULE, "event-rule")

    def _tick(self) -> None:
        """Advance the brand-bar spinner while a turn runs."""
        self._spin = (self._spin + 1) % len(self.SPINNER)
        self._refresh_status()

    def _refresh_status(self, *, passed: bool | None = None, route: str = "") -> None:
        session = self.session_id[:8] if self.session_id else "new"
        parts = [f"session: {session}", f"turn {self.turn}"]
        if route:
            parts.append(f"last: {route}")
        if passed is not None:
            parts.append("✔ passed" if passed else "✖ not verified")
        if self._busy:
            parts.append("working…")
        if self._pending_approvals:
            parts.append(f"⚠ {self._pending_approvals} approval(s) waiting")
        self.query_one("#status", Static).update("  ·  ".join(parts))
        self._refresh_brand()

    def _refresh_brand(self) -> None:
        """The left side of the brand bar: the wolf, the state and the spinner."""
        if self._busy:
            frame = self.SPINNER[self._spin % len(self.SPINNER)]
            state = f"[#fbbf24]{frame} working[/]"
        elif self._pending_approvals:
            state = "[#f87171]● needs approval[/]"
        else:
            state = "[#34d399]● ready[/]"
        self.query_one("#brand", Static).update(f"🐺 WQ Agent   {state}")

    def _start_spinner(self) -> None:
        if self._spinner_timer is None:
            self._spinner_timer = self.set_interval(0.12, self._tick)

    def _stop_spinner(self) -> None:
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None

    # -- approval ------------------------------------------------------------
    def on_approval_requested_message(self, message: ApprovalRequestedMessage) -> None:
        """Show the modal for a risky command and feed the answer back to the tool."""
        gate = message.gate
        self._pending_approvals += 1
        self._append(f"⚠ approval required: {_short(gate.request.command, 140)}", "event-handoff")
        self._refresh_status()

        def decided(approved: bool | None) -> None:
            gate.resolve(bool(approved))
            self._pending_approvals = max(0, self._pending_approvals - 1)
            self._append(
                "✔ approved" if approved else "✖ denied",
                "event-ok" if approved else "event-error",
            )
            self._refresh_status()

        self.push_screen(ApprovalModal(gate, workspace=self.workspace), decided)

    # -- actions -------------------------------------------------------------
    def action_clear_stream(self) -> None:
        stream = self.query_one("#stream", VerticalScroll)
        for child in list(stream.children):
            child.remove()
