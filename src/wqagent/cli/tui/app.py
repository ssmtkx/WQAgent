"""The WQ Agent TUI, built with `Textual <https://textual.textualize.io>`_.

Skeleton **C** -- a two-pane layout whose plan sidebar folds away::

    +----------------------------------------------------------------------+
    | WQ  Agent           8f3c1d2e . turn 2 . workflow          * ready    |  brand bar
    +--------------------------+-------------------------------------------+
    | Plan                     | Activity                                  |
    | =====--------------------| >  handoff  planner -> codeAgent           |
    | 3 / 7 done               | >  file_write -> app/main.py               |
    | v  scaffold FastAPI app  | v  file_write : written (412 lines)        |
    | *  implement auth        | >  checkpoint  running/codeAgent (4 files) |
    | o  add API docs          | >  chat : here is the FastAPI scaffold...  |
    | x  deploy to staging     | *  intent  chat [0.82]  quick question     |
    |                          | +---------------------------------------+  |
    |                          | | >  ask WQ Agent...                    |  |
    |                          | +---------------------------------------+  |
    +--------------------------+-------------------------------------------+
    | Enter send   Ctrl+L clear   Ctrl+B plan   Ctrl+C quit        .\\demo  |  key bar
    +----------------------------------------------------------------------+

The sidebar is :data:`PLAN_WIDTH` columns wide and toggles with ``Ctrl+B``.
Below :data:`NARROW_WIDTH` columns it collapses by itself: 38 columns of plan
inside an 80-column terminal leaves the event log nothing but wrapped scraps.

The agent itself runs in a **background thread** (one thread per turn), and every
event it yields is posted to the UI thread as an :class:`AgentEventMessage`; the
UI then updates the plan pane, the activity log, the brand bar or the approval
modal. Each submitted line is one
:func:`~wqagent.core.agent.stream_session_events` turn, so the conversation
continues exactly like the ``wq`` CLI does.

Every container height is an explicit number of rows. A ``height: auto`` around a
``1fr`` child is what collapsed this screen to a blank rectangle on Textual 8, so
:func:`test_layout_has_real_geometry` pins the result down.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Input, ProgressBar, Static

from wqagent.cli.tui.approval import (
    ApprovalBar,
    ApprovalGate,
    ApprovalRequestedMessage,
    TuiApprovalHandler,
)
from wqagent.cli.tui.diff_view import DiffBlock
from wqagent.cli.tui.stream_view import LiveReply

__all__ = ["AgentEventMessage", "WQAgentTuiApp", "format_event", "keybar_text"]

#: Event kinds that should not clutter the stream (they are bookkeeping).
QUIET_EVENTS = frozenset({"node_end", "memory", "ai_message", "trace_saved"})

#: Columns reserved for the plan sidebar while it is expanded.
PLAN_WIDTH = 38
#: Terminals narrower than this fold the sidebar away on their own.
NARROW_WIDTH = 100

#: One glyph per todo state, in the design's vocabulary (text, never emoji).
TODO_MARKS = {"completed": "✓", "in_progress": "●", "pending": "○", "blocked": "✕"}
#: Todo state -> CSS class, so colours stay in the stylesheet.
TODO_CLASSES = {
    "completed": "todo-completed",
    "in_progress": "todo-active",
    "pending": "todo-pending",
    "blocked": "todo-blocked",
}

#: Argument keys worth showing as the target of a tool call, in order.
_TARGET_KEYS = ("file_path", "path", "command", "pattern", "query", "url", "name")


# -- event formatting --------------------------------------------------------
def format_event(event: Mapping[str, Any]) -> tuple[str, str]:
    """One activity line for an agent event: ``(text, css_class)``.

    The markers follow the design: ``↳`` handoff, ``›`` tool call, ``✓``/``✕``
    result, ``»`` metadata, ``●`` routing.
    """
    kind = str(event.get("type") or "event")

    if kind == "user":
        return f"› {event.get('text') or ''}", "event-user"
    if kind == "tool_call":
        return f"› {event.get('name')} → {_target(event.get('args'))}", "event-tool"
    if kind == "tool_result":
        result = event.get("result") if isinstance(event.get("result"), Mapping) else {}
        ok = bool(result.get("ok", True))
        detail = _short(result.get("result") if ok else result.get("error"), 140)
        mark = "✓" if ok else "✕"
        return f"{mark} {event.get('name')} : {detail}".rstrip(), (
            "event-ok" if ok else "event-error"
        )
    if kind == "handoff":
        instruction = _short(event.get("instruction"), 60)
        line = f"↳  handoff  {event.get('from')} → {event.get('to')}"
        return (f"{line}  ·  {instruction}" if instruction else line), "event-handoff"
    if kind == "intent":
        confidence = float(event.get("intent_confidence") or 0.0)
        reason = _short(event.get("intent_reason"), 80)
        line = f"●  intent  {event.get('intent_route')} [{confidence:.2f}]"
        return (f"{line}  {reason}" if reason else line), "event-intent"
    if kind == "chat_response":
        return f"»  chat : {_short(event.get('chat_response'), 160)}", "event-chat"
    if kind == "checkpoint_saved":
        return (
            f"»  checkpoint  {event.get('status')}/{event.get('latest_node')} "
            f"({event.get('files')} files)",
            "event-meta",
        )
    if kind == "context_compressed":
        return (
            f"»  context  {event.get('messages_before')} msgs/"
            f"{event.get('tokens_before')} → {event.get('tokens_after')} tokens",
            "event-meta",
        )
    if kind == "todo_update":
        return _plan_line(event.get("todos")), "event-plan"
    if kind == "plan_snapshot":
        return f"»  plan  {_short(event.get('plan_summary'), 90)}", "event-plan"
    if kind == "search_results":
        return (
            f"›  search  {len(event.get('results') or [])} result(s) "
            f"for {event.get('query')!r}",
            "event-tool",
        )
    if kind == "session_saved":
        return (
            f"»  session  turn {event.get('turn')} → {event.get('reply_turn')} "
            f"({event.get('route')})",
            "event-meta",
        )
    if kind == "turn_failed":
        return f"✕  turn failed : {_short(event.get('error'), 200)}", "event-error"
    if kind == "turn_finished":
        return "", "event-meta"
    return f"»  {kind} : {_short(_json(event), 140)}", "event-meta"


def _plan_line(todos: Any) -> str:
    """``»  plan  2 / 7 done  ·  ● implement auth`` for one todo update."""
    if not isinstance(todos, (list, tuple)) or not todos:
        return "»  plan  (empty)"
    done = sum(1 for todo in todos if _status(todo) == "completed")
    line = f"»  plan  {done} / {len(todos)} done"
    active = next((todo for todo in todos if _status(todo) == "in_progress"), None)
    if isinstance(active, Mapping):
        label = _short(active.get("content") or active.get("id"), 50)
        line += f"  ·  ● {label}"
    return line


def _status(todo: Any) -> str:
    return str(todo.get("status") or "pending") if isinstance(todo, Mapping) else "pending"


def _target(value: Any) -> str:
    """The most meaningful argument of a tool call (a path, a command, ...)."""
    if isinstance(value, Mapping):
        for key in _TARGET_KEYS:
            if key in value:
                return _short(value[key], 70)
        for item in list(value.values())[:1]:
            return _short(item, 70)
        return ""
    return _short(value, 70)


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
    return text[: limit - 1].rstrip() + "…"


def keybar_text(*, plan_open: bool = True) -> Text:
    """The key hints, drawn as chips (a one-row bar cannot draw borders)."""
    hints = (
        ("Enter", "send"),
        ("Ctrl+L", "clear"),
        ("Ctrl+B", "plan ▾" if plan_open else "plan ▸"),
        ("Ctrl+D", "diff"),
        ("Ctrl+C", "quit"),
    )
    text = Text()
    for index, (key, label) in enumerate(hints):
        if index:
            text.append("  ")
        text.append(f" {key} ", style="bold #0F6E56 on #dde5ee")
        text.append(f" {label}", style="#475569")
    return text


def _tail(value: Any, limit: int) -> str:
    """A path shortened from the left (the last segments matter, not the drive)."""
    text = str(value or "")
    return text if len(text) <= limit else "…" + text[-(limit - 1) :]


class AgentEventMessage(Message):
    """One agent event, delivered on the UI thread."""

    def __init__(self, event: Mapping[str, Any]) -> None:
        self.event = dict(event)
        super().__init__()


class WQAgentTuiApp(App[None]):
    """The interactive front end: talk to the agent, watch what it does."""

    CSS = """
    /* ---- surfaces: white body, hairline rules, one grey key bar ---------- */
    Screen {
        layout: vertical;
        background: #ffffff;
        color: #1f2933;
    }

    /* ---- brand bar ------------------------------------------------------- */
    #brandbar {
        height: 3;
        padding: 0 1;
        background: #ffffff;
        border-bottom: solid #e6e9ed;
        align: left middle;
    }
    #chip {
        width: auto;
        height: 1;
        padding: 0 1;
        background: #0F6E56;
        color: #ffffff;
        text-style: bold;
    }
    #brand {
        width: auto;
        height: 1;
        margin: 0 0 0 1;
        color: #1f2933;
        text-style: bold;
    }
    #status {
        width: 1fr;
        height: 1;
        content-align: center middle;
        color: #64748b;
    }
    #ready {
        width: auto;
        height: 1;
        color: #15803D;
        text-style: bold;
    }

    /* ---- the two panes --------------------------------------------------- */
    #panes { height: 1fr; }
    #planpane {
        width: 38;
        height: 1fr;
        padding: 0 1;
        border-right: solid #e6e9ed;
    }
    #activitypane {
        width: 1fr;
        height: 1fr;
        padding: 0 1;
    }
    .pane-title {
        height: 1;
        margin: 1 0 0 0;
        color: #1f2933;
        text-style: bold;
    }

    /* ---- plan pane: one thin bar, a caption, then a row per todo --------- */
    #plan-progress {
        height: 1;
        margin: 0;
        background: #ffffff;
    }
    #plan-progress Bar { width: 1fr; }
    #plan-progress Bar > .bar--bar { color: #0F6E56; background: #e2e8f0; }
    #plan-progress Bar > .bar--complete { color: #15803D; background: #e2e8f0; }
    #plan-caption {
        height: 1;
        color: #64748b;
    }
    #plan {
        height: 1fr;
        scrollbar-size-vertical: 0;
    }
    #plan Static { padding: 0; }
    .todo-completed { color: #9aa0a6; }
    .todo-active { color: #1f2933; text-style: bold; }
    .todo-pending { color: #64748b; }
    .todo-blocked { color: #B91C1C; text-style: bold; }

    /* ---- activity pane: the log, then the composer at its foot ----------- */
    #stream {
        height: 1fr;
        padding: 0;
        background: #ffffff;
        scrollbar-size-vertical: 1;
        scrollbar-color: #cbd5e1;
        scrollbar-background: #ffffff;
    }
    #inputbar {
        height: 3;
        margin: 0 0 1 0;
        background: #ffffff;
        border: round #d8dee6;
    }
    #inputbar:focus-within { border: round #0F6E56; }
    #prompt-prefix {
        width: 3;
        height: 1;
        content-align: center middle;
        color: #0F6E56;
        text-style: bold;
    }
    #prompt {
        height: 1;
        border: none;
        background: #ffffff;
        color: #1f2933;
    }
    #prompt:focus { border: none; }
    #prompt:disabled { color: #9aa0a6; }

    /* ---- key bar: two rows so the hairline does not eat the text --------- */
    #keybar {
        height: 2;
        padding: 0 1;
        background: #eef2f6;
        border-top: solid #e6e9ed;
        align: left middle;
    }
    #keys {
        width: 1fr;
        height: 1;
    }
    #workspace {
        width: auto;
        height: 1;
        color: #9aa0a6;
    }

    /* ---- event styles (light surface, semantic colours, no emoji) -------- */
    .event-user { color: #0f172a; text-style: bold; }
    .event-tool { color: #0E7490; }
    .event-ok { color: #15803D; }
    .event-error { color: #B91C1C; }
    .event-handoff { color: #6D28D9; }
    .event-intent { color: #1D4ED8; }
    .event-chat { color: #1f2933; }
    .event-plan { color: #64748b; }
    .event-meta { color: #64748b; }
    .event-final { color: #15803D; text-style: bold; }
    """

    theme = "textual-light"
    TITLE = "WQ Agent"
    SUB_TITLE = "workspace-scoped coding agent"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+b", "toggle_plan", "Plan"),
        Binding("ctrl+d", "toggle_diff", "Diff"),
        Binding("ctrl+l", "clear_stream", "Clear stream"),
    ]

    #: Spinner frames for the "working" indicator in the brand bar.
    SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    #: Lines kept in the activity log (the oldest are dropped).
    STREAM_LIMIT = 400

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
        self._approval_queue: list[ApprovalGate] = []
        self._active_gate: ApprovalGate | None = None
        self._approval_bar: ApprovalBar | None = None
        self._diffs: list[DiffBlock] = []
        #: The line a streamed reply is being written into (``None`` when idle).
        self._live: LiveReply | None = None
        self._live_text = ""
        self._token_buffer = ""
        self._flush_timer: Any = None
        self._last_reply = ""
        self._spin = 0
        self._spinner_timer: Any = None
        self._route = ""
        self._passed: bool | None = None
        #: What the user asked for; a narrow terminal overrides it, not the reverse.
        self._want_plan = True
        #: Set when the user forces the sidebar open on a terminal that auto-folded it.
        self._ignore_narrow = False

    # -- layout --------------------------------------------------------------
    def compose(self) -> ComposeResult:
        with Horizontal(id="brandbar"):
            yield Static(" WQ ", id="chip")
            yield Static("Agent", id="brand")
            yield Static(id="status")
            yield Static(id="ready")
        with Horizontal(id="panes"):
            with Vertical(id="planpane"):
                yield Static("Plan", classes="pane-title")
                yield ProgressBar(
                    total=1,
                    show_percentage=False,
                    show_eta=False,
                    id="plan-progress",
                )
                yield Static("no plan yet", id="plan-caption")
                with VerticalScroll(id="plan"):
                    yield from ()
            with Vertical(id="activitypane"):
                yield Static("Activity", classes="pane-title")
                yield VerticalScroll(id="stream")
                with Horizontal(id="inputbar"):
                    yield Static("›", id="prompt-prefix")
                    yield Input(placeholder="ask WQ Agent…", id="prompt")
        with Horizontal(id="keybar"):
            yield Static(keybar_text(), id="keys")
            yield Static(_tail(self.workspace, 30), id="workspace")

    def on_mount(self) -> None:
        self.theme = "textual-light"
        self.dark = False  # guarantee a light surface regardless of default
        self._apply_plan_visibility()
        self._apply_keybar()
        self._render_header()
        self._append("Ready — describe a task or ask a question.", "event-meta")
        self.query_one("#prompt", Input).focus()

    def on_resize(self, event: events.Resize) -> None:
        self._apply_plan_visibility()
        self._apply_keybar()

    # -- the collapsible sidebar ---------------------------------------------
    def _plan_visible(self) -> bool:
        width = self.size.width
        return self._want_plan and (width >= NARROW_WIDTH or self._ignore_narrow)

    def _apply_plan_visibility(self) -> None:
        """Fold the plan pane away on narrow terminals (or when the user asked)."""
        if not self.is_mounted:
            return
        if self.size.width >= NARROW_WIDTH:
            self._ignore_narrow = False  # widening re-arms the automatic rule
        visible = self._plan_visible()
        pane = self.query_one("#planpane", Vertical)
        if pane.display != visible:
            pane.display = visible
            self._apply_keybar()  # the hint reads "plan ▾" / "plan ▸"

    def action_toggle_plan(self) -> None:
        """``Ctrl+B``: hide the sidebar, or force it open on a terminal that folded it."""
        if self._plan_visible():
            self._want_plan = False
            self._ignore_narrow = False
        else:
            self._want_plan = True
            self._ignore_narrow = self.size.width < NARROW_WIDTH
        self._apply_plan_visibility()

    def action_toggle_diff(self) -> None:
        """``Ctrl+D``: open or close the most recent patch in the activity log."""
        blocks = [block for block in self._diffs if block.parent is not None]
        if not blocks:
            return
        block = blocks[-1]
        block.toggle()
        block.scroll_visible(animate=False)

    def _apply_keybar(self) -> None:
        """Fit the key hints first; the workspace path gets whatever is left."""
        if not self.is_mounted:
            return
        keys = keybar_text(plan_open=self._plan_visible())
        self.query_one("#keys", Static).update(keys)
        hint = self.query_one("#workspace", Static)
        space = self.size.width - 4 - keys.cell_len  # padding 1 each side + a gap
        hint.display = space >= 18  # never let the path eat a key hint
        if hint.display:
            hint.update(_tail(self.workspace, min(30, space)))

    # -- the turn ------------------------------------------------------------
    def on_input_submitted(self, event: Input.Submitted) -> None:
        task = str(event.value or "").strip()
        event.input.value = ""
        if not task:
            return
        if self._busy:
            self._append("still working on the previous turn — wait for it", "event-error")
            return
        self._start_turn(task)

    def _start_turn(self, task: str) -> None:
        """One session turn, on its own thread."""
        self._busy = True
        self._last_reply = ""
        self._append(f"› {task}", "event-user")
        self._start_spinner()
        self._render_header()
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
        if kind == "token":
            self._on_token(str(event.get("text") or ""))
            return

        # anything else ends a reply that was still streaming
        final = ""
        if kind == "chat_response":
            final = str(event.get("chat_response") or "")
        elif kind == "final_answer":
            final = str(event.get("content") or "")
        streamed = self._close_stream(final)

        if kind == "turn_finished":
            self._busy = False
            self._stop_spinner()
            self._render_header()
            return
        if kind == "final_answer":
            self._passed = bool(event.get("passed"))
            self._route = str(event.get("route") or self._route)
            # the chat route reports the same text twice: do not print it again
            if not (streamed or (final.strip() and final.strip() == self._last_reply.strip())):
                self._append(f"»  answer : {_short(final, 400)}", "event-final")
            self._last_reply = final or self._last_reply
            self._render_header()
            return
        if kind == "chat_response":
            self._last_reply = final
            if not streamed:
                self._append(format_event(event)[0], "event-chat")
            return
        if kind == "session_saved":
            self.session_id = str(event.get("session_id") or "")
            self.turn = int(event.get("turn_index") or self.turn)
            self._route = str(event.get("route") or self._route)
            self._render_header()
            self._append(format_event(event)[0], "event-meta")
            return
        if self._updates_plan(event):
            self._update_plan(event)
            if kind == "todo_update":
                self._append(_plan_line(event.get("todos")), "event-plan")
            return
        if kind in QUIET_EVENTS:
            return
        if kind == "tool_result" and isinstance(event.get("diff"), Mapping):
            self._mount_diff(event)
            return
        text, css = format_event(event)
        if text:
            self._append(text, css)

    def _mount_diff(self, event: Mapping[str, Any]) -> None:
        """A file-changing tool call: the summary line plus a collapsible patch."""
        diff = event.get("diff")
        if not isinstance(diff, Mapping):
            return
        text, css = format_event(event)
        block = DiffBlock(dict(diff), header=text, css=css)
        self._append_widget(block)
        self._diffs.append(block)

    # -- streaming replies ---------------------------------------------------
    def _on_token(self, text: str) -> None:
        """One model delta: open the live line if needed, then buffer the text."""
        if not text:
            return
        if self._live is not None and self._live.parent is None:
            self._live = None  # the log was trimmed under it
        if self._live is None:
            live = LiveReply()
            self._live = live
            self._live_text = ""
            self._append_widget(live)
        self._token_buffer += text
        if self._flush_timer is None:
            # ~20 repaints a second: a fast stream must not starve the event loop
            self._flush_timer = self.set_timer(0.05, self._flush_tokens)

    def _flush_tokens(self) -> None:
        self._flush_timer = None
        block = self._live
        if block is None or block.parent is None or not self._token_buffer:
            return
        self._live_text += self._token_buffer
        self._token_buffer = ""
        block.update_text(self._live_text)
        self.query_one("#stream", VerticalScroll).scroll_end(animate=False)

    def _close_stream(self, final_text: str = "") -> bool:
        """Finish the live line, keeping (or replacing) its text; True if one was open."""
        block = self._live
        if block is None:
            return False
        if self._token_buffer:
            self._live_text += self._token_buffer
            self._token_buffer = ""
        if final_text.strip():
            self._live_text = final_text
        if block.parent is not None:
            block.update_text(self._live_text)
        self._live = None
        if self._flush_timer is not None:
            self._flush_timer.stop()
            self._flush_timer = None
        return True

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
        done = sum(1 for todo in self.plan if _status(todo) == "completed")
        total = len(self.plan)
        self.query_one("#plan-progress", ProgressBar).update(
            total=max(total, 1), progress=done
        )
        self.query_one("#plan-caption", Static).update(
            f"{done} / {total} done" if total else "no plan yet"
        )
        panel = self.query_one("#plan", VerticalScroll)
        for child in list(panel.children):
            child.remove()
        for todo in self.plan:
            status = _status(todo)
            label = _short(todo.get("content") or todo.get("id") or "", PLAN_WIDTH - 6)
            panel.mount(
                Static(
                    f"{TODO_MARKS.get(status, '○')}  {label}",
                    classes=f"todo {TODO_CLASSES.get(status, 'todo-pending')}",
                )
            )

    def _append(self, text: str, css: str = "event-meta") -> None:
        """Add one line to the (scrolling) activity log."""
        text = str(text or "")
        if not text.strip():
            return
        self._append_widget(Static(text, classes=css))

    def _append_widget(self, widget: Static) -> None:
        """Mount one row in the log, keep it bounded, and follow the tail."""
        stream = self.query_one("#stream", VerticalScroll)
        stream.mount(widget)
        self._trim_stream()
        stream.scroll_end(animate=False)

    def _trim_stream(self) -> None:
        """Drop the oldest lines once the log is over :data:`STREAM_LIMIT`."""
        stream = self.query_one("#stream", VerticalScroll)
        children = list(stream.children)
        for stale in children[: max(0, len(children) - self.STREAM_LIMIT)]:
            stale.remove()
        self._diffs = [block for block in self._diffs if block.parent is not None]

    # -- brand bar -----------------------------------------------------------
    def _tick(self) -> None:
        """Advance the brand-bar spinner while a turn runs."""
        self._spin = (self._spin + 1) % len(self.SPINNER)
        self._render_ready()

    def _render_header(self) -> None:
        """Session, turn and route in the middle; the run's state on the right."""
        session = self.session_id[:8] if self.session_id else "new"
        parts = [session, f"turn {self.turn}"]
        if self._route:
            parts.append(self._route)
        self.query_one("#status", Static).update(" · ".join(parts))
        self._render_ready()

    def _render_ready(self) -> None:
        if self._busy:
            frame = self.SPINNER[self._spin % len(self.SPINNER)]
            ready = Text(f"{frame} working", style="bold #B45309")
        elif self._pending_approvals:
            ready = Text("● approval", style="bold #B91C1C")
        else:
            ready = Text("● ready", style="bold #15803D")
        self.query_one("#ready", Static).update(ready)

    def _start_spinner(self) -> None:
        if self._spinner_timer is None:
            self._spinner_timer = self.set_interval(0.12, self._tick)

    def _stop_spinner(self) -> None:
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None

    # -- approval ------------------------------------------------------------
    def on_approval_requested_message(self, message: ApprovalRequestedMessage) -> None:
        """Queue a risky command; the inline bar shows one request at a time."""
        self._approval_queue.append(message.gate)
        self._append(
            f"▲  approval required : {_short(message.gate.request.command, 140)}",
            "event-handoff",
        )
        self._sync_pending()
        bar = self._approval_bar
        if bar is not None:  # a second request arrived: show the queue depth
            bar.queued = len(self._approval_queue)
            bar.render_request()
        self._show_next_approval()

    def _show_next_approval(self) -> None:
        """Mount the bar for the next queued gate, if nothing is being asked."""
        if self._active_gate is not None or not self._approval_queue:
            return
        gate = self._approval_queue.pop(0)
        self._active_gate = gate
        bar = ApprovalBar(gate, workspace=self.workspace, queued=len(self._approval_queue))
        self._approval_bar = bar
        self.query_one("#activitypane").mount(bar, before=self.query_one("#inputbar"))
        self.query_one("#prompt", Input).disabled = True  # nobody types while a gate waits
        self._sync_pending()
        self._render_ready()

    def _finish_approval(self, approved: bool) -> None:
        """Answer the active gate, drop the bar, and show the next one if any."""
        gate = self._active_gate
        if gate is None:
            return
        self._active_gate = None
        gate.resolve(approved)  # releases the tool thread
        self._append(
            "✓  approved" if approved else "✕  denied",
            "event-ok" if approved else "event-error",
        )
        bar = self._approval_bar
        self._approval_bar = None
        if bar is not None:
            bar.remove()
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = False
        prompt.focus()
        self._sync_pending()
        self._show_next_approval()
        self._render_ready()

    def on_approval_bar_decided(self, message: ApprovalBar.Decided) -> None:
        self._finish_approval(message.approved)

    def on_key(self, event: events.Key) -> None:
        """While a gate waits, the whole screen answers ``y`` / ``n`` / ``esc``."""
        if self._active_gate is None:
            return
        if event.key == "y":
            event.stop()
            self._finish_approval(True)
        elif event.key in ("n", "escape"):
            event.stop()
            self._finish_approval(False)

    def _sync_pending(self) -> None:
        self._pending_approvals = len(self._approval_queue) + (1 if self._active_gate else 0)

    # -- actions -------------------------------------------------------------
    def action_clear_stream(self) -> None:
        self._close_stream()
        stream = self.query_one("#stream", VerticalScroll)
        for child in list(stream.children):
            child.remove()
        self._diffs.clear()
