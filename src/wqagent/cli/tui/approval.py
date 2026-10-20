"""Approval inside the TUI: the inline prompt and the gate that connects it to the tool thread.

The bash tool asks for approval **from its own thread** and must not continue until a
human answered, while the prompt can only exist on the UI thread. :class:`ApprovalGate`
is the middle ground:

1. the tool thread calls :class:`TuiApprovalHandler`, which creates a gate and posts an
   :class:`ApprovalRequestedMessage`;
2. the app mounts an :class:`ApprovalBar` above the composer and, when a key answers it,
   calls ``gate.resolve(approved)``;
3. the tool thread sits in ``gate.wait()`` until that happens - with the run's approval
   timeout (``WQ_APPROVAL_TIMEOUT``) as the backstop, because a run nobody answers must
   end up *denied* rather than stuck.

The prompt is a **bar, not a modal**: it slides in three rows above the input, keeps the
conversation readable, and never swaps the screen out from under the user.
"""

from __future__ import annotations

import threading
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.events import Resize
from textual.message import Message
from textual.widgets import Static

from wqagent.core.approval import (
    ApprovalDecision,
    ApprovalRequest,
    approval_timeout_from_env,
)

__all__ = [
    "ApprovalBar",
    "ApprovalGate",
    "ApprovalRequestedMessage",
    "TuiApprovalHandler",
]


class ApprovalGate:
    """A one-shot handshake between the tool thread and the UI thread."""

    def __init__(self, request: ApprovalRequest, *, timeout: float | None = None) -> None:
        self.request = request
        self.timeout = approval_timeout_from_env() if timeout is None else float(timeout)
        self._done = threading.Event()
        self._decision: ApprovalDecision | None = None

    @property
    def resolved(self) -> bool:
        """True once someone answered (or the wait timed out)."""
        return self._done.is_set()

    @property
    def decision(self) -> ApprovalDecision | None:
        """The decision, if there is one yet."""
        return self._decision

    def resolve(self, approved: bool, reason: str = "") -> ApprovalDecision:
        """Record the answer. The first one wins; safe from any thread."""
        if self._done.is_set():
            return self._decision or ApprovalDecision(False, "no decision")
        default_reason = "approved in the TUI" if approved else "denied in the TUI"
        self._decision = ApprovalDecision(approved=bool(approved), reason=reason or default_reason)
        self._done.set()
        return self._decision

    def wait(self, timeout: float | None = None) -> ApprovalDecision:
        """Block until the gate is resolved; a timeout counts as *denied*."""
        limit = self.timeout if timeout is None else float(timeout)
        if not self._done.wait(None if limit <= 0 else limit):
            self.resolve(False, f"no answer within {limit:g}s")
        return self._decision or ApprovalDecision(False, "no decision")


class ApprovalRequestedMessage(Message):
    """Posted from the tool thread when a risky command needs a decision."""

    def __init__(self, gate: ApprovalGate) -> None:
        self.gate = gate
        super().__init__()


class ApprovalBar(Vertical):
    """Three rows above the composer: what would run, why it is risky, and the keys.

    ``y`` approves, ``n``/``esc`` deny. The app owns the keys (so they work whichever
    widget has focus) and calls :meth:`decided`; the bar only renders and reports.
    """

    DEFAULT_CSS = """
    ApprovalBar {
        height: 4;
        padding: 0 1;
        background: #fffbeb;
        border-top: solid #f59e0b;
    }
    ApprovalBar Static { height: 1; }
    """

    class Decided(Message):
        """The operator pressed a key; the app resolves the gate."""

        def __init__(self, approved: bool) -> None:
            self.approved = approved
            super().__init__()

    def __init__(
        self,
        gate: ApprovalGate,
        *,
        workspace: Any = "",
        queued: int = 0,
    ) -> None:
        super().__init__()
        self.gate = gate
        self.workspace = str(workspace or "")
        self.queued = int(queued)

    def compose(self) -> ComposeResult:
        yield Static(id="approval-title")
        yield Static(id="approval-command")
        yield Static(id="approval-hint")

    def on_mount(self) -> None:
        self.render_request()

    def on_resize(self, event: Resize) -> None:
        self.render_request()

    # -- rendering -----------------------------------------------------------
    def render_request(self) -> None:
        """Rewrite the three rows to fit the current width."""
        if len(self.children) < 3:  # called before the rows were composed
            return
        # on_mount runs before the first layout, where self.size is still 0 wide
        width = max(20, (self.size.width or self.app.size.width) - 2)
        request = self.gate.request

        title = Text()
        title.append("▲  approval required", style="bold #B45309")
        title.append(f"  ·  {request.tool_name}", style="#92400E")
        if self.queued:
            title.append(f"  ·  {self.queued} more waiting", style="#B45309")
        self.query_one("#approval-title", Static).update(_fit(title, width))

        command = Text()
        command.append("$ ", style="#92400E")
        command.append(_fit(Text(str(request.command)), width - 2).plain, style="#1f2933")
        lines = Text()
        lines.append_text(command)
        risk = str(request.risk_reason or "")
        if risk:
            lines.append(f"   ({risk})", style="#92400E")
        self.query_one("#approval-command", Static).update(_fit(lines, width))

        hint = Text()
        hint.append(" y ", style="bold #ffffff on #B45309")
        hint.append(" approve    ", style="#7c2d12")
        hint.append(" n ", style="bold #ffffff on #92400E")
        hint.append(" deny    ", style="#7c2d12")
        hint.append(" esc ", style="bold #ffffff on #92400E")
        hint.append(" deny", style="#7c2d12")
        if self.gate.timeout > 0:
            note = f"      no answer in {self.gate.timeout:g}s = denied"
            if hint.cell_len + len(note) <= width:  # never truncate the keys away
                hint.append(note, style="#a16207")
        self.query_one("#approval-hint", Static).update(_fit(hint, width))

    # -- input ---------------------------------------------------------------
    def decide(self, approved: bool) -> None:
        """Report the decision to the app (it resolves the gate and unmounts the bar)."""
        self.post_message(self.Decided(approved))


def _fit(text: Text, width: int) -> Text:
    """Truncate a one-line Text to *width* cells, keeping its styles."""
    if width <= 0 or text.cell_len <= width:
        return text
    fitted = text.copy()
    fitted.truncate(width - 1, overflow="ellipsis")
    return fitted


class TuiApprovalHandler:
    """``approval_handler`` for a TUI run: post the request, block the tool thread."""

    def __init__(self, app: Any, *, timeout: float | None = None) -> None:
        self.app = app
        self.timeout = timeout

    def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        gate = ApprovalGate(request, timeout=self.timeout)
        try:
            self.app.post_message(ApprovalRequestedMessage(gate))
        except Exception as exc:  # noqa: BLE001 - never leave the tool thread hanging
            return ApprovalDecision(False, f"the TUI could not ask ({type(exc).__name__}: {exc})")
        return gate.wait()
