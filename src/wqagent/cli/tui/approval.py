"""Approval inside the TUI: the modal and the gate that connects it to the tool thread.

The bash tool asks for approval **from its own thread** and must not continue until a
human answered, while the dialog can only exist on the UI thread. :class:`ApprovalGate`
is the middle ground:

1. the tool thread calls :class:`TuiApprovalHandler`, which creates a gate and posts an
   :class:`ApprovalRequestedMessage`;
2. the app pushes :class:`ApprovalModal` and, when it is dismissed, calls
   ``gate.resolve(approved)``;
3. the tool thread sits in ``gate.wait()`` until that happens - with the run's approval
   timeout (``WQ_APPROVAL_TIMEOUT``) as the backstop, because a run nobody answers must
   end up *denied* rather than stuck.
"""

from __future__ import annotations

import threading
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from wqagent.core.approval import (
    ApprovalDecision,
    ApprovalRequest,
    approval_timeout_from_env,
)

__all__ = [
    "ApprovalGate",
    "ApprovalModal",
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


class ApprovalModal(ModalScreen[bool]):
    """The dialog: what would run, why it is risky, and Approve/Deny.

    ``Y``/Enter approves, ``N``/Escape denies; the buttons do the same.
    """

    BINDINGS = [
        Binding("y", "approve", "Approve"),
        Binding("enter", "approve", "Approve", show=False),
        Binding("n", "deny", "Deny"),
        Binding("escape", "deny", "Deny", show=False),
    ]

    def __init__(self, gate: ApprovalGate, *, workspace: Any = "") -> None:
        super().__init__()
        self.gate = gate
        self.workspace = str(workspace or "")

    def compose(self) -> ComposeResult:
        request = self.gate.request
        with Vertical(id="approval-dialog"):
            yield Static("⚠  approval required", id="approval-title")
            yield Static(f"tool      : {request.tool_name}", classes="approval-row")
            yield Static(f"risk      : {request.risk_reason}", classes="approval-row")
            yield Static(f"approval  : {request.id}", classes="approval-row")
            if self.workspace:
                yield Static(f"workspace : {self.workspace}", classes="approval-row")
            yield Static(f"$ {request.command}", id="approval-command")
            with Horizontal(id="approval-buttons"):
                yield Button("Approve (Y)", variant="success", id="approval-approve")
                yield Button("Deny (N)", variant="error", id="approval-deny")
            yield Static(
                f"Y / Enter approves · N / Esc denies"
                + (
                    f" · no answer within {self.gate.timeout:g}s = denied"
                    if self.gate.timeout > 0
                    else ""
                ),
                id="approval-hint",
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "approval-approve")

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)


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
