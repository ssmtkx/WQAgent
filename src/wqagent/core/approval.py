"""Risk classification and approval for shell commands.

The bash tool has two different guards:

* **Destructive patterns** (:data:`~wqagent.core.state.DEFAULT_BLOCKED_COMMAND_PATTERNS`)
  are refused outright - they reach far beyond the workspace.
* **Risky commands** (this module) are *allowed*, but only with approval: installing
  packages, downloading from the network, or starting a long-running server. What
  "approval" means is a per-run policy:

  ``inline`` (default)
      Ask the human, through the handler the runtime was given. Without a handler the
      command is **denied**: an unattended run must not install packages on its own.
  ``auto``
      Run it, and mark the outcome with ``requires_approval=True`` so the transcript
      shows what happened.
  ``deny``
      Refuse it outright.

The policy comes from ``RuntimeState.approval_mode`` (``WQ_APPROVAL_MODE`` sets the
default), and decisions are recorded on the :class:`~wqagent.tools.bash_tool.ShellOutcome`
that the model sees::

    "$ pip install requests"
    exit_code: 0
    requires_approval: True (auto-approved: Python package installation)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from uuid import uuid4

__all__ = [
    "APPROVAL_MODE_ENV",
    "APPROVAL_TIMEOUT_ENV",
    "DEFAULT_APPROVAL_MODE",
    "DEFAULT_APPROVAL_TIMEOUT_SECONDS",
    "RISK_PATTERNS",
    "VALID_APPROVAL_MODES",
    "ApprovalDecision",
    "ApprovalRequest",
    "approval_mode_from_env",
    "approval_timeout_from_env",
    "classify_command_risk",
    "new_approval_request",
    "normalize_approval_mode",
]

#: ``(pattern, reason)`` pairs for commands that need approval before they run.
RISK_PATTERNS: list[tuple[str, str]] = [
    (r"(?:^|&&|\|\||;)\s*(?:python\s+-m\s+)?pip\s+install\b", "Python package installation"),
    (r"(?:^|&&|\|\||;)\s*uv\s+add\b", "Project dependency change with uv add"),
    (r"(?:^|&&|\|\||;)\s*uv\s+sync\b", "Dependency synchronization with uv sync"),
    (r"(?:^|&&|\|\||;)\s*uv\s+pip\s+install\b", "Python package installation with uv pip"),
    (r"(?:^|&&|\|\||;)\s*npm\s+install\b", "Node package installation"),
    (r"(?:^|&&|\|\||;)\s*pnpm\s+install\b", "Node package installation"),
    (r"(?:^|&&|\|\||;)\s*yarn\s+(?:install\b|add\b)", "Node package installation"),
    (r"(?:^|&&|\|\||;)\s*(?:curl|wget)\b", "Network download command"),
    (r"(?:^|&&|\|\||;)\s*uvicorn\b", "Long-running development server"),
    (r"(?:^|&&|\|\||;)\s*python\s+-m\s+http\.server\b", "Long-running development server"),
]

#: Compiled once; matching is case-insensitive because Windows shells are.
_COMPILED_RISK_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), reason) for pattern, reason in RISK_PATTERNS
)

#: Approval policies a run may use.
VALID_APPROVAL_MODES: frozenset[str] = frozenset({"inline", "auto", "deny"})
#: Policy used when none was chosen (and for unusable values).
DEFAULT_APPROVAL_MODE = "inline"
#: Environment variable that chooses the policy, e.g. ``WQ_APPROVAL_MODE=auto``.
APPROVAL_MODE_ENV = "WQ_APPROVAL_MODE"
#: Environment variable that bounds how long an ``inline`` prompt waits, in seconds
#: (``0`` waits forever). A timeout counts as "no".
APPROVAL_TIMEOUT_ENV = "WQ_APPROVAL_TIMEOUT"
#: Seconds an ``inline`` prompt waits for an answer before denying.
DEFAULT_APPROVAL_TIMEOUT_SECONDS = 300.0


def classify_command_risk(command: str) -> str | None:
    """Return the risk reason when *command* needs approval, otherwise ``None``.

    Only the *start* of a command (or of a chained one: after ``&&``, ``||`` or
    ``;``) counts, so a mere mention such as ``echo "pip install x"`` is not risky.
    """
    text = str(command or "").strip()
    if not text:
        return None
    for pattern, reason in _COMPILED_RISK_PATTERNS:
        if pattern.search(text):
            return reason
    return None


def normalize_approval_mode(mode: str | None) -> str:
    """The mode to use: missing, blank or unknown values fall back to ``"inline"``."""
    text = str(mode or "").strip().lower()
    return text if text in VALID_APPROVAL_MODES else DEFAULT_APPROVAL_MODE


def approval_mode_from_env(default: str = DEFAULT_APPROVAL_MODE) -> str:
    """The mode from ``WQ_APPROVAL_MODE``, or *default* (itself normalized)."""
    raw = os.environ.get(APPROVAL_MODE_ENV, "")
    return normalize_approval_mode(raw) if raw.strip() else normalize_approval_mode(default)


def approval_timeout_from_env(
    default: float = DEFAULT_APPROVAL_TIMEOUT_SECONDS,
) -> float:
    """Seconds an ``inline`` prompt waits, from ``WQ_APPROVAL_TIMEOUT``.

    ``0`` (or a negative value) means "wait forever"; an unusable value falls back to
    *default*, so a typo can never turn the prompt into an instant "no".
    """
    raw = os.environ.get(APPROVAL_TIMEOUT_ENV, "")
    if not raw.strip():
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return 0.0 if value <= 0 else value


@dataclass(frozen=True)
class ApprovalRequest:
    """One risky command waiting for a decision."""

    id: str
    command: str
    risk_reason: str
    tool_name: str = "BashTool"


@dataclass(frozen=True)
class ApprovalDecision:
    """What the approver decided, and why."""

    approved: bool
    reason: str = ""


def new_approval_request(
    command: str, risk_reason: str, *, tool_name: str = "BashTool"
) -> ApprovalRequest:
    """Build a request with a short, unique id (``approval-1a2b3c4d``)."""
    return ApprovalRequest(
        id=f"approval-{uuid4().hex[:8]}",
        command=str(command or ""),
        risk_reason=str(risk_reason or ""),
        tool_name=str(tool_name or "BashTool"),
    )
