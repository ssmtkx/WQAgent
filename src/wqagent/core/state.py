"""Runtime state shared by the agent loop and every tool.

``RuntimeState`` owns the workspace root: tools never accept a path without
running it through :meth:`RuntimeState.resolve`, so the workspace boundary is
enforced in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from wqagent.core.approval import (
    ApprovalDecision,
    ApprovalRequest,
    approval_mode_from_env,
    normalize_approval_mode,
)
from wqagent.core.checkpoint import checkpoint_mode_from_env, normalize_checkpoint_mode
from wqagent.core.paths import (
    is_within_workspace,
    relative_to_workspace,
    resolve_in_workspace,
    resolve_workspace,
)
from wqagent.core.trace import (
    new_trace_id,
    normalize_trace_mode,
    trace_mode_from_env,
)

__all__ = [
    "RuntimeState",
    "create_runtime",
    "truncate_text",
    "DEFAULT_MAX_OUTPUT_CHARS",
    "DEFAULT_MAX_SCAN_BYTES",
    "DEFAULT_READ_LINES",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_BLOCKED_COMMAND_PATTERNS",
]

DEFAULT_MAX_OUTPUT_CHARS = 20_000
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_READ_LINES = 2_000
DEFAULT_MAX_SCAN_BYTES = 5 * 1024 * 1024

#: Commands refused by :class:`~wqagent.tools.bash_tool.BashTool`. They are
#: destructive far beyond the workspace, which is the only thing this agent may
#: touch. Override per run with ``RuntimeState(blocked_command_patterns=(...))``.
DEFAULT_BLOCKED_COMMAND_PATTERNS: tuple[str, ...] = (
    # rm aimed at the filesystem root or the home directory
    r"\brm\s+(?:-[a-zA-Z]+\s+)*(?:/|~|\$HOME|\$\{HOME\})(?:\S*)?(?:\s|$)",
    # cmd.exe: del /f /s /q C:\
    r"\bdel\s+/[a-zA-Z]*[sq][a-zA-Z]*\s+[a-zA-Z]:\\(?:\s|$|\*)",
    # PowerShell: Remove-Item -Recurse -Force C:\
    r"\bremove-item\b(?=[^\n]*-recurse)(?=[^\n]*-force)[^\n]*[a-zA-Z]:\\(?:\s|$|\*)",
    r"\bformat\s+[a-zA-Z]:",
    r"\bmkfs(?:\.\w+)?\b",
    r"\bdiskpart\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bhalt\b",
    # fork bomb
    r":\(\)\s*\{[^\n]*\}\s*;\s*:",
)


def truncate_text(text: str, limit: int, *, label: str = "output") -> str:
    """Clip *text* to roughly *limit* characters, keeping head and tail."""
    if limit <= 0 or len(text) <= limit:
        return text
    omitted = len(text) - limit
    marker = f"\n... [{label} truncated: {omitted} of {len(text)} characters omitted] ...\n"
    if limit <= len(marker) + 200:
        return text[:limit] + marker
    head = int((limit - len(marker)) * 0.7)
    tail = limit - len(marker) - head
    return text[:head] + marker + text[-tail:]


@dataclass
class RuntimeState:
    """Everything a tool needs to know about the current run.

    Attributes:
        workspace: Root directory every tool is confined to. Created on demand.
        timeout_seconds: Default timeout for shell commands and subprocesses.
        max_output_chars: Tool output is clipped to this budget before it is
            handed back to the model.
        max_read_lines: Default page size of ``file_read``.
        max_scan_bytes: ``grep`` skips files larger than this.
        blocked_command_patterns: Regexes ``bash`` refuses to run.
        approval_mode: Policy for *risky* commands - ``inline`` (ask the handler, deny
            without one), ``auto`` (run, marked) or ``deny`` (refuse). ``None`` means
            "not specified": ``WQ_APPROVAL_MODE`` (else ``inline``) is used.
        approval_handler: Called with an :class:`~wqagent.core.approval.ApprovalRequest`
            in ``inline`` mode; the CLI passes an interactive prompt. ``None`` denies.
        checkpoint_mode: What to save for a resume - ``light`` (metadata, recovery
            guide, git snapshot), ``strict`` (plus the full state and every event) or
            ``off``. ``None`` means "not specified": ``WQ_CHECKPOINT_MODE`` (else
            ``light``) is used.
        trace_mode: What to record about the run - ``summary``/``on`` (counters +
            timeline), ``full`` (plus ``events.jsonl``) or ``off``. ``None`` means
            "not specified": ``WQ_TRACE_MODE`` (else ``summary``) is used.
        trace_id: Identifier of this run's trace directory; generated when empty.
        resume_from: Workspace a run was resumed from, empty for a fresh run.
        session_id: Conversation session this run belongs to (``core/session.py``);
            empty means "no session", which the memory layer treats as a default.
        session_turn: Turn number inside that session (0 = not recorded yet).
        started_at: UTC timestamp of the run (useful for logs).
    """

    workspace: Path
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS
    max_read_lines: int = DEFAULT_READ_LINES
    max_scan_bytes: int = DEFAULT_MAX_SCAN_BYTES
    blocked_command_patterns: tuple[str, ...] = DEFAULT_BLOCKED_COMMAND_PATTERNS
    approval_mode: str | None = None
    approval_handler: Callable[[ApprovalRequest], ApprovalDecision] | None = field(
        default=None, repr=False, compare=False
    )
    checkpoint_mode: str | None = None
    trace_mode: str | None = None
    trace_id: str = ""
    resume_from: str = ""
    session_id: str = ""
    session_turn: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        self.workspace = resolve_workspace(self.workspace, create=True)
        self.approval_mode = (
            approval_mode_from_env()
            if self.approval_mode is None
            else normalize_approval_mode(self.approval_mode)
        )
        self.checkpoint_mode = (
            checkpoint_mode_from_env()
            if self.checkpoint_mode is None
            else normalize_checkpoint_mode(self.checkpoint_mode)
        )
        self.trace_mode = (
            trace_mode_from_env() if self.trace_mode is None else normalize_trace_mode(self.trace_mode)
        )
        if not self.trace_id:
            self.trace_id = new_trace_id()
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_output_chars < 500:
            raise ValueError("max_output_chars must be at least 500")
        if self.max_read_lines < 1:
            raise ValueError("max_read_lines must be positive")
        if self.max_scan_bytes < 1:
            raise ValueError("max_scan_bytes must be positive")

    # -- paths ---------------------------------------------------------------
    def resolve(
        self,
        path: str | Path,
        *,
        must_exist: bool = False,
        kind: str | None = None,
    ) -> Path:
        """Resolve *path* inside :attr:`workspace` (see :func:`resolve_in_workspace`)."""
        return resolve_in_workspace(self.workspace, path, must_exist=must_exist, kind=kind)

    def contains(self, path: str | Path) -> bool:
        """True when *path* is the workspace itself or inside it."""
        return is_within_workspace(self.workspace, path)

    def relative(self, path: str | Path) -> str:
        """Display form of *path* (relative to the workspace when possible)."""
        return relative_to_workspace(self.workspace, path)

    # -- output --------------------------------------------------------------
    def clip(self, text: str, *, label: str = "output", limit: int | None = None) -> str:
        """Clip tool output to :attr:`max_output_chars` characters."""
        return truncate_text(text, self.max_output_chars if limit is None else limit, label=label)

    def summary(self) -> str:
        """One-line description of the run, used by the CLI banner."""
        return (
            f"workspace={self.workspace} timeout={self.timeout_seconds}s "
            f"max_output={self.max_output_chars} chars approval={self.approval_mode}"
        )


def create_runtime(
    workspace: str | Path,
    *,
    approval_mode: str | None = "inline",
    approval_handler: Callable[[ApprovalRequest], ApprovalDecision] | None = None,
    checkpoint_mode: str | None = "light",
    trace_mode: str | None = "on",
    resume_from: str | Path | None = None,
    session: Mapping[str, Any] | None = None,
    **options: Any,
) -> RuntimeState:
    """Build the :class:`RuntimeState` of one run.

    The mode arguments are the *defaults* of a run; pass ``None`` for any of them to
    fall back to its environment variable (``WQ_APPROVAL_MODE``,
    ``WQ_CHECKPOINT_MODE``, ``WQ_TRACE_MODE``).

    Args:
        workspace: Workspace root (created on demand).
        approval_mode: ``inline`` / ``auto`` / ``deny`` for risky shell commands.
        approval_handler: Consulted in ``inline`` mode; ``None`` denies.
        checkpoint_mode: ``light`` / ``strict`` / ``off``.
        trace_mode: ``summary`` (or its alias ``on``) / ``full`` / ``off``.
        resume_from: Workspace to resume; it becomes the run's workspace.
        session: A session record (``core/session.py``); its id and turn number are
            carried on the runtime so memory and reports can label the run.
        **options: Any other :class:`RuntimeState` field (``timeout_seconds``, ...).

    Returns:
        A :class:`RuntimeState` ready to hand to the tools and the workflow.
    """
    target = resume_from if resume_from is not None else workspace
    state = RuntimeState(
        workspace=target,
        approval_mode=approval_mode,
        approval_handler=approval_handler,
        checkpoint_mode=checkpoint_mode,
        trace_mode=trace_mode,
        **options,
    )
    if resume_from is not None:
        state.resume_from = str(state.workspace)
    if session:
        state.session_id = str(session.get("session_id") or "")
        state.session_turn = int(session.get("turn_index") or 0)
    return state
