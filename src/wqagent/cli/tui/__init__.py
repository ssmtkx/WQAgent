"""The Textual TUI (optional: ``pip install "wqagent[tui]"``).

Importing this package never imports Textual, so the plain CLI keeps working when the
extra is not installed; :func:`run_tui` reports exactly that instead of crashing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["TEXTUAL_MISSING_HINT", "print_logo", "run_tui", "textual_available"]

TEXTUAL_MISSING_HINT = (
    "the TUI needs the optional 'textual' dependency: "
    'pip install "wqagent[tui]"   (or: pip install textual)'
)


def print_logo(console: Any = None, **options: Any) -> None:
    """Draw the startup logo (``logo.py`` itself needs no Textual, only rich)."""
    from wqagent.cli.tui.logo import print_logo as _print_logo

    _print_logo(console, **options)


def textual_available() -> bool:
    """True when Textual can be imported."""
    try:
        import textual  # noqa: F401
    except ImportError:
        return False
    return True


def run_tui(
    *,
    workspace: str | Path | None = None,
    model: Any | None = None,
    **options: Any,
) -> int:
    """Launch the TUI and block until the user quits; returns the exit code.

    Args:
        workspace: Workspace the session and the tools live in.
        model: Chat model handed to every node.
        **options: Forwarded to :class:`~wqagent.cli.tui.app.WQAgentTuiApp`
            (``approval_mode``, ``checkpoint_mode``, ``trace_mode``, ``runner``, ...).
    """
    try:
        from wqagent.cli.tui.app import WQAgentTuiApp
    except ImportError as exc:  # textual is an optional extra
        from rich.console import Console

        Console(stderr=True).print(
            f"[bold red]error:[/] {TEXTUAL_MISSING_HINT}\n[dim]({exc})[/]"
        )
        return 2

    # the logo is drawn before the UI takes over the screen (rich; animated on a TTY)
    from wqagent.cli.tui.logo import print_logo as _print_logo

    _print_logo()
    WQAgentTuiApp(workspace=workspace, model=model, **options).run()
    return 0
