"""The diff block that sits in the activity stream for a file-changing tool call.

A ``file_write`` / ``file_edit`` produces one line plus a patch. The patch is
collapsed by default -- the log stays scannable -- and ``Ctrl+D`` opens the most
recent block::

    ✓ file_write : app/main.py   +12 −3   ▸
    │ a/app/main.py
    │ b/app/main.py
    │ @@ -10,3 +10,5 @@
    │ +import auth
"""

from __future__ import annotations

from typing import Any, Mapping

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

__all__ = ["DiffBlock", "render_patch"]

#: Patch lines shown before the block summarises the rest.
MAX_PATCH_LINES = 240

_ADD = "#166534 on #e8f7ee"
_DEL = "#991b1b on #fdeaea"
_HUNK = "bold #0369a1"
_FILE = "#94a3b8"
_CONTEXT = "#64748b"


def render_patch(change: Mapping[str, Any]) -> Text:
    """Colour a unified diff: green additions, red deletions, blue hunk headers."""
    body = str(change.get("text") or "")
    lines = body.splitlines()
    text = Text()
    for index, line in enumerate(lines[:MAX_PATCH_LINES]):
        if index:
            text.append("\n")
        if line.startswith(("+++", "---")):
            text.append(line, style=_FILE)
        elif line.startswith("@@"):
            text.append(line, style=_HUNK)
        elif line.startswith("+"):
            text.append(line, style=_ADD)
        elif line.startswith("-"):
            text.append(line, style=_DEL)
        else:
            text.append(line, style=_CONTEXT)
    if len(lines) > MAX_PATCH_LINES:
        text.append(f"\n… ({len(lines) - MAX_PATCH_LINES} more lines)")
    return text


class DiffBlock(Vertical):
    """One file's change: a summary line, and the patch behind ``Ctrl+D``."""

    DEFAULT_CSS = """
    DiffBlock {
        height: auto;
        margin: 0;
    }
    DiffBlock #diff-header { height: 1; }
    DiffBlock #diff-body {
        height: auto;
        padding: 0 0 0 2;
    }
    """

    def __init__(
        self,
        change: Mapping[str, Any],
        *,
        header: str = "",
        css: str = "event-ok",
    ) -> None:
        super().__init__()
        self.change = dict(change)
        self.header_text = header
        self.css_class = css
        self.expanded = False

    @property
    def added(self) -> int:
        return int(self.change.get("added") or 0)

    @property
    def removed(self) -> int:
        return int(self.change.get("removed") or 0)

    def compose(self) -> ComposeResult:
        yield Static(id="diff-header", classes=self.css_class)
        yield Static(render_patch(self.change), id="diff-body")

    def on_mount(self) -> None:
        self.query_one("#diff-body", Static).display = False
        self.render_header()

    # -- rendering -----------------------------------------------------------
    def render_header(self) -> None:
        text = Text()
        text.append(self.header_text or str(self.change.get("path") or "diff"))
        text.append(f"   +{self.added}", style="bold #15803D")
        text.append(f" −{self.removed}", style="bold #B91C1C")
        if self.change.get("truncated"):
            text.append("  (truncated)", style="#a16207")
        text.append("   ▾" if self.expanded else "   ▸", style="#9aa0a6")
        self.query_one("#diff-header", Static).update(text)

    # -- interaction ---------------------------------------------------------
    def toggle(self) -> bool:
        """Open or close the patch; returns the new state."""
        self.expanded = not self.expanded
        self.query_one("#diff-body", Static).display = self.expanded
        self.render_header()
        return self.expanded
