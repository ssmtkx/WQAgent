"""The line a streamed reply is written into.

A plain wrapped ``Static`` puts continuation lines back at column 0, where they
read like new events. :class:`LiveReply` keeps the ``»  chat :`` prefix in a
gutter column so the whole reply stays visually one block::

    »  chat : FastAPI 的最小骨架现在包含三个文件：
              `app/main.py`（路由）、`app/auth.py`（依赖）
              和 `tests/test_api.py`。
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static

__all__ = ["LiveReply"]

#: Width of the prefix column, i.e. how far the text is indented.
GUTTER_WIDTH = 10


class LiveReply(Horizontal):
    """A growing reply: the prefix in a gutter, the text wrapping beside it."""

    DEFAULT_CSS = f"""
    LiveReply {{
        height: auto;
        margin: 0;
    }}
    LiveReply > #live-gutter {{
        width: {GUTTER_WIDTH};
        height: 1;
        color: #64748b;
    }}
    LiveReply > #live-body {{
        width: 1fr;
        height: auto;
        color: #1f2933;
    }}
    """

    def compose(self) -> ComposeResult:
        yield Static("»  chat :", id="live-gutter")
        yield Static("", id="live-body")

    @property
    def body(self) -> Static:
        return self.query_one("#live-body", Static)

    def update_text(self, text: str) -> None:
        """Show *text* (the reply so far); wrapping happens in the text column."""
        self.body.update(Text(text))
