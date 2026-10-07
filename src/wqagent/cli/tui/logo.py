"""The startup logo: an ASCII wordmark, rendered with rich, with a short reveal.

    🐺 WQ Agent
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     Stage 6 · MultiAgent + Context/Harness
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Everything degrades gracefully, because this runs before the UI exists:

* a console that cannot encode the box-drawing art (or the 🐺) gets the pure-ASCII
  variant (:func:`logo_lines` picks it, or pass ``ascii_only=True``);
* the animation is skipped when the output is not a terminal, when ``WQ_NO_ANIMATION``
  is set, or when more than ``MAX_ANIMATION_SECONDS`` would be spent on it;
* ``WQ_NO_LOGO=1`` prints nothing at all.

Show it yourself with::

    from rich.console import Console
    from wqagent.cli.tui.logo import print_logo

    print_logo(Console())                    # animated when it is a terminal
    print_logo(Console(), animate=False)     # one static block
    print_logo(Console(), ascii_only=True)   # pure ASCII
"""

from __future__ import annotations

import os
import time
from typing import Any, Sequence

from rich.console import Console, Group, RenderableType
from rich.text import Text

__all__ = [
    "LOGO_ASCII",
    "LOGO_BOX",
    "NO_ANIMATION_ENV",
    "NO_LOGO_ENV",
    "SUBTITLE",
    "TITLE",
    "TITLE_ASCII",
    "animate_logo",
    "logo_enabled",
    "logo_lines",
    "logo_text",
    "print_logo",
]

#: Box-drawing wordmark ("WQ").
LOGO_BOX: tuple[str, ...] = (
    "██╗    ██╗ ██████╗ ",
    "██║    ██║██╔═══██╗",
    "██║ █╗ ██║██║   ██║",
    "██║███╗██║██║▄▄ ██║",
    "╚███╔███╔╝╚██████╔╝",
    " ╚══╝╚══╝  ╚══▀▀═╝ ",
)
#: Pure-ASCII wordmark, for consoles that cannot encode the above.
LOGO_ASCII: tuple[str, ...] = (
    "##   ##  #### ",
    "##   ## ##  ##",
    "## # ## ##  ##",
    "### ### ##  ##",
    " ## ##  ##  ##",
    "  #   #  #### ",
)
TITLE = "🐺 WQ Agent"
TITLE_ASCII = "WQ Agent"
SUBTITLE = "Stage 6 · MultiAgent + Context/Harness"
SUBTITLE_ASCII = "Stage 6 - MultiAgent + Context/Harness"
#: Width of the horizontal rules, matching the subtitle line.
RULE_WIDTH = 38
RULE = "━" * RULE_WIDTH
RULE_ASCII = "-" * RULE_WIDTH

#: Set this to ``1``/``true`` to skip the reveal animation.
NO_ANIMATION_ENV = "WQ_NO_ANIMATION"
#: Set this to ``1``/``true`` to print no logo at all.
NO_LOGO_ENV = "WQ_NO_LOGO"
#: Seconds per frame of the reveal.
FRAME_SECONDS = 0.045
#: Never spend more than this on the animation.
MAX_ANIMATION_SECONDS = 1.5

#: Gradient used for the wordmark (left to right).
GRADIENT = ("#22d3ee", "#38bdf8", "#818cf8", "#c084fc", "#f472b6")
#: Colour of a character the reveal sweep is passing over.
SWEEP_STYLE = "bold white on #1e293b"


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def logo_enabled() -> bool:
    """False when ``WQ_NO_LOGO`` asks for no logo at all."""
    return not _truthy(os.environ.get(NO_LOGO_ENV))


def animation_enabled() -> bool:
    """False when ``WQ_NO_ANIMATION`` asks for the static logo."""
    return not _truthy(os.environ.get(NO_ANIMATION_ENV))


def supports(text: str, console: Console | None = None) -> bool:
    """True when *console* can write *text* as it is."""
    stream = (console or Console()).file
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError, TypeError):
        return False
    return True


def _ascii_only(console: Console | None, ascii_only: bool | None) -> bool:
    if ascii_only is not None:
        return ascii_only
    return not (supports(TITLE, console) and supports(LOGO_BOX[0], console))


def logo_lines(*, ascii_only: bool | None = None, console: Console | None = None) -> list[str]:
    """The logo as plain lines (the shape of the banner in the README)."""
    plain = _ascii_only(console, ascii_only)
    title = TITLE_ASCII if plain else TITLE
    subtitle = SUBTITLE_ASCII if plain else SUBTITLE
    rule = RULE_ASCII if plain else RULE
    art = LOGO_ASCII if plain else LOGO_BOX
    return [f"  {title}", f"  {rule}", f"   {subtitle}", f"  {rule}", *(f"  {row}" for row in art)]


def logo_text(
    *, ascii_only: bool | None = None, console: Console | None = None, sweep: int | None = None
) -> Text:
    """The logo as a coloured :class:`~rich.text.Text`.

    Args:
        ascii_only: Force the pure-ASCII variant (default: detect from the console).
        console: Console whose encoding decides the variant.
        sweep: Column the reveal sweep is passing over (``None`` = finished), which
            paints that column in a highlight style.
    """
    plain = _ascii_only(console, ascii_only)
    art = LOGO_ASCII if plain else LOGO_BOX
    text = Text()
    text.append("  ")
    text.append(TITLE_ASCII if plain else TITLE, style="bold #facc15")
    text.append("\n  ")
    text.append(RULE_ASCII if plain else RULE, style="bold #38bdf8")
    text.append("\n   ")
    text.append(SUBTITLE_ASCII if plain else SUBTITLE, style="italic #94a3b8")
    text.append("\n  ")
    text.append(RULE_ASCII if plain else RULE, style="bold #38bdf8")
    text.append("\n")
    for row in art:
        text.append("  ")
        for column, char in enumerate(row):
            if sweep is not None and column >= sweep:
                text.append(char, style="dim #475569")  # not revealed yet
                continue
            colour = GRADIENT[min(len(GRADIENT) - 1, column * len(GRADIENT) // max(1, len(row)))]
            style = SWEEP_STYLE if sweep is not None and column == sweep - 1 else colour
            text.append(char, style=f"bold {style}" if style != SWEEP_STYLE else style)
        text.append("\n")
    return text


def render_logo(
    *, ascii_only: bool | None = None, console: Console | None = None, sweep: int | None = None
) -> RenderableType:
    """A renderable version of the logo (handy inside ``Live``/panels)."""
    return Group(logo_text(ascii_only=ascii_only, console=console, sweep=sweep))


def animate_logo(
    console: Console,
    *,
    ascii_only: bool | None = None,
    frame_seconds: float = FRAME_SECONDS,
) -> None:
    """Reveal the logo line by line with a colour sweep; never raises."""
    from rich.live import Live

    plain = _ascii_only(console, ascii_only)
    width = max(len(row) for row in (LOGO_ASCII if plain else LOGO_BOX))
    frames = width + 2  # the sweep, then the finished logo
    try:
        with Live(
            render_logo(ascii_only=plain, console=console, sweep=0),
            console=console,
            refresh_per_second=max(10, int(1 / max(frame_seconds, 0.01))),
            transient=False,
        ) as live:
            for frame in range(1, frames + 1):
                live.update(
                    render_logo(
                        ascii_only=plain,
                        console=console,
                        sweep=frame if frame < frames else None,
                    )
                )
                if frame < frames and frame_seconds > 0:
                    time.sleep(frame_seconds)
    except Exception:  # noqa: BLE001 - a logo must never break startup
        # the animated path failed (no terminal control, a broken stream, ...): fall back
        # to one static block, and swallow even a failure *there*
        try:
            print_logo(console, ascii_only=plain, animate=False)
        except Exception:  # noqa: BLE001 - nothing about a logo may stop the run
            return


def print_logo(
    console: Console | None = None,
    *,
    animate: bool | None = None,
    ascii_only: bool | None = None,
    frame_seconds: float = FRAME_SECONDS,
) -> None:
    """Print the startup logo; animated only when that makes sense.

    Args:
        console: Console to draw on (a real terminal gets the animation).
        animate: ``True``/``False`` to force it; ``None`` animates only when the console
            is a terminal, ``WQ_NO_ANIMATION`` is unset, and the reveal is short enough.
        ascii_only: Force the pure-ASCII variant.
        frame_seconds: Seconds per animation frame.
    """
    if not logo_enabled():
        return
    target = console or Console()
    if animate is None:
        budget = frame_seconds * (max(len(row) for row in LOGO_BOX) + 2)
        animate = bool(
            getattr(target, "is_terminal", False)
            and animation_enabled()
            and budget <= MAX_ANIMATION_SECONDS
        )
    if animate:
        animate_logo(target, ascii_only=ascii_only, frame_seconds=frame_seconds)
        return
    target.print(logo_text(ascii_only=ascii_only, console=target))


def subtitle_line(*, ascii_only: bool | None = None) -> str:
    """Just the ``Stage 6 · …`` line (used by the TUI's header block)."""
    return SUBTITLE_ASCII if ascii_only else SUBTITLE


def frames_of(console: Console, *, ascii_only: bool | None = None) -> Sequence[RenderableType]:
    """Every animation frame, without printing anything (useful for tests/previews)."""
    plain = _ascii_only(console, ascii_only)
    width = max(len(row) for row in (LOGO_ASCII if plain else LOGO_BOX))
    return [render_logo(ascii_only=plain, console=console, sweep=frame) for frame in range(width + 1)]
