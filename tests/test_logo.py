"""Tests for the startup logo (``cli/tui/logo.py``) - rich only, no Textual needed."""

from __future__ import annotations

import io
import time
from typing import Any

import pytest
from rich.console import Console

from wqagent.cli.tui.logo import (
    FRAME_SECONDS, LOGO_ASCII, LOGO_BOX, MAX_ANIMATION_SECONDS, NO_ANIMATION_ENV, NO_LOGO_ENV,
    RULE, RULE_ASCII, SUBTITLE, SUBTITLE_ASCII, TITLE, TITLE_ASCII, animate_logo,
    animation_enabled, frames_of, logo_enabled, logo_lines, logo_text, print_logo,
    subtitle_line, supports,
)


def _console(**kwargs: Any) -> tuple[Console, io.StringIO]:
    stream = io.StringIO()
    options: dict[str, Any] = {"file": stream, "width": 90, "force_terminal": False, "no_color": True}
    options.update(kwargs)
    return Console(**options), stream


def _printed(
    *,
    animate: bool | None = None,
    ascii_only: bool | None = None,
    frame_seconds: float = FRAME_SECONDS,
    **console_options: Any,
) -> str:
    """What ``print_logo`` writes to a fresh console (``animate=None`` = the auto rule)."""
    console, stream = _console(**console_options)
    print_logo(console, animate=animate, ascii_only=ascii_only, frame_seconds=frame_seconds)
    return stream.getvalue()


# -- the lines ---------------------------------------------------------------
@pytest.mark.parametrize(
    ("ascii_only", "title", "subtitle", "rule", "art", "pure_ascii"),
    [
        (False, TITLE, SUBTITLE, RULE, LOGO_BOX, False),
        (True, TITLE_ASCII, SUBTITLE_ASCII, RULE_ASCII, LOGO_ASCII, True),
    ],
)
def test_logo_lines_match_the_banner_shape(
    ascii_only: bool, title: str, subtitle: str, rule: str, art: tuple[str, ...], pure_ascii: bool
) -> None:
    lines = logo_lines(ascii_only=ascii_only)

    assert lines[0] == f"  {title}"
    assert lines[1] == f"  {rule}" and lines[3] == f"  {rule}"
    assert lines[2] == f"   {subtitle}"
    assert lines[4:] == [f"  {row}" for row in art]
    assert "\n".join(lines).isascii() is pure_ascii, "the ASCII variant must encode anywhere"


def test_logo_lines_picks_ascii_for_a_narrow_console() -> None:
    gbk = io.TextIOWrapper(io.BytesIO(), encoding="gbk")
    console = Console(file=gbk, width=90, force_terminal=False)

    lines = logo_lines(console=console)

    assert lines[0] == f"  {TITLE_ASCII}"
    assert "\n".join(lines).isascii()


def test_supports_and_subtitle_line() -> None:
    utf8 = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    gbk = io.TextIOWrapper(io.BytesIO(), encoding="gbk")

    assert supports(TITLE, Console(file=utf8)) is True
    assert supports(TITLE, Console(file=gbk)) is False
    assert subtitle_line(ascii_only=True) == SUBTITLE_ASCII
    assert subtitle_line() == SUBTITLE


# -- the renderable ----------------------------------------------------------
def test_logo_text_plain_content() -> None:
    text = logo_text(ascii_only=False)

    assert TITLE in text.plain
    assert SUBTITLE in text.plain
    assert RULE in text.plain
    assert LOGO_BOX[0].strip()[:4] in text.plain
    assert len(text.spans) > 5, "the logo should be coloured"
    assert any("bold" in str(span.style) for span in text.spans)


def test_logo_text_sweep_dims_what_is_not_revealed_yet() -> None:
    revealed = logo_text(ascii_only=True, sweep=None)
    partial = logo_text(ascii_only=True, sweep=4)

    assert partial.plain == revealed.plain  # same shape, different styling
    styles = {str(span.style) for span in partial.spans}
    assert any("dim" in style for style in styles)
    full_styles = {str(span.style) for span in revealed.spans}
    assert not any("dim" in style for style in full_styles)
    # one column is highlighted while the sweep passes over it
    assert any("on #1e293b" in str(span.style) for span in partial.spans)


def test_frames_cover_the_whole_wordmark() -> None:
    console, _ = _console()

    frames = frames_of(console, ascii_only=False)

    assert len(frames) == max(len(row) for row in LOGO_BOX) + 1
    assert all(frame is not None for frame in frames)  # every frame renders


# -- printing ----------------------------------------------------------------
@pytest.mark.parametrize("animate", [False, None])  # animate=None: a StringIO is no terminal
def test_print_logo_static_renders_the_banner(animate: bool | None) -> None:
    output = _printed(animate=animate)

    assert TITLE in output and SUBTITLE in output
    assert LOGO_BOX[0].rstrip() in output
    assert "\x1b[" not in output  # nothing terminal-specific in a plain render
    assert output.count(TITLE) == 1


@pytest.mark.parametrize(
    ("name", "value", "check", "expected"),
    [
        (NO_LOGO_ENV, "1", logo_enabled, False),
        (NO_LOGO_ENV, "0", logo_enabled, True),
        (NO_ANIMATION_ENV, "yes", animation_enabled, False),
    ],
)
def test_environment_switches(
    name: str, value: str, check: Any, expected: bool, monkeypatch: Any
) -> None:
    """``WQ_NO_LOGO`` / ``WQ_NO_ANIMATION`` gate the banner and the reveal."""
    monkeypatch.setenv(name, value)

    assert check() is expected


def test_print_logo_is_silent_when_disabled(monkeypatch: Any) -> None:
    monkeypatch.setenv(NO_LOGO_ENV, "1")

    assert _printed(animate=False) == ""


def test_print_logo_can_be_forced_ascii() -> None:
    assert _printed(animate=False, ascii_only=True).isascii()


def test_print_logo_animates_when_asked() -> None:
    """On a terminal the reveal paints several frames; it also stays quick."""
    started = time.monotonic()
    output = _printed(force_terminal=True, animate=True, frame_seconds=0)
    elapsed = time.monotonic() - started

    assert TITLE in output and SUBTITLE in output
    assert "\x1b[" in output, "a terminal gets cursor control from Live"
    assert elapsed < MAX_ANIMATION_SECONDS, f"the animation must stay quick ({elapsed:.2f}s)"


def test_animate_logo_without_a_terminal_still_prints_the_banner() -> None:
    console, stream = _console()

    animate_logo(console, frame_seconds=0)

    assert TITLE in stream.getvalue()


def test_animate_logo_never_raises_on_a_broken_console() -> None:
    class Exploding(Console):
        def print(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("no console here")

    console = Exploding(file=io.StringIO(), force_terminal=False)

    animate_logo(console, ascii_only=True, frame_seconds=0)  # must not raise


def test_animation_budget_is_frames_times_delay() -> None:
    # the auto rule uses this budget, so a big delay must simply disable the animation
    assert _printed(frame_seconds=MAX_ANIMATION_SECONDS).count(TITLE) == 1
