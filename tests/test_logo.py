"""Tests for the startup logo (``cli/tui/logo.py``) - rich only, no Textual needed."""

from __future__ import annotations

import io
import time
from typing import Any

import pytest
from rich.console import Console

from wqagent.cli.tui.logo import (
    LOGO_ASCII,
    LOGO_BOX,
    MAX_ANIMATION_SECONDS,
    NO_ANIMATION_ENV,
    NO_LOGO_ENV,
    RULE,
    RULE_ASCII,
    SUBTITLE,
    SUBTITLE_ASCII,
    TITLE,
    TITLE_ASCII,
    animate_logo,
    animation_enabled,
    frames_of,
    logo_enabled,
    logo_lines,
    logo_text,
    print_logo,
    subtitle_line,
    supports,
)


def _console(**kwargs: Any) -> tuple[Console, io.StringIO]:
    stream = io.StringIO()
    options: dict[str, Any] = {"file": stream, "width": 90, "force_terminal": False, "no_color": True}
    options.update(kwargs)
    return Console(**options), stream


# -- the lines ---------------------------------------------------------------
def test_logo_lines_match_the_banner_shape() -> None:
    lines = logo_lines(ascii_only=False)

    assert lines[0] == f"  {TITLE}"
    assert lines[1] == f"  {RULE}" and lines[3] == f"  {RULE}"
    assert lines[2] == f"   {SUBTITLE}"
    assert lines[4:] == [f"  {row}" for row in LOGO_BOX]


def test_logo_lines_ascii_variant_is_pure_ascii() -> None:
    lines = logo_lines(ascii_only=True)

    assert lines[0] == f"  {TITLE_ASCII}"
    assert lines[2] == f"   {SUBTITLE_ASCII}"
    assert lines[1] == f"  {RULE_ASCII}"
    assert lines[4:] == [f"  {row}" for row in LOGO_ASCII]
    blob = "\n".join(lines)
    assert blob.isascii(), "the ASCII variant must encode anywhere"


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
def test_print_logo_static_writes_the_banner() -> None:
    console, stream = _console()

    print_logo(console, animate=False)

    output = stream.getvalue()
    assert TITLE in output and SUBTITLE in output
    assert LOGO_BOX[0].rstrip() in output
    assert "\x1b[" not in output  # nothing terminal-specific in a plain render
    assert output.count(TITLE) == 1


def test_print_logo_default_is_static_on_a_non_terminal() -> None:
    console, stream = _console()

    print_logo(console)  # animate=None: a StringIO is not a terminal

    assert stream.getvalue().count(TITLE) == 1


def test_print_logo_respects_the_environment(monkeypatch: Any) -> None:
    console, stream = _console()
    monkeypatch.setenv(NO_LOGO_ENV, "1")

    print_logo(console, animate=False)

    assert stream.getvalue() == ""
    assert logo_enabled() is False

    monkeypatch.setenv(NO_LOGO_ENV, "0")
    assert logo_enabled() is True
    monkeypatch.setenv(NO_ANIMATION_ENV, "yes")
    assert animation_enabled() is False


def test_print_logo_can_be_forced_ascii() -> None:
    console, stream = _console()

    print_logo(console, animate=False, ascii_only=True)

    assert stream.getvalue().isascii()


def test_print_logo_animates_when_asked() -> None:
    """On a terminal the reveal paints several frames; it also stays quick."""
    stream = io.StringIO()
    console = Console(file=stream, width=90, force_terminal=True, no_color=True)

    started = time.monotonic()
    print_logo(console, animate=True, frame_seconds=0)
    elapsed = time.monotonic() - started

    output = stream.getvalue()
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
    console, stream = _console()
    print_logo(console, frame_seconds=MAX_ANIMATION_SECONDS)  # huge frames -> static

    assert stream.getvalue().count(TITLE) == 1
