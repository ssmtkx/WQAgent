"""Tests for the file tools (read, write, edit)."""

from __future__ import annotations

from pathlib import Path

from wqagent.core.state import RuntimeState
from wqagent.tools.file_tools import FileEditTool, FileReadTool, FileWriteTool


def test_write_creates_file_and_parents(state: RuntimeState) -> None:
    result = FileWriteTool(state).run("src/pkg/mod.py", "print('hi')\n")
    assert result.startswith("Created src/pkg/mod.py")
    assert (state.workspace / "src" / "pkg" / "mod.py").read_text(encoding="utf-8") == "print('hi')\n"


def test_write_overwrites_existing_file(state: RuntimeState) -> None:
    tool = FileWriteTool(state)
    tool.run("notes.txt", "old\n")
    result = tool.run("notes.txt", "new\n")
    assert result.startswith("Overwrote notes.txt")
    assert (state.workspace / "notes.txt").read_text(encoding="utf-8") == "new\n"


def test_write_refuses_to_escape_the_workspace(state: RuntimeState) -> None:
    result = FileWriteTool(state).run("../outside.txt", "nope")
    assert result.startswith("Error:")
    assert not (state.workspace.parent / "outside.txt").exists()


def test_write_refuses_to_replace_a_directory(state: RuntimeState) -> None:
    (state.workspace / "pkg").mkdir()
    assert FileWriteTool(state).run("pkg", "x").startswith("Error:")


def test_read_returns_numbered_lines(state: RuntimeState) -> None:
    (state.workspace / "a.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    output = FileReadTool(state).run("a.txt")
    assert "a.txt: showing lines 1-3" in output
    assert "1\tone" in output
    assert "3\tthree" in output


def test_read_honours_offset_and_limit(state: RuntimeState) -> None:
    (state.workspace / "a.txt").write_text(
        "".join(f"line{i}\n" for i in range(1, 11)), encoding="utf-8"
    )
    output = FileReadTool(state).run("a.txt", offset=3, limit=2)
    assert "showing lines 3-4" in output
    assert "line3" in output and "line4" in output
    assert "line5" not in output
    assert "continue with offset=5" in output


def test_read_past_end_of_file_reports_total(state: RuntimeState) -> None:
    (state.workspace / "a.txt").write_text("one\ntwo\n", encoding="utf-8")
    output = FileReadTool(state).run("a.txt", offset=99)
    assert output.startswith("Error:")
    assert "2 lines" in output


def test_read_missing_file_and_binary_file(state: RuntimeState) -> None:
    assert FileReadTool(state).run("missing.txt").startswith("Error:")
    (state.workspace / "blob.bin").write_bytes(b"\x00\x01\x02binary")
    assert "binary file" in FileReadTool(state).run("blob.bin")


def test_read_escape_is_rejected(state: RuntimeState) -> None:
    assert FileReadTool(state).run("../../etc/passwd").startswith("Error:")


def test_edit_replaces_a_unique_match(state: RuntimeState) -> None:
    (state.workspace / "app.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    result = FileEditTool(state).run("app.py", "x = 1", "x = 42")
    assert result.startswith("Edited app.py")
    assert "line 1" in result
    assert (state.workspace / "app.py").read_text(encoding="utf-8") == "x = 42\ny = 2\n"


def test_edit_reports_missing_match(state: RuntimeState) -> None:
    (state.workspace / "app.py").write_text("x = 1\n", encoding="utf-8")
    result = FileEditTool(state).run("app.py", "nope", "y")
    assert result.startswith("Error:")
    assert "not found" in result
    assert (state.workspace / "app.py").read_text(encoding="utf-8") == "x = 1\n"


def test_edit_refuses_ambiguous_match(state: RuntimeState) -> None:
    (state.workspace / "app.py").write_text("a = 1\nb = 2\na = 1\n", encoding="utf-8")
    result = FileEditTool(state).run("app.py", "a = 1", "a = 9")
    assert result.startswith("Error:")
    assert "matched 2 times" in result
    assert "lines 1, 3" in result
    assert (state.workspace / "app.py").read_text(encoding="utf-8").count("a = 1") == 2


def _read_raw(path: Path) -> str:
    """Read a file without newline translation (Path.read_text gained it in 3.13)."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def test_edit_adapts_lf_snippets_for_crlf_files(state: RuntimeState) -> None:
    target = state.workspace / "win.txt"
    target.write_text("one\r\ntwo\r\n", encoding="utf-8", newline="")
    result = FileEditTool(state).run("win.txt", "one\ntwo", "ONE\nTWO")
    assert result.startswith("Edited win.txt")
    assert "CRLF" in result
    assert _read_raw(target) == "ONE\r\nTWO\r\n"


def test_edit_keeps_crlf_endings_for_single_line_matches(state: RuntimeState) -> None:
    target = state.workspace / "win.txt"
    target.write_text("one\r\ntwo\r\n", encoding="utf-8", newline="")
    result = FileEditTool(state).run("win.txt", "two", "TWO")
    assert result.startswith("Edited win.txt")
    assert _read_raw(target) == "one\r\nTWO\r\n"


def test_edit_rejects_empty_or_identical_snippets(state: RuntimeState) -> None:
    (state.workspace / "app.py").write_text("x = 1\n", encoding="utf-8")
    tool = FileEditTool(state)
    assert tool.run("app.py", "", "y").startswith("Error:")
    assert tool.run("app.py", "x = 1", "x = 1").startswith("Error:")


def test_edit_escape_is_rejected(state: RuntimeState) -> None:
    assert FileEditTool(state).run("../../etc/hosts", "a", "b").startswith("Error:")
