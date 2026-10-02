"""Tests for the grep tool (built-in scanner; ripgrep is disabled by conftest)."""

from __future__ import annotations

from wqagent.core.state import RuntimeState
from wqagent.tools.grep_tool import GrepTool


def _seed(state: RuntimeState) -> None:
    src = state.workspace / "src"
    src.mkdir(exist_ok=True)
    (src / "app.py").write_text("def main():\n    return 1\n", encoding="utf-8")
    (src / "helper.py").write_text("def helper():\n    return 2\n", encoding="utf-8")
    (state.workspace / "notes.md").write_text("def main is documented\n", encoding="utf-8")


def test_finds_matches_across_the_tree(state: RuntimeState) -> None:
    _seed(state)
    output = GrepTool(state).run("def main", ".")
    assert "src/app.py:1: def main():" in output
    assert "notes.md:1: def main is documented" in output
    assert "helper.py" not in output
    assert output.startswith("2 matches")


def test_glob_filters_files(state: RuntimeState) -> None:
    _seed(state)
    output = GrepTool(state).run("def ", ".", glob="*.py")
    assert "src/app.py" in output
    assert "src/helper.py" in output
    assert "notes.md" not in output


def test_glob_with_a_directory_prefix(state: RuntimeState) -> None:
    _seed(state)
    output = GrepTool(state).run("def ", ".", glob="src/*.py")
    assert "src/app.py" in output
    assert "notes.md" not in output


def test_head_limit_truncates_and_is_reported(state: RuntimeState) -> None:
    (state.workspace / "many.txt").write_text(
        "".join(f"hit {i}\n" for i in range(20)), encoding="utf-8"
    )
    output = GrepTool(state).run("hit", ".", head_limit=5)
    assert output.count("many.txt:") == 5
    assert "stopped at head_limit=5" in output


def test_ignore_case(state: RuntimeState) -> None:
    (state.workspace / "a.txt").write_text("Hello World\n", encoding="utf-8")
    assert "a.txt" not in GrepTool(state).run("hello", ".")
    assert "a.txt:1: Hello World" in GrepTool(state).run("hello", ".", ignore_case=True)


def test_single_file_scope(state: RuntimeState) -> None:
    _seed(state)
    output = GrepTool(state).run("def ", "src/app.py")
    assert "src/app.py" in output
    assert "helper.py" not in output


def test_no_matches_and_invalid_regex(state: RuntimeState) -> None:
    _seed(state)
    assert GrepTool(state).run("does-not-exist", ".").startswith("No matches")
    assert GrepTool(state).run("([unclosed", ".").startswith("Error:")


def test_skips_binary_hidden_and_vendor_directories(state: RuntimeState) -> None:
    (state.workspace / "blob.bin").write_bytes(b"\x00needle")
    (state.workspace / ".hidden.txt").write_text("needle\n", encoding="utf-8")
    vendor = state.workspace / "node_modules"
    vendor.mkdir()
    (vendor / "dep.js").write_text("needle\n", encoding="utf-8")
    keep = state.workspace / "keep.txt"
    keep.write_text("needle\n", encoding="utf-8")

    output = GrepTool(state).run("needle", ".")
    assert "keep.txt" in output
    assert "blob.bin" not in output
    assert ".hidden.txt" not in output
    assert "node_modules" not in output
    assert "skipped 1 binary file" in output


def test_escape_is_rejected(state: RuntimeState) -> None:
    assert GrepTool(state).run("x", "../").startswith("Error:")
    assert GrepTool(state).run("x", "does-not-exist").startswith("Error:")
