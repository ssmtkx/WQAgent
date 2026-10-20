"""Tests for the diff captured around file-changing tool calls (``core/diff.py``)."""

from __future__ import annotations

from pathlib import Path

from wqagent.core.diff import capture, finish, unified_diff
from wqagent.core.state import RuntimeState
from wqagent.tools.file_tools import FileEditTool, FileWriteTool


def test_capture_ignores_tools_that_do_not_touch_files(state: RuntimeState) -> None:
    assert capture(FileWriteTool(state), {"file_path": "a.py"}) is not None
    assert capture(object(), {"file_path": "a.py"}) is None
    assert capture(FileWriteTool(state), {}) is None  # no path to look at


def test_capture_sees_through_the_langchain_wrapper(state: RuntimeState) -> None:
    """``build_tools()`` returns StructuredTools; their state hides in ``.func``."""
    wrapped = FileWriteTool(state).as_tool()

    snapshot = capture(wrapped, {"file_path": "a.py"})

    assert snapshot is not None and snapshot.display == "a.py"


def test_an_edit_produces_one_added_and_one_removed_line(state: RuntimeState) -> None:
    path = state.workspace / "app.py"
    path.write_text("one\ntwo\nthree\n", encoding="utf-8")
    tool = FileEditTool(state)

    snapshot = capture(tool, {"file_path": "app.py"})
    tool.run(file_path="app.py", old_text="two", new_text="TWO")
    change = finish(snapshot)

    assert change is not None
    assert change.path == "app.py"
    assert (change.added, change.removed) == (1, 1)
    assert "-two" in change.text and "+TWO" in change.text
    assert change.truncated is False


def test_nothing_changed_means_no_diff(state: RuntimeState) -> None:
    path = state.workspace / "app.py"
    path.write_text("one\n", encoding="utf-8")

    snapshot = capture(FileWriteTool(state), {"file_path": "app.py"})

    assert finish(snapshot) is None  # nobody wrote anything
    assert finish(None) is None


def test_a_new_file_diffs_from_dev_null(state: RuntimeState) -> None:
    tool = FileWriteTool(state)

    snapshot = capture(tool, {"file_path": "app/new.py"})
    tool.run(file_path="app/new.py", content="a\nb\n")
    change = finish(snapshot)

    assert change is not None
    assert (change.added, change.removed) == (2, 0)
    assert "/dev/null" in change.text


def test_binary_files_are_not_diffed(state: RuntimeState) -> None:
    (state.workspace / "blob.bin").write_bytes(b"\x00\x01\x02")

    snapshot = capture(FileWriteTool(state), {"file_path": "blob.bin"})

    assert snapshot is not None
    assert snapshot.text is None  # present, but not text: no before/after diff


def test_unified_diff_truncates_a_huge_patch() -> None:
    before = "\n".join(f"old line {number}" for number in range(2000))
    after = "\n".join(f"new line {number}" for number in range(2000))

    change = unified_diff(before, after, "big.txt")

    assert change is not None
    assert change.truncated is True
    assert "more diff lines" in change.text
    assert change.as_dict()["truncated"] is True


def test_unified_diff_of_identical_text_is_none() -> None:
    assert unified_diff("same", "same", "x.py") is None


# -- the loop that feeds the UI ----------------------------------------------
def test_react_loop_attaches_the_diff_to_the_event_but_not_to_the_model(
    state: RuntimeState,
) -> None:
    from helpers import ScriptedModel, tool_call
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from wqagent.core.agent import react_loop
    from wqagent.tools import build_tools

    (state.workspace / "app.py").write_text("one\ntwo\n", encoding="utf-8")
    model = ScriptedModel(
        [
            tool_call(
                "file_edit",
                {"file_path": "app.py", "old_text": "two", "new_text": "TWO"},
            ),
            AIMessage(content="done"),
        ]
    )

    events = list(
        react_loop([HumanMessage(content="edit it")], model=model, tools=build_tools(state))
    )

    results = [event for event in events if event["type"] == "tool_result"]
    assert results, "the loop never reported the tool result"
    change = results[0]["diff"]
    assert change is not None
    assert change["path"] == "app.py"
    assert "+TWO" in change["text"]

    # the patch is for the UI only: the model gets the tool's own summary
    tool_message = next(m for m in model.seen[-1] if isinstance(m, ToolMessage))
    assert '"diff"' not in tool_message.content
    assert "TWO" not in tool_message.content.split("Edited")[0]
