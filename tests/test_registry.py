"""Tests for the tool registry: the StructuredTool list handed to bind_tools()."""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from wqagent.core.state import RuntimeState
from wqagent.tools import build_tool_instances, build_tools, describe_tools, tool_names

EXPECTED_TOOLS = ["file_read", "file_write", "file_edit", "grep", "bash"]


def test_build_tools_returns_structured_tools(state: RuntimeState) -> None:
    tools = build_tools(state)
    assert [tool.name for tool in tools] == EXPECTED_TOOLS
    assert all(isinstance(tool, StructuredTool) for tool in tools)
    assert all(tool.description for tool in tools)


def test_argument_schemas_are_exposed(state: RuntimeState) -> None:
    tools = {tool.name: tool for tool in build_tools(state)}
    assert set(tools["file_read"].args) == {"file_path", "offset", "limit"}
    assert set(tools["file_edit"].args) == {"file_path", "old_text", "new_text"}
    assert set(tools["grep"].args) == {
        "pattern",
        "path",
        "glob",
        "head_limit",
        "ignore_case",
    }
    assert set(tools["bash"].args) == {"command", "timeout_seconds"}


def test_tools_invoke_against_the_workspace(state: RuntimeState) -> None:
    tools = {tool.name: tool for tool in build_tools(state)}
    assert tools["file_write"].invoke(
        {"file_path": "hello.txt", "content": "hi\n"}
    ).startswith("Created")
    assert "hi" in tools["file_read"].invoke({"file_path": "hello.txt"})
    assert "hello.txt" in tools["grep"].invoke({"pattern": "hi"})
    assert "exit_code: 0" in tools["bash"].invoke({"command": "echo ok"})


def test_tools_enforce_the_workspace_boundary(state: RuntimeState) -> None:
    tools = {tool.name: tool for tool in build_tools(state)}
    assert tools["file_read"].invoke({"file_path": "../escape.txt"}).startswith("Error:")
    assert tools["file_write"].invoke(
        {"file_path": "../escape.txt", "content": "x"}
    ).startswith("Error:")


def test_each_state_gets_its_own_tool_instances(state: RuntimeState) -> None:
    first = build_tool_instances(state)
    second = build_tool_instances(state)
    assert first is not second
    assert all(instance.state is state for instance in first)
    assert [instance.name for instance in first] == EXPECTED_TOOLS
    assert tool_names() == EXPECTED_TOOLS
    assert [info["name"] for info in describe_tools()] == EXPECTED_TOOLS
