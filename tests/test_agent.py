"""Tests for the agent loop, driven by a scripted fake model (no API calls)."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from helpers import ScriptedModel, tool_call
from wqagent.core.agent import Agent
from wqagent.core.state import RuntimeState
from wqagent.tools import build_tools


class BoomTool:
    """A tool that always raises, to test error feedback."""

    name = "boom"

    def invoke(self, args: dict[str, Any]) -> str:  # noqa: ARG002
        raise RuntimeError("kaboom")


def test_agent_binds_tools_and_executes_them(state: RuntimeState) -> None:
    model = ScriptedModel(
        [
            tool_call("file_write", {"file_path": "hello.py", "content": "print('hi')\n"}),
            AIMessage(content="Created hello.py."),
        ]
    )
    agent = Agent(model, build_tools(state), state)

    result = agent.run("create hello.py")

    assert result.completed
    assert result.text == "Created hello.py."
    assert result.steps == 2
    assert [record.name for record in result.tool_calls] == ["file_write"]
    assert (state.workspace / "hello.py").read_text(encoding="utf-8") == "print('hi')\n"
    assert model.bound[0][0] == "file_read"
    # the tool result was fed back to the model on the second round
    assert any(isinstance(message, ToolMessage) for message in model.seen[1])


def test_agent_reports_unknown_tools_without_crashing(state: RuntimeState) -> None:
    model = ScriptedModel(
        [
            tool_call("nope", {"a": 1}),
            AIMessage(content="done"),
        ]
    )
    result = Agent(model, build_tools(state), state).run("do something")
    assert result.tool_calls[0].output.startswith("Error: unknown tool 'nope'")
    assert result.completed


def test_agent_turns_tool_exceptions_into_feedback(state: RuntimeState) -> None:
    model = ScriptedModel([tool_call("boom", {}), AIMessage(content="recovered")])
    result = Agent(model, [BoomTool()], state).run("explode")
    assert result.tool_calls[0].output == "Error: RuntimeError: kaboom"
    assert result.text == "recovered"


def test_agent_stops_at_the_step_limit(state: RuntimeState) -> None:
    model = ScriptedModel([tool_call("file_read", {"file_path": "x.txt"}) for _ in range(3)])
    result = Agent(model, build_tools(state), state, max_steps=3).run("loop forever")
    assert not result.completed
    assert result.stopped_early
    assert result.steps == 3
    assert "Stopped after 3 steps" in result.text


def test_agent_emits_progress_events(state: RuntimeState) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    model = ScriptedModel(
        [
            tool_call("file_write", {"file_path": "a.txt", "content": "a\n"}),
            AIMessage(content="ok"),
        ]
    )
    agent = Agent(
        model,
        build_tools(state),
        state,
        on_event=lambda event, payload: events.append((event, payload)),
    )
    agent.run("write a file")

    kinds = [event for event, _ in events]
    assert kinds == [
        "ai_message",
        "tool_call",
        "tool_result",
        "ai_message",
        "final_answer",
    ]
    assert events[1][1]["name"] == "file_write"


def test_system_prompt_mentions_the_workspace(state: RuntimeState) -> None:
    agent = Agent(ScriptedModel([AIMessage(content="ok")]), build_tools(state), state)
    assert str(state.workspace) in agent.system_prompt
    assert "file_edit" in agent.system_prompt


def test_missing_model_responses_surface_as_errors(state: RuntimeState) -> None:
    with pytest.raises(AssertionError):
        Agent(ScriptedModel([]), build_tools(state), state).run("nothing scripted")
