"""Tests for the ReAct event stream: ``stream_agent_events`` and ``execute_tool``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from wqagent.core.agent import DEFAULT_MAX_LOOPS, execute_tool, stream_agent_events
from wqagent.prompts.stage1 import ACTOR_PROMPT


class ScriptedModel:
    """Replays scripted responses and records the message lists it receives."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.seen: list[list[BaseMessage]] = []
        self.bound: list[Any] | None = None

    def bind_tools(self, tools: list[Any]) -> "ScriptedModel":
        self.bound = list(tools)
        return self

    def invoke(self, messages: list[BaseMessage]) -> Any:
        self.seen.append(list(messages))
        if not self.responses:
            raise AssertionError("ScriptedModel ran out of scripted responses")
        return self.responses.pop(0)


def _tool_call(name: str, args: dict[str, Any], call_id: str = "call_1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _kinds(events: list[dict[str, Any]]) -> list[str]:
    return [str(event["type"]) for event in events]


def test_streams_the_full_event_sequence(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    model = ScriptedModel(
        [
            _tool_call("file_write", {"file_path": "hello.py", "content": "print('hi')\n"}),
            AIMessage(content="Created hello.py."),
        ]
    )

    events = list(stream_agent_events("create hello.py", workspace=workspace, model=model))

    assert _kinds(events) == [
        "ai_message",
        "tool_call",
        "tool_result",
        "ai_message",
        "final_answer",
    ]
    assert events[0]["tool_calls"] == ["file_write"]
    assert events[1]["name"] == "file_write"
    assert events[1]["args"] == {"file_path": "hello.py", "content": "print('hi')\n"}
    assert events[2]["result"]["ok"] is True
    assert "Created hello.py" in events[2]["result"]["result"]
    assert events[3]["content"] == "Created hello.py."
    assert events[4]["content"] == "Created hello.py."
    assert events[4]["stopped_early"] is False
    assert events[4]["loop"] == 2
    assert (workspace / "hello.py").read_text(encoding="utf-8") == "print('hi')\n"


def test_uses_actor_prompt_and_binds_every_tool(tmp_path: Path) -> None:
    model = ScriptedModel([AIMessage(content="nothing to do")])
    list(stream_agent_events("say hi", workspace=tmp_path / "ws", model=model))

    assert [tool.name for tool in model.bound or []] == [
        "file_read",
        "file_write",
        "file_edit",
        "grep",
        "bash",
    ]
    first = model.seen[0]
    assert isinstance(first[0], SystemMessage)
    assert first[0].content == ACTOR_PROMPT
    assert isinstance(first[1], HumanMessage)
    assert first[1].content == "say hi"


def test_tool_results_are_fed_back_as_json_tool_messages(tmp_path: Path) -> None:
    model = ScriptedModel(
        [_tool_call("bash", {"command": "echo hi"}), AIMessage(content="done")]
    )
    list(stream_agent_events("say hi", workspace=tmp_path / "ws", model=model))

    tool_message = model.seen[1][-1]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.tool_call_id == "call_1"
    payload = json.loads(str(tool_message.content))
    assert payload["ok"] is True
    assert payload["tool"] == "bash"
    assert "hi" in payload["result"]


def test_creates_the_workspace_when_it_is_missing(tmp_path: Path) -> None:
    workspace = tmp_path / "fresh" / "nested" / "ws"
    model = ScriptedModel([AIMessage(content="ok")])
    list(stream_agent_events("noop", workspace=workspace, model=model))
    assert workspace.is_dir()


def test_tool_level_errors_are_reported_and_the_loop_continues(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            _tool_call("file_read", {"file_path": "../escape.txt"}),
            AIMessage(content="Recovered: that path is outside the workspace."),
        ]
    )
    events = list(stream_agent_events("read outside", workspace=tmp_path / "ws", model=model))

    result = events[2]["result"]
    assert result["ok"] is False
    assert "escapes the workspace" in result["error"]
    assert events[-1]["stopped_early"] is False


def test_unknown_tools_are_reported(tmp_path: Path) -> None:
    model = ScriptedModel([_tool_call("nope", {}), AIMessage(content="ok")])
    events = list(stream_agent_events("x", workspace=tmp_path / "ws", model=model))

    assert events[2]["result"]["ok"] is False
    assert "unknown tool 'nope'" in events[2]["result"]["error"]


def test_raising_tools_are_reported(tmp_path: Path) -> None:
    class Boom:
        name = "boom"

        def invoke(self, args: dict[str, Any]) -> str:  # noqa: ARG002
            raise RuntimeError("kaboom")

    model = ScriptedModel([_tool_call("boom", {}), AIMessage(content="ok")])
    events = list(
        stream_agent_events(
            "explode", workspace=tmp_path / "ws", model=model, tools=[Boom()]
        )
    )
    assert events[2]["result"] == {
        "ok": False,
        "tool": "boom",
        "error": "RuntimeError: kaboom",
    }


def test_stops_after_max_loops(tmp_path: Path) -> None:
    model = ScriptedModel([_tool_call("bash", {"command": "echo loop"}) for _ in range(3)])
    events = list(
        stream_agent_events("loop", workspace=tmp_path / "ws", model=model, max_loops=3)
    )

    final = events[-1]
    assert final["type"] == "final_answer"
    assert final["stopped_early"] is True
    assert final["loop"] == 3
    assert "Stopped after 3 steps" in final["content"]
    assert sum(1 for event in events if event["type"] == "tool_call") == 3
    assert sum(1 for event in events if event["type"] == "ai_message") == 3


def test_default_loop_budget_is_ten() -> None:
    assert DEFAULT_MAX_LOOPS == 10


def test_history_and_on_event_are_populated(tmp_path: Path) -> None:
    seen: list[str] = []
    history: list[BaseMessage] = []
    model = ScriptedModel([AIMessage(content="done")])

    list(
        stream_agent_events(
            "hi",
            workspace=tmp_path / "ws",
            model=model,
            history=history,
            on_event=lambda kind, payload: seen.append(kind),
        )
    )

    assert [type(message).__name__ for message in history] == [
        "SystemMessage",
        "HumanMessage",
        "AIMessage",
    ]
    assert seen == ["ai_message", "final_answer"]


def test_model_is_created_from_env_when_not_injected(
    tmp_path: Path, monkeypatch: Any
) -> None:
    created: dict[str, Any] = {}
    model = ScriptedModel([AIMessage(content="ok")])

    def fake_create_model(**kwargs: Any) -> ScriptedModel:
        created.update(kwargs)
        created["called"] = True
        return model

    monkeypatch.setattr("wqagent.core.agent.create_model", fake_create_model)
    events = list(stream_agent_events("hi", workspace=tmp_path / "ws"))

    assert created.get("called") is True
    assert _kinds(events) == ["ai_message", "final_answer"]


def test_state_and_tool_overrides_win(tmp_path: Path) -> None:
    from wqagent.core.state import RuntimeState

    state = RuntimeState(workspace=tmp_path / "ws", timeout_seconds=7)
    model = ScriptedModel([AIMessage(content="ok")])
    list(
        stream_agent_events(
            "hi",
            workspace=tmp_path / "other",
            state=state,
            model=model,
            tools=[],
            system_prompt="custom",
        )
    )

    assert not (tmp_path / "other").exists()  # `state` took precedence
    assert model.bound is None  # no tools -> nothing to bind
    assert model.seen[0][0].content == "custom"


def test_execute_tool_contract() -> None:
    class Echo:
        name = "echo"

        def invoke(self, args: dict[str, Any]) -> str:
            return "Error: nope" if args.get("fail") else f"echo {args.get('text', '')}"

    tools = {"echo": Echo()}
    assert execute_tool("echo", {"text": "hi"}, tools) == {
        "ok": True,
        "tool": "echo",
        "result": "echo hi",
    }
    assert execute_tool("echo", {"fail": True}, tools) == {
        "ok": False,
        "tool": "echo",
        "error": "nope",
    }
    unknown = execute_tool("missing", {}, tools)
    assert unknown["ok"] is False
    assert "unknown tool 'missing'" in unknown["error"]
    assert "echo" in unknown["error"]  # lists what is available
