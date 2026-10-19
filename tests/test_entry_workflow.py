"""Tests for the entry workflow: intent routing and the light chat branch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from helpers import (
    ScriptedModel,
    answer,
    chat_script,
    graph_state,
    passing_script,
    route_answer,
    workflow_script,
)
from wqagent.core.agent import stream_agent_events
from wqagent.graph.nodes import (
    DEFAULT_INTENT_ROUTE,
    INTENT_CONFIDENCE_THRESHOLD,
    INTENT_ROUTES,
    chat_responder_node,
    intent_route_fn,
    intent_router_node,
)
from wqagent.graph.workflow import (
    ENTRY_NODE_NAMES,
    ENTRY_ROUTE_TARGETS,
    build_entry_workflow,
)
from wqagent.prompts.entry import CHAT_RESPONDER_PROMPT, INTENT_ROUTER_PROMPT

CHAT_REPLY = "你好！我是 WQ Agent。"


def _run(
    tmp_path: Path, script: list[Any], task: str = "你好", **kwargs: Any
) -> list[dict[str, Any]]:
    """Stream *script* through the run entry point in ``tmp_path/ws``."""
    workspace = tmp_path / "ws"
    return list(stream_agent_events(task, workspace=workspace, model=ScriptedModel(script), **kwargs))


# -- prompts -----------------------------------------------------------------
def test_prompts_are_the_configured_ones() -> None:
    assert "intent router" in INTENT_ROUTER_PROMPT
    assert "If uncertain, choose workflow." in INTENT_ROUTER_PROMPT
    assert '{"route":"chat"|"workflow","reason":"brief reason","confidence":0.0}' in INTENT_ROUTER_PROMPT
    for word in ("继续", "修一下", "运行测试"):
        assert word in INTENT_ROUTER_PROMPT  # the follow-up rule

    assert "lightweight chat node" in CHAT_RESPONDER_PROMPT
    assert "Do not claim that you read files" in CHAT_RESPONDER_PROMPT


def test_entry_constants() -> None:
    assert ENTRY_NODE_NAMES == ("intent_router", "chat_responder")
    assert ENTRY_ROUTE_TARGETS == {"chat_responder": "chat_responder", "planner": "__end__"}
    assert INTENT_ROUTES == ("chat", "workflow")
    assert DEFAULT_INTENT_ROUTE == "workflow"
    assert INTENT_CONFIDENCE_THRESHOLD == 0.55


# -- intent_route_fn ---------------------------------------------------------
@pytest.mark.parametrize(
    "route, expected",
    [("chat", "chat_responder"), ("workflow", "planner"), ("", "planner"), (None, "planner")],
)
def test_intent_route_fn(route: Any, expected: str) -> None:
    assert intent_route_fn({"intent_route": route}) == expected


# -- the router node ---------------------------------------------------------
def test_router_returns_the_three_fields(tmp_path: Path) -> None:
    model = ScriptedModel([route_answer("chat", reason="greeting", confidence=0.93)])

    update = intent_router_node(graph_state(root=tmp_path / "ws"), model=model)

    assert update == {
        "intent_route": "chat",
        "intent_reason": "greeting",
        "intent_confidence": 0.93,
    }
    assert model.bound == []  # routing never binds tools


def test_router_sees_the_user_input_and_the_session_context(tmp_path: Path) -> None:
    state = graph_state("继续修一下那个 bug", root=tmp_path / "ws")
    state["context_summary"] = "earlier: wrote utils.py"
    state["todos"] = [{"id": "t1", "content": "fix it", "status": "in_progress", "note": ""}]
    state["messages"] = [AIMessage(content="previous answer")]
    model = ScriptedModel([route_answer("workflow")])

    intent_router_node(state, model=model)

    instruction = model.seen[-1][1].content
    assert INTENT_ROUTER_PROMPT in model.seen[-1][0].content
    assert "继续修一下那个 bug" in instruction
    assert "earlier: wrote utils.py" in instruction
    assert "t1=in_progress" in instruction
    assert "previous answer" in instruction


def test_router_defaults_to_workflow_below_the_threshold(tmp_path: Path) -> None:
    model = ScriptedModel([route_answer("chat", confidence=0.54)])

    update = intent_router_node(graph_state(root=tmp_path / "ws"), model=model)

    assert update["intent_route"] == "workflow"
    assert "confidence 0.54 below 0.55" in update["intent_reason"]
    assert update["intent_confidence"] == 0.54  # the model's own number is kept


@pytest.mark.parametrize(
    "text",
    ["", "not json", '{"route":"nonsense","confidence":0.99}', '{"confidence":0.9}', "[]"],
)
def test_router_defaults_to_workflow_on_anything_unusable(tmp_path: Path, text: str) -> None:
    state = graph_state(root=tmp_path / "ws")
    update = intent_router_node(state, model=ScriptedModel([answer(text)]))

    assert update["intent_route"] == "workflow"
    # an unusable *route* keeps whatever confidence was parsed, but never routes to chat
    assert 0.0 <= update["intent_confidence"] <= 1.0
    assert update["intent_reason"]


@pytest.mark.parametrize("value", ["0.8", None, "abc", float("nan"), 5, -1])
def test_router_survives_odd_confidence_values(tmp_path: Path, value: Any) -> None:
    payload = '{"route":"chat","reason":"x","confidence":%s}' % (
        '"%s"' % value if isinstance(value, str) else value
    )

    update = intent_router_node(
        graph_state(root=tmp_path / "ws"), model=ScriptedModel([answer(payload)])
    )

    assert 0.0 <= update["intent_confidence"] <= 1.0


def test_router_never_raises(tmp_path: Path) -> None:
    class Boom(ScriptedModel):
        def invoke(self, messages: Any) -> Any:
            raise RuntimeError("provider down")

    update = intent_router_node(graph_state(root=tmp_path / "ws"), model=Boom([]))

    assert update["intent_route"] == "workflow"
    assert "router failed (RuntimeError: provider down)" in update["intent_reason"]


def test_router_emits_an_intent_event(tmp_path: Path) -> None:
    seen: list[dict[str, Any]] = []
    model = ScriptedModel([route_answer("chat")])

    intent_router_node(graph_state(root=tmp_path / "ws"), model=model, on_event=seen.append)

    assert [event["type"] for event in seen] == ["intent"]
    assert seen[0]["intent_route"] == "chat"
    assert seen[0]["node"] == "intent_router"


# -- the chat node -----------------------------------------------------------
def test_chat_node_answers_without_tools(tmp_path: Path) -> None:
    model = ScriptedModel([answer("我是 WQ Agent，可以帮你写代码。")])

    update = chat_responder_node(graph_state("你是谁？", root=tmp_path / "ws"), model=model)

    assert update["chat_response"] == "我是 WQ Agent，可以帮你写代码。"
    assert update["final_answer"] == update["chat_response"]
    assert model.bound == []  # no tools at all
    assert CHAT_RESPONDER_PROMPT in model.seen[-1][0].content
    assert "你是谁？" in model.seen[-1][1].content


def test_chat_node_falls_back_to_text_on_failure(tmp_path: Path) -> None:
    class Boom(ScriptedModel):
        def invoke(self, messages: Any) -> Any:
            raise RuntimeError("provider down")

    update = chat_responder_node(graph_state(root=tmp_path / "ws"), model=Boom([]))

    assert "chat unavailable" in update["chat_response"]
    assert update["final_answer"] == update["chat_response"]


def test_chat_node_handles_an_empty_answer(tmp_path: Path) -> None:
    update = chat_responder_node(
        graph_state(root=tmp_path / "ws"), model=ScriptedModel([answer("   ")])
    )

    assert update["chat_response"].strip()
    assert update["final_answer"] == update["chat_response"]


# -- the graph ---------------------------------------------------------------
def test_entry_graph_chat_path(tmp_path: Path) -> None:
    app = build_entry_workflow(model=ScriptedModel(chat_script(CHAT_REPLY)))

    result = app.invoke(graph_state("你好", root=tmp_path / "ws"))

    assert result["intent_route"] == "chat"
    assert result["final_answer"] == CHAT_REPLY


def test_entry_graph_workflow_path_ends_immediately(tmp_path: Path) -> None:
    app = build_entry_workflow(model=ScriptedModel([route_answer("workflow")]))

    result = app.invoke(graph_state("写个 hello.py", root=tmp_path / "ws"))

    assert result["intent_route"] == "workflow"
    assert not result.get("chat_response")  # the chat node never ran
    assert not result.get("final_answer")


def test_entry_graph_streams_the_node_events(tmp_path: Path) -> None:
    app = build_entry_workflow(model=ScriptedModel([route_answer("chat"), answer("hi")]))

    nodes = [
        node
        for mode, chunk in app.stream(graph_state("你好", root=tmp_path / "ws"), stream_mode=["updates"])
        for node in chunk
    ]

    assert nodes == ["intent_router", "chat_responder"]


# -- wiring into the run entry point ----------------------------------------
def test_stream_router_is_off_by_default(tmp_path: Path) -> None:
    """The library keeps its old behaviour: no router unless asked."""
    events = _run(tmp_path, workflow_script(router=False), task="say hi", trace_mode="off")

    assert not [event for event in events if event["type"] == "intent"]
    assert events[-1]["route"] == "workflow"
    assert [event["node"] for event in events if event["type"] == "node_end"][0] == "planner"


def test_stream_chat_route_skips_the_workflow(tmp_path: Path) -> None:
    events = _run(tmp_path, chat_script(CHAT_REPLY), entry_router=True, trace_mode="off")

    kinds = [event["type"] for event in events]
    assert "intent" in kinds and "chat_response" in kinds
    nodes = [event["node"] for event in events if event["type"] == "node_end"]
    assert nodes == ["intent_router", "chat_responder"]  # no planner, no verifier
    intent = next(event for event in events if event["type"] == "intent")
    assert intent["intent_route"] == "chat" and intent["intent_confidence"] == 0.95
    final = events[-1]
    assert final["type"] == "final_answer"
    assert final["route"] == "chat" and final["passed"] is True
    assert final["content"] == CHAT_REPLY
    assert final["attempts"] == 0 and final["stopped_early"] is False


def test_stream_workflow_route_uses_the_normal_pipeline(tmp_path: Path) -> None:
    script = [route_answer("workflow", confidence=0.8), *passing_script()]
    events = _run(tmp_path, script, task="写个 hello.py", entry_router=True)

    nodes = [event["node"] for event in events if event["type"] == "node_end"]
    assert nodes[0] == "intent_router"
    assert nodes[1:] == [
        "planner",
        "context_monitor",
        "verifier",
        "context_monitor",
        "final",
    ]
    assert events[-1]["route"] == "workflow" and events[-1]["passed"] is True


def test_chat_run_still_writes_checkpoint_and_trace(tmp_path: Path) -> None:
    _run(tmp_path, chat_script(CHAT_REPLY), entry_router=True, trace_mode="full")

    checkpoint = tmp_path / "ws" / ".wqagent" / "checkpoints" / "checkpoint.json"
    assert checkpoint.is_file()
    assert '"status": "passed"' in checkpoint.read_text(encoding="utf-8")
    traces = list((tmp_path / "ws" / ".wqagent" / "traces").iterdir())
    assert len(traces) == 1
    assert (traces[0] / "trace.json").is_file()
