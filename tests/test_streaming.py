"""Tests for the streaming helpers (``core/streaming.py``) and the chat node."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk

from helpers import ScriptedModel, chat_script
from wqagent.core.state import RuntimeState
from wqagent.core.streaming import merge_chunks, streaming_invoke
from wqagent.graph.state import new_graph_state


class StreamingFake:
    """A model that streams *chunks*; ``invoke`` returns them joined."""

    def __init__(self, chunks: list[str], *, explode: bool = False) -> None:
        self.chunks = chunks
        self.explode = explode
        self.streamed = 0

    def stream(self, messages: Any) -> Any:
        self.streamed += 1
        for piece in self.chunks:
            if self.explode:
                raise RuntimeError("stream broke")
            yield AIMessageChunk(content=piece)

    def invoke(self, messages: Any) -> AIMessage:
        return AIMessage(content="".join(self.chunks))


class PlainModel:
    """No streaming support at all (like the scripted test models)."""

    def invoke(self, messages: Any) -> AIMessage:
        return AIMessage(content="whole answer")


def test_streaming_forwards_every_delta_and_returns_the_whole_message() -> None:
    seen: list[str] = []

    message = streaming_invoke(StreamingFake(["Hel", "lo"]), [], seen.append)

    assert seen == ["Hel", "lo"]
    assert message.content == "Hello"


def test_a_model_without_stream_falls_back_to_invoke() -> None:
    seen: list[str] = []

    message = streaming_invoke(PlainModel(), [], seen.append)

    assert seen == []
    assert message.content == "whole answer"


def test_a_broken_stream_falls_back_without_duplicating_text() -> None:
    seen: list[str] = []

    message = streaming_invoke(StreamingFake(["Hel", "lo"], explode=True), [], seen.append)

    assert seen == []  # nothing reached the UI before the fallback
    assert message.content == "Hello"


def test_merge_chunks_folds_tool_call_deltas() -> None:
    chunks = [
        AIMessageChunk(
            content="",
            tool_call_chunks=[{"name": "file_write", "args": '{"file_path": "a.py"', "id": "c1", "index": 0}],
        ),
        AIMessageChunk(
            content="",
            tool_call_chunks=[{"name": None, "args": '}', "id": None, "index": 0}],
        ),
    ]

    merged = merge_chunks(chunks)

    assert merged.tool_calls and merged.tool_calls[0]["name"] == "file_write"
    assert merged.tool_calls[0]["args"] == {"file_path": "a.py"}


def test_chat_responder_streams_transient_token_events(
    state: RuntimeState, monkeypatch: Any
) -> None:
    from wqagent.graph import nodes

    tokens: list[dict[str, Any]] = []
    monkeypatch.setattr(nodes, "stream_writer", lambda: tokens.append)

    graph_state = new_graph_state("你好", state)
    update = nodes.chat_responder_node(graph_state, model=StreamingFake(["你", "好"]))

    assert update["chat_response"] == "你好"
    assert [event["text"] for event in tokens] == ["你", "好"]
    assert all(event["transient"] is True for event in tokens)
    assert all(event["type"] == "token" for event in tokens)


def test_chat_responder_does_not_stream_without_a_sink(
    state: RuntimeState, monkeypatch: Any
) -> None:
    from wqagent.graph import nodes

    monkeypatch.setattr(nodes, "stream_writer", lambda: None)
    model = StreamingFake(["你", "好"])

    update = nodes.chat_responder_node(new_graph_state("你好", state), model=model)

    assert update["chat_response"] == "你好"
    assert model.streamed == 0  # no listener: one plain call instead


# -- end to end through the entry graph --------------------------------------
class StreamingScript(ScriptedModel):
    """A scripted model whose replies also stream, in three-character pieces."""

    def stream(self, messages: Any) -> Any:
        text = str(getattr(self.invoke(messages), "content", ""))
        for start in range(0, len(text), 3):
            yield AIMessageChunk(content=text[start : start + 3])


def test_a_chat_route_run_streams_its_reply(tmp_path: Path) -> None:
    from wqagent.core.agent import stream_agent_events

    workspace = tmp_path / "ws"
    workspace.mkdir()
    model = StreamingScript(chat_script("你好，我是 WQ"))

    events = list(
        stream_agent_events(
            "hi",
            workspace=workspace,
            model=model,
            trace_mode="on",
            entry_router=True,  # the chat route lives in the entry graph
        )
    )

    tokens = [event for event in events if event["type"] == "token"]
    assert tokens, "the chat route never streamed"
    assert "".join(event["text"] for event in tokens) == "你好，我是 WQ"
    assert all(event["transient"] is True for event in tokens)

    final = next(event for event in events if event["type"] == "final_answer")
    assert final["content"] == "你好，我是 WQ"


def test_token_events_stay_out_of_the_trace(tmp_path: Path) -> None:
    """Deltas are for the screen; the run's trail records the answer, not the drips."""
    import json

    from wqagent.core.agent import stream_agent_events

    workspace = tmp_path / "ws"
    workspace.mkdir()
    model = StreamingScript(chat_script("streamed reply"))

    events = list(
        stream_agent_events(
            "hi", workspace=workspace, model=model, trace_mode="full", entry_router=True
        )
    )
    assert any(event["type"] == "token" for event in events)

    (trace_dir,) = (workspace / ".wqagent" / "traces").iterdir()
    logged = [
        json.loads(line)
        for line in (trace_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert logged
    assert not [line for line in logged if line.get("type") == "token"]
