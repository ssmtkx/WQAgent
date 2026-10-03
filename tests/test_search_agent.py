"""Tests for the search specialist (``run_search_agent``)."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from wqagent.agents.search_agent import (
    DEFAULT_MAX_LOOPS,
    build_search_messages,
    run_search_agent,
)
from wqagent.core.state import RuntimeState
from wqagent.graph import new_graph_state
from wqagent.prompts.stage3 import SEARCH_AGENT_PROMPT
from wqagent.tools.web_search_tool import MISSING_KEY_ERROR, WebSearchTool

FASTAPI_RESULTS = [
    {
        "title": "FastAPI",
        "url": "https://fastapi.tiangolo.com/",
        "content": "FastAPI framework.",
        "score": 0.97,
    }
]
PYDANTIC_RESULTS = [
    {
        "title": "Pydantic",
        "url": "https://docs.pydantic.dev/",
        "content": "Data validation.",
        "score": 0.9,
    },
    {
        "title": "FastAPI (again)",
        "url": "https://fastapi.tiangolo.com/",  # duplicate URL: must be de-duplicated
        "content": "Same page.",
        "score": 0.5,
    },
]


class ScriptedModel:
    """Replays responses; records what was bound and what it received."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.bound: list[list[str]] = []
        self.seen: list[list[Any]] = []

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "ScriptedModel":
        self.bound.append([tool.name for tool in tools])
        return self

    def invoke(self, messages: list[Any]) -> Any:
        self.seen.append(list(messages))
        if not self.responses:
            raise AssertionError("ScriptedModel ran out of responses")
        return self.responses.pop(0)


class FakeSearchTool(WebSearchTool):
    """A WebSearchTool that answers from a table instead of the network."""

    def __init__(
        self,
        results_by_query: dict[str, list[dict[str, Any]]] | None = None,
        *,
        ok: bool = True,
        error: str = "",
    ) -> None:
        super().__init__(api_key="fake")
        self.results_by_query = results_by_query or {}
        self.ok = ok
        self.error = error
        self.queries: list[str] = []

    def search(
        self, query: str, *, max_results: int = 5, search_depth: str = "basic"
    ) -> dict[str, Any]:
        self.queries.append(query)
        if not self.ok:
            return {"ok": False, "query": query, "answer": "", "results": [], "error": self.error}
        return {
            "ok": True,
            "query": query,
            "answer": f"answer for {query}",
            "results": self.results_by_query.get(query, FASTAPI_RESULTS),
            "error": "",
        }


class PlainTextTool:
    """A tool that returns something the agent cannot parse as a search payload."""

    name = "web_search"

    def as_tool(self) -> Any:
        from langchain_core.tools import StructuredTool

        def run(query: str) -> str:
            return "just some text, not JSON"

        return StructuredTool.from_function(
            func=run, name=self.name, description="search"
        )


def _tool_call(query: str, call_id: str = "c1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "web_search",
                "args": {"query": query},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


@pytest.fixture()
def state(tmp_path):
    return new_graph_state("build a FastAPI service", RuntimeState(workspace=tmp_path / "ws"))


# -- messages ----------------------------------------------------------------
def test_messages_are_prompt_task_instruction_and_notes(state: Any) -> None:
    state["research_notes"] = "FastAPI is a Python web framework."

    messages = build_search_messages(state, "which version should we pin?")

    assert messages[0].content == SEARCH_AGENT_PROMPT
    human = messages[1].content
    assert "build a FastAPI service" in human
    assert "which version should we pin?" in human
    assert "FastAPI is a Python web framework." in human


def test_messages_survive_a_missing_state_and_instruction() -> None:
    messages = build_search_messages(None, "")

    assert messages[0].content == SEARCH_AGENT_PROMPT
    assert "(no task given)" in messages[1].content


# -- the happy path ----------------------------------------------------------
def test_collects_queries_sources_and_answers(state: Any) -> None:
    model = ScriptedModel(
        [
            _tool_call("fastapi latest version", "c1"),
            _tool_call("pydantic v2 docs", "c2"),
            AIMessage(content="FastAPI 0.115 with Pydantic v2."),
        ]
    )
    tool = FakeSearchTool({"pydantic v2 docs": PYDANTIC_RESULTS})
    events: list[dict[str, Any]] = []

    result = run_search_agent(state, "which versions?", model=model, tool=tool, writer=events.append)

    assert result["ok"] is True and result["error"] == ""
    assert result["queries"] == ["fastapi latest version", "pydantic v2 docs"]
    # sources from both searches, de-duplicated by url
    assert [source["url"] for source in result["sources"]] == [
        "https://fastapi.tiangolo.com/",
        "https://docs.pydantic.dev/",
    ]
    assert result["answers"] == [
        "answer for fastapi latest version",
        "answer for pydantic v2 docs",
    ]
    assert result["summary"] == "FastAPI 0.115 with Pydantic v2."

    # the model only ever got the search tool
    assert model.bound == [["web_search"]]
    # the tool result travelled back as a ToolMessage
    assert any(isinstance(message, ToolMessage) for message in result["messages"])

    # writer saw one tool_call + one search_results per search
    assert [event["type"] for event in events] == [
        "tool_call",
        "search_results",
        "tool_call",
        "search_results",
    ]
    assert result["tool_events"] == events
    assert events[0]["args"] == {"query": "fastapi latest version"}
    assert events[1]["ok"] is True and events[1]["results"] == FASTAPI_RESULTS


def test_falls_back_to_a_digest_when_the_model_hits_max_loops(state: Any) -> None:
    model = ScriptedModel([_tool_call(f"query {i}", f"c{i}") for i in range(5)])
    tool = FakeSearchTool()

    result = run_search_agent(state, "research", model=model, tool=tool, max_loops=2)

    assert len(tool.queries) == 2  # never exceeded the budget
    assert result["ok"] is True
    assert "answer for query 0" in result["summary"]  # digest of the collected answers
    assert "https://fastapi.tiangolo.com/" in result["summary"]


def test_a_model_that_answers_without_searching(state: Any) -> None:
    model = ScriptedModel([AIMessage(content="I already know this.")])

    result = run_search_agent(
        state, "easy question", model=model, tool=FakeSearchTool()
    )

    assert result["ok"] is True
    assert result["summary"] == "I already know this."
    assert result["queries"] == [] and result["sources"] == []
    assert result["tool_events"] == []


# -- failure paths -----------------------------------------------------------
def test_missing_tavily_key_is_reported(state: Any, monkeypatch: Any) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    # the developer's .env must not inject a key here (it would hit the network)
    monkeypatch.setattr("wqagent.providers.load_env", lambda *args, **kwargs: None)
    model = ScriptedModel([_tool_call("fastapi version"), AIMessage(content="not sure")])

    result = run_search_agent(
        state, "research", model=model, tool=WebSearchTool()  # no key, no fake
    )

    assert result["ok"] is False
    assert result["error"] == MISSING_KEY_ERROR
    assert result["queries"] == ["fastapi version"]
    assert result["sources"] == []
    # the failure was streamed, and the run still produced everything else
    assert [event["type"] for event in result["tool_events"]] == [
        "tool_call",
        "search_results",
    ]
    assert result["tool_events"][1]["ok"] is False
    assert result["tool_events"][1]["error"] == MISSING_KEY_ERROR


def test_search_errors_are_kept_but_a_summary_still_returns(state: Any) -> None:
    model = ScriptedModel([_tool_call("fastapi version"), AIMessage(content="Best effort answer.")])
    tool = FakeSearchTool(ok=False, error="RuntimeError: 429 rate limit")

    result = run_search_agent(state, "research", model=model, tool=tool)

    assert result["ok"] is False  # every search failed
    assert "429 rate limit" in result["error"]
    assert result["summary"] == "Best effort answer."  # the model's text is still returned


def test_partial_search_success_stays_ok(state: Any) -> None:
    class FlakyTool(FakeSearchTool):
        def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
            if query == "second":
                return {"ok": False, "query": query, "answer": "", "results": [], "error": "boom"}
            return super().search(query, **kwargs)

    model = ScriptedModel(
        [_tool_call("first", "c1"), _tool_call("second", "c2"), AIMessage(content="Partial.")]
    )

    result = run_search_agent(state, "research", model=model, tool=FlakyTool())

    assert result["ok"] is True  # one good search is enough to be useful
    assert len(result["sources"]) == 1


def test_unparsable_tool_output_is_tolerated(state: Any) -> None:
    model = ScriptedModel([_tool_call("anything"), AIMessage(content="Hmm.")])

    result = run_search_agent(
        state, "research", model=model, tool=PlainTextTool()  # type: ignore[arg-type]
    )

    assert result["ok"] is False
    assert "unparsable tool result" in result["error"]
    assert result["tool_events"][1]["ok"] is False


def test_default_budget_is_four() -> None:
    assert DEFAULT_MAX_LOOPS == 4
