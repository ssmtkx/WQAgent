"""Tests for ``WebSearchTool`` (Tavily-backed, with every failure as data)."""

from __future__ import annotations

from typing import Any

import pytest

from wqagent.tools.web_search_tool import (
    MISSING_KEY_ERROR,
    WebSearchTool,
)

TAVILY_RESPONSE = {
    "query": "fastapi latest version",
    "answer": "FastAPI 0.115 is the current release.",
    "results": [
        {
            "title": "FastAPI",
            "url": "https://fastapi.tiangolo.com/",
            "content": "FastAPI framework, high performance.",
            "score": 0.97,
        },
        {
            "title": "Releases",
            "url": "https://github.com/fastapi/fastapi/releases",
            "content": "Release notes.",
            "score": 0.81,
        },
    ],
}


class FakeTavilyClient:
    """Records the call and replays a canned response."""

    calls: list[dict[str, Any]] = []
    response: Any = TAVILY_RESPONSE
    error: Exception | None = None

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def search(self, **kwargs: Any) -> Any:
        type(self).calls.append(kwargs)
        if type(self).error is not None:
            raise type(self).error
        return type(self).response


@pytest.fixture(autouse=True)
def fake_tavily(monkeypatch: pytest.MonkeyPatch) -> type[FakeTavilyClient]:
    import tavily

    FakeTavilyClient.calls = []
    FakeTavilyClient.response = TAVILY_RESPONSE
    FakeTavilyClient.error = None
    monkeypatch.setattr(tavily, "TavilyClient", FakeTavilyClient, raising=False)
    return FakeTavilyClient


@pytest.fixture(autouse=True)
def no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    # never let the developer's own .env inject a key into these tests
    monkeypatch.setattr("wqagent.providers.load_env", lambda *args, **kwargs: None)


def test_search_returns_the_documented_shape() -> None:
    result = WebSearchTool(api_key="tvly-test").search("fastapi latest version")

    assert result["ok"] is True
    assert result["query"] == "fastapi latest version"
    assert result["answer"].startswith("FastAPI 0.115")
    assert result["error"] == ""
    assert result["results"] == TAVILY_RESPONSE["results"]
    assert set(result["results"][0]) == {"title", "url", "content", "score"}


def test_search_forwards_the_arguments() -> None:
    WebSearchTool(api_key="tvly-test").search(
        "anything", max_results=3, search_depth="advanced"
    )

    call = FakeTavilyClient.calls[0]
    assert call["query"] == "anything"
    assert call["max_results"] == 3
    assert call["search_depth"] == "advanced"
    assert call["include_answer"] is True


def test_search_clamps_bad_arguments() -> None:
    tool = WebSearchTool(api_key="tvly-test")

    tool.search("q", max_results=999)  # type: ignore[arg-type]
    tool.search("q", max_results="not a number")  # type: ignore[arg-type]
    tool.search("q", search_depth="warp-speed")

    assert FakeTavilyClient.calls[0]["max_results"] == 20
    assert FakeTavilyClient.calls[1]["max_results"] == 5
    assert FakeTavilyClient.calls[2]["search_depth"] == "basic"


def test_missing_api_key_is_reported_not_raised() -> None:
    result = WebSearchTool().search("anything")

    assert result["ok"] is False
    assert result["error"] == MISSING_KEY_ERROR == "missing TAVILY_API_KEY"
    assert result["query"] == "anything"
    assert result["results"] == [] and result["answer"] == ""
    assert FakeTavilyClient.calls == []  # never reached the network


def test_key_can_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-from-env")

    result = WebSearchTool().search("anything")

    assert result["ok"] is True
    assert FakeTavilyClient.calls  # reached the client


def test_empty_query_is_reported() -> None:
    result = WebSearchTool(api_key="tvly-test").search("   ")

    assert result["ok"] is False and "empty query" in result["error"]
    assert FakeTavilyClient.calls == []


def test_tavily_errors_become_data() -> None:
    FakeTavilyClient.error = RuntimeError("429 rate limit")

    result = WebSearchTool(api_key="tvly-test").search("anything")

    assert result["ok"] is False
    assert result["error"] == "RuntimeError: 429 rate limit"
    assert result["results"] == []


def test_unexpected_response_shape_is_tolerated() -> None:
    tool = WebSearchTool(api_key="tvly-test")

    FakeTavilyClient.response = "not a dict"
    assert tool.search("q")["ok"] is False

    FakeTavilyClient.response = {"query": "q", "answer": None, "results": "nope"}
    result = tool.search("q")
    assert result["ok"] is True and result["results"] == [] and result["answer"] == ""

    FakeTavilyClient.response = {
        "results": [
            None,
            42,
            {"title": "T", "url": "u", "content": "c", "score": "0.5"},
            {"url": "u2"},  # missing title/content/score
        ]
    }
    normalised = tool.search("q")["results"]
    assert [item["url"] for item in normalised] == ["u", "u2"]
    assert normalised[0]["score"] == 0.5
    assert normalised[1] == {"title": "", "url": "u2", "content": "", "score": 0.0}


def test_missing_package_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "tavily":
            raise ImportError("No module named 'tavily'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = WebSearchTool(api_key="tvly-test").search("anything")

    assert result["ok"] is False
    assert "tavily-python is not installed" in result["error"]


def test_langchain_wrapper_exposes_the_schema() -> None:
    tool = WebSearchTool(api_key="tvly-test").as_tool()

    assert tool.name == "web_search"
    assert set(tool.args) == {"query", "max_results", "search_depth"}

    result = tool.invoke({"query": "fastapi latest version"})

    assert result["ok"] is True  # dict return survives StructuredTool.invoke
    assert len(result["results"]) == 2
