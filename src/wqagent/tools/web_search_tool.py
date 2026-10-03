"""Web search through Tavily - the search specialist's only external capability.

The tool never raises: a missing ``TAVILY_API_KEY``, a missing ``tavily-python``
install, an empty query or a Tavily/network error all come back as
``{"ok": False, "error": ...}`` so a research run degrades instead of crashing a
graph.

Result shape (every key is always present)::

    {"ok": True,
     "query": "...",
     "answer": "short generated answer",
     "results": [{"title": ..., "url": ..., "content": ..., "score": 0.93}]}
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Mapping

from pydantic import BaseModel, Field

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.tools import StructuredTool

__all__ = [
    "DEFAULT_MAX_RESULTS",
    "MISSING_KEY_ERROR",
    "SEARCH_DEPTHS",
    "WebSearchInput",
    "WebSearchTool",
]

#: Environment variable holding the Tavily API key.
API_KEY_ENV = "TAVILY_API_KEY"

#: Returned (with ``ok: False``) when no key is configured.
MISSING_KEY_ERROR = "missing TAVILY_API_KEY"

DEFAULT_MAX_RESULTS = 5
MAX_MAX_RESULTS = 20

#: Values Tavily accepts for ``search_depth`` (the first one is the default).
SEARCH_DEPTHS = ("basic", "advanced")


class WebSearchInput(BaseModel):
    """Arguments accepted by :class:`WebSearchTool`."""

    query: str = Field(description="Natural-language search query.")
    max_results: int = Field(
        default=DEFAULT_MAX_RESULTS,
        ge=1,
        le=MAX_MAX_RESULTS,
        description="How many results to ask for.",
    )
    search_depth: str = Field(
        default="basic",
        description="'basic' is fast, 'advanced' crawls deeper for harder questions.",
    )


def _empty(query: str = "", error: str = "") -> dict[str, Any]:
    """A result dict with every documented key, for the failure paths."""
    return {"ok": False, "query": query, "answer": "", "results": [], "error": error}


def _score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class WebSearchTool:
    """Search the public web and return sources with their content."""

    name: str = "web_search"
    description: str = (
        "Search the public web and return the top results with title, URL, snippet and "
        "relevance score, plus a short generated answer. Use it for facts that are not in "
        "the workspace: library versions, API references, error messages, current best "
        "practice."
    )
    args_schema: type[BaseModel] = WebSearchInput

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def as_tool(self) -> "StructuredTool":
        """Wrap this implementation in a LangChain ``StructuredTool``."""
        from langchain_core.tools import StructuredTool

        return StructuredTool.from_function(
            func=self.run,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
        )

    def run(
        self,
        query: str,
        max_results: int = DEFAULT_MAX_RESULTS,
        search_depth: str = "basic",
    ) -> dict[str, Any]:
        """Tool entry point: the same dict as :meth:`search`."""
        return self.search(
            query, max_results=max_results, search_depth=search_depth
        )

    def search(
        self,
        query: str,
        *,
        max_results: int = DEFAULT_MAX_RESULTS,
        search_depth: str = "basic",
    ) -> dict[str, Any]:
        """Run one search and return the structured result; never raises."""
        query = str(query or "").strip()
        if not query:
            return _empty(error="empty query")

        api_key = self._api_key()
        if not api_key:
            return _empty(query, MISSING_KEY_ERROR)

        try:
            from tavily import TavilyClient
        except ImportError:
            return _empty(
                query, "tavily-python is not installed (pip install tavily-python)"
            )

        try:
            limit = max(1, min(int(max_results), MAX_MAX_RESULTS))
        except (TypeError, ValueError):
            limit = DEFAULT_MAX_RESULTS
        depth = search_depth if search_depth in SEARCH_DEPTHS else SEARCH_DEPTHS[0]

        try:
            client = TavilyClient(api_key=api_key)
            raw = client.search(
                query=query,
                max_results=limit,
                search_depth=depth,
                include_answer=True,
            )
        except Exception as exc:  # noqa: BLE001 - network/API errors become data
            return _empty(query, f"{type(exc).__name__}: {exc}")

        if not isinstance(raw, Mapping):
            return _empty(query, "unexpected response from Tavily")
        return {
            "ok": True,
            "query": str(raw.get("query") or query),
            "answer": str(raw.get("answer") or ""),
            "results": self._normalise(raw.get("results")),
            "error": "",
        }

    # -- internals -----------------------------------------------------------
    def _api_key(self) -> str:
        """Explicit key, else ``TAVILY_API_KEY`` (loading ``.env`` first)."""
        if self.api_key and self.api_key.strip():
            return self.api_key.strip()
        try:
            from wqagent.providers import load_env

            load_env()
        except Exception:  # pragma: no cover - .env is optional
            pass
        return os.environ.get(API_KEY_ENV, "").strip()

    @staticmethod
    def _normalise(raw: Any) -> list[dict[str, Any]]:
        """Keep only the documented fields, whatever Tavily sent back."""
        results: list[dict[str, Any]] = []
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, Mapping):
                continue
            results.append(
                {
                    "title": str(item.get("title") or ""),
                    "url": str(item.get("url") or ""),
                    "content": str(item.get("content") or ""),
                    "score": _score(item.get("score")),
                }
            )
        return results
