"""The search specialist (``searchAgent``).

:func:`run_search_agent` is a small ReAct agent whose only capability is web
search. A planner or actor node calls it to gather external facts::

    result = run_search_agent(state, "what is the current stable FastAPI version?")
    if result["ok"]:
        state["research_notes"] = (state["research_notes"] + "\\n" + result["summary"]).strip()

It builds ``SystemMessage(SEARCH_AGENT_PROMPT) + HumanMessage(task + instruction +
research notes)``, lets the tool-bound model search the web up to ``max_loops``
times, and reports what it found. Like the rest of the project it never raises:
failures (no API key, search errors, a model that returns nothing usable) come
back as ``ok=False`` plus an ``error``.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from wqagent.core.agent import last_ai_text, react_loop, write_event
from wqagent.prompts.stage3 import SEARCH_AGENT_PROMPT
from wqagent.providers import create_model
from wqagent.tools.web_search_tool import WebSearchTool

__all__ = ["DEFAULT_MAX_LOOPS", "build_search_messages", "run_search_agent"]

#: How many search rounds the agent may run by default.
DEFAULT_MAX_LOOPS = 4

#: Event sink: called with ``{"type": "tool_call"|"search_results", ...}``.
EventWriter = Callable[[dict[str, Any]], None]


def build_search_messages(
    state: Mapping[str, Any] | None, instruction: str
) -> list[BaseMessage]:
    """Build the agent's opening messages: prompt + task + instruction + notes."""
    source = state or {}
    task = str(source.get("task") or "").strip() or "(no task given)"
    notes = str(source.get("research_notes") or "").strip()
    request = str(instruction or "").strip()

    lines = [f"Task:\n{task}"]
    if request:
        lines.append(f"Research request:\n{request}")
    if notes:
        lines.append(
            "Research notes collected so far (build on them, do not repeat them):\n"
            f"{notes}"
        )
    lines.append(
        "Search the web as needed, then answer with a concise research summary and the "
        "source URLs you used."
    )
    return [
        SystemMessage(content=SEARCH_AGENT_PROMPT),
        HumanMessage(content="\n\n".join(lines)),
    ]


def run_search_agent(
    state: Mapping[str, Any] | None,
    instruction: str,
    *,
    writer: EventWriter | None = None,
    max_loops: int = DEFAULT_MAX_LOOPS,
    model: Any | None = None,
    tool: WebSearchTool | None = None,
) -> dict[str, Any]:
    """Research *instruction* for *state* and return the findings.

    Args:
        state: The graph state (or any mapping) holding ``task`` and the
            ``research_notes`` gathered so far.
        instruction: What to research, in natural language.
        writer: Event sink; defaults to LangGraph's custom stream writer when the
            agent runs inside a graph.
        max_loops: Maximum search rounds.
        model: Chat model; defaults to ``create_model()`` (``.env``).
        tool: Search tool; defaults to :class:`WebSearchTool`.

    Returns:
        ``{"ok": bool, "error": str, "summary": str, "queries": [...],
        "answers": [...], "sources": [{"title", "url", "content", "score"}],
        "messages": [...], "tool_events": [...]}``
    """
    chat = model if model is not None else create_model()
    search_tool = tool if tool is not None else WebSearchTool()
    messages = build_search_messages(state, instruction)

    queries: list[str] = []
    answers: list[str] = []
    sources: list[dict[str, Any]] = []
    tool_events: list[dict[str, Any]] = []

    def emit(event: dict[str, Any]) -> None:
        tool_events.append(event)
        write_event(writer, event)

    def on_event(kind: str, payload: dict[str, Any]) -> None:
        loop = int(payload.get("loop") or 0)
        if kind == "tool_call":
            emit(
                {
                    "type": "tool_call",
                    "name": str(payload.get("name") or ""),
                    "args": dict(payload.get("args") or {}),
                    "loop": loop,
                }
            )
        elif kind == "tool_result":
            data = _search_payload(payload.get("result"))
            query = str(data.get("query") or "").strip()
            if query and query not in queries:
                queries.append(query)
            answer = str(data.get("answer") or "").strip()
            if answer:
                answers.append(answer)
            for item in data.get("results") or []:
                if not isinstance(item, Mapping) or not item.get("url"):
                    continue
                if any(item.get("url") == existing.get("url") for existing in sources):
                    continue
                sources.append(dict(item))
            emit(
                {
                    "type": "search_results",
                    "ok": bool(data.get("ok")),
                    "query": query or str(payload.get("name") or ""),
                    "answer": answer,
                    "results": list(data.get("results") or []),
                    "error": str(data.get("error") or ""),
                    "loop": loop,
                }
            )

    for _ in react_loop(
        messages, model=chat, tools=[search_tool.as_tool()], max_loops=max_loops, on_event=on_event
    ):
        pass

    failed = [
        event
        for event in tool_events
        if event["type"] == "search_results" and not event.get("ok")
    ]
    error = ""
    if failed and not sources:
        error = str(failed[0].get("error") or "web search failed")

    summary = last_ai_text(messages) or _digest(answers, sources)
    return {
        "ok": not error and bool(summary or sources),
        "error": error,
        "summary": summary,
        "queries": queries,
        "answers": answers,
        "sources": sources,
        "messages": messages,
        "tool_events": tool_events,
    }


def _search_payload(result: Any) -> dict[str, Any]:
    """Unwrap ``{"ok", "tool", "result"}`` into the tool's own result dict."""
    if not isinstance(result, Mapping):
        return {"ok": False, "error": "unexpected tool result"}
    if result.get("ok") is False:
        return {
            "ok": False,
            "error": str(result.get("error") or "tool failed"),
            "query": "",
            "answer": "",
            "results": [],
        }
    inner: Any = result.get("result")
    if isinstance(inner, Mapping):
        return dict(inner)
    if isinstance(inner, str):
        try:
            parsed = json.loads(inner)
        except json.JSONDecodeError:
            return {"ok": False, "error": "unparsable tool result"}
        return dict(parsed) if isinstance(parsed, Mapping) else {"ok": False, "error": "unexpected tool result"}
    return {"ok": False, "error": "empty tool result"}


def _digest(answers: Sequence[str], sources: Sequence[Mapping[str, Any]]) -> str:
    """A fallback summary when the model never wrote one (e.g. it hit max_loops)."""
    lines: list[str] = []
    if answers:
        lines.append("\n".join(answers))
    for source in sources:
        title = str(source.get("title") or source.get("url") or "").strip()
        url = str(source.get("url") or "").strip()
        if url:
            lines.append(f"- {title}: {url}" if title else f"- {url}")
    return "\n".join(lines).strip()
