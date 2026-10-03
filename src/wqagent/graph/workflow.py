"""The compiled WQ Agent workflow.

    planner ──▶ verifier ──┬─(passed / budget spent)─▶ final ──▶ END
      ▲                    │
      └────(retry: revise the plan and delegate the fix)───────┘

The planner is a supervisor: it owns the plan and delegates research and
implementation to the specialist agents through its tools
(``call_search_agent`` / ``call_code_agent``), so there is no separate actor node.

Usage::

    from wqagent.core.state import RuntimeState
    from wqagent.graph import build_workflow, new_graph_state

    app = build_workflow()
    result = app.invoke(new_graph_state("write hello.py and run it", RuntimeState("./demo")))
    print(result["passed"], result["final_answer"])
"""

from __future__ import annotations

import functools
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from wqagent.graph.nodes import (
    final_node,
    planner_node,
    verifier_node,
    verifier_route,
)
from wqagent.graph.state import WQGraphState

__all__ = ["NODE_NAMES", "ROUTE_TARGETS", "build_workflow"]

#: Nodes of the workflow, in execution order.
NODE_NAMES: tuple[str, ...] = ("planner", "verifier", "final")

#: Where :func:`~wqagent.graph.nodes.verifier_route` may send the run.
ROUTE_TARGETS: dict[str, str] = {"final": "final", "planner": "planner"}


def _bind(node: Callable[..., dict[str, Any]], **options: Any) -> Any:
    """Bind node options; with nothing set, keep the bare function."""
    chosen = {key: value for key, value in options.items() if value is not None}
    return functools.partial(node, **chosen) if chosen else node


def build_workflow(
    checkpointer: Any | None = None,
    *,
    model: Any | None = None,
    planner_loops: int | None = None,
    verifier_loops: int | None = None,
    agent_model: Any | None = None,
) -> Any:
    """Build and compile the plan/delegate/verify/final workflow.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. ``InMemorySaver()``)
            enabling persistence, ``get_state`` and resume.
        model: Chat model for the graph's own nodes; defaults to the nodes' lazy
            ``create_model()`` (``.env`` configuration).
        planner_loops: Tool rounds the planner may use per attempt (default 10).
        verifier_loops: Read-only inspection loops the verifier may use (default 6).
        agent_model: Model handed to the specialist agents (searchAgent /
            codeAgent); defaults to their own lazy ``create_model()``.

    Returns:
        The compiled graph - call ``invoke``/``stream`` with a
        :class:`~wqagent.graph.state.WQGraphState` built by
        :func:`~wqagent.graph.state.new_graph_state`.
    """
    graph = StateGraph(WQGraphState)
    graph.add_node(
        "planner", _bind(planner_node, model=model, max_loops=planner_loops, agent_model=agent_model)
    )
    graph.add_node(
        "verifier", _bind(verifier_node, model=model, max_loops=verifier_loops)
    )
    graph.add_node("final", _bind(final_node, model=model))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "verifier")
    graph.add_conditional_edges("verifier", verifier_route, ROUTE_TARGETS)
    graph.add_edge("final", END)

    if checkpointer is None:
        return graph.compile()
    return graph.compile(checkpointer=checkpointer)
