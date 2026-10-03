"""LangGraph workflow for WQ Agent.

* :mod:`wqagent.graph.state` - the shared state every node reads and writes.
* :mod:`wqagent.graph.nodes` - planner / verifier / final nodes, the specialist
  handoff tools and the router.
* :mod:`wqagent.graph.workflow` - :func:`build_workflow`, which wires and compiles them.

Example::

    from wqagent.core.state import RuntimeState
    from wqagent.graph import build_workflow, new_graph_state

    app = build_workflow()
    result = app.invoke(new_graph_state("write hello.py and run it", RuntimeState("./demo")))

The names below are re-exported **lazily** (PEP 562). Importing this package must
not pull in ``nodes``/``workflow``: :mod:`wqagent.tools.graph_tools` imports
``wqagent.graph.state``, which first executes this ``__init__``, and an eager
import of ``nodes`` (which imports the specialist agents, which import
``graph_tools``) would re-enter ``graph_tools`` while it is still initialising.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from wqagent.graph.nodes import (  # noqa: F401
        DEFAULT_ACTOR_LOOPS,
        DEFAULT_PLANNER_LOOPS,
        DEFAULT_VERIFIER_LOOPS,
        VERIFY_OUTPUT_CHARS,
        CallCodeAgentTool,
        CallSearchAgentTool,
        actor_node,
        final_node,
        format_final_answer,
        get_model,
        planner_node,
        run_verification_commands,
        set_model,
        verifier_node,
        verifier_route,
    )
    from wqagent.graph.state import (  # noqa: F401
        DEFAULT_MAX_ATTEMPTS,
        TODO_STATUSES,
        AgentHandoff,
        SourceItem,
        TodoItem,
        VerificationCheck,
        VerificationResult,
        WQGraphState,
        make_todo,
        new_graph_state,
        todo_progress,
    )
    from wqagent.graph.workflow import NODE_NAMES, ROUTE_TARGETS, build_workflow  # noqa: F401

#: name -> module that defines it (loaded on first access).
_LAZY_EXPORTS: dict[str, str] = {
    # submodules
    "nodes": "wqagent.graph.nodes",
    "state": "wqagent.graph.state",
    "workflow": "wqagent.graph.workflow",
    # nodes
    "CallCodeAgentTool": "wqagent.graph.nodes",
    "CallSearchAgentTool": "wqagent.graph.nodes",
    "DEFAULT_ACTOR_LOOPS": "wqagent.graph.nodes",
    "DEFAULT_PLANNER_LOOPS": "wqagent.graph.nodes",
    "DEFAULT_VERIFIER_LOOPS": "wqagent.graph.nodes",
    "VERIFY_OUTPUT_CHARS": "wqagent.graph.nodes",
    "actor_node": "wqagent.graph.nodes",
    "final_node": "wqagent.graph.nodes",
    "format_final_answer": "wqagent.graph.nodes",
    "get_model": "wqagent.graph.nodes",
    "planner_node": "wqagent.graph.nodes",
    "run_verification_commands": "wqagent.graph.nodes",
    "set_model": "wqagent.graph.nodes",
    "verifier_node": "wqagent.graph.nodes",
    "verifier_route": "wqagent.graph.nodes",
    # state
    "AgentHandoff": "wqagent.graph.state",
    "DEFAULT_MAX_ATTEMPTS": "wqagent.graph.state",
    "SourceItem": "wqagent.graph.state",
    "TODO_STATUSES": "wqagent.graph.state",
    "TodoItem": "wqagent.graph.state",
    "VerificationCheck": "wqagent.graph.state",
    "VerificationResult": "wqagent.graph.state",
    "WQGraphState": "wqagent.graph.state",
    "make_todo": "wqagent.graph.state",
    "new_graph_state": "wqagent.graph.state",
    "todo_progress": "wqagent.graph.state",
    # workflow
    "NODE_NAMES": "wqagent.graph.workflow",
    "ROUTE_TARGETS": "wqagent.graph.workflow",
    "build_workflow": "wqagent.graph.workflow",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Import and cache a re-exported name on first access."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # cache, so later lookups skip this hook
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
