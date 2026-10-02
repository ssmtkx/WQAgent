"""LangGraph workflow for WQ Agent.

* :mod:`wqagent.graph.state` - the shared state every node reads and writes.
* :mod:`wqagent.graph.nodes` - planner / actor / verifier / final nodes and the router.
* :mod:`wqagent.graph.workflow` - :func:`build_workflow`, which wires and compiles them.

Example::

    from wqagent.core.state import RuntimeState
    from wqagent.graph import build_workflow, new_graph_state

    app = build_workflow()
    result = app.invoke(new_graph_state("write hello.py and run it", RuntimeState("./demo")))
"""

from wqagent.graph.nodes import (
    DEFAULT_ACTOR_LOOPS,
    DEFAULT_VERIFIER_LOOPS,
    VERIFY_OUTPUT_CHARS,
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
from wqagent.graph.state import (
    DEFAULT_MAX_ATTEMPTS,
    TODO_STATUSES,
    TodoItem,
    VerificationCheck,
    VerificationResult,
    WQGraphState,
    make_todo,
    new_graph_state,
    todo_progress,
)
from wqagent.graph.workflow import NODE_NAMES, ROUTE_TARGETS, build_workflow

__all__ = [
    "DEFAULT_ACTOR_LOOPS",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_VERIFIER_LOOPS",
    "NODE_NAMES",
    "ROUTE_TARGETS",
    "TODO_STATUSES",
    "VERIFY_OUTPUT_CHARS",
    "TodoItem",
    "VerificationCheck",
    "VerificationResult",
    "WQGraphState",
    "actor_node",
    "build_workflow",
    "final_node",
    "format_final_answer",
    "get_model",
    "make_todo",
    "new_graph_state",
    "planner_node",
    "run_verification_commands",
    "set_model",
    "todo_progress",
    "verifier_node",
    "verifier_route",
]
