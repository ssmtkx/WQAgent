"""Stage 4 prompts - context-window management.

Stage 1 is the (pre-stage-3) actor, stage 2 the final node, stage 3 the
supervising planner / verifier / specialists. Stage 4 keeps a long run inside the
model's window:

* :data:`CONTEXT_COMPRESSION_PROMPT` - the compressor node
  (:func:`wqagent.graph.nodes.context_compressor_node`), which rewrites the message
  transcript into one compact summary plus structured state.
"""

from __future__ import annotations

__all__ = ["CONTEXT_COMPRESSION_PROMPT"]

CONTEXT_COMPRESSION_PROMPT = """You are the context_compressor node in WQ Agent stage 4.

Your job is to compress the graph context so the task can continue with a much
smaller message window.

Keep everything needed to resume work:
- user task and active goal
- current plan, todos, acceptance criteria, verification commands
- completed work and current files/artifacts
- important tool findings and command results
- research notes and source URLs
- latest verifier failure and recommended next step
- risks, blockers, and assumptions

Remove redundant transcript detail:
- repeated tool calls
- long stdout/stderr
- duplicate search snippets
- stale intermediate reasoning

Return only JSON with these keys:
- summary
- active_goal
- completed_work
- open_todos
- important_files
- tool_findings
- sources
- next_steps
- risks
"""
