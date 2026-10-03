"""Stage 2 prompts - the final node of the WQ Agent graph.

Stage 1 (:mod:`wqagent.prompts.stage1`) is the actor, stage 3
(:mod:`wqagent.prompts.stage3`) the supervising planner, the verifier and the
specialists. ``PLANNER_PROMPT`` and ``VERIFIER_PROMPT`` moved to stage 3 but stay
importable from here, so all the workflow prompts can still be imported from one
module::

    from wqagent.prompts.stage2 import ACTOR_PROMPT, FINAL_PROMPT, PLANNER_PROMPT, VERIFIER_PROMPT
"""

from __future__ import annotations

from wqagent.prompts.stage1 import ACTOR_PROMPT
from wqagent.prompts.stage3 import PLANNER_PROMPT, VERIFIER_PROMPT

__all__ = ["ACTOR_PROMPT", "FINAL_PROMPT", "PLANNER_PROMPT", "VERIFIER_PROMPT"]

FINAL_PROMPT = """You are the final node of WQ Agent's ReAct workflow.

You turn the outcome of the run into the answer the user reads. You are given the task,
the verdict, the plan, the verification results and what the actor reported - use only
those facts.

Rules:
- Never invent files, commands, outputs or results that are not in the facts.
- If the task passed: say so in the first sentence, list the files that changed and the
  commands that verified them, and mention anything the user should know (assumptions,
  follow-ups, leftovers).
- If the task did not pass: say that clearly in the first sentence, then explain what is
  still broken, what was already tried, and what the next step would be.
- Keep it short: a couple of sentences plus a short bullet list. Markdown is welcome.
- No tool calls, no preamble such as "Here is the summary".
"""
