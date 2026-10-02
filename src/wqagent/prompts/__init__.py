"""Prompt templates, grouped by workflow stage.

* :mod:`wqagent.prompts.stage1` - the actor node of the ReAct workflow
  (:data:`~wqagent.prompts.stage1.ACTOR_PROMPT`).
* :mod:`wqagent.prompts.stage2` - the planner, verifier and final nodes around
  it (:data:`~wqagent.prompts.stage2.PLANNER_PROMPT`,
  :data:`~wqagent.prompts.stage2.VERIFIER_PROMPT`,
  :data:`~wqagent.prompts.stage2.FINAL_PROMPT`).

All four are importable from :mod:`wqagent.prompts.stage2` as well.
"""

from wqagent.prompts.stage1 import ACTOR_PROMPT
from wqagent.prompts.stage2 import (
    FINAL_PROMPT,
    PLANNER_PROMPT,
    VERIFIER_PROMPT,
)

__all__ = ["ACTOR_PROMPT", "FINAL_PROMPT", "PLANNER_PROMPT", "VERIFIER_PROMPT"]
