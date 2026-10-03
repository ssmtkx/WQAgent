"""Prompt templates, grouped by workflow stage.

* :mod:`wqagent.prompts.stage1` - the (pre-stage-3) actor node
  (:data:`~wqagent.prompts.stage1.ACTOR_PROMPT`).
* :mod:`wqagent.prompts.stage2` - the final node
  (:data:`~wqagent.prompts.stage2.FINAL_PROMPT`).
* :mod:`wqagent.prompts.stage3` - the supervising planner, the verifier and the
  specialist agents (:data:`~wqagent.prompts.stage3.PLANNER_PROMPT`,
  :data:`~wqagent.prompts.stage3.VERIFIER_PROMPT`,
  :data:`~wqagent.prompts.stage3.SEARCH_AGENT_PROMPT`,
  :data:`~wqagent.prompts.stage3.CODE_AGENT_PROMPT`).
* :mod:`wqagent.prompts.stage4` - context-window management
  (:data:`~wqagent.prompts.stage4.CONTEXT_COMPRESSION_PROMPT`).
"""

from wqagent.prompts.stage1 import ACTOR_PROMPT
from wqagent.prompts.stage2 import FINAL_PROMPT
from wqagent.prompts.stage3 import (
    CODE_AGENT_PROMPT,
    PLANNER_PROMPT,
    SEARCH_AGENT_PROMPT,
    VERIFIER_PROMPT,
)
from wqagent.prompts.stage4 import CONTEXT_COMPRESSION_PROMPT

__all__ = [
    "ACTOR_PROMPT",
    "CODE_AGENT_PROMPT",
    "CONTEXT_COMPRESSION_PROMPT",
    "FINAL_PROMPT",
    "PLANNER_PROMPT",
    "SEARCH_AGENT_PROMPT",
    "VERIFIER_PROMPT",
]
