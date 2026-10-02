"""Prompt templates, grouped by workflow stage.

* :mod:`wqagent.prompts.stage1` - the actor node of the ReAct workflow
  (:data:`~wqagent.prompts.stage1.ACTOR_PROMPT`).
"""

from wqagent.prompts.stage1 import ACTOR_PROMPT

__all__ = ["ACTOR_PROMPT"]
