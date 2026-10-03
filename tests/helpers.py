"""Shared test doubles and scripted responses.

Since stage 3 the planner supervises two specialist agents through tools, so a
full run needs a longer script than before (planner -> codeAgent -> planner ->
verifier -> final). These builders keep the test files readable.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

__all__ = [
    "ScriptedModel",
    "answer",
    "code_call",
    "passing_script",
    "plan_call",
    "report_call",
    "search_call",
    "todo_update_call",
    "tool_call",
    "write_call",
]


class ScriptedModel:
    """Replays scripted responses; records what was bound and what it received."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.bound: list[list[str]] = []
        self.seen: list[list[Any]] = []

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "ScriptedModel":
        self.bound.append([tool.name for tool in tools])
        return self

    def invoke(self, messages: list[Any]) -> Any:
        self.seen.append(list(messages))
        if not self.responses:
            raise AssertionError("ScriptedModel ran out of scripted responses")
        return self.responses.pop(0)

    @property
    def last_message(self) -> Any:
        return self.seen[-1][-1] if self.seen else None


def tool_call(name: str, args: dict[str, Any], call_id: str = "c1") -> AIMessage:
    """One assistant turn that calls *name*."""
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}]
    )


def answer(text: str) -> AIMessage:
    """A plain assistant turn (no tool calls)."""
    return AIMessage(content=text)


def plan_call(
    *,
    summary: str = "do the thing",
    todos: list[dict[str, Any]] | None = None,
    criteria: list[str] | None = None,
    commands: list[str] | None = None,
    **overrides: Any,
) -> AIMessage:
    """A planner ``todo_write`` call."""
    payload: dict[str, Any] = {
        "plan_summary": summary,
        "todos": [{"id": "t1", "content": "do the thing"}] if todos is None else todos,
        "acceptance_criteria": ["it works"] if criteria is None else criteria,
        "verification_commands": ["echo ok"] if commands is None else commands,
    }
    payload.update(overrides)
    return tool_call("todo_write", payload)


def search_call(instruction: str = "look up the version", call_id: str = "c2") -> AIMessage:
    """A planner ``call_search_agent`` call."""
    return tool_call("call_search_agent", {"instruction": instruction}, call_id)


def code_call(instruction: str = "implement it", call_id: str = "c3") -> AIMessage:
    """A planner ``call_code_agent`` call."""
    return tool_call("call_code_agent", {"instruction": instruction}, call_id)


def write_call(
    file_path: str = "hello.py",
    content: str = "print('Hello')\n",
    call_id: str = "c4",
) -> AIMessage:
    """A codeAgent ``file_write`` call."""
    return tool_call("file_write", {"file_path": file_path, "content": content}, call_id)


def todo_update_call(*updates: dict[str, Any], call_id: str = "c5") -> AIMessage:
    """A codeAgent ``todo_update`` call."""
    return tool_call("todo_update", {"updates": list(updates)}, call_id)


def report_call(passed: bool = True, **overrides: Any) -> AIMessage:
    """A verifier ``verification_report`` call."""
    payload: dict[str, Any] = {
        "passed": passed,
        "reason": "checked the workspace" if passed else "the command failed",
        "checks": [{"name": "it works", "passed": passed, "detail": "looked at it"}],
        "recommended_next_instruction": "" if passed else "fix it",
    }
    payload.update(overrides)
    return tool_call("verification_report", payload, "c9")


def passing_script(
    *,
    instruction: str = "write hello.py",
    file_path: str = "hello.py",
    content: str = "print('Hello')\n",
    commands: list[str] | None = None,
    plan_overrides: dict[str, Any] | None = None,
    final_text: str | None = None,
) -> list[Any]:
    """The eight responses a passing planner -> codeAgent -> verifier -> final run needs."""
    return [
        plan_call(commands=commands, **(plan_overrides or {})),
        code_call(instruction),
        write_call(file_path, content),
        answer(f"Wrote {file_path}."),
        answer("Delegated the implementation to codeAgent."),
        answer("Inspected the workspace."),
        report_call(True),
        answer(final_text or f"Done: {file_path} written and verified."),
    ]
