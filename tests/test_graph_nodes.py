"""Tests for the graph nodes: planner, actor, verifier and the router."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from wqagent.core.state import RuntimeState
from wqagent.graph.memory import (
    DEFAULT_CONTEXT_TOKEN_LIMIT,
    HANDOFF_TEXT_CHARS,
    NOTEPAD_FILE_CHARS,
    RESEARCH_CHARS,
    SUMMARY_CHARS,
    read_todos,
    write_todos,
)
from wqagent.graph.nodes import (
    COMPRESS_TRANSCRIPT_CHARS,
    actor_node,
    context_compressor_node,
    context_compressor_route,
    context_monitor_node,
    context_monitor_route,
    get_model,
    planner_node,
    run_verification_commands,
    set_model,
    verifier_node,
    verifier_route,
)
from wqagent.graph.state import make_todo, new_graph_state
from wqagent.prompts.stage1 import ACTOR_PROMPT
from wqagent.prompts.stage3 import PLANNER_PROMPT, VERIFIER_PROMPT
from wqagent.prompts.stage4 import CONTEXT_COMPRESSION_PROMPT
from wqagent.tools.bash_tool import SHELL_NAME, shell_guidance


class ScriptedModel:
    """Replays scripted responses; records what was bound and what it received."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.bound: list[tuple[list[str], Any]] = []
        self.seen: list[list[Any]] = []

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "ScriptedModel":
        self.bound.append(([tool.name for tool in tools], kwargs.get("tool_choice")))
        return self

    def invoke(self, messages: list[Any]) -> Any:
        self.seen.append(list(messages))
        if not self.responses:
            raise AssertionError("ScriptedModel ran out of scripted responses")
        return self.responses.pop(0)


def _tool_call(name: str, args: dict[str, Any], call_id: str = "c1") -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}]
    )


def _state(tmp_path: Path, **extra: Any):
    return new_graph_state("write hello.py", RuntimeState(workspace=tmp_path / "ws"), **extra)


@pytest.fixture(autouse=True)
def _reset_module_model():
    """Keep the module-level model from leaking between tests."""
    set_model(None)
    yield
    set_model(None)


# -- planner (supervisor) ----------------------------------------------------
def test_planner_creates_the_plan(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            _tool_call(
                "todo_write",
                {
                    "plan_summary": "write the file, then run it",
                    "todos": [
                        {"id": "t1", "content": "write hello.py"},
                        {"id": "t2", "content": "run it", "note": "python hello.py"},
                    ],
                    "acceptance_criteria": ["hello.py prints Hello"],
                    "verification_commands": ["python hello.py"],
                },
            ),
            AIMessage(content="Planned two steps; nothing to delegate yet."),
        ]
    )

    update = planner_node(_state(tmp_path), model=model)

    assert update["plan_summary"] == "write the file, then run it"
    assert [todo["id"] for todo in update["todos"]] == ["t1", "t2"]
    assert [todo["status"] for todo in update["todos"]] == ["pending", "pending"]
    assert update["todos"][1]["note"] == "python hello.py"
    assert update["acceptance_criteria"] == ["hello.py prints Hello"]
    assert update["verification_commands"] == ["python hello.py"]
    assert update["last_actor_summary"] == "Planned two steps; nothing to delegate yet."
    assert update["agent_handoffs"] == []

    # the supervisor binds its three tools - no forced tool_choice any more
    names, tool_choice = model.bound[0]
    assert names == ["todo_write", "call_search_agent", "call_code_agent"]
    assert tool_choice is None
    sent = model.seen[0]
    assert sent[0].content == PLANNER_PROMPT
    assert "write hello.py" in sent[1].content


def test_planner_revises_the_plan_after_a_failure(tmp_path: Path) -> None:
    state = _state(
        tmp_path,
        todos=[make_todo("t1", "write hello.py", status="completed"), make_todo("t2", "run it")],
        last_error="command failed: python hello.py (exit code 1)\n  NameError: name 'hello' is not defined",
    )
    model = ScriptedModel(
        [
            _tool_call(
                "todo_write",
                {
                    "plan_summary": "fix the NameError, then rerun",
                    "todos": [
                        {"id": "t1", "content": "write hello.py", "status": "completed"},
                        {"id": "t2", "content": "run it", "status": "pending"},
                        {"id": "t3", "content": "fix the NameError in hello.py"},
                    ],
                    "acceptance_criteria": ["hello.py prints Hello"],
                    "verification_commands": ["python hello.py"],
                },
            ),
            AIMessage(content="Revised the plan and delegated the fix."),
        ]
    )

    update = planner_node(state, model=model)

    prompt = model.seen[0][1].content
    assert "FAILED" in prompt
    assert "exit code 1" in prompt and "NameError" in prompt
    assert "[x] t1" in prompt  # the completed todo is shown as completed
    assert "delegate only the missing fix" in prompt
    assert [todo["id"] for todo in update["todos"]] == ["t1", "t2", "t3"]
    assert update["todos"][0]["status"] == "completed"


def test_planner_delegates_research_then_implementation(tmp_path: Path) -> None:
    """The supervisor handoff: searchAgent first, codeAgent second."""
    from wqagent.tools.web_search_tool import WebSearchTool

    class FakeSearchTool(WebSearchTool):
        def __init__(self) -> None:
            super().__init__(api_key="fake")
            self.queries: list[str] = []

        def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
            self.queries.append(query)
            return {
                "ok": True,
                "query": query,
                "answer": "FastAPI 0.115 is current.",
                "results": [
                    {
                        "title": "FastAPI",
                        "url": "https://fastapi.tiangolo.com/",
                        "content": "docs",
                        "score": 0.9,
                    }
                ],
                "error": "",
            }

    search_tool = FakeSearchTool()
    model = ScriptedModel(
        [
            _tool_call(
                "todo_write",
                {
                    "plan_summary": "research then build",
                    "todos": [{"id": "t1", "content": "pin fastapi"}],
                },
            ),
            _tool_call("call_search_agent", {"instruction": "which fastapi version?"}, "c2"),
            # the search agent runs on the same model
            _tool_call("web_search", {"query": "which fastapi version?"}, "s1"),
            AIMessage(content="FastAPI 0.115 is current."),
            _tool_call("call_code_agent", {"instruction": "pin fastapi==0.115"}, "c3"),
            _tool_call("file_write", {"file_path": "requirements.txt", "content": "fastapi==0.115\n"}),
            AIMessage(content="Wrote requirements.txt."),
            AIMessage(content="Researched the version and pinned it."),
        ]
    )
    events: list[dict[str, Any]] = []

    update = planner_node(
        _state(tmp_path), model=model, agent_model=model, search_tool=search_tool,
        on_event=events.append,
    )

    assert search_tool.queries == ["which fastapi version?"]
    assert (tmp_path / "ws" / "requirements.txt").read_text(encoding="utf-8") == "fastapi==0.115\n"

    # research effects landed in the state update
    assert "FastAPI 0.115 is current." in update["research_notes"]
    assert "https://fastapi.tiangolo.com/" in update["research_notes"]
    assert [source["url"] for source in update["sources"]] == ["https://fastapi.tiangolo.com/"]

    # both handoffs were logged, with their instructions and results
    assert [(h["from_agent"], h["to_agent"]) for h in update["agent_handoffs"]] == [
        ("planner", "searchAgent"),
        ("planner", "codeAgent"),
    ]
    assert update["agent_handoffs"][0]["instruction"] == "which fastapi version?"
    assert update["agent_handoffs"][1]["result"] == "Wrote requirements.txt."

    # the implementer's summary and transcript are kept
    assert update["code_agent_summary"] == "Wrote requirements.txt."
    assert update["todos"][0]["status"] == "pending"  # nobody updated the todos
    assert update["messages"]  # codeAgent's transcript

    # the handoff events were streamed, tagged with the specialist that ran
    kinds = [(event["type"], event.get("node")) for event in events]
    assert ("handoff", "planner") in kinds
    assert ("tool_call", "searchAgent") in kinds
    assert ("tool_call", "codeAgent") in kinds
    handoffs = [event for event in events if event["type"] == "handoff"]
    assert handoffs[0]["to"] == "searchAgent"
    assert handoffs[1]["to"] == "codeAgent"
    assert handoffs[1]["instruction"] == "pin fastapi==0.115"


def test_planner_appends_to_the_existing_handoff_log(tmp_path: Path) -> None:
    state = _state(
        tmp_path,
        agent_handoffs=[{"from_agent": "planner", "to_agent": "searchAgent", "instruction": "old"}],
    )
    model = ScriptedModel(
        [
            _tool_call("call_code_agent", {"instruction": "do it"}, "c1"),
            AIMessage(content="codeAgent did it."),
            AIMessage(content="done"),
        ]
    )

    update = planner_node(state, model=model, agent_model=model)

    assert [h["instruction"] for h in update["agent_handoffs"]] == ["old", "do it"]


def test_planner_keeps_the_previous_todos_when_the_model_answers_nothing(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path, todos=[make_todo("t1", "keep me")], last_error="boom")
    model = ScriptedModel([AIMessage(content="I have no idea")])

    update = planner_node(state, model=model)

    assert [todo["id"] for todo in update["todos"]] == ["t1"]


def test_planner_requires_a_runtime() -> None:
    with pytest.raises(ValueError, match="runtime"):
        planner_node({"task": "x"})  # type: ignore[arg-type]


# -- actor -------------------------------------------------------------------
def test_actor_executes_the_plan_and_records_todo_updates(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    state = new_graph_state(
        "write hello.py",
        RuntimeState(workspace=workspace),
        plan_summary="write, then run",
        todos=[make_todo("t1", "write hello.py"), make_todo("t2", "run it")],
        acceptance_criteria=["prints Hello"],
        verification_commands=["echo done"],
    )
    model = ScriptedModel(
        [
            _tool_call("file_write", {"file_path": "hello.py", "content": "print('Hello')\n"}),
            _tool_call(
                "todo_update",
                {"updates": [{"id": "t1", "status": "completed", "note": "written"}]},
            ),
            AIMessage(content="Wrote hello.py and ran it."),
        ]
    )
    events: list[dict[str, Any]] = []

    update = actor_node(state, model=model, on_event=events.append)

    assert (workspace / "hello.py").read_text(encoding="utf-8") == "print('Hello')\n"
    assert update["last_actor_summary"] == "Wrote hello.py and ran it."
    assert {todo["id"]: todo["status"] for todo in update["todos"]} == {
        "t1": "completed",
        "t2": "pending",
    }
    assert update["todos"][0]["note"] == "written"

    # only this node's contribution goes back into the graph transcript
    kinds = [message.type for message in update["messages"]]
    assert kinds[0] == "human" and "system" not in kinds
    assert any(isinstance(message, ToolMessage) for message in update["messages"])

    # the actor bound the five workspace tools plus todo_update
    names, _ = model.bound[0]
    assert names == ["file_read", "file_write", "file_edit", "grep", "bash", "todo_update"]
    assert model.seen[0][0].content == ACTOR_PROMPT

    # every ReAct event was forwarded
    assert [event["type"] for event in events][0] == "ai_message"
    assert [event["name"] for event in events if event["type"] == "tool_call"] == [
        "file_write",
        "todo_update",
    ]
    assert "execute the plan" in model.seen[0][2 - 1].content.lower()


def test_actor_replays_the_transcript_but_returns_only_the_delta(tmp_path: Path) -> None:
    state = _state(
        tmp_path,
        messages=[HumanMessage(content="first attempt"), AIMessage(content="did something")],
    )
    model = ScriptedModel([AIMessage(content="done now")])

    update = actor_node(state, model=model)

    sent = model.seen[0]
    assert sent[0].content == ACTOR_PROMPT
    assert [message.content for message in sent[1:3]] == ["first attempt", "did something"]
    assert [message.type for message in update["messages"]] == ["human", "ai"]
    assert update["messages"][-1].content == "done now"


def test_actor_reports_the_step_limit_in_the_summary(tmp_path: Path) -> None:
    model = ScriptedModel([_tool_call("bash", {"command": "echo tick"}) for _ in range(3)])

    update = actor_node(_state(tmp_path), model=model, max_loops=3)

    assert "Stopped after 3 steps" in update["last_actor_summary"]


def test_actor_ignores_updates_for_unknown_todos(tmp_path: Path) -> None:
    state = _state(tmp_path, todos=[make_todo("t1", "do it")])
    model = ScriptedModel(
        [
            _tool_call("todo_update", {"updates": [{"id": "ghost", "status": "completed"}]}),
            AIMessage(content="done"),
        ]
    )

    update = actor_node(state, model=model)

    assert [todo["id"] for todo in update["todos"]] == ["t1"]
    assert update["todos"][0]["status"] == "pending"
    assert "unknown todo id 'ghost'" in update["last_actor_summary"]


# -- verifier ----------------------------------------------------------------
def test_verifier_passes_and_marks_the_plan_completed(tmp_path: Path) -> None:
    state = _state(
        tmp_path,
        plan_summary="write, then run",
        todos=[make_todo("t1", "write hello.py"), make_todo("t2", "run it")],
        acceptance_criteria=["prints Hello"],
        verification_commands=["echo ok"],
    )
    model = ScriptedModel(
        [
            _tool_call("file_read", {"file_path": "hello.py"}),  # read-only inspection
            AIMessage(content="I read the file."),
            _tool_call(
                "verification_report",
                {
                    "passed": True,
                    "reason": "the file exists and the command succeeded",
                    "checks": [{"name": "prints Hello", "passed": True, "detail": "saw print"}],
                    "recommended_next_instruction": "",
                },
            ),
        ]
    )

    update = verifier_node(state, model=model)

    assert update["passed"] is True
    assert update["attempts"] == 1
    assert update["last_error"] == ""
    assert [todo["status"] for todo in update["todos"]] == ["completed", "completed"]
    assert update["verification_results"][0]["command"] == "echo ok"
    assert update["verification_results"][0]["ok"] is True
    assert update["verification_checks"][0]["name"] == "prints Hello"

    # the inspection agent only got read-only tools...
    assert model.bound[0][0] == ["file_read", "grep"]
    # ...and the verdict came from a forced verification_report call
    assert model.bound[-1] == (["verification_report"], "verification_report")
    assert model.seen[0][0].content == VERIFIER_PROMPT
    assert "exit_code=0" in model.seen[0][1].content


def test_verifier_survives_a_tool_its_prompt_names_but_it_does_not_have(
    tmp_path: Path,
) -> None:
    """The prompt mentions NotepadReadTool / web_search; a call must only degrade."""
    state = _state(tmp_path, todos=[make_todo("t1", "do it")], verification_commands=["echo ok"])
    model = ScriptedModel(
        [
            _tool_call("NotepadReadTool", {}, "v1"),
            _tool_call("web_search", {"query": "anything"}, "v2"),
            AIMessage(content="no notepad and no web tool here; judging from the files"),
            _tool_call(
                "verification_report",
                {
                    "passed": True,
                    "reason": "the file is there and the command succeeded",
                    "checks": [{"name": "it works", "passed": True, "detail": "exit 0"}],
                    "recommended_next_instruction": "",
                },
            ),
        ]
    )

    update = verifier_node(state, model=model)

    # unknown tools are fed back as errors, the verdict still lands
    assert update["passed"] is True
    assert model.seen[1][-1].type == "tool"  # the NotepadReadTool failure came back
    assert "unknown tool" in model.seen[1][-1].content
    assert "unknown tool" in model.seen[2][-1].content  # ...and the web_search one


def test_verifier_overrules_a_pass_when_a_command_fails(tmp_path: Path) -> None:
    state = _state(tmp_path, todos=[make_todo("t1", "do it")], verification_commands=["exit 3"])
    model = ScriptedModel(
        [
            AIMessage(content="looks fine to me"),
            _tool_call(
                "verification_report",
                {
                    "passed": True,
                    "reason": "the actor said it is done",
                    "checks": [],
                    "recommended_next_instruction": "",
                },
            ),
        ]
    )

    update = verifier_node(state, model=model)

    assert update["passed"] is False
    assert "overruled" in update["last_error"]
    assert update["verification_results"][0]["exit_code"] == 3
    assert update["verification_results"][0]["ok"] is False
    assert [todo["status"] for todo in update["todos"]] == ["pending"]


def test_verifier_records_the_next_instruction_on_failure(tmp_path: Path) -> None:
    state = _state(tmp_path, verification_commands=["echo boom; exit 1"])
    model = ScriptedModel(
        [
            AIMessage(content="inspecting"),
            _tool_call(
                "verification_report",
                {
                    "passed": False,
                    "reason": "the command failed",
                    "checks": [{"name": "runs cleanly", "passed": False, "detail": "exit 1"}],
                    "recommended_next_instruction": "fix the import error in hello.py",
                },
            ),
        ]
    )

    update = verifier_node(state, model=model)

    assert update["passed"] is False
    assert "the command failed" in update["last_error"]
    assert "command failed: echo boom; exit 1 (exit code 1)" in update["last_error"]
    assert "Recommended next instruction: fix the import error in hello.py" in update["last_error"]
    assert update["verification_checks"][0]["passed"] is False
    assert update["attempts"] == 1


def test_verifier_fails_safely_without_a_verdict(tmp_path: Path) -> None:
    model = ScriptedModel(
        [AIMessage(content="not sure"), AIMessage(content="still not sure"), AIMessage(content="nope")]
    )

    update = verifier_node(_state(tmp_path), model=model)

    assert update["passed"] is False
    assert "no verdict" in update["last_error"]


def test_run_verification_commands_records_outcomes(tmp_path: Path) -> None:
    runtime = RuntimeState(workspace=tmp_path / "ws")

    results = run_verification_commands(
        runtime, ["echo hi", "exit 2", "   ", "rm -rf /"]
    )

    assert len(results) == 3  # blanks are skipped
    assert results[0]["ok"] is True and results[0]["exit_code"] == 0
    assert "hi" in results[0]["stdout"]
    assert results[1]["ok"] is False and results[1]["exit_code"] == 2
    assert results[2]["ok"] is False and "safety filter" in results[2]["stderr"]


# -- routing -----------------------------------------------------------------
def test_verifier_route() -> None:
    assert verifier_route({"passed": True, "attempts": 1, "max_attempts": 3}) == "final"
    assert verifier_route({"passed": False, "attempts": 1, "max_attempts": 3}) == "planner"
    assert verifier_route({"passed": False, "attempts": 3, "max_attempts": 3}) == "final"
    assert verifier_route({"passed": False, "attempts": 9, "max_attempts": 3}) == "final"
    assert verifier_route({"passed": False}) == "planner"  # attempts defaults to 0 < 3
    assert verifier_route({}) == "planner"


# -- context monitor ---------------------------------------------------------
class CountingModel:
    """A model whose token counter is scripted (or broken, for the fallback)."""

    def __init__(self, count: Any = 10) -> None:
        self.count = count
        self.seen: list[list[Any]] = []

    def get_num_tokens_from_messages(self, messages: list[Any]) -> Any:
        self.seen.append(list(messages))
        if isinstance(self.count, Exception):
            raise self.count
        return self.count


def test_context_monitor_counts_with_the_models_tokenizer(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state["messages"] = [HumanMessage(content="do the thing")]
    model = CountingModel(12_345)

    update = context_monitor_node(state, model=model)

    assert update == {
        "context_token_count": 12_345,
        "context_should_compress": False,
        "context_next_node": "verifier",  # default
    }
    # the model was asked to count the conversation *plus* the memory payload
    counted = model.seen[0]
    assert [message.type for message in counted] == ["human", "system"]
    assert "Layered memory:" in counted[-1].content
    assert '"rules"' in counted[-1].content


def test_context_monitor_falls_back_to_characters(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state["messages"] = [HumanMessage(content="x" * 4_000)]
    model = CountingModel(RuntimeError("no tokenizer here"))

    update = context_monitor_node(state, model=model)

    # exactly len(text) // 4 over the message text plus the memory payload
    text = "x" * 4_000 + "\n" + model.seen[0][-1].content
    assert update["context_token_count"] == len(text) // 4
    assert update["context_token_count"] > 1_000
    assert update["context_should_compress"] is False


def test_context_monitor_falls_back_without_the_method(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state["messages"] = [HumanMessage(content="y" * 800)]

    update = context_monitor_node(state, model=ScriptedModel([]))

    assert update["context_token_count"] >= 200


def test_context_monitor_handles_a_useless_count(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state["messages"] = [HumanMessage(content="z" * 400)]

    for count in (None, -5, "not a number"):
        update = context_monitor_node(state, model=CountingModel(count))
        assert update["context_token_count"] >= 100, count


def test_context_monitor_flags_an_over_budget_context(tmp_path: Path) -> None:
    state = _state(tmp_path)
    model = CountingModel(5_000)

    over = context_monitor_node(state, model=model)
    assert over["context_should_compress"] is False

    state["context_token_limit"] = 4_999
    assert context_monitor_node(state, model=model)["context_should_compress"] is True

    state["context_token_limit"] = 5_000  # exactly at the limit is not over it
    assert context_monitor_node(state, model=model)["context_should_compress"] is False


def test_context_monitor_uses_the_default_limit_and_echoes_the_next_node(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state["context_next_node"] = "planner"

    update = context_monitor_node(state, model=CountingModel(1_000))

    assert update["context_next_node"] == "planner"
    assert DEFAULT_CONTEXT_TOKEN_LIMIT == 400_000
    # a huge count still does not compress without an explicit smaller limit
    assert context_monitor_node(state, model=CountingModel(399_999))["context_should_compress"] is False
    assert context_monitor_node(state, model=CountingModel(400_001))["context_should_compress"] is True


def test_context_monitor_reuses_the_stored_snapshot(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state["memory_snapshot"] = {
        "rules": {"scope": "workspace", "storage": "internal", "rules": ["stored"]},
        "working_memory": {"node": "planner", "task": "stored task"},
        "history_summary_store": {},
    }
    model = CountingModel(7)

    context_monitor_node(state, model=model)

    payload = model.seen[0][-1].content
    assert "stored task" in payload
    assert "stored" in payload


@pytest.mark.parametrize("limit", ["abc", None, "", 0, {"nope": 1}])
def test_context_monitor_survives_a_junk_limit(limit: Any, tmp_path: Path) -> None:
    state = _state(tmp_path)
    state["context_token_limit"] = limit  # type: ignore[typeddict-item]

    update = context_monitor_node(state, model=CountingModel(1_000))

    # unparsable (or unset) budgets fall back to the default, they never raise
    assert update["context_should_compress"] is False
    assert update["context_token_count"] == 1_000


def test_context_monitor_needs_a_runtime() -> None:
    with pytest.raises(ValueError, match="RuntimeState"):
        context_monitor_node({"task": "no runtime"}, model=CountingModel(1))  # type: ignore[arg-type]


def test_context_monitor_route() -> None:
    assert context_monitor_route({"passed": True}) == "final"
    # a pass wins over an over-budget context and a spent budget
    assert (
        context_monitor_route({"passed": True, "context_should_compress": True, "attempts": 9})
        == "final"
    )
    # a spent attempt budget stops the run instead of looping planner -> verifier
    assert context_monitor_route({"passed": False, "attempts": 3, "max_attempts": 3}) == "final"
    assert (
        context_monitor_route(
            {"attempts": 3, "max_attempts": 3, "context_should_compress": True}
        )
        == "final"
    )
    assert context_monitor_route({"context_should_compress": True}) == "context_compressor"
    assert context_monitor_route({"attempts": 1, "max_attempts": 3, "context_next_node": "planner"}) == "planner"
    assert context_monitor_route({"context_next_node": "verifier"}) == "verifier"
    # unset / empty -> the safe default
    assert context_monitor_route({}) == "verifier"
    assert context_monitor_route({"context_next_node": "  "}) == "verifier"


def test_context_compressor_route() -> None:
    assert context_compressor_route({"context_next_node": "verifier"}) == "verifier"
    assert context_compressor_route({"context_next_node": "planner"}) == "planner"
    assert context_compressor_route({"context_next_node": "final"}) == "final"
    # missing or blank -> verification is what normally follows a compression
    assert context_compressor_route({}) == "verifier"
    assert context_compressor_route({"context_next_node": None}) == "verifier"  # type: ignore[typeddict-item]


def test_context_monitor_does_not_compress_twice_without_progress(tmp_path: Path) -> None:
    """An unreachable limit must not loop monitor -> compressor for ever."""
    state = _state(tmp_path, context_token_limit=1)
    model = CountingModel(5_000)

    first = context_monitor_node(state, model=model)
    assert first["context_should_compress"] is True

    # the compressor tried and landed at 5_000 tokens - still over the limit
    state["compression_events"] = [{"tokens_before": 9_000, "tokens_after": 5_000}]
    state["context_token_limit"] = 1
    assert context_monitor_node(state, model=CountingModel(5_000))["context_should_compress"] is False
    # ...even after new work grew the context again: the floor is still too high
    assert context_monitor_node(state, model=CountingModel(6_000))["context_should_compress"] is False

    # a compression that DID fit the budget keeps the mechanism alive for later growth
    state["compression_events"] = [{"tokens_before": 9_000, "tokens_after": 400}]
    state["context_token_limit"] = 1_000
    assert context_monitor_node(state, model=CountingModel(500))["context_should_compress"] is False
    assert context_monitor_node(state, model=CountingModel(1_500))["context_should_compress"] is True


def test_context_monitor_ignores_junk_compression_events(tmp_path: Path) -> None:
    state = _state(tmp_path, context_token_limit=1)
    state["compression_events"] = ["junk", {"tokens_after": "nope"}]  # type: ignore[list-item]

    update = context_monitor_node(state, model=CountingModel(5_000))

    assert update["context_should_compress"] is True  # cannot prove futility -> compress


def test_planner_and_verifier_direct_the_context_monitor(tmp_path: Path) -> None:
    plan_model = ScriptedModel(
        [
            _tool_call("todo_write", {"plan_summary": "p", "todos": [{"id": "t1", "content": "c"}]}),
            AIMessage(content="planned"),
        ]
    )
    assert planner_node(_state(tmp_path), model=plan_model)["context_next_node"] == "verifier"

    failing = ScriptedModel(
        [
            AIMessage(content="inspected"),
            _tool_call(
                "verification_report",
                {"passed": False, "reason": "nope", "checks": [], "recommended_next_instruction": "fix"},
            ),
        ]
    )
    failed = verifier_node(_state(tmp_path, verification_commands=["exit 1"]), model=failing)
    assert failed["passed"] is False
    assert failed["context_next_node"] == "planner"

    passing = ScriptedModel(
        [
            AIMessage(content="inspected"),
            _tool_call(
                "verification_report",
                {"passed": True, "reason": "fine", "checks": [], "recommended_next_instruction": ""},
            ),
        ]
    )
    verified = verifier_node(_state(tmp_path, verification_commands=["echo ok"]), model=passing)
    assert verified["passed"] is True
    assert "context_next_node" not in verified  # nothing follows but the final node


# -- context compressor ------------------------------------------------------
COMPRESSION_PAYLOAD: dict[str, Any] = {
    "summary": "compressed: hello.py is written, verification still fails",
    "active_goal": "make python hello.py exit 0",
    "completed_work": "hello.py written (22 bytes)",
    "open_todos": ["t1: rerun the program"],
    "important_files": ["hello.py"],
    "tool_findings": ["python 3.12.7 is on PATH"],
    "sources": ["https://docs.test/python"],
    "next_steps": ["run python hello.py and check exit code"],
    "risks": "the verifier command exits 1",
}


def _compression_json(payload: Any = None) -> str:
    if payload is None:
        payload = COMPRESSION_PAYLOAD
    return payload if isinstance(payload, str) else json.dumps(payload)


class CompressorModel:
    """Returns a scripted answer to the compression request and counts tokens."""

    def __init__(self, answer: Any = None, *, tokens: Any = 20) -> None:
        self.answer = answer if answer is not None else AIMessage(content=_compression_json())
        self.tokens = tokens
        self.seen: list[list[Any]] = []

    def invoke(self, messages: list[Any]) -> Any:
        self.seen.append(list(messages))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer

    def get_num_tokens_from_messages(self, messages: list[Any]) -> Any:
        if isinstance(self.tokens, Exception):
            raise self.tokens
        return self.tokens


def _compressible_state(tmp_path: Path, **extra: Any):
    state = _state(
        tmp_path,
        todos=[make_todo("t1", "write hello.py")],
        plan_summary="write hello.py then run it",
        acceptance_criteria=["it prints Hello"],
        verification_commands=["python hello.py"],
        attempts=1,
        **extra,
    )
    state["messages"] = [
        HumanMessage(content="write hello.py"),
        AIMessage(content="I will write the file"),
        _tool_call("file_write", {"file_path": "hello.py", "content": "print('Hello')\n"}),
        ToolMessage(content="Created hello.py", tool_call_id="c1"),
        AIMessage(content="now run it"),
        ToolMessage(content="exit code 1: NameError", tool_call_id="c2"),
    ]
    return state


def test_context_compressor_replaces_the_transcript(tmp_path: Path) -> None:
    state = _compressible_state(tmp_path)
    model = CompressorModel()

    update = context_compressor_node(state, model=model)

    messages = update["messages"]
    assert len(messages) == 2
    assert isinstance(messages[0], RemoveMessage)
    assert messages[0].id == REMOVE_ALL_MESSAGES
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == COMPRESSION_PAYLOAD["summary"]

    # the summary becomes the new context in every field that carries it
    assert update["context_summary"] == COMPRESSION_PAYLOAD["summary"]
    assert update["history_summary"] == COMPRESSION_PAYLOAD["summary"]
    assert update["context_should_compress"] is False
    assert update["context_token_count"] == 20
    assert update["context_next_node"] == "verifier"  # echoed/defaulted
    assert update["memory_snapshot"] == {}  # stale derived data is dropped


def test_context_compressor_applies_the_replace_to_a_real_graph(tmp_path: Path) -> None:
    """add_messages must drop the old transcript and keep only the summary."""
    from functools import partial

    from langgraph.graph import END, START, StateGraph

    from wqagent.graph import WQGraphState

    state = _compressible_state(tmp_path)
    graph = StateGraph(WQGraphState)
    graph.add_node("compress", partial(context_compressor_node, model=CompressorModel()))
    graph.add_edge(START, "compress")
    graph.add_edge("compress", END)

    result = graph.compile().invoke(state, {"configurable": {}})

    assert [message.type for message in result["messages"]] == ["ai"]
    assert result["messages"][0].content == COMPRESSION_PAYLOAD["summary"]
    assert result["context_summary"] == COMPRESSION_PAYLOAD["summary"]


def test_context_compressor_sends_the_prompt_and_the_context(tmp_path: Path) -> None:
    state = _compressible_state(tmp_path, research_notes="found: use print()")

    context_compressor_node(state, model=CompressorModel())
    model = CompressorModel()
    context_compressor_node(state, model=model)

    sent = model.seen[0]
    assert [message.type for message in sent] == ["system", "human"]
    assert sent[0].content == CONTEXT_COMPRESSION_PROMPT
    # task, transcript and the layered memory all reached the compressor
    assert "write hello.py" in sent[1].content
    assert "[tool] exit code 1: NameError" in sent[1].content
    assert "Layered memory:" in sent[1].content
    assert "found: use print()" in sent[1].content


def test_context_compressor_caps_the_fields_it_writes_back(tmp_path: Path) -> None:
    state = _compressible_state(
        tmp_path,
        research_notes="r" * 5_000,
        code_agent_summary="c" * 5_000,
        agent_handoffs=[{"instruction": "i" * 900} for _ in range(9)],
        sources=[{"title": "Docs", "url": "https://docs.test/a", "content": "body", "score": 0.9}],
    )

    update = context_compressor_node(state, model=CompressorModel())

    assert len(update["research_notes"]) <= RESEARCH_CHARS + 3
    assert update["research_notes"].endswith("...")
    assert len(update["code_agent_summary"]) <= SUMMARY_CHARS + 3
    assert len(update["agent_handoffs"]) == 6  # newest six
    assert len(update["agent_handoffs"][0]["instruction"]) <= HANDOFF_TEXT_CHARS + 3
    # structured sources stay, the model's urls are appended as bare urls
    assert update["sources"][0] == {"title": "Docs", "url": "https://docs.test/a"}
    assert {"title": "", "url": "https://docs.test/python"} in update["sources"]


def test_context_compressor_ignores_junk_source_urls(tmp_path: Path) -> None:
    state = _compressible_state(tmp_path)
    model = CompressorModel(
        AIMessage(content=_compression_json({**COMPRESSION_PAYLOAD, "sources": ["not a url", 42, "ok.test"]}))
    )

    update = context_compressor_node(state, model=model)

    assert [source["url"] for source in update["sources"]] == ["ok.test"]


def test_context_compressor_appends_a_compression_event(tmp_path: Path) -> None:
    state = _compressible_state(
        tmp_path, compression_events=[{"node": "old", "attempt": 0, "summary": "previous"}]
    )
    model = CompressorModel(tokens=123)

    update = context_compressor_node(state, model=model)
    events = update["compression_events"]

    assert [event["summary"] for event in events][0] == "previous"  # history kept
    assert len(events) == 2
    event = events[-1]
    assert event["node"] == "context_compressor"
    assert event["attempt"] == 1
    assert event["tokens_before"] == 123 and event["tokens_after"] == 123
    assert event["token_limit"] == DEFAULT_CONTEXT_TOKEN_LIMIT
    assert event["summary"] == COMPRESSION_PAYLOAD["summary"]
    # the state list was not mutated in place
    assert len(state["compression_events"]) == 1


def test_context_compressor_writes_history_summary(tmp_path: Path) -> None:
    state = _compressible_state(tmp_path)

    context_compressor_node(state, model=CompressorModel())

    text = (tmp_path / "ws" / "HISTORY_SUMMARY.md").read_text(encoding="utf-8")
    assert text.startswith("# History summary")
    assert COMPRESSION_PAYLOAD["summary"] in text
    for heading, item in (
        ("## Active goal", "make python hello.py exit 0"),
        ("## Completed work", "hello.py written (22 bytes)"),
        ("## Open todos", "- t1: rerun the program"),
        ("## Important files", "- hello.py"),
        ("## Tool findings", "- python 3.12.7 is on PATH"),
        ("## Sources", "- https://docs.test/python"),
        ("## Next steps", "- run python hello.py and check exit code"),
        ("## Risks", "the verifier command exits 1"),
    ):
        assert heading in text, heading
        assert item in text, item
    assert "compressed by context_compressor_node" in text


def test_context_compressor_reports_persistence_in_the_event(tmp_path: Path) -> None:
    state = _compressible_state(tmp_path)
    events: list[dict[str, Any]] = []

    context_compressor_node(state, model=CompressorModel(), on_event=events.append)

    compressed = [event for event in events if event["type"] == "context_compressed"]
    assert len(compressed) == 1
    assert compressed[0]["node"] == "context_compressor"
    assert compressed[0]["ok"] is True
    assert compressed[0]["persisted"]["ok"] is True
    assert compressed[0]["history_path"] == "HISTORY_SUMMARY.md"
    assert compressed[0]["tokens_after"] == 20


def test_context_compressor_survives_a_failed_write(tmp_path: Path, monkeypatch: Any) -> None:
    import wqagent.graph.nodes as node_module

    state = _compressible_state(tmp_path)
    monkeypatch.setattr(
        node_module,
        "_write_history_summary",
        lambda runtime, content: {"ok": False, "path": "HISTORY_SUMMARY.md", "error": "OSError: nope"},
    )
    events: list[dict[str, Any]] = []

    update = context_compressor_node(state, model=CompressorModel(), on_event=events.append)

    assert update["context_summary"] == COMPRESSION_PAYLOAD["summary"]  # still compressed
    assert update["messages"][1].content == COMPRESSION_PAYLOAD["summary"]
    assert events[-1]["persisted"]["ok"] is False
    assert not (tmp_path / "ws" / "HISTORY_SUMMARY.md").exists()


@pytest.mark.parametrize(
    "answer",
    [
        AIMessage(content="```json\n" + _compression_json() + "\n```"),
        AIMessage(content="Here you go:\n" + _compression_json() + "\nHope that helps!"),
    ],
)
def test_context_compressor_tolerates_fenced_and_prose_json(answer: AIMessage, tmp_path: Path) -> None:
    update = context_compressor_node(_compressible_state(tmp_path), model=CompressorModel(answer))

    assert update["context_summary"] == COMPRESSION_PAYLOAD["summary"]
    assert update["compression_events"][-1]["summary"] == COMPRESSION_PAYLOAD["summary"]


@pytest.mark.parametrize(
    "answer",
    [
        RuntimeError("provider exploded"),
        AIMessage(content="I am not going to answer in JSON."),
        AIMessage(content="null"),
        AIMessage(content='{"active_goal": "no summary key"}'),
    ],
)
def test_context_compressor_falls_back_to_a_deterministic_summary(
    answer: Any, tmp_path: Path
) -> None:
    state = _compressible_state(tmp_path, last_error="command failed: exit 1")
    events: list[dict[str, Any]] = []

    update = context_compressor_node(state, model=CompressorModel(answer), on_event=events.append)

    # the transcript is still replaced - an over-budget context must not survive
    assert update["messages"][0].id == REMOVE_ALL_MESSAGES
    summary = update["context_summary"]
    assert "write hello.py" in summary  # task
    assert "python hello.py" in summary  # verification command / plan
    assert "command failed: exit 1" in summary  # last failure
    assert "compressed deterministically" in summary
    assert update["context_should_compress"] is False
    assert events[-1]["ok"] is False
    assert update["compression_events"][-1]["summary"] == summary


def test_context_compressor_never_grows_past_the_state(tmp_path: Path) -> None:
    """A bloated transcript is replaced by a much smaller context."""
    state = _compressible_state(tmp_path)
    state["messages"] = [HumanMessage(content="x" * 40_000)] + state["messages"]
    model = CountingModel(RuntimeError("no tokenizer"))

    before = context_monitor_node(state, model=model)["context_token_count"]
    update = context_compressor_node(state, model=CompressorModel())
    after = update["context_token_count"]

    assert before > 10_000
    assert after < before


def test_context_compressor_trims_a_huge_transcript_request(tmp_path: Path) -> None:
    state = _compressible_state(tmp_path)
    state["messages"] = [HumanMessage(content="HEAD" + "x" * 200_000 + "TAIL")]
    model = CompressorModel()

    context_compressor_node(state, model=model)

    request = model.seen[0][1].content
    assert "HEAD" in request and "TAIL" in request
    assert "characters of transcript dropped" in request
    assert len(request) < COMPRESS_TRANSCRIPT_CHARS + 30_000


def test_context_compressor_writes_the_notepad(tmp_path: Path) -> None:
    """Compressing is when the runtime distils the durable notes."""
    state = _compressible_state(tmp_path)
    events: list[dict[str, Any]] = []

    context_compressor_node(state, model=CompressorModel(), on_event=events.append)

    text = (tmp_path / "ws" / "NOTEPAD.md").read_text(encoding="utf-8")
    assert text.startswith("# Notepad")
    assert "## Compression #1 - make python hello.py exit 0" in text
    # the durable half of the nine keys...
    assert "**Completed work**\nhello.py written (22 bytes)" in text
    assert "- hello.py" in text and "- python 3.12.7 is on PATH" in text
    assert "- https://docs.test/python" in text
    assert "- run python hello.py and check exit code" in text
    assert "**Risks**\nthe verifier command exits 1" in text
    # ...but not the compressed history, which belongs to HISTORY_SUMMARY.md
    assert COMPRESSION_PAYLOAD["summary"] not in text
    assert "open_todos" not in text
    assert "tokens 20 -> 20" in text  # provenance comment

    compressed = [event for event in events if event["type"] == "context_compressed"][-1]
    assert compressed["notepad"]["ok"] is True
    assert compressed["notepad_path"] == "NOTEPAD.md"


def test_context_compressor_appends_to_an_existing_notepad(tmp_path: Path) -> None:
    runtime = RuntimeState(workspace=tmp_path / "ws")
    state = _compressible_state(tmp_path)
    runtime.workspace.mkdir(parents=True, exist_ok=True)
    (runtime.workspace / "NOTEPAD.md").write_text(
        "# Notepad\n\n## Earlier note\n\nkeep me\n", encoding="utf-8"
    )
    state["compression_events"] = [{"node": "context_compressor", "attempt": 0}]

    context_compressor_node(state, model=CompressorModel())

    text = (runtime.workspace / "NOTEPAD.md").read_text(encoding="utf-8")
    assert text.count("# Notepad") == 1  # the title is not repeated
    assert "## Earlier note" in text and "keep me" in text  # nothing is lost
    assert "## Compression #2" in text  # numbered after the recorded compression
    assert text.index("Earlier note") < text.index("Compression #2")  # newest last


def test_context_compressor_caps_the_notepad(tmp_path: Path) -> None:
    runtime = RuntimeState(workspace=tmp_path / "ws")
    runtime.workspace.mkdir(parents=True, exist_ok=True)
    (runtime.workspace / "NOTEPAD.md").write_text(
        "# Notepad\n\n" + "".join(f"## Old #{n}\n\n{'x' * 2000}\n\n" for n in range(10)),
        encoding="utf-8",
    )
    state = _compressible_state(tmp_path)

    context_compressor_node(state, model=CompressorModel())

    text = (runtime.workspace / "NOTEPAD.md").read_text(encoding="utf-8")
    assert len(text) <= NOTEPAD_FILE_CHARS + 2000  # bounded, newest sections kept
    assert "## Old #9" in text  # the newest old section survives
    assert "## Compression #1" in text


def test_context_compressor_without_a_model_summary_writes_no_notepad(tmp_path: Path) -> None:
    """A deterministic fallback has no durable fields to distil."""
    state = _compressible_state(tmp_path)

    update = context_compressor_node(state, model=CompressorModel(AIMessage(content="not json")))

    assert update["context_summary"]  # the run was still compressed
    assert not (tmp_path / "ws" / "NOTEPAD.md").exists()


def test_compressor_events_report_both_files(tmp_path: Path) -> None:
    state = _compressible_state(tmp_path)
    events: list[dict[str, Any]] = []

    context_compressor_node(state, model=CompressorModel(), on_event=events.append)

    event = [item for item in events if item["type"] == "context_compressed"][-1]
    assert event["persisted"]["ok"] is True
    assert event["notepad"]["ok"] is True
    assert event["history_path"] == "HISTORY_SUMMARY.md"
    assert event["notepad_path"] == "NOTEPAD.md"


def test_context_compressor_normalises_stringified_lists(tmp_path: Path) -> None:
    """Models often stringify list fields; they must not land in the notepad as text."""
    payload = {
        **COMPRESSION_PAYLOAD,
        "important_files": "[]",
        "tool_findings": '["python 3.12 is on PATH", "exit code 0"]',
        "sources": "",
        "next_steps": "not json at all",
    }
    state = _compressible_state(tmp_path)

    context_compressor_node(state, model=CompressorModel(AIMessage(content=json.dumps(payload))))

    text = (tmp_path / "ws" / "NOTEPAD.md").read_text(encoding="utf-8")
    assert "**Important files**" not in text  # an empty list adds nothing
    assert "**Sources**" not in text
    assert "- python 3.12 is on PATH" in text and "- exit code 0" in text
    assert "**Next steps**\n- not json at all" in text  # a real sentence is kept
    assert "[]" not in text


def test_context_compressor_needs_a_runtime() -> None:
    with pytest.raises(ValueError, match="RuntimeState"):
        context_compressor_node({"task": "x"}, model=CompressorModel())  # type: ignore[arg-type]


# -- layered memory injection ------------------------------------------------
def test_planner_injects_the_layered_memory(tmp_path: Path) -> None:
    state = _state(tmp_path, research_notes="found: use python", todos=[make_todo("t1", "do it")])
    model = ScriptedModel([AIMessage(content="planned")])
    events: list[dict[str, Any]] = []

    update = planner_node(state, model=model, on_event=events.append)

    prompt = model.seen[0][1].content
    assert "Layered memory" in prompt
    assert '"rules"' in prompt and '"working_memory"' in prompt
    assert "found: use python" in prompt  # research notes ride along
    assert "[ ] t1: do it" in prompt  # the planning brief is still there

    memory_events = [event for event in events if event["type"] == "memory"]
    assert len(memory_events) == 1
    assert memory_events[0]["node"] == "planner"
    assert memory_events[0]["memory"]["working_memory"]["task"] == "write hello.py"
    # the snapshot this node showed the model is carried in the update
    assert update["memory_snapshot"]["working_memory"]["node"] == "planner"


def test_verifier_injects_the_layered_memory(tmp_path: Path) -> None:
    state = _state(tmp_path, verification_commands=["echo ok"])
    model = ScriptedModel(
        [
            AIMessage(content="inspected"),
            _tool_call(
                "verification_report",
                {"passed": True, "reason": "fine", "checks": [], "recommended_next_instruction": ""},
            ),
        ]
    )
    events: list[dict[str, Any]] = []

    update = verifier_node(state, model=model, on_event=events.append)

    prompt = model.seen[0][1].content
    assert "Layered memory" in prompt and '"working_memory"' in prompt
    assert "exit_code=0" in prompt  # the command evidence is still there
    assert [event["node"] for event in events if event["type"] == "memory"] == ["verifier"]
    assert update["memory_snapshot"]["working_memory"]["node"] == "verifier"


def test_injected_memory_carries_the_durable_files(tmp_path: Path) -> None:
    state = _state(tmp_path)
    (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ws" / "NOTEPAD.md").write_text("durable: always use python3", encoding="utf-8")
    (tmp_path / "ws" / "HISTORY_SUMMARY.md").write_text("history: earlier attempt", encoding="utf-8")
    model = ScriptedModel([AIMessage(content="planned")])

    planner_node(state, model=model)

    prompt = model.seen[0][1].content
    assert "durable: always use python3" in prompt
    assert "history: earlier attempt" in prompt
    assert '"notepad_exists": true' in prompt
    assert '"history_exists": true' in prompt


def test_memory_event_is_written_to_the_graph_stream(tmp_path: Path) -> None:
    """The streamed copy is tagged, so a consumer knows which node injected it."""
    state = _state(tmp_path)
    streamed: list[dict[str, Any]] = []

    planner_node(state, model=ScriptedModel([AIMessage(content="planned")]), writer=streamed.append)

    memory_events = [event for event in streamed if event["type"] == "memory"]
    assert len(memory_events) == 1
    assert memory_events[0]["node"] == "planner"
    assert memory_events[0]["digest"].endswith("notepad=no, history=no")


# -- TODO.md is the working plan state ---------------------------------------
def test_planner_persists_the_plan_to_todo_md(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            _tool_call(
                "todo_write",
                {
                    "plan_summary": "write then run",
                    "todos": [{"id": "t1", "content": "write hello.py"}],
                    "acceptance_criteria": ["prints Hello"],
                    "verification_commands": ["python hello.py"],
                },
            ),
            AIMessage(content="planned"),
        ]
    )

    update = planner_node(_state(tmp_path), model=model)

    text = (tmp_path / "ws" / "TODO.md").read_text(encoding="utf-8")
    assert "**Task:** write hello.py" in text
    assert "write then run" in text
    assert "- [ ] t1: write hello.py" in text
    assert "- `python hello.py`" in text
    plan = read_todos(RuntimeState(workspace=tmp_path / "ws"))
    assert plan["todos"] == update["todos"]
    assert plan["acceptance_criteria"] == ["prints Hello"]


def test_verifier_persists_completed_todos(tmp_path: Path) -> None:
    state = _state(
        tmp_path, todos=[make_todo("t1", "write hello.py")], verification_commands=["echo ok"]
    )
    model = ScriptedModel(
        [
            AIMessage(content="inspected"),
            _tool_call(
                "verification_report",
                {"passed": True, "reason": "fine", "checks": [], "recommended_next_instruction": ""},
            ),
        ]
    )

    update = verifier_node(state, model=model)

    assert update["passed"] is True
    assert "- [x] t1: write hello.py" in (tmp_path / "ws" / "TODO.md").read_text(encoding="utf-8")


def test_code_agent_updates_todo_md_while_working(tmp_path: Path) -> None:
    """The code agent's persist hook keeps the file current mid-run."""
    model = ScriptedModel(
        [
            _tool_call("call_code_agent", {"instruction": "write it"}, "c2"),
            _tool_call("file_write", {"file_path": "hello.py", "content": "print(1)\n"}, "c3"),
            _tool_call("todo_update", {"updates": [{"id": "t1", "status": "completed"}]}, "c4"),
            AIMessage(content="wrote hello.py"),
            AIMessage(content="done"),
        ]
    )

    planner_node(_state(tmp_path, todos=[make_todo("t1", "write hello.py")]), model=model, agent_model=model)

    text = (tmp_path / "ws" / "TODO.md").read_text(encoding="utf-8")
    assert "- [x] t1: write hello.py" in text  # written by the specialist's hook


def test_planner_resumes_the_plan_from_todo_md(tmp_path: Path) -> None:
    runtime = RuntimeState(workspace=tmp_path / "ws")
    write_todos(
        runtime,
        [make_todo("t1", "write hello.py", status="completed"), make_todo("t2", "run it")],
        task="write hello.py",
        plan_summary="write then run",
        acceptance_criteria=["prints Hello"],
        verification_commands=["python hello.py"],
    )
    model = ScriptedModel([AIMessage(content="continuing")])

    update = planner_node(new_graph_state("write hello.py", runtime), model=model)

    prompt = model.seen[0][1].content
    assert "left an unfinished plan for this exact task in TODO.md" in prompt
    assert "[x] t1: write hello.py" in prompt  # the recorded progress is loaded
    assert update["plan_summary"] == "write then run"
    assert [todo["status"] for todo in update["todos"]] == ["completed", "pending"]
    assert update["acceptance_criteria"] == ["prints Hello"]


def test_planner_ignores_a_todo_md_from_another_task(tmp_path: Path) -> None:
    runtime = RuntimeState(workspace=tmp_path / "ws")
    write_todos(runtime, [make_todo("t9", "something else")], task="a different task")
    model = ScriptedModel([AIMessage(content="fresh plan")])

    update = planner_node(new_graph_state("write hello.py", runtime), model=model)

    assert "unfinished plan" not in model.seen[0][1].content
    assert update["todos"] == []  # nothing was adopted


def test_planner_ignores_a_finished_plan(tmp_path: Path) -> None:
    """A completed TODO.md is not a resume point - it is just the last run's record."""
    runtime = RuntimeState(workspace=tmp_path / "ws")
    write_todos(
        runtime,
        [make_todo("t1", "write hello.py", status="completed")],
        task="write hello.py",
    )
    model = ScriptedModel([AIMessage(content="fresh plan")])

    update = planner_node(new_graph_state("write hello.py", runtime), model=model)

    assert "unfinished plan" not in model.seen[0][1].content
    assert update["todos"] == []


def test_planner_keeps_the_state_plan_over_todo_md(tmp_path: Path) -> None:
    runtime = RuntimeState(workspace=tmp_path / "ws")
    write_todos(runtime, [make_todo("t9", "stale")], task="write hello.py")
    state = new_graph_state("write hello.py", runtime, todos=[make_todo("t1", "current")])

    planner_node(state, model=ScriptedModel([AIMessage(content="planned")]))

    assert read_todos(runtime)["todos"][0]["id"] == "t1"  # the state wins, the file is rewritten


# -- shell guidance for verification commands --------------------------------
def test_planner_brief_names_the_shell(tmp_path: Path) -> None:
    model = ScriptedModel([AIMessage(content="planned")])

    planner_node(_state(tmp_path), model=model)

    prompt = model.seen[0][1].content
    assert SHELL_NAME in prompt
    assert "Verification commands run in" in prompt
    # concrete, machine-specific advice rather than a generic reminder
    assert shell_guidance() in prompt
    if os.name == "nt":
        assert "&&" in prompt


# -- model plumbing ----------------------------------------------------------
def test_get_model_is_lazy_and_cached(tmp_path: Path, monkeypatch: Any) -> None:
    created: list[str] = []
    model = ScriptedModel([AIMessage(content="ok")])

    def fake_create_model(**kwargs: Any) -> ScriptedModel:
        created.append("called")
        return model

    monkeypatch.setattr("wqagent.graph.nodes.create_model", fake_create_model)

    first = get_model()
    second = get_model()

    assert first is second is model
    assert created == ["called"]  # created once

    sentinel = object()
    set_model(sentinel)
    assert get_model() is sentinel


# -- the whole loop, wired into a real graph ---------------------------------
def test_graph_retries_after_a_failed_verification(tmp_path: Path) -> None:
    """planner -> verifier (fails) -> planner -> verifier (passes) -> END."""
    from langgraph.graph import END, START, StateGraph

    from wqagent.graph import WQGraphState

    workspace = tmp_path / "ws"
    runtime = RuntimeState(workspace=workspace)
    model = ScriptedModel(
        [
            # attempt 1: plan + delegate, with a verification command that fails
            _tool_call(
                "todo_write",
                {
                    "plan_summary": "write hello.py and run it",
                    "todos": [{"id": "t1", "content": "write hello.py"}],
                    "acceptance_criteria": ["hello.py runs cleanly"],
                    "verification_commands": ["exit 1"],
                },
            ),
            _tool_call("call_code_agent", {"instruction": "write hello.py"}, "c2"),
            _tool_call("file_write", {"file_path": "hello.py", "content": "print('Hello')\n"}),
            AIMessage(content="Wrote hello.py."),
            AIMessage(content="Delegated the implementation."),
            AIMessage(content="Inspected the file, it looks right."),
            _tool_call(
                "verification_report",
                {
                    "passed": False,
                    "reason": "the verification command exited non-zero",
                    "checks": [
                        {"name": "hello.py runs cleanly", "passed": False, "detail": "exit 1"}
                    ],
                    "recommended_next_instruction": "re-run the program and make it exit 0",
                },
            ),
            # attempt 2: revised plan (a command that passes) and the missing fix
            _tool_call(
                "todo_write",
                {
                    "plan_summary": "verify by actually running the program",
                    "todos": [{"id": "t1", "content": "write hello.py", "status": "completed"}],
                    "acceptance_criteria": ["hello.py runs cleanly"],
                    "verification_commands": ["echo ok"],
                },
            ),
            _tool_call("call_code_agent", {"instruction": "make it exit 0"}, "c3"),
            AIMessage(content="Re-ran the program, it works."),
            AIMessage(content="Delegated the fix."),
            AIMessage(content="Verified the file and the command output."),
            _tool_call(
                "verification_report",
                {
                    "passed": True,
                    "reason": "the program prints Hello and the command succeeded",
                    "checks": [{"name": "hello.py runs cleanly", "passed": True, "detail": "exit 0"}],
                    "recommended_next_instruction": "",
                },
            ),
        ]
    )
    set_model(model)

    graph = StateGraph(WQGraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("verifier", verifier_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "verifier")
    graph.add_conditional_edges(
        "verifier", verifier_route, {"planner": "planner", "final": END}
    )

    result = graph.compile().invoke(new_graph_state("write hello.py", runtime, max_attempts=3))

    assert result["passed"] is True
    assert result["attempts"] == 2
    assert result["last_error"] == ""
    assert [todo["status"] for todo in result["todos"]] == ["completed"]
    # the last verification run replaced the failing one (last-value-wins channel)
    assert [check["ok"] for check in result["verification_results"]] == [True]
    assert result["verification_results"][0]["command"] == "echo ok"
    assert (workspace / "hello.py").read_text(encoding="utf-8") == "print('Hello')\n"
    # every delegation is in the handoff log, both rounds included
    assert [handoff["to_agent"] for handoff in result["agent_handoffs"]] == [
        "codeAgent",
        "codeAgent",
    ]
    # the specialists' transcripts landed in the graph transcript
    assert any(message.type == "tool" for message in result["messages"])
    assert result["code_agent_summary"] == "Re-ran the program, it works."
    # exactly the thirteen model calls the two rounds need - no extras
    assert len(model.seen) == 13
    assert model.responses == []


def test_graph_gives_up_after_max_attempts(tmp_path: Path) -> None:
    """A verifier that always fails must not loop forever."""
    from langgraph.graph import END, START, StateGraph

    from wqagent.graph import WQGraphState

    runtime = RuntimeState(workspace=tmp_path / "ws")
    responses: list[Any] = []
    for attempt in range(2):  # max_attempts=2 -> two planner rounds
        responses.extend(
            [
                _tool_call(
                    "todo_write",
                    {
                        "plan_summary": f"attempt {attempt + 1}",
                        "todos": [{"id": "t1", "content": "do the thing"}],
                        "acceptance_criteria": ["it works"],
                        "verification_commands": ["exit 1"],
                    },
                ),
                AIMessage(content=f"planner round {attempt + 1}"),
                AIMessage(content="inspected"),
                _tool_call(
                    "verification_report",
                    {
                        "passed": False,
                        "reason": "still failing",
                        "checks": [],
                        "recommended_next_instruction": "try again",
                    },
                ),
            ]
        )
    model = ScriptedModel(responses)
    set_model(model)

    graph = StateGraph(WQGraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("verifier", verifier_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "verifier")
    graph.add_conditional_edges(
        "verifier", verifier_route, {"planner": "planner", "final": END}
    )

    result = graph.compile().invoke(new_graph_state("do the thing", runtime, max_attempts=2))

    assert result["passed"] is False
    assert result["attempts"] == 2  # stopped exactly at the budget
    assert "still failing" in result["last_error"]
    assert len(model.seen) == 8
    assert model.responses == []
