"""Tests for the ``wq`` CLI: flags, config resolution and an end-to-end run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import wqagent.cli.app as cli_module
from langchain_core.messages import AIMessage, BaseMessage
from typer.testing import CliRunner

from wqagent import __version__
from wqagent.cli.app import app
from wqagent.core.state import RuntimeState
from wqagent.providers import ProviderConfigError

runner = CliRunner()


class ScriptedModel:
    """Fake chat model: replays scripted responses, records bind_tools() and calls."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.bound: list[list[str]] = []
        self.seen: list[list[BaseMessage]] = []

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "ScriptedModel":
        self.bound.append([tool.name for tool in tools])
        return self

    def invoke(self, messages: list[BaseMessage]) -> Any:
        self.seen.append(list(messages))
        if not self.responses:
            raise AssertionError("ScriptedModel ran out of responses")
        return self.responses.pop(0)


def _tool_call(name: str, args: dict[str, Any], call_id: str = "c1") -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}]
    )


def _router_answer(route: str = "workflow", confidence: float = 0.9) -> AIMessage:
    """The entry router's answer (the CLI runs it before the workflow)."""
    return AIMessage(
        content='{"route":"%s","reason":"scripted","confidence":%s}' % (route, confidence)
    )


def _passing_script(
    file_path: str = "hello.py", content: str = "print('hi')\n", *, router: bool = True
) -> list[Any]:
    """Router + the eight responses a passing planner -> codeAgent -> verifier -> final run needs."""
    return [
        *([_router_answer()] if router else []),
        _tool_call(
            "todo_write",
            {
                "plan_summary": f"write {file_path}, then run it",
                "todos": [{"id": "t1", "content": f"write {file_path}"}],
                "acceptance_criteria": [f"{file_path} runs cleanly"],
                "verification_commands": ["echo ok"],
            },
        ),
        _tool_call("call_code_agent", {"instruction": f"write {file_path}"}, "c2"),
        _tool_call("file_write", {"file_path": file_path, "content": content}, "c3"),
        AIMessage(content=f"Created {file_path}."),
        AIMessage(content="Delegated the implementation."),
        AIMessage(content="Inspected the file."),
        _tool_call(
            "verification_report",
            {
                "passed": True,
                "reason": "the file exists and the command succeeded",
                "checks": [{"name": f"{file_path} runs cleanly", "passed": True, "detail": "exit 0"}],
                "recommended_next_instruction": "",
            },
        ),
        AIMessage(content=f"Done: created {file_path} and verified it."),
    ]


def _use_scripted(monkeypatch: Any, responses: list[Any]) -> ScriptedModel:
    """Patch ``create_model`` so the CLI runs against a scripted model."""
    model = ScriptedModel(list(responses))
    monkeypatch.setattr(cli_module, "create_model", lambda **kwargs: model)
    return model


def _combined(result: Any) -> str:
    """stdout + stderr, whatever the installed click version captures."""
    text = result.output or ""
    try:
        text += result.stderr or ""
    except (ValueError, AttributeError):  # pragma: no cover - older click
        pass
    return text


def _flat(text: str) -> str:
    """Collapse line wrapping and padding so assertions survive narrow output."""
    return " ".join(text.split())


def _clear_llm_env(monkeypatch: Any) -> None:
    for name in (
        "API_KEY",
        "OPENAI_API_KEY",
        "MODEL",
        "OPENAI_MODEL",
        "BASE_URL",
        "OPENAI_BASE_URL",
        "WQ_MODEL",
        "WQ_BASE_URL",
        "WQ_API_KEY",
        "WQ_WORKSPACE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_list_tools() -> None:
    result = runner.invoke(app, ["--list-tools"])
    assert result.exit_code == 0
    for name in ("file_read", "file_write", "file_edit", "grep", "bash"):
        assert name in result.output


def test_list_tools_verbose_shows_arguments() -> None:
    result = runner.invoke(app, ["--list-tools", "--verbose"])
    assert result.exit_code == 0
    assert "old_text" in result.output
    assert "timeout_seconds" in result.output


def test_missing_task_is_a_usage_error() -> None:
    assert runner.invoke(app, []).exit_code == 2


def test_show_config_reports_resolved_settings(tmp_path: Path, monkeypatch: Any) -> None:
    """Real environment variables must win over a .env file, and nothing is created."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODEL", "gpt-4o-mini")
    monkeypatch.setenv("BASE_URL", "https://api.example.com/v1")
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = runner.invoke(app, ["--show-config"])

    assert result.exit_code == 0
    flat = _flat(result.output)
    assert "gpt-4o-mini" in flat
    assert "https://api.example.com/v1" in flat
    assert "api key" in flat
    assert not (tmp_path / "workspace").exists()


def test_show_config_respects_flags(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _clear_llm_env(monkeypatch)
    result = runner.invoke(
        app,
        [
            "--show-config",
            "-w",
            "custom-dir",
            "--model",
            "qwen2.5-coder:7b",
            "--base-url",
            "http://localhost:8000/v1",
        ],
    )
    assert result.exit_code == 0
    flat = _flat(result.output)
    assert "qwen2.5-coder:7b" in flat
    assert "custom-dir" in flat
    assert "will be created" in flat
    assert "not created" not in flat


def test_missing_api_key_is_reported(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)

    def explode(**kwargs: Any) -> Any:
        raise ProviderConfigError("no API key found. Put API_KEY=... in a .env file")

    monkeypatch.setattr(cli_module, "create_model", explode)
    result = runner.invoke(app, ["do something", "-w", str(tmp_path / "ws")])
    assert result.exit_code == 1
    assert "no API key found" in _combined(result)


def test_run_end_to_end_with_a_scripted_model(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    workspace = tmp_path / "ws"
    _use_scripted(monkeypatch, _passing_script())

    result = runner.invoke(app, ["create hello.py", "-w", str(workspace)])

    assert result.exit_code == 0
    assert (workspace / "hello.py").read_text(encoding="utf-8") == "print('hi')\n"
    output = _flat(result.output)
    assert "Done: created hello.py and verified it." in output
    # every node and the delegated specialist reported itself (labels fall back to
    # ASCII when the captured stream cannot encode the emoji)
    for marker in ("Planner", "codeAgent", "Verifier", "Final"):
        assert marker in output, marker
    assert "verdict" in output and "passed" in output


def test_renderer_prints_the_handoff_and_specialist_headers(monkeypatch: Any) -> None:
    """The handoff line and the specialist header, rendered into a clean stream."""
    from io import StringIO

    from rich.console import Console

    stream = StringIO()
    monkeypatch.setattr(
        cli_module, "console", Console(file=stream, width=200, force_terminal=False)
    )
    monkeypatch.setattr(cli_module, "_can_encode", lambda text: True)  # emoji path
    renderer = cli_module._NodeRenderer(verbose=False)

    renderer.render(
        {"type": "handoff", "node": "planner", "from": "planner", "to": "codeAgent",
         "instruction": "write hello.py"}
    )
    renderer.render(
        {"type": "tool_call", "node": "codeAgent", "name": "file_write",
         "args": {"file_path": "hello.py", "content": "print('hi')\n"}}
    )
    renderer.render(
        {"type": "node_end", "node": "planner",
         "update": {"plan_summary": "write it", "todos": [{"id": "t1", "content": "write it"}],
                    "verification_commands": ["echo ok"],
                    "agent_handoffs": [{"to_agent": "codeAgent", "instruction": "write hello.py"}],
                    "last_actor_summary": "Delegated the implementation."}}
    )

    text = stream.getvalue()
    assert "↪ planner → 🛠 codeAgent" in text
    assert "   write hello.py" in text
    assert "🛠 codeAgent" in text  # the specialist introduced itself on its first event
    assert "📋 Planner" in text
    assert "-> file_write hello.py" in text
    assert "$ echo ok" in text


def test_renderer_prints_the_context_monitor(tmp_path: Path, monkeypatch: Any) -> None:
    from io import StringIO

    from rich.console import Console

    stream = StringIO()
    monkeypatch.setattr(
        cli_module, "console", Console(file=stream, width=200, force_terminal=False)
    )
    monkeypatch.setattr(cli_module, "_can_encode", lambda text: False)  # ASCII path
    renderer = cli_module._NodeRenderer(verbose=False)

    renderer.render(
        {
            "type": "node_end",
            "node": "context_monitor",
            "update": {
                "context_token_count": 1200,
                "context_token_limit": 400_000,
                "context_should_compress": False,
                "context_next_node": "verifier",
            },
        }
    )
    renderer.render(
        {
            "type": "node_end",
            "node": "context_monitor",
            "update": {
                "context_token_count": 401_000,
                "context_should_compress": True,
                "context_next_node": "planner",
            },
        }
    )

    text = _flat(stream.getvalue())
    assert "Context" in text
    assert "1200/400000 tokens (ok) -> next: verifier" in text
    assert "401000 tokens (over budget) -> next: planner" in text


def test_renderer_prints_the_context_compressor(monkeypatch: Any) -> None:
    from io import StringIO

    from rich.console import Console

    stream = StringIO()
    monkeypatch.setattr(
        cli_module, "console", Console(file=stream, width=200, force_terminal=False)
    )
    monkeypatch.setattr(cli_module, "_can_encode", lambda text: False)  # ASCII path
    renderer = cli_module._NodeRenderer(verbose=False)

    renderer.render(
        {
            "type": "context_compressed",
            "node": "context_compressor",
            "ok": True,
            "error": "",
            "tokens_before": 412_000,
            "tokens_after": 3_100,
            "messages_before": 24,
            "messages_after": 1,
            "persisted": {"ok": True, "path": "HISTORY_SUMMARY.md", "error": ""},
            "notepad": {"ok": True, "path": "NOTEPAD.md", "error": ""},
            "history_path": "HISTORY_SUMMARY.md",
            "notepad_path": "NOTEPAD.md",
        }
    )
    renderer.render(
        {
            "type": "node_end",
            "node": "context_compressor",
            "update": {
                "context_summary": "compressed: hello.py written",
                "compression_events": [
                    {
                        "tokens_before": 412_000,
                        "tokens_after": 3_100,
                        "messages_before": 24,
                        "messages_after": 1,
                        "token_limit": 400_000,
                        "summary": "compressed",
                    }
                ],
            },
        }
    )
    # a fallback compression says so
    renderer.render(
        {
            "type": "context_compressed",
            "node": "context_compressor",
            "ok": False,
            "error": "RuntimeError: provider exploded",
            "tokens_before": 10,
            "tokens_after": 20,
            "persisted": {"ok": False, "error": "OSError: read-only"},
        }
    )

    text = _flat(stream.getvalue())
    assert "Compressor" in text
    # before/after comparison: both message count and tokens
    assert "compressed 24 msgs/412000 tokens -> 1 msg/3100 tokens [HISTORY_SUMMARY.md, NOTEPAD.md]" in text
    assert "24 msgs/412000 tokens -> 1 msg/3100 tokens (limit 400000)" in text
    assert "compressed: hello.py written" not in text  # only in verbose mode
    assert "compressed 10 -> 20 tokens (history not persisted)" in text
    assert "fallback summary: RuntimeError: provider exploded" in text


def test_renderer_prints_the_injected_memory(monkeypatch: Any) -> None:
    from io import StringIO

    from rich.console import Console

    stream = StringIO()
    monkeypatch.setattr(
        cli_module, "console", Console(file=stream, width=200, force_terminal=False)
    )
    monkeypatch.setattr(cli_module, "_can_encode", lambda text: False)  # ASCII path
    renderer = cli_module._NodeRenderer(verbose=False)

    renderer.render(
        {
            "type": "memory",
            "node": "planner",
            "digest": "1 todos, 0 sources, 1 handoffs, notepad=yes, history=no",
            "chars": 1517,
            "memory": {"rules": {}, "working_memory": {}, "history_summary_store": {}},
        }
    )

    text = _flat(stream.getvalue())
    assert "Planner" in text  # the header opens on the node's first event
    assert "memory[planner]: 1 todos, 0 sources, 1 handoffs, notepad=yes, history=no (1517 chars)" in text


def _risky_bash_script(command: str = "pip install nothing-here") -> list[Any]:
    """A run whose agent calls bash with a risky command."""
    return [
        _router_answer(),
        _tool_call(
            "todo_write",
            {
                "plan_summary": "install something",
                "todos": [{"id": "t1", "content": "install it"}],
                "acceptance_criteria": ["it is installed"],
                "verification_commands": ["echo ok"],
            },
            "r1",
        ),
        _tool_call("call_code_agent", {"instruction": "install it"}, "r2"),
        _tool_call("bash", {"command": command}, "r3"),
        AIMessage(content="Ran the install."),
        AIMessage(content="Done with the install."),
        AIMessage(content="Inspected."),
        _tool_call(
            "verification_report",
            {
                "passed": True,
                "reason": "fine",
                "checks": [],
                "recommended_next_instruction": "",
            },
            "r9",
        ),
        AIMessage(content="Done: handled the install request."),
    ]


def test_cli_approval_prompt_denies_when_the_user_says_no(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.chdir(tmp_path)
    _use_scripted(monkeypatch, _risky_bash_script())

    result = runner.invoke(
        app,
        ["install something", "-w", str(tmp_path / "ws"), "--approval-mode", "inline"],
        input="n\n",
    )

    output = _flat(result.output)
    assert "approval required" in output
    assert "Python package installation" in output
    assert "command not run - approval required" in output
    assert "declined at the prompt" in output
    assert result.exit_code == 0  # verification still passed, so the run itself is fine


def test_cli_approval_prompt_runs_when_the_user_says_yes(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.chdir(tmp_path)
    _use_scripted(monkeypatch, _risky_bash_script("npm install --dry-run"))

    result = runner.invoke(
        app,
        ["install something", "-w", str(tmp_path / "ws"), "--approval-mode", "inline"],
        input="y\n",
    )

    output = _flat(result.output)
    assert "approval required" in output
    assert "requires_approval: True (approved by the user" in output
    assert "Node package installation" in output


def test_cli_approval_mode_flag_and_env(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WQ_APPROVAL_MODE", raising=False)

    shown = runner.invoke(app, ["--show-config", "-w", str(tmp_path / "ws")])
    assert "approval : inline (default)" in _flat(shown.output)

    monkeypatch.setenv("WQ_APPROVAL_MODE", "deny")
    from_env = runner.invoke(app, ["--show-config", "-w", str(tmp_path / "ws")])
    assert "approval : deny (env)" in _flat(from_env.output)

    explicit = runner.invoke(
        app, ["--show-config", "-w", str(tmp_path / "ws"), "--approval-mode", "auto"]
    )
    assert "approval : auto (cli)" in _flat(explicit.output)

    # the value is a Literal: an unusable one is rejected instead of guessed at
    junk = runner.invoke(
        app, ["--show-config", "-w", str(tmp_path / "ws"), "--approval-mode", "junk"]
    )
    assert junk.exit_code == 2
    assert "not one of 'inline', 'auto', 'deny'" in _combined(junk)
    assert "Usage:" in _combined(junk)

    # ... including one that arrives through the environment
    monkeypatch.setenv("WQ_APPROVAL_MODE", "sideways")
    from_env_junk = runner.invoke(app, ["--show-config", "-w", str(tmp_path / "ws")])
    assert from_env_junk.exit_code == 2
    assert "WQ_APPROVAL_MODE" in _combined(from_env_junk)


def test_cli_mode_options_are_case_insensitive(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WQ_TRACE_MODE", raising=False)

    result = runner.invoke(
        app,
        [
            "--show-config",
            "-w",
            str(tmp_path / "ws"),
            "--approval-mode",
            "DENY",
            "--checkpoint-mode",
            "Strict",
            "--trace-mode",
            "ON",
        ],
    )

    assert result.exit_code == 0
    output = _flat(result.output)
    assert "approval : deny (cli)" in output
    assert "checkpoints : strict (cli)" in output
    assert "traces : summary (cli)" in output  # `on` is the friendly spelling


def test_cli_approval_deny_mode_blocks_the_command(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _use_scripted(monkeypatch, _risky_bash_script("uv sync"))

    result = runner.invoke(
        app, ["install something", "-w", str(tmp_path / "ws"), "--approval-mode", "deny"]
    )

    output = _flat(result.output)
    assert "approval required" in output
    assert "mode is 'deny'" in output
    assert "command not run - approval required" in output


# -- checkpoints / resume ----------------------------------------------------
def _checkpointed_workspace(tmp_path: Path) -> Path:
    """A workspace as a crashed run would leave it: a checkpoint + a stale file."""
    from wqagent.core.checkpoint import CheckpointManager
    from wqagent.graph.state import make_todo, new_graph_state

    workspace = tmp_path / "ws"
    runtime = RuntimeState(workspace=workspace, checkpoint_mode="light")
    state = new_graph_state("write hello.py", runtime, todos=[make_todo("t1", "write hello.py")])
    state["last_error"] = "command failed: python hello.py"
    (workspace / "hello.py").write_text("print('hi')\n", encoding="utf-8")
    saved = CheckpointManager(runtime, "write hello.py").save(
        state, status="running", latest_node="verifier"
    )
    assert saved is not None and saved["git_ok"], saved
    # the workspace moved on after the checkpoint
    (workspace / "hello.py").write_text("scribbled over\n", encoding="utf-8")
    return workspace


def test_cli_run_writes_checkpoints(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _use_scripted(monkeypatch, _passing_script())

    result = runner.invoke(app, ["say hi", "-w", str(tmp_path / "ws"), "-v"])

    assert result.exit_code == 0
    root = tmp_path / "ws" / ".wqagent" / "checkpoints"
    assert (root / "checkpoint.json").is_file()
    assert (root / "RECOVERY.md").is_file()
    payload = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
    assert payload["status"] == "passed"  # the closing checkpoint records the outcome
    assert "checkpoint passed/final" in _flat(result.output)


def test_cli_checkpoint_mode_off_writes_nothing(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _use_scripted(monkeypatch, _passing_script())

    result = runner.invoke(
        app,
        [
            "say hi",
            "-w",
            str(tmp_path / "ws"),
            "--checkpoint-mode",
            "off",
            "--trace-mode",
            "off",
        ],
    )

    assert result.exit_code == 0
    hidden = tmp_path / "ws" / ".wqagent"
    assert not (hidden / "checkpoints").exists()
    assert not (hidden / "traces").exists()
    # the only things that may appear are the child temp dir and the session record
    assert sorted(path.name for path in hidden.iterdir()) in ([], ["tmp"], ["session"], ["session", "tmp"])


def test_cli_run_writes_a_trace(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _use_scripted(monkeypatch, _passing_script())

    result = runner.invoke(app, ["say hi", "-w", str(tmp_path / "ws"), "-v"])

    assert result.exit_code == 0, result.output
    traces = tmp_path / "ws" / ".wqagent" / "traces"
    trace_dirs = [path for path in traces.iterdir() if path.is_dir()]
    assert len(trace_dirs) == 1
    trace_id = trace_dirs[0].name
    assert trace_id.startswith("wq-")
    assert f"id {trace_id}" in _flat(result.output)  # the banner shows the run's id
    assert "trace " in _flat(result.output)  # the verbose trace line
    payload = json.loads((trace_dirs[0] / "trace.json").read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["tool_calls"] >= 1 and payload["node_visits"]["planner"] == 1
    assert (trace_dirs[0] / "timeline.md").is_file()
    assert not (trace_dirs[0] / "events.jsonl").exists()  # summary mode


def test_cli_trace_mode_full_and_off(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _use_scripted(monkeypatch, _passing_script())

    full = runner.invoke(app, ["say hi", "-w", str(tmp_path / "ws"), "--trace-mode", "full"])

    assert full.exit_code == 0
    events = list((tmp_path / "ws" / ".wqagent" / "traces").glob("*/events.jsonl"))
    assert len(events) == 1
    lines = events[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 5 and json.loads(lines[0])["type"] == "run_start"

    _use_scripted(monkeypatch, _passing_script())
    off = runner.invoke(
        app, ["say hi", "-w", str(tmp_path / "other"), "--trace-mode", "off"]
    )
    assert off.exit_code == 0
    assert not (tmp_path / "other" / ".wqagent" / "traces").exists()


def test_cli_strict_mode_saves_state_and_events(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _use_scripted(monkeypatch, _passing_script())

    result = runner.invoke(
        app, ["say hi", "-w", str(tmp_path / "ws"), "--checkpoint-mode", "strict"]
    )

    assert result.exit_code == 0
    root = tmp_path / "ws" / ".wqagent" / "checkpoints"
    assert (root / "state.json").is_file()
    lines = (root / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 5  # every streamed event was appended
    assert all(json.loads(line)["type"] for line in lines)


def test_cli_resume_restores_and_continues(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    workspace = _checkpointed_workspace(tmp_path)
    # resuming skips the entry router, so the script starts at the planner
    _use_scripted(monkeypatch, _passing_script(router=False))

    result = runner.invoke(app, ["--resume", str(workspace), "--max-attempts", "1"])

    output = _flat(result.output)
    assert result.exit_code == 0, output
    assert "resumed" in output
    assert "task : write hello.py" in output
    assert "status : running (latest node: verifier)" in output
    assert "restored : 1 file(s)" in output
    assert "warning :" in output  # the stale file was overwritten
    # the snapshot came back and the run continued from the saved plan
    assert (workspace / "hello.py").read_text(encoding="utf-8") == "print('hi')\n"
    assert "Done: created hello.py" in output


def test_cli_resume_without_a_checkpoint_fails(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["--resume", str(tmp_path / "empty-ws")])

    assert result.exit_code == 1
    assert "no checkpoint to resume" in _combined(result)
    assert "no checkpoint.json" in _combined(result)


def test_cli_checkpoint_mode_flag_and_env(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("WQ_CHECKPOINT_MODE", raising=False)

    shown = runner.invoke(app, ["--show-config", "-w", str(tmp_path / "ws")])
    assert "checkpoints : light" in _flat(shown.output)

    monkeypatch.setenv("WQ_CHECKPOINT_MODE", "strict")
    from_env = runner.invoke(app, ["--show-config", "-w", str(tmp_path / "ws")])
    assert "checkpoints : strict" in _flat(from_env.output)

    explicit = runner.invoke(
        app, ["--show-config", "-w", str(tmp_path / "ws"), "--checkpoint-mode", "off"]
    )
    assert "checkpoints : off" in _flat(explicit.output)


# -- stalling and timeouts ---------------------------------------------------
def test_heartbeat_explains_a_silent_run(monkeypatch: Any) -> None:
    """Silence must be explained: no events for a while -> say what we are waiting for."""
    import time as time_module

    from io import StringIO

    from rich.console import Console

    stream = StringIO()
    monkeypatch.setattr(
        cli_module, "error_console", Console(file=stream, width=200, force_terminal=False)
    )
    monkeypatch.setattr(cli_module, "_can_encode", lambda text: False)

    with cli_module._Heartbeat(interval=0.2) as heartbeat:
        heartbeat.touch("codeAgent")
        time_module.sleep(0.8)

    text = stream.getvalue()
    assert "still in codeAgent" in text
    assert "no events for" in text
    assert text.count("still in") >= 1


def test_heartbeat_stays_quiet_while_events_flow(monkeypatch: Any) -> None:
    import time as time_module

    from io import StringIO

    from rich.console import Console

    stream = StringIO()
    monkeypatch.setattr(
        cli_module, "error_console", Console(file=stream, width=200, force_terminal=False)
    )

    with cli_module._Heartbeat(interval=0.5) as heartbeat:
        for _ in range(6):
            heartbeat.touch("planner")
            time_module.sleep(0.1)

    assert stream.getvalue() == ""


def test_heartbeat_can_be_disabled(monkeypatch: Any) -> None:
    import time as time_module

    from io import StringIO

    from rich.console import Console

    stream = StringIO()
    monkeypatch.setattr(
        cli_module, "error_console", Console(file=stream, width=200, force_terminal=False)
    )

    with cli_module._Heartbeat(interval=0.1, enabled=False):
        time_module.sleep(0.4)

    assert stream.getvalue() == ""


@pytest.mark.parametrize(
    "value, expected",
    [(None, 120.0), ("", 120.0), ("45", 45.0), ("junk", 120.0), ("0", 120.0), ("-3", 120.0)],
)
def test_request_timeout_from_env(monkeypatch: Any, value: Any, expected: float) -> None:
    if value is None:
        monkeypatch.delenv(cli_module.REQUEST_TIMEOUT_ENV, raising=False)
    else:
        monkeypatch.setenv(cli_module.REQUEST_TIMEOUT_ENV, value)

    assert cli_module.request_timeout_from_env(120.0) == expected


def test_cli_passes_the_request_timeout_to_the_model(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.chdir(tmp_path)
    seen: dict[str, Any] = {}

    def fake_create_model(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return ScriptedModel(_passing_script())

    monkeypatch.setattr(cli_module, "create_model", fake_create_model)

    result = runner.invoke(
        app, ["say hi", "-w", str(tmp_path / "ws"), "--request-timeout", "45"]
    )

    assert result.exit_code == 0, result.output
    assert seen["timeout"] == 45.0
    assert "request timeout 45s" in _flat(result.output)  # the banner says so

    shown = runner.invoke(
        app, ["--show-config", "-w", str(tmp_path / "ws"), "--request-timeout", "45"]
    )
    assert "model/request: 45s per attempt" in _flat(shown.output)


def test_cli_request_timeout_comes_from_the_environment(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(cli_module.REQUEST_TIMEOUT_ENV, "77")
    seen: dict[str, Any] = {}

    def fake_create_model(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return ScriptedModel(_passing_script())

    monkeypatch.setattr(cli_module, "create_model", fake_create_model)

    result = runner.invoke(app, ["say hi", "-w", str(tmp_path / "ws")])

    assert result.exit_code == 0, result.output
    assert seen["timeout"] == 77.0


def test_failure_hint_explains_a_timed_out_provider() -> None:
    hint = cli_module._failure_hint(TimeoutError("Request timed out."), 120.0)

    assert "Request timed out." in hint
    assert "did not answer within 120s" in hint
    assert "--request-timeout" in hint
    assert "resumable" in hint


def test_failure_hint_leaves_other_errors_alone() -> None:
    assert cli_module._failure_hint(ValueError("nope"), 120.0) == "ValueError: nope"


def test_cli_model_timeout_keeps_the_trail(tmp_path: Path, monkeypatch: Any) -> None:
    """A provider timeout: clear message, and the trace/checkpoint are still written."""
    monkeypatch.chdir(tmp_path)

    class TimedOut(ScriptedModel):
        calls = 0

        def invoke(self, messages: Any) -> Any:
            type(self).calls += 1
            if type(self).calls > 1:
                raise TimeoutError("Request timed out.")
            return super().invoke(messages)

    monkeypatch.setattr(cli_module, "create_model", lambda **kwargs: TimedOut(_passing_script()))

    result = runner.invoke(app, ["say hi", "-w", str(tmp_path / "ws")])

    assert result.exit_code == 1
    output = _combined(result)
    assert "Request timed out." in output
    assert "did not answer within 120s" in output
    assert "resume with" in output
    traces = list((tmp_path / "ws" / ".wqagent" / "traces").iterdir())
    assert len(traces) == 1, "the trace must exist even though the run failed"
    assert json.loads((traces[0] / "trace.json").read_text(encoding="utf-8"))["status"] == "failed"
    payload = json.loads(
        (tmp_path / "ws" / ".wqagent" / "checkpoints" / "checkpoint.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["status"] == "failed"


def test_heartbeat_names_the_running_command(monkeypatch: Any) -> None:
    """Regression: silence during a long command blamed the model and hid the command."""
    from io import StringIO

    from rich.console import Console

    stream = StringIO()
    monkeypatch.setattr(
        cli_module, "error_console", Console(file=stream, width=200, force_terminal=False)
    )
    monkeypatch.setattr(cli_module, "_can_encode", lambda text: False)
    monkeypatch.setattr(
        cli_module,
        "describe_running_commands",
        lambda: [
            {
                "command": "python -m pip install -r requirements.txt",
                "elapsed": 95.4,
                "timeout": 120.0,
            }
        ],
    )
    heartbeat = cli_module._Heartbeat(interval=30.0)

    text = heartbeat._explain(35.0)

    assert "bash has been running for 95s" in text
    assert "timeout 120s" in text
    assert "python -m pip install -r requirements.txt" in text
    assert "raise --timeout" in text
    assert "waiting for the model" not in text


def test_heartbeat_mentions_extra_commands(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        cli_module,
        "describe_running_commands",
        lambda: [
            {"command": "pip install a", "elapsed": 40.0, "timeout": 120.0},
            {"command": "pip install b", "elapsed": 12.0, "timeout": 120.0},
        ],
    )

    text = cli_module._Heartbeat(interval=30.0)._explain(35.0)

    assert "(+1 more)" in text


def test_heartbeat_falls_back_to_the_model_wording(monkeypatch: Any) -> None:
    monkeypatch.setattr(cli_module, "describe_running_commands", lambda: [])
    heartbeat = cli_module._Heartbeat(interval=30.0)
    heartbeat.touch("codeAgent")

    text = heartbeat._explain(45.0)

    assert "still in codeAgent" in text and "no events for 45s" in text
    assert "waiting for the model" in text


def test_heartbeat_survives_a_broken_registry(monkeypatch: Any) -> None:
    def boom() -> Any:
        raise RuntimeError("registry exploded")

    monkeypatch.setattr(cli_module, "describe_running_commands", boom)

    text = cli_module._Heartbeat(interval=30.0)._explain(35.0)

    assert "still in" in text  # a status line must never break the run


def test_heartbeat_explains_a_command_with_no_output(monkeypatch: Any) -> None:
    """`0 chars of output` for half a minute is itself a diagnosis, not just silence."""
    monkeypatch.setattr(
        cli_module,
        "describe_running_commands",
        lambda: [
            {
                "command": 'python -m pip install --no-input "fastapi>=0.110"',
                "elapsed": 208.0,
                "timeout": 600.0,
                "tail": [],
                "bytes": 0,
            }
        ],
    )

    text = cli_module._Heartbeat(interval=30.0)._explain(35.0)

    assert "208s" in text and "0 chars of output" in text
    assert "no output at all yet" in text
    assert "slow package index" in text


def test_heartbeat_skips_that_hint_while_output_flows(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        cli_module,
        "describe_running_commands",
        lambda: [
            {
                "command": "python -m pip install fastapi",
                "elapsed": 208.0,
                "timeout": 600.0,
                "tail": ["Collecting fastapi", "Downloading fastapi-0.110.0-py3-none-any.whl"],
                "bytes": 4096,
            }
        ],
    )

    text = cli_module._Heartbeat(interval=30.0)._explain(35.0)

    assert "no output at all yet" not in text
    assert "Downloading fastapi" in text


def test_cli_answers_conversation_without_the_workflow(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The entry router: a greeting is answered directly, with no plan and no tools."""
    monkeypatch.chdir(tmp_path)
    _use_scripted(monkeypatch, [_router_answer("chat", 0.95), AIMessage(content="你好！我是 WQ Agent。")])

    result = runner.invoke(app, ["你好", "-w", str(tmp_path / "ws")])

    output = _flat(result.output)
    assert result.exit_code == 0, output
    assert "你好！我是 WQ Agent。" in output
    assert "chat (no tools, no workspace changes)" in output
    assert "no tools, no workspace changes" in output
    # the workflow never ran
    assert "Planner" not in output and "Verifier" not in output


def test_cli_tui_flag_hands_over_to_the_tui(tmp_path: Path, monkeypatch: Any) -> None:
    """``--tui`` launches the Textual UI instead of the plain runner (never a real UI)."""
    monkeypatch.chdir(tmp_path)
    captured: dict[str, Any] = {}

    def fake_run_tui(**kwargs: Any) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("wqagent.cli.tui.run_tui", fake_run_tui)
    monkeypatch.setattr(cli_module, "create_model", lambda **kwargs: ScriptedModel([]))

    result = runner.invoke(
        app,
        [
            "你好",
            "-w",
            str(tmp_path / "ws"),
            "--tui",
            "--approval-mode",
            "auto",
            "--trace-mode",
            "off",
            "--max-attempts",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["workspace"] == (tmp_path / "ws").resolve()
    assert captured["approval_mode"] == "auto"
    assert captured["trace_mode"] == "off"
    assert captured["max_attempts"] == 2
    assert captured["entry_router"] is True
    assert captured["model"] is not None
    # the plain runner never started: no session turn is recorded by the CLI itself
    assert not (tmp_path / "ws" / ".wqagent" / "session").exists()


def test_cli_tui_needs_no_task(tmp_path: Path, monkeypatch: Any) -> None:
    """`wq --tui` opens the UI without a task; the missing-task error is for plain runs."""
    monkeypatch.chdir(tmp_path)
    captured: dict[str, Any] = {}
    monkeypatch.setattr("wqagent.cli.tui.run_tui", lambda **kwargs: captured.update(kwargs) or 0)
    monkeypatch.setattr(cli_module, "create_model", lambda **kwargs: ScriptedModel([]))

    result = runner.invoke(app, ["--tui", "-w", str(tmp_path / "ws")])

    assert result.exit_code == 0, result.output
    assert captured["workspace"] == (tmp_path / "ws").resolve()

    plain = runner.invoke(app, ["-w", str(tmp_path / "ws")])
    assert plain.exit_code == 2
    assert "missing task" in _combined(plain)


def test_cli_no_router_always_runs_the_workflow(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    # no router response in the script: with --no-router the planner gets the first one
    _use_scripted(monkeypatch, _passing_script(router=False))

    result = runner.invoke(
        app, ["say hi", "-w", str(tmp_path / "ws"), "--no-router"]
    )

    output = _flat(result.output)
    assert result.exit_code == 0, output
    assert "Planner" in output
    assert "chat (no tools" not in output


def test_cli_shows_the_route_in_verbose_mode(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _use_scripted(monkeypatch, _passing_script())

    result = runner.invoke(app, ["say hi", "-w", str(tmp_path / "ws"), "-v"])

    output = _flat(result.output)
    assert "route: workflow (plan it)" in output
    assert "scripted" in output  # the router's reason


def test_cli_resume_skips_the_router(tmp_path: Path, monkeypatch: Any) -> None:
    """A resumed run is clearly work: no routing call is spent on it."""
    monkeypatch.chdir(tmp_path)
    workspace = _checkpointed_workspace(tmp_path)
    model = _use_scripted(monkeypatch, _passing_script(router=False))

    result = runner.invoke(app, ["--resume", str(workspace), "--max-attempts", "1"])

    assert result.exit_code == 0, result.output
    # the first call is the planner, not the router, and every scripted response fits
    assert "planner" in model.seen[0][0].content.lower()
    assert len(model.seen) == 8
    assert "route:" not in _flat(result.output)


def test_cli_records_the_session_across_runs(tmp_path: Path, monkeypatch: Any) -> None:
    """Two conversations in the same workspace share one session and count turns."""
    monkeypatch.chdir(tmp_path)
    workspace = tmp_path / "ws"

    _use_scripted(monkeypatch, [_router_answer("chat", 0.95), AIMessage(content="你好！")])
    first = runner.invoke(app, ["你好", "-w", str(workspace)])
    assert first.exit_code == 0, first.output

    _use_scripted(
        monkeypatch,
        [_router_answer("chat", 0.95), AIMessage(content="我是 WQ Agent，可以帮你干活。")],
    )
    second = runner.invoke(app, ["你是谁", "-w", str(workspace)])
    assert second.exit_code == 0, second.output

    session_file = workspace / ".wqagent" / "session" / "session.json"
    summary_file = workspace / ".wqagent" / "session" / "SESSION_SUMMARY.md"
    assert session_file.is_file() and summary_file.is_file()
    payload = json.loads(session_file.read_text(encoding="utf-8"))
    assert payload["turn_index"] == 4  # user+assistant for both runs
    assert [item["role"] for item in payload["recent_turns"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert [item["route"] for item in payload["recent_turns"] if item["role"] == "assistant"] == [
        "chat",
        "chat",
    ]
    assert payload["recent_turns"][0]["content"] == "你好"
    assert payload["recent_turns"][2]["content"] == "你是谁"
    summary = summary_file.read_text(encoding="utf-8")
    assert "你好" in summary and "### 4. assistant (chat)" in summary


def test_cli_session_context_reaches_the_router(tmp_path: Path, monkeypatch: Any) -> None:
    """The second run's router prompt carries the first run's exchange."""
    monkeypatch.chdir(tmp_path)
    workspace = tmp_path / "ws"
    _use_scripted(monkeypatch, [_router_answer("chat"), AIMessage(content="hello")])
    runner.invoke(app, ["帮我看下项目结构", "-w", str(workspace)])

    model = _use_scripted(
        monkeypatch, [_router_answer("workflow")] + _passing_script(router=False)
    )
    result = runner.invoke(app, ["继续", "-w", str(workspace)])

    assert result.exit_code == 0, result.output
    instruction = model.seen[0][1].content  # the router's own call
    assert "帮我看下项目结构" in instruction  # the previous user turn
    assert "session:" in instruction and "turn:" in instruction


def test_cli_session_is_not_written_when_the_run_fails(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A failed run keeps the user's turn (so the session is not lost) but records no reply."""
    monkeypatch.chdir(tmp_path)
    workspace = tmp_path / "ws"

    class Boom(ScriptedModel):
        def invoke(self, messages: Any) -> Any:
            raise RuntimeError("provider exploded")

    monkeypatch.setattr(cli_module, "create_model", lambda **kwargs: Boom([]))

    result = runner.invoke(app, ["do something", "-w", str(workspace)])

    assert result.exit_code == 1
    payload = json.loads(
        (workspace / ".wqagent" / "session" / "session.json").read_text(encoding="utf-8")
    )
    assert [item["role"] for item in payload["recent_turns"]] == ["user"]
    assert payload["turn_index"] == 1


def test_default_workspace_is_created(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _use_scripted(monkeypatch, _passing_script())

    result = runner.invoke(app, ["say hi"])

    assert result.exit_code == 0
    assert (tmp_path / "workspace" / "hello.py").is_file()


def test_workspace_env_var_is_honoured(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "from-env"
    monkeypatch.setenv("WQ_WORKSPACE", str(target))
    _use_scripted(monkeypatch, _passing_script())

    result = runner.invoke(app, ["say hi"])

    assert result.exit_code == 0
    assert (target / "hello.py").is_file()


def test_failed_verification_exits_with_an_error(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _use_scripted(
        monkeypatch,
        [
            _tool_call(
                "todo_write",
                {
                    "plan_summary": "try the thing",
                    "todos": [{"id": "t1", "content": "do the thing"}],
                    "acceptance_criteria": ["it works"],
                    "verification_commands": ["exit 1"],
                },
            ),
            _tool_call("call_code_agent", {"instruction": "do the thing"}, "c2"),
            AIMessage(content="Tried the thing."),
            AIMessage(content="Delegated it."),
            AIMessage(content="It still fails."),
            _tool_call(
                "verification_report",
                {
                    "passed": False,
                    "reason": "the command exits 1",
                    "checks": [{"name": "it works", "passed": False, "detail": "exit 1"}],
                    "recommended_next_instruction": "fix it",
                },
            ),
            AIMessage(content="Not done: `exit 1` still fails."),
        ],
    )

    result = runner.invoke(
        app, ["do the thing", "--max-attempts", "1", "-w", str(tmp_path / "ws")]
    )

    assert result.exit_code == 1
    output = _flat(result.output)
    # the answer goes through the Markdown renderer, so inline-code backticks are gone
    assert "Not done: exit 1 still fails." in output
    assert "Verifier (failed)" in output or "❌ Verifier" in output
    assert "failed" in output  # the verdict row


def test_planner_step_limit_is_reported(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    script: list[Any] = [
        _router_answer(),
        *[
            _tool_call(
                "todo_write",
                {
                    "plan_summary": "loop forever",
                    "todos": [{"id": "t1", "content": "keep going"}],
                    "acceptance_criteria": ["never"],
                    "verification_commands": ["echo ok"],
                },
                f"c{i}",
            )
            for i in range(3)
        ],
    ]
    script.extend(
        [
            AIMessage(content="Inspected."),
            _tool_call(
                "verification_report",
                {
                    "passed": False,
                    "reason": "nothing was produced",
                    "checks": [],
                    "recommended_next_instruction": "actually do the work",
                },
            ),
            AIMessage(content="Not done."),
        ]
    )
    _use_scripted(monkeypatch, script)

    result = runner.invoke(
        app,
        ["loop", "--max-steps", "3", "--max-attempts", "1", "-w", str(tmp_path / "ws")],
    )

    assert result.exit_code == 1
    assert "Stopped after 3 steps" in _flat(result.output)


def test_model_failures_are_reported(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli_module, "create_model", lambda **kwargs: ScriptedModel([])
    )
    result = runner.invoke(app, ["boom", "-w", str(tmp_path / "ws")])
    assert result.exit_code == 1
    assert "AssertionError" in _combined(result)
