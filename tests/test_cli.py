"""Tests for the ``wq`` CLI: flags, config resolution and an end-to-end run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import wqagent.cli.app as cli_module
from typer.testing import CliRunner

from helpers import (
    ScriptedModel,
    answer,
    chat_script,
    passing_script,
    plan_call,
    report_call,
    route_answer,
    tool_call,
    workflow_script,
)
from wqagent import __version__
from wqagent.cli.app import app
from wqagent.core.state import RuntimeState
from wqagent.providers import ProviderConfigError

runner = CliRunner()
HELLO_CONTENT = "print('hi')\n"
FINAL_TEXT = "Done: created hello.py and verified it."


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
    _use_scripted(monkeypatch, workflow_script(content=HELLO_CONTENT, final_text=FINAL_TEXT))

    result = runner.invoke(app, ["create hello.py", "-w", str(workspace)])

    assert result.exit_code == 0
    assert (workspace / "hello.py").read_text(encoding="utf-8") == HELLO_CONTENT
    output = _flat(result.output)
    assert FINAL_TEXT in output
    # every node and the delegated specialist reported itself (labels fall back to
    # ASCII when the captured stream cannot encode the emoji)
    for marker in ("Planner", "codeAgent", "Verifier", "Final"):
        assert marker in output, marker
    assert "verdict" in output and "passed" in output


# -- the event renderer ------------------------------------------------------
def _rendered(monkeypatch: Any, events: list[dict[str, Any]], *, emoji: bool) -> str:
    """Render *events* into a captured console and return the raw text."""
    from io import StringIO

    from rich.console import Console

    stream = StringIO()
    monkeypatch.setattr(
        cli_module, "console", Console(file=stream, width=200, force_terminal=False)
    )
    monkeypatch.setattr(cli_module, "_can_encode", lambda text: emoji)
    renderer = cli_module._NodeRenderer(verbose=False)
    for event in events:
        renderer.render(event)
    return stream.getvalue()


_HANDOFF_EVENTS: list[dict[str, Any]] = [
    {
        "type": "handoff",
        "node": "planner",
        "from": "planner",
        "to": "codeAgent",
        "instruction": "write hello.py",
    },
    {
        "type": "tool_call",
        "node": "codeAgent",
        "name": "file_write",
        "args": {"file_path": "hello.py", "content": "print('hi')\n"},
    },
    {
        "type": "node_end",
        "node": "planner",
        "update": {
            "plan_summary": "write it",
            "todos": [{"id": "t1", "content": "write it"}],
            "verification_commands": ["echo ok"],
            "agent_handoffs": [{"to_agent": "codeAgent", "instruction": "write hello.py"}],
            "last_actor_summary": "Delegated the implementation.",
        },
    },
]

_CONTEXT_MONITOR_EVENTS: list[dict[str, Any]] = [
    {
        "type": "node_end",
        "node": "context_monitor",
        "update": {
            "context_token_count": 1200,
            "context_token_limit": 400_000,
            "context_should_compress": False,
            "context_next_node": "verifier",
        },
    },
    {
        "type": "node_end",
        "node": "context_monitor",
        "update": {
            "context_token_count": 401_000,
            "context_should_compress": True,
            "context_next_node": "planner",
        },
    },
]

_CONTEXT_COMPRESSOR_EVENTS: list[dict[str, Any]] = [
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
    },
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
    },
    # a fallback compression says so
    {
        "type": "context_compressed",
        "node": "context_compressor",
        "ok": False,
        "error": "RuntimeError: provider exploded",
        "tokens_before": 10,
        "tokens_after": 20,
        "persisted": {"ok": False, "error": "OSError: read-only"},
    },
]

_MEMORY_EVENTS: list[dict[str, Any]] = [
    {
        "type": "memory",
        "node": "planner",
        "digest": "1 todos, 0 sources, 1 handoffs, notepad=yes, history=no",
        "chars": 1517,
        "memory": {"rules": {}, "working_memory": {}, "history_summary_store": {}},
    }
]


@pytest.mark.parametrize(
    ("events", "emoji", "raw", "collapsed", "absent"),
    [
        (
            _HANDOFF_EVENTS,
            True,
            [
                "↪ planner → 🛠 codeAgent",
                "   write hello.py",
                "🛠 codeAgent",  # the specialist introduced itself on its first event
                "📋 Planner",
                "-> file_write hello.py",
                "$ echo ok",
            ],
            [],
            [],
        ),
        (
            _CONTEXT_MONITOR_EVENTS,
            False,
            [],
            [
                "Context",
                "1200/400000 tokens (ok) -> next: verifier",
                "401000 tokens (over budget) -> next: planner",
            ],
            [],
        ),
        (
            _CONTEXT_COMPRESSOR_EVENTS,
            False,
            [],
            [
                "Compressor",
                # before/after comparison: both message count and tokens
                "compressed 24 msgs/412000 tokens -> 1 msg/3100 tokens "
                "[HISTORY_SUMMARY.md, NOTEPAD.md]",
                "24 msgs/412000 tokens -> 1 msg/3100 tokens (limit 400000)",
                "compressed 10 -> 20 tokens (history not persisted)",
                "fallback summary: RuntimeError: provider exploded",
            ],
            ["compressed: hello.py written"],  # only in verbose mode
        ),
        (
            _MEMORY_EVENTS,
            False,
            [],
            [
                "Planner",  # the header opens on the node's first event
                "memory[planner]: 1 todos, 0 sources, 1 handoffs, notepad=yes, "
                "history=no (1517 chars)",
            ],
            [],
        ),
    ],
)
def test_renderer_prints_the_node_events(
    monkeypatch: Any,
    events: list[dict[str, Any]],
    emoji: bool,
    raw: list[str],
    collapsed: list[str],
    absent: list[str],
) -> None:
    """Every node event renders the line the operator needs to read."""
    text = _rendered(monkeypatch, events, emoji=emoji)

    for expected in raw:
        assert expected in text, expected
    for expected in collapsed:
        assert expected in _flat(text), expected
    for unexpected in absent:
        assert unexpected not in _flat(text), unexpected


def _risky_bash_script(command: str = "pip install nothing-here") -> list[Any]:
    """A run whose agent calls bash with a risky command."""
    return [
        route_answer(),
        plan_call(
            summary="install something",
            todos=[{"id": "t1", "content": "install it"}],
            criteria=["it is installed"],
        ),
        tool_call("call_code_agent", {"instruction": "install it"}, "r2"),
        tool_call("bash", {"command": command}, "r3"),
        answer("Ran the install."),
        answer("Done with the install."),
        answer("Inspected."),
        report_call(True, reason="fine", checks=[]),
        answer("Done: handled the install request."),
    ]


@pytest.mark.parametrize(
    ("command", "reply", "expected"),
    [
        (
            "pip install nothing-here",
            "n\n",
            [
                "approval required",
                "Python package installation",
                "command not run - approval required",
                "declined at the prompt",
            ],
        ),
        (
            "npm install --dry-run",
            "y\n",
            ["approval required", "requires_approval: True (approved by the user",
             "Node package installation"],
        ),
    ],
)
def test_cli_approval_prompt(
    tmp_path: Path, monkeypatch: Any, command: str, reply: str, expected: list[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    _use_scripted(monkeypatch, _risky_bash_script(command))

    result = runner.invoke(
        app,
        ["install something", "-w", str(tmp_path / "ws"), "--approval-mode", "inline"],
        input=reply,
    )

    output = _flat(result.output)
    for marker in expected:
        assert marker in output, marker
    assert result.exit_code == 0  # verification still passed, so the run itself is fine


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
    (workspace / "hello.py").write_text(HELLO_CONTENT, encoding="utf-8")
    saved = CheckpointManager(runtime, "write hello.py").save(
        state, status="running", latest_node="verifier"
    )
    assert saved is not None and saved["git_ok"], saved
    # the workspace moved on after the checkpoint
    (workspace / "hello.py").write_text("scribbled over\n", encoding="utf-8")
    return workspace


def test_cli_run_writes_checkpoints(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _use_scripted(monkeypatch, workflow_script())

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
    _use_scripted(monkeypatch, workflow_script())

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
    _use_scripted(monkeypatch, workflow_script())

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
    _use_scripted(monkeypatch, workflow_script())

    full = runner.invoke(app, ["say hi", "-w", str(tmp_path / "ws"), "--trace-mode", "full"])

    assert full.exit_code == 0
    events = list((tmp_path / "ws" / ".wqagent" / "traces").glob("*/events.jsonl"))
    assert len(events) == 1
    lines = events[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) > 5 and json.loads(lines[0])["type"] == "run_start"

    _use_scripted(monkeypatch, workflow_script())
    off = runner.invoke(
        app, ["say hi", "-w", str(tmp_path / "other"), "--trace-mode", "off"]
    )
    assert off.exit_code == 0
    assert not (tmp_path / "other" / ".wqagent" / "traces").exists()


def test_cli_strict_mode_saves_state_and_events(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _use_scripted(monkeypatch, workflow_script())

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
    _use_scripted(monkeypatch, passing_script(content=HELLO_CONTENT, final_text=FINAL_TEXT))

    result = runner.invoke(app, ["--resume", str(workspace), "--max-attempts", "1"])

    output = _flat(result.output)
    assert result.exit_code == 0, output
    assert "resumed" in output
    assert "task : write hello.py" in output
    assert "status : running (latest node: verifier)" in output
    assert "restored : 1 file(s)" in output
    assert "warning :" in output  # the stale file was overwritten
    # the snapshot came back and the run continued from the saved plan
    assert (workspace / "hello.py").read_text(encoding="utf-8") == HELLO_CONTENT
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
@pytest.mark.parametrize(
    ("options", "nodes", "pause", "expected"),
    [
        ({"interval": 0.2}, ["codeAgent"], 0.8, ["still in codeAgent", "no events for"]),
        ({"interval": 0.5}, ["planner"] * 6, 0.1, []),
        ({"interval": 0.1, "enabled": False}, [], 0.4, []),
    ],
)
def test_heartbeat_speaks_only_while_the_run_is_quiet(
    monkeypatch: Any,
    options: dict[str, Any],
    nodes: list[str],
    pause: float,
    expected: list[str],
) -> None:
    """Silence must be explained - but only when it is real, and never when disabled."""
    import time as time_module
    from io import StringIO

    from rich.console import Console

    stream = StringIO()
    monkeypatch.setattr(
        cli_module, "error_console", Console(file=stream, width=200, force_terminal=False)
    )
    monkeypatch.setattr(cli_module, "_can_encode", lambda text: False)

    with cli_module._Heartbeat(**options) as heartbeat:
        for node in nodes:
            heartbeat.touch(node)
            time_module.sleep(pause)

    text = stream.getvalue()
    for marker in expected:
        assert marker in text, marker
    if expected:
        assert text.count("still in") >= 1
    else:
        assert text == ""


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


@pytest.mark.parametrize(
    ("env", "args", "expected", "banner"),
    [
        (None, ["--request-timeout", "45"], 45.0, "request timeout 45s"),
        ("77", [], 77.0, ""),
    ],
)
def test_cli_request_timeout_reaches_the_model(
    tmp_path: Path,
    monkeypatch: Any,
    env: str | None,
    args: list[str],
    expected: float,
    banner: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    if env is None:
        monkeypatch.delenv(cli_module.REQUEST_TIMEOUT_ENV, raising=False)
    else:
        monkeypatch.setenv(cli_module.REQUEST_TIMEOUT_ENV, env)
    seen: dict[str, Any] = {}

    def fake_create_model(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return ScriptedModel(workflow_script())

    monkeypatch.setattr(cli_module, "create_model", fake_create_model)

    result = runner.invoke(app, ["say hi", "-w", str(tmp_path / "ws"), *args])

    assert result.exit_code == 0, result.output
    assert seen["timeout"] == expected
    if banner:
        assert banner in _flat(result.output)  # the banner says so
        shown = runner.invoke(app, ["--show-config", "-w", str(tmp_path / "ws"), *args])
        assert "model/request: 45s per attempt" in _flat(shown.output)


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

    monkeypatch.setattr(cli_module, "create_model", lambda **kwargs: TimedOut(workflow_script()))

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


def _broken_registry() -> list[dict[str, Any]]:
    raise RuntimeError("registry exploded")


@pytest.mark.parametrize(
    ("registry", "node", "idle", "expected", "unexpected"),
    [
        # the usual reason for silence: a long shell command, not the model
        (
            lambda: [
                {
                    "command": "python -m pip install -r requirements.txt",
                    "elapsed": 95.4,
                    "timeout": 120.0,
                }
            ],
            None,
            35.0,
            [
                "bash has been running for 95s",
                "timeout 120s",
                "python -m pip install -r requirements.txt",
                "raise --timeout",
            ],
            ["waiting for the model"],
        ),
        (
            lambda: [
                {"command": "pip install a", "elapsed": 40.0, "timeout": 120.0},
                {"command": "pip install b", "elapsed": 12.0, "timeout": 120.0},
            ],
            None,
            35.0,
            ["(+1 more)"],
            [],
        ),
        (
            lambda: [],
            "codeAgent",
            45.0,
            ["still in codeAgent", "no events for 45s", "waiting for the model"],
            [],
        ),
        (_broken_registry, None, 35.0, ["still in"], []),  # a status line never breaks the run
        (
            lambda: [
                {
                    "command": 'python -m pip install --no-input "fastapi>=0.110"',
                    "elapsed": 208.0,
                    "timeout": 600.0,
                    "tail": [],
                    "bytes": 0,
                }
            ],
            None,
            35.0,
            ["208s", "0 chars of output", "no output at all yet", "slow package index"],
            [],
        ),
        (
            lambda: [
                {
                    "command": "python -m pip install fastapi",
                    "elapsed": 208.0,
                    "timeout": 600.0,
                    "tail": ["Collecting fastapi", "Downloading fastapi-0.110.0-py3-none-any.whl"],
                    "bytes": 4096,
                }
            ],
            None,
            35.0,
            ["Downloading fastapi"],
            ["no output at all yet"],
        ),
    ],
)
def test_heartbeat_explains_what_it_is_waiting_for(
    monkeypatch: Any,
    registry: Any,
    node: str | None,
    idle: float,
    expected: list[str],
    unexpected: list[str],
) -> None:
    from io import StringIO

    from rich.console import Console

    monkeypatch.setattr(
        cli_module, "error_console", Console(file=StringIO(), width=200, force_terminal=False)
    )
    monkeypatch.setattr(cli_module, "_can_encode", lambda text: False)
    monkeypatch.setattr(cli_module, "describe_running_commands", registry)
    heartbeat = cli_module._Heartbeat(interval=30.0)
    if node:
        heartbeat.touch(node)

    text = heartbeat._explain(idle)

    for marker in expected:
        assert marker in text, marker
    for marker in unexpected:
        assert marker not in text, marker


def test_cli_answers_conversation_without_the_workflow(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The entry router: a greeting is answered directly, with no plan and no tools."""
    monkeypatch.chdir(tmp_path)
    _use_scripted(monkeypatch, chat_script("你好！我是 WQ Agent。", confidence=0.95))

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
    _use_scripted(monkeypatch, passing_script())

    result = runner.invoke(
        app, ["say hi", "-w", str(tmp_path / "ws"), "--no-router"]
    )

    output = _flat(result.output)
    assert result.exit_code == 0, output
    assert "Planner" in output
    assert "chat (no tools" not in output


def test_cli_shows_the_route_in_verbose_mode(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _use_scripted(monkeypatch, workflow_script())

    result = runner.invoke(app, ["say hi", "-w", str(tmp_path / "ws"), "-v"])

    output = _flat(result.output)
    assert "route: workflow (plan it)" in output
    assert "scripted" in output  # the router's reason


def test_cli_resume_skips_the_router(tmp_path: Path, monkeypatch: Any) -> None:
    """A resumed run is clearly work: no routing call is spent on it."""
    monkeypatch.chdir(tmp_path)
    workspace = _checkpointed_workspace(tmp_path)
    model = _use_scripted(monkeypatch, passing_script())

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

    _use_scripted(monkeypatch, chat_script("你好！", confidence=0.95))
    first = runner.invoke(app, ["你好", "-w", str(workspace)])
    assert first.exit_code == 0, first.output

    _use_scripted(monkeypatch, chat_script("我是 WQ Agent，可以帮你干活。", confidence=0.95))
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
    _use_scripted(monkeypatch, chat_script())
    runner.invoke(app, ["帮我看下项目结构", "-w", str(workspace)])

    model = _use_scripted(monkeypatch, workflow_script())
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


@pytest.mark.parametrize(
    ("from_env", "target"),
    [(None, "workspace"), ("from-env", "from-env")],
)
def test_workspace_default_and_env_var(
    tmp_path: Path, monkeypatch: Any, from_env: str | None, target: str
) -> None:
    monkeypatch.chdir(tmp_path)
    expected = tmp_path / target
    if from_env:
        monkeypatch.setenv("WQ_WORKSPACE", str(expected))
    _use_scripted(monkeypatch, workflow_script())

    result = runner.invoke(app, ["say hi"])

    assert result.exit_code == 0
    assert (expected / "hello.py").is_file()


def test_failed_verification_exits_with_an_error(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    _use_scripted(
        monkeypatch,
        [
            plan_call(summary="try the thing", commands=["exit 1"]),
            tool_call("call_code_agent", {"instruction": "do the thing"}, "c2"),
            answer("Tried the thing."),
            answer("Delegated it."),
            answer("It still fails."),
            report_call(False, reason="the command exits 1"),
            answer("Not done: `exit 1` still fails."),
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
        route_answer(),
        *[
            plan_call(
                summary="loop forever",
                todos=[{"id": "t1", "content": "keep going"}],
                criteria=["never"],
            )
            for _ in range(3)
        ],
        answer("Inspected."),
        report_call(
            False,
            reason="nothing was produced",
            checks=[],
            recommended_next_instruction="actually do the work",
        ),
        answer("Not done."),
    ]
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
