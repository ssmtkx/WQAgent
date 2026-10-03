"""Tests for the ``wq`` CLI: flags, config resolution and an end-to-end run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import wqagent.cli.app as cli_module
from langchain_core.messages import AIMessage, BaseMessage
from typer.testing import CliRunner

from wqagent import __version__
from wqagent.cli.app import app
from wqagent.providers import ProviderConfigError

runner = CliRunner()


class ScriptedModel:
    """Fake chat model: replays scripted responses, records bind_tools()."""

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.bound: list[list[str]] = []

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> "ScriptedModel":
        self.bound.append([tool.name for tool in tools])
        return self

    def invoke(self, messages: list[BaseMessage]) -> Any:
        if not self.responses:
            raise AssertionError("ScriptedModel ran out of responses")
        return self.responses.pop(0)


def _tool_call(name: str, args: dict[str, Any], call_id: str = "c1") -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}]
    )


def _passing_script(file_path: str = "hello.py", content: str = "print('hi')\n") -> list[Any]:
    """The eight responses a passing planner -> codeAgent -> verifier -> final run needs."""
    return [
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
