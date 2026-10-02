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
        self.bound: list[Any] | None = None

    def bind_tools(self, tools: list[Any]) -> "ScriptedModel":
        self.bound = list(tools)
        return self

    def invoke(self, messages: list[BaseMessage]) -> Any:
        if not self.responses:
            raise AssertionError("ScriptedModel ran out of responses")
        return self.responses.pop(0)


def _tool_call(name: str, args: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": "call_1", "type": "tool_call"}]
    )


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
    monkeypatch.setattr(
        cli_module,
        "create_model",
        lambda **kwargs: ScriptedModel(
            [
                _tool_call("file_write", {"file_path": "hello.py", "content": "print('hi')\n"}),
                AIMessage(content="Created hello.py."),
            ]
        ),
    )
    result = runner.invoke(app, ["create hello.py", "-w", str(workspace)])

    assert result.exit_code == 0
    assert (workspace / "hello.py").read_text(encoding="utf-8") == "print('hi')\n"
    assert "Created hello.py." in result.output


def test_default_workspace_is_created(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli_module,
        "create_model",
        lambda **kwargs: ScriptedModel([AIMessage(content="nothing to do")]),
    )
    result = runner.invoke(app, ["say hi"])
    assert result.exit_code == 0
    assert (tmp_path / "workspace").is_dir()


def test_workspace_env_var_is_honoured(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "from-env"
    monkeypatch.setenv("WQ_WORKSPACE", str(target))
    monkeypatch.setattr(
        cli_module,
        "create_model",
        lambda **kwargs: ScriptedModel([AIMessage(content="ok")]),
    )
    result = runner.invoke(app, ["say hi"])
    assert result.exit_code == 0
    assert target.is_dir()


def test_step_limit_exits_with_an_error(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli_module,
        "create_model",
        lambda **kwargs: ScriptedModel(
            [_tool_call("file_read", {"file_path": "missing.txt"}) for _ in range(3)]
        ),
    )
    result = runner.invoke(app, ["loop", "--max-steps", "3", "-w", str(tmp_path / "ws")])
    assert result.exit_code == 1
    assert "Stopped after 3 steps" in result.output


def test_model_failures_are_reported(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli_module, "create_model", lambda **kwargs: ScriptedModel([])
    )
    result = runner.invoke(app, ["boom", "-w", str(tmp_path / "ws")])
    assert result.exit_code == 1
    assert "AssertionError" in _combined(result)
