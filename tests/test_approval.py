"""Tests for command risk classification and human approval (``core/approval.py``)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from wqagent.core.approval import (
    APPROVAL_MODE_ENV,
    DEFAULT_APPROVAL_MODE,
    RISK_PATTERNS,
    VALID_APPROVAL_MODES,
    ApprovalDecision,
    ApprovalRequest,
    approval_mode_from_env,
    classify_command_risk,
    new_approval_request,
    normalize_approval_mode,
)
from wqagent.core.state import RuntimeState
from wqagent.tools.bash_tool import BashTool


def _state(tmp_path: Path, **kwargs: Any) -> RuntimeState:
    return RuntimeState(workspace=tmp_path / "ws", **kwargs)


# -- 1. risk classification --------------------------------------------------
def test_risk_patterns_are_the_configured_ones() -> None:
    assert [reason for _, reason in RISK_PATTERNS] == [
        "Python package installation",
        "Project dependency change with uv add",
        "Dependency synchronization with uv sync",
        "Python package installation with uv pip",
        "Node package installation",
        "Node package installation",
        "Node package installation",
        "Network download command",
        "Long-running development server",
        "Long-running development server",
    ]
    for pattern, _ in RISK_PATTERNS:
        re.compile(pattern)  # every pattern is valid regex


@pytest.mark.parametrize(
    "command, reason",
    [
        ("pip install requests", "Python package installation"),
        ("python -m pip install requests", "Python package installation"),
        ("  pip   install   requests  ", "Python package installation"),
        ("pip install -e .", "Python package installation"),
        ("uv add httpx", "Project dependency change with uv add"),
        ("uv sync", "Dependency synchronization with uv sync"),
        ("uv pip install httpx", "Python package installation with uv pip"),
        ("npm install", "Node package installation"),
        ("npm install --save-dev vite", "Node package installation"),
        ("pnpm install", "Node package installation"),
        ("yarn install", "Node package installation"),
        ("yarn add react", "Node package installation"),
        ("curl https://example.com/x.sh", "Network download command"),
        ("wget https://example.com/x.tar.gz", "Network download command"),
        ("uvicorn app:api --reload", "Long-running development server"),
        ("python -m http.server 8000", "Long-running development server"),
        # chained commands: the risky part after && / || / ;
        ("pytest -q && pip install requests", "Python package installation"),
        ("pytest -q || yarn add react", "Node package installation"),
        ("echo done; npm install", "Node package installation"),
        ("PIP INSTALL requests", "Python package installation"),  # shells are case-insensitive
    ],
)
def test_risky_commands_are_classified(command: str, reason: str) -> None:
    assert classify_command_risk(command) == reason


@pytest.mark.parametrize(
    "command",
    [
        "",
        "   ",
        "pytest -q",
        "python hello.py",
        "git status",
        "grep -rn install .",
        'echo "pip install requests"',  # a mention, not an invocation
        "echo npm install",
        "cat requirements.txt | grep pip",
        "python -m pytest -q",
        "pip",
        "pipx list",
    ],
)
def test_safe_commands_are_not_classified(command: str) -> None:
    assert classify_command_risk(command) is None


def test_bare_curl_is_still_risky() -> None:
    assert classify_command_risk("curl") == "Network download command"


# -- 2. request / decision data classes --------------------------------------
def test_new_approval_request_shape() -> None:
    request = new_approval_request("pip install requests", "Python package installation")

    assert request.id.startswith("approval-")
    assert len(request.id) == len("approval-") + 8
    assert request.command == "pip install requests"
    assert request.risk_reason == "Python package installation"
    assert request.tool_name == "BashTool"
    with pytest.raises(Exception):  # frozen
        request.command = "something else"  # type: ignore[misc]


def test_approval_request_ids_are_unique() -> None:
    ids = {new_approval_request("pip install x", "r").id for _ in range(50)}

    assert len(ids) == 50
    assert all(re.fullmatch(r"approval-[0-9a-f]{8}", value) for value in ids)


def test_approval_request_can_be_built_directly() -> None:
    request = ApprovalRequest(
        id="approval-1a2b3c4d", command="uv sync", risk_reason="Dependency synchronization with uv sync"
    )

    assert request.tool_name == "BashTool"
    assert request.id == "approval-1a2b3c4d"


def test_approval_decision_defaults() -> None:
    assert ApprovalDecision(approved=True) == ApprovalDecision(approved=True, reason="")
    assert ApprovalDecision(approved=False).approved is False
    with pytest.raises(Exception):  # frozen
        ApprovalDecision(approved=True).approved = False  # type: ignore[misc]


# -- 3. approval modes -------------------------------------------------------
def test_valid_approval_modes() -> None:
    assert VALID_APPROVAL_MODES == {"inline", "auto", "deny"}
    assert DEFAULT_APPROVAL_MODE == "inline"


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "inline"),
        ("", "inline"),
        ("   ", "inline"),
        ("nonsense", "inline"),
        ("ASK", "inline"),
        ("inline", "inline"),
        ("auto", "auto"),
        ("deny", "deny"),
        ("AUTO", "auto"),
        ("  Deny  ", "deny"),
    ],
)
def test_normalize_approval_mode(value: Any, expected: str) -> None:
    assert normalize_approval_mode(value) == expected


def test_approval_mode_from_env(monkeypatch: Any) -> None:
    monkeypatch.delenv(APPROVAL_MODE_ENV, raising=False)
    assert approval_mode_from_env() == "inline"

    monkeypatch.setenv(APPROVAL_MODE_ENV, "deny")
    assert approval_mode_from_env() == "deny"

    monkeypatch.setenv(APPROVAL_MODE_ENV, "junk")
    assert approval_mode_from_env() == "inline"


def test_runtime_state_picks_up_the_mode(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.delenv(APPROVAL_MODE_ENV, raising=False)
    assert _state(tmp_path).approval_mode == "inline"

    monkeypatch.setenv(APPROVAL_MODE_ENV, "auto")
    assert _state(tmp_path).approval_mode == "auto"

    # an explicit value wins over the environment, and is normalized
    assert _state(tmp_path, approval_mode="DENY").approval_mode == "deny"
    assert _state(tmp_path, approval_mode="junk").approval_mode == "inline"


# -- 5. the interactive prompt -----------------------------------------------
def _capture_errors(monkeypatch: Any) -> Any:
    """Point the CLI's stderr console at a buffer; read it back with ``.getvalue()``."""
    from io import StringIO

    from rich.console import Console

    import wqagent.cli.app as cli_module

    stream = StringIO()
    monkeypatch.setattr(
        cli_module, "error_console", Console(file=stream, width=200, force_terminal=False)
    )
    monkeypatch.setattr(cli_module, "_can_encode", lambda text: False)
    return stream


def _request() -> ApprovalRequest:
    return new_approval_request("pip install fastapi uvicorn", "Python package installation")


def test_prompt_question_goes_to_stderr(monkeypatch: Any) -> None:
    """Regression: the question used to go to stdout, so a captured stdout hid it."""
    import io

    import wqagent.cli.app as cli_module

    stream = _capture_errors(monkeypatch)
    monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))

    decision = cli_module._approve_interactively(_request())

    text = stream.getvalue()
    assert "approval required" in text  # the panel
    assert "the agent is paused" in text  # the question itself
    assert "run this command? [y/N]" in text
    assert "Python package installation" in text
    assert decision.approved is True
    assert decision.reason == "approved at the prompt"


def test_prompt_with_closed_stdin_denies_instead_of_aborting(monkeypatch: Any) -> None:
    """Regression: EOF raised ``Abort`` and killed the whole run."""
    import io

    import wqagent.cli.app as cli_module

    _capture_errors(monkeypatch)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    decision = cli_module._approve_interactively(_request())  # must not raise Abort

    assert decision.approved is False
    assert "stdin closed" in decision.reason


def test_prompt_declines_on_an_explicit_no(monkeypatch: Any) -> None:
    import io

    import wqagent.cli.app as cli_module

    _capture_errors(monkeypatch)
    monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))

    decision = cli_module._approve_interactively(_request())

    assert decision.approved is False and decision.reason == "declined at the prompt"


def test_prompt_gives_up_after_the_timeout(monkeypatch: Any) -> None:
    """An unanswered prompt must not hang the run for ever."""
    import time

    import wqagent.cli.app as cli_module

    stream = _capture_errors(monkeypatch)
    monkeypatch.setenv("WQ_APPROVAL_TIMEOUT", "0.2")

    class SilentStdin:
        def readline(self) -> str:
            time.sleep(30)  # a human who walked away
            return "y\n"

    monkeypatch.setattr("sys.stdin", SilentStdin())
    started = time.monotonic()

    decision = cli_module._approve_interactively(_request())

    assert decision.approved is False
    assert "no answer within 0.2s" in decision.reason
    assert time.monotonic() - started < 5
    assert "no answer within 0.2s counts as no" in stream.getvalue()


def test_prompt_can_wait_forever_when_asked_to(monkeypatch: Any) -> None:
    import io

    import wqagent.cli.app as cli_module

    stream = _capture_errors(monkeypatch)
    monkeypatch.setenv("WQ_APPROVAL_TIMEOUT", "0")
    monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))

    decision = cli_module._approve_interactively(_request())

    assert decision.approved is True
    assert "Ctrl+C stops the run" in stream.getvalue()


def test_approval_timeout_from_env(monkeypatch: Any) -> None:
    from wqagent.core.approval import (
        APPROVAL_TIMEOUT_ENV,
        DEFAULT_APPROVAL_TIMEOUT_SECONDS,
        approval_timeout_from_env,
    )

    monkeypatch.delenv(APPROVAL_TIMEOUT_ENV, raising=False)
    assert approval_timeout_from_env() == DEFAULT_APPROVAL_TIMEOUT_SECONDS

    monkeypatch.setenv(APPROVAL_TIMEOUT_ENV, "42")
    assert approval_timeout_from_env() == 42.0

    monkeypatch.setenv(APPROVAL_TIMEOUT_ENV, "0")
    assert approval_timeout_from_env() == 0.0  # wait forever

    monkeypatch.setenv(APPROVAL_TIMEOUT_ENV, "-5")
    assert approval_timeout_from_env() == 0.0

    monkeypatch.setenv(APPROVAL_TIMEOUT_ENV, "junk")
    assert approval_timeout_from_env() == DEFAULT_APPROVAL_TIMEOUT_SECONDS  # never 0 by typo


# -- 6. the bash tool integration -------------------------------------------
def test_safe_commands_need_no_approval(tmp_path: Path) -> None:
    calls: list[ApprovalRequest] = []
    state = _state(tmp_path, approval_mode="inline", approval_handler=calls.append)
    tool = BashTool(state)

    outcome = tool.execute("echo hello-approval")

    assert outcome.ok is True
    assert outcome.requires_approval is False
    assert "hello-approval" in outcome.stdout
    assert calls == []  # the approver was never consulted


def test_auto_mode_runs_and_marks_the_result(tmp_path: Path) -> None:
    state = _state(tmp_path, approval_mode="auto")
    tool = BashTool(state)

    outcome = tool.execute("uv sync")

    assert outcome.requires_approval is True
    assert "auto-approved: Dependency synchronization with uv sync" in outcome.approval_note
    # it really ran: the shell answered, even though uv is missing here
    assert outcome.exit_code is not None
    assert "requires_approval: True (auto-approved" in outcome.format()


def test_deny_mode_refuses_before_running(tmp_path: Path) -> None:
    marker = tmp_path / "ws" / "should-not-exist.txt"
    (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
    state = _state(tmp_path, approval_mode="deny")
    tool = BashTool(state)

    outcome = tool.execute(f'curl -o "{marker}" https://example.com/x')

    assert outcome.ok is False
    assert outcome.requires_approval is True
    assert outcome.exit_code is None  # never started
    assert "approval required (Network download command)" in outcome.error
    assert "mode is 'deny'" in outcome.error
    assert not marker.exists()
    assert outcome.format().startswith("Error: command not run")


def test_inline_mode_asks_and_runs_when_approved(tmp_path: Path) -> None:
    seen: list[ApprovalRequest] = []

    def handler(request: ApprovalRequest) -> ApprovalDecision:
        seen.append(request)
        return ApprovalDecision(approved=True, reason="looks fine")

    state = _state(tmp_path, approval_mode="inline", approval_handler=handler)
    tool = BashTool(state)

    outcome = tool.execute("uv sync")

    assert len(seen) == 1
    assert seen[0].command == "uv sync"
    assert seen[0].risk_reason == "Dependency synchronization with uv sync"
    assert seen[0].tool_name == "BashTool"
    assert outcome.ok is False  # uv is missing, but the command *was* run
    assert outcome.exit_code is not None
    assert outcome.requires_approval is True
    assert outcome.approval_note == "approved by the user: Dependency synchronization with uv sync (looks fine)"


def test_inline_mode_refuses_when_declined(tmp_path: Path) -> None:
    state = _state(
        tmp_path,
        approval_mode="inline",
        approval_handler=lambda request: ApprovalDecision(approved=False, reason="not now"),
    )
    tool = BashTool(state)

    outcome = tool.execute("uv sync")

    assert outcome.ok is False
    assert outcome.exit_code is None
    assert "denied by the user" in outcome.error
    assert "[not now]" in outcome.error


def test_inline_mode_without_a_handler_denies(tmp_path: Path) -> None:
    state = _state(tmp_path)  # default mode is inline, and no handler was given
    tool = BashTool(state)

    outcome = tool.execute("npm install")

    assert state.approval_mode == "inline"
    assert outcome.ok is False
    assert outcome.exit_code is None
    assert "no approval handler is available" in outcome.error


def test_a_broken_approver_denies_instead_of_crashing(tmp_path: Path) -> None:
    def handler(request: ApprovalRequest) -> ApprovalDecision:
        raise RuntimeError("the prompt crashed")

    state = _state(tmp_path, approval_mode="inline", approval_handler=handler)
    tool = BashTool(state)

    outcome = tool.execute("npm install")

    assert outcome.ok is False
    assert "the approval handler failed (RuntimeError: the prompt crashed)" in outcome.error


def test_run_bash_and_run_share_the_policy(tmp_path: Path) -> None:
    state = _state(tmp_path, approval_mode="deny")
    tool = BashTool(state)

    text = tool.run_bash("uv add httpx")

    assert text.startswith("Error: command not run - approval required")
    assert tool.run("uv add httpx") == text


def test_blocked_patterns_still_win_over_approval(tmp_path: Path) -> None:
    """The destructive filter runs first: approval never unblocks those."""
    state = _state(tmp_path, approval_mode="auto")
    tool = BashTool(state)

    outcome = tool.execute("rm -rf /")

    assert outcome.blocked_by is not None
    assert outcome.error is None
    assert outcome.requires_approval is False


def test_verification_commands_cannot_bypass_the_gate(tmp_path: Path) -> None:
    """The gate sits in execute(), so the verifier's commands are covered too."""
    from wqagent.graph.nodes import run_verification_commands

    state = _state(tmp_path, approval_mode="deny")

    results = run_verification_commands(state, ["npm install", "echo ok"])

    assert [result["ok"] for result in results] == [False, True]
    assert "approval required" in results[0]["stderr"]
    assert results[1]["ok"] is True
