"""Tests for command risk classification and human approval (``core/approval.py``)."""

from __future__ import annotations

import io
import re
import time
from pathlib import Path
from typing import Any

import pytest
import wqagent.cli.app as cli_module
from rich.console import Console

from helpers import runtime_for, workspace_dir
from wqagent.core.approval import (
    APPROVAL_MODE_ENV,
    APPROVAL_TIMEOUT_ENV,
    DEFAULT_APPROVAL_MODE,
    DEFAULT_APPROVAL_TIMEOUT_SECONDS,
    RISK_PATTERNS,
    VALID_APPROVAL_MODES,
    ApprovalDecision,
    ApprovalRequest,
    approval_mode_from_env,
    approval_timeout_from_env,
    classify_command_risk,
    new_approval_request,
    normalize_approval_mode,
)
from wqagent.graph.nodes import run_verification_commands
from wqagent.tools.bash_tool import BashTool


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
        # risky: a pattern must start the command (or follow && / || / ;)
        ("pip install requests", "Python package installation"),
        ("python -m pip install requests", "Python package installation"),
        ("  pip   install   requests  ", "Python package installation"),
        ("pip install -e .", "Python package installation"),
        ("PIP INSTALL requests", "Python package installation"),  # shells are case-insensitive
        ("uv add httpx", "Project dependency change with uv add"),
        ("uv sync", "Dependency synchronization with uv sync"),
        ("uv pip install httpx", "Python package installation with uv pip"),
        ("npm install", "Node package installation"),
        ("npm install --save-dev vite", "Node package installation"),
        ("pnpm install", "Node package installation"),
        ("yarn install", "Node package installation"),
        ("yarn add react", "Node package installation"),
        ("curl", "Network download command"),  # bare curl is still risky
        ("curl https://example.com/x.sh", "Network download command"),
        ("wget https://example.com/x.tar.gz", "Network download command"),
        ("uvicorn app:api --reload", "Long-running development server"),
        ("python -m http.server 8000", "Long-running development server"),
        # chained commands: the risky part after && / || / ;
        ("pytest -q && pip install requests", "Python package installation"),
        ("pytest -q || yarn add react", "Node package installation"),
        ("echo done; npm install", "Node package installation"),
        # safe: a mention, not an invocation
        ("", None),
        ("   ", None),
        ("pytest -q", None),
        ("python hello.py", None),
        ("git status", None),
        ("grep -rn install .", None),
        ('echo "pip install requests"', None),
        ("echo npm install", None),
        ("cat requirements.txt | grep pip", None),
        ("python -m pytest -q", None),
        ("pip", None),
        ("pipx list", None),
    ],
)
def test_command_risk_classification(command: str, reason: str | None) -> None:
    assert classify_command_risk(command) == reason


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
    assert runtime_for(tmp_path / "ws").approval_mode == "inline"

    monkeypatch.setenv(APPROVAL_MODE_ENV, "auto")
    assert runtime_for(tmp_path / "ws").approval_mode == "auto"

    # an explicit value wins over the environment, and is normalized
    assert runtime_for(tmp_path / "ws", approval_mode="DENY").approval_mode == "deny"
    assert runtime_for(tmp_path / "ws", approval_mode="junk").approval_mode == "inline"


# -- 5. the interactive prompt -----------------------------------------------
def _capture_errors(monkeypatch: Any) -> Any:
    """Point the CLI's stderr console at a buffer; read it back with ``.getvalue()``."""
    stream = io.StringIO()
    monkeypatch.setattr(
        cli_module, "error_console", Console(file=stream, width=200, force_terminal=False)
    )
    monkeypatch.setattr(cli_module, "_can_encode", lambda text: False)
    return stream


def _request() -> ApprovalRequest:
    return new_approval_request("pip install fastapi uvicorn", "Python package installation")


@pytest.mark.parametrize(
    "typed, approved, reason, fragments",
    [
        # Regression: the question used to go to stdout, so a captured stdout hid it
        (
            "y\n",
            True,
            "approved at the prompt",
            (
                "approval required",  # the panel
                "the agent is paused",  # the question itself
                "run this command? [y/N]",
                "Python package installation",
            ),
        ),
        # Regression: EOF raised ``Abort`` and killed the whole run
        ("", False, "no answer at the prompt (stdin closed?)", ()),
        ("n\n", False, "declined at the prompt", ()),
    ],
)
def test_prompt_answer_decides(
    monkeypatch: Any, typed: str, approved: bool, reason: str, fragments: tuple[str, ...]
) -> None:
    """The panel and the question go to stderr; y, n and a closed stdin decide."""
    stream = _capture_errors(monkeypatch)
    monkeypatch.setattr("sys.stdin", io.StringIO(typed))

    decision = cli_module._approve_interactively(_request())

    assert decision.approved is approved
    assert decision.reason == reason
    text = stream.getvalue()
    for fragment in fragments:
        assert fragment in text


def test_prompt_gives_up_after_the_timeout(monkeypatch: Any) -> None:
    """An unanswered prompt must not hang the run for ever."""
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
    stream = _capture_errors(monkeypatch)
    monkeypatch.setenv("WQ_APPROVAL_TIMEOUT", "0")
    monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))

    decision = cli_module._approve_interactively(_request())

    assert decision.approved is True
    assert "Ctrl+C stops the run" in stream.getvalue()


def test_approval_timeout_from_env(monkeypatch: Any) -> None:
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
    state = runtime_for(tmp_path / "ws", approval_mode="inline", approval_handler=calls.append)
    tool = BashTool(state)

    outcome = tool.execute("echo hello-approval")

    assert outcome.ok is True
    assert outcome.requires_approval is False
    assert "hello-approval" in outcome.stdout
    assert calls == []  # the approver was never consulted


def test_auto_mode_runs_and_marks_the_result(tmp_path: Path) -> None:
    tool = BashTool(runtime_for(tmp_path / "ws", approval_mode="auto"))

    outcome = tool.execute("uv sync")

    assert outcome.requires_approval is True
    assert "auto-approved: Dependency synchronization with uv sync" in outcome.approval_note
    # it really ran: the shell answered, even though uv is missing here
    assert outcome.exit_code is not None
    assert "requires_approval: True (auto-approved" in outcome.format()


def test_deny_mode_refuses_before_running(tmp_path: Path) -> None:
    marker = workspace_dir(tmp_path) / "should-not-exist.txt"
    tool = BashTool(runtime_for(tmp_path / "ws", approval_mode="deny"))

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

    tool = BashTool(runtime_for(tmp_path / "ws", approval_mode="inline", approval_handler=handler))

    outcome = tool.execute("uv sync")

    assert len(seen) == 1
    assert seen[0].command == "uv sync"
    assert seen[0].risk_reason == "Dependency synchronization with uv sync"
    assert seen[0].tool_name == "BashTool"
    assert outcome.ok is False  # uv is missing, but the command *was* run
    assert outcome.exit_code is not None
    assert outcome.requires_approval is True
    assert outcome.approval_note == "approved by the user: Dependency synchronization with uv sync (looks fine)"


def _declines(request: ApprovalRequest) -> ApprovalDecision:
    """An approver who said no."""
    return ApprovalDecision(approved=False, reason="not now")


def _raises(request: ApprovalRequest) -> ApprovalDecision:
    """An approver whose prompt crashed."""
    raise RuntimeError("the prompt crashed")


@pytest.mark.parametrize(
    "options, handler, command, fragments",
    [
        # no handler and no explicit mode: the default (inline) policy denies
        ({}, None, "npm install", ("no approval handler is available",)),
        ({"approval_mode": "inline"}, _declines, "uv sync", ("denied by the user", "[not now]")),
        (
            {"approval_mode": "inline"}, _raises, "npm install",
            ("the approval handler failed (RuntimeError: the prompt crashed)",),
        ),
    ],
)
def test_inline_mode_denies_without_a_yes(
    tmp_path: Path, options: dict[str, Any], handler: Any, command: str, fragments: tuple[str, ...]
) -> None:
    if handler is not None:
        options = {**options, "approval_handler": handler}
    state = runtime_for(tmp_path / "ws", **options)

    outcome = BashTool(state).execute(command)

    assert state.approval_mode == "inline"
    assert outcome.ok is False
    assert outcome.exit_code is None
    for fragment in fragments:
        assert fragment in outcome.error


def test_run_bash_and_run_share_the_policy(tmp_path: Path) -> None:
    tool = BashTool(runtime_for(tmp_path / "ws", approval_mode="deny"))

    text = tool.run_bash("uv add httpx")

    assert text.startswith("Error: command not run - approval required")
    assert tool.run("uv add httpx") == text


def test_blocked_patterns_still_win_over_approval(tmp_path: Path) -> None:
    """The destructive filter runs first: approval never unblocks those."""
    tool = BashTool(runtime_for(tmp_path / "ws", approval_mode="auto"))

    outcome = tool.execute("rm -rf /")

    assert outcome.blocked_by is not None
    assert outcome.error is None
    assert outcome.requires_approval is False


def test_verification_commands_cannot_bypass_the_gate(tmp_path: Path) -> None:
    """The gate sits in execute(), so the verifier's commands are covered too."""
    state = runtime_for(tmp_path / "ws", approval_mode="deny")

    results = run_verification_commands(state, ["npm install", "echo ok"])

    assert [result["ok"] for result in results] == [False, True]
    assert "approval required" in results[0]["stderr"]
    assert results[1]["ok"] is True
