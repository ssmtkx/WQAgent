"""Tests for the bash tool: output capture, exit codes, timeout, safety filter."""

from __future__ import annotations

import os

from wqagent.core.state import RuntimeState
from wqagent.tools.bash_tool import SHELL_NAME, BashTool, shell_guidance


def test_description_tells_the_model_which_shell_it_gets() -> None:
    assert SHELL_NAME in BashTool.description
    assert "syntax" in BashTool.description


def test_shell_guidance_matches_the_platform() -> None:
    guidance = shell_guidance()

    assert SHELL_NAME in guidance or os.name != "nt"
    if os.name == "nt":
        # the constructs that actually broke live runs on Windows PowerShell
        for trap in ("&&", "test", "python3"):
            assert trap in guidance, trap
    else:
        assert "POSIX" in guidance


def _sleep(seconds: int) -> str:
    return f"Start-Sleep -Seconds {seconds}" if os.name == "nt" else f"sleep {seconds}"


def _print_cwd() -> str:
    return "Get-Location" if os.name == "nt" else "pwd"


def test_runs_a_command_and_captures_stdout(state: RuntimeState) -> None:
    output = BashTool(state).run("echo hello-wq")
    assert "exit_code: 0" in output
    assert "hello-wq" in output
    assert "stdout:" in output


def test_reports_non_zero_exit_codes(state: RuntimeState) -> None:
    assert "exit_code: 3" in BashTool(state).run("exit 3")


def test_captures_stderr(state: RuntimeState) -> None:
    command = (
        "[Console]::Error.WriteLine('boom')"
        if os.name == "nt"
        else "echo boom 1>&2"
    )
    output = BashTool(state).run(command)
    assert "stderr:" in output
    assert "boom" in output


def test_runs_inside_the_workspace(state: RuntimeState) -> None:
    output = BashTool(state).run(_print_cwd())
    assert str(state.workspace) in output.replace("\\", "\\")


def test_command_can_write_into_the_workspace(state: RuntimeState) -> None:
    command = "Set-Content -Path made.txt -Value hi" if os.name == "nt" else "echo hi > made.txt"
    BashTool(state).run(command)
    assert (state.workspace / "made.txt").exists()


def test_timeout_kills_the_command(state: RuntimeState) -> None:
    output = BashTool(state).run(_sleep(30), timeout_seconds=1)
    assert "timed out after 1s" in output
    assert "exit_code" in output


def test_destructive_commands_are_refused(state: RuntimeState) -> None:
    output = BashTool(state).run("rm -rf /")
    assert output.startswith("Error:")
    assert "safety filter" in output


def test_empty_command_is_rejected(state: RuntimeState) -> None:
    assert BashTool(state).run("   ").startswith("Error:")


def test_output_is_clipped_to_the_runtime_budget(state: RuntimeState) -> None:
    state.max_output_chars = 600
    command = (
        "1..200 | ForEach-Object { 'x' * 60 }" if os.name == "nt" else "yes x | head -n 200"
    )
    output = BashTool(state).run(command)
    assert "bash output truncated" in output
    assert len(output) < 1_500
