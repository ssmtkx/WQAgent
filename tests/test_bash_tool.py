"""Tests for the bash tool: output capture, exit codes, timeout, safety filter."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pytest

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


def test_child_env_points_relative_temp_dirs_inside_the_workspace(
    state: RuntimeState, monkeypatch: Any
) -> None:
    """Regression: `TEMP=.` made pip litter the workspace with pip-* directories."""
    tool = BashTool(state)
    monkeypatch.setenv("TEMP", ".")
    monkeypatch.setenv("TMP", "relative-tmp")
    monkeypatch.delenv("TMPDIR", raising=False)

    env = tool._child_env()

    for name in ("TEMP", "TMP", "TMPDIR"):
        assert Path(env[name]).is_absolute(), name
        assert Path(env[name]).is_dir(), name
        assert Path(env[name]).is_relative_to(state.workspace), name
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["PYTHONUNBUFFERED"] == "1"  # a buffered child looks like a hung one
    assert env["PIP_NO_INPUT"] == "1"
    assert env["PIP_DISABLE_PIP_VERSION_CHECK"] == "1"
    assert env["WQ_WORKSPACE"] == str(state.workspace)


def test_child_env_keeps_the_users_own_pip_settings(
    state: RuntimeState, monkeypatch: Any
) -> None:
    monkeypatch.setenv("PIP_NO_INPUT", "0")
    monkeypatch.setenv("PIP_DISABLE_PIP_VERSION_CHECK", "0")

    env = BashTool(state)._child_env()

    assert env["PIP_NO_INPUT"] == "0"
    assert env["PIP_DISABLE_PIP_VERSION_CHECK"] == "0"


def test_child_env_keeps_an_absolute_temp(state: RuntimeState, monkeypatch: Any) -> None:
    own_temp = state.workspace.parent / "own-temp"
    own_temp.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TEMP", str(own_temp))

    assert BashTool(state)._child_env()["TEMP"] == str(own_temp)


def test_a_child_temp_dir_never_lands_in_the_workspace(
    state: RuntimeState, monkeypatch: Any
) -> None:
    """What pip does: create a temp dir. It must not end up in the workspace root."""
    monkeypatch.setenv("TEMP", ".")
    monkeypatch.setenv("TMP", ".")
    tool = BashTool(state)
    code = (
        "import pathlib, tempfile; "
        "pathlib.Path(tempfile.mkdtemp(prefix='pip-install-')).mkdir(exist_ok=True); "
        "print('made a temp dir')"
    )

    outcome = tool.execute(f'python -c "{code}"')

    assert outcome.ok is True, outcome.format()
    assert "made a temp dir" in outcome.stdout
    leftovers = sorted(
        path.name for path in state.workspace.iterdir() if path.name.startswith("pip")
    )
    assert leftovers == [], f"temp dirs leaked into the workspace: {leftovers}"
    staged = state.workspace / ".wqagent" / "tmp"
    assert any(staged.iterdir())
    # leave the temporary tree tidy (pytest's own cleanup trips over these dirs)
    for path in sorted(staged.iterdir(), reverse=True):
        shutil.rmtree(path, ignore_errors=True)


def test_output_is_readable_while_the_command_runs(state: RuntimeState) -> None:
    """The point of live readers: you can see what a long command is doing."""
    import threading
    import time as time_module

    from wqagent.tools.bash_tool import describe_running_commands

    if os.name == "nt":
        command = (
            "Write-Output 'Collecting fastapi'; Start-Sleep -Seconds 6; "
            "Write-Output 'Building wheel for pydantic-core'"
        )
    else:
        command = "echo Collecting fastapi; sleep 6; echo Built"
    result: dict[str, Any] = {}
    worker = threading.Thread(
        target=lambda: result.setdefault("value", BashTool(state).execute(command))
    )
    worker.start()
    seen: list[dict[str, Any]] = []
    for _ in range(60):
        described = describe_running_commands()
        if described:
            seen.append(described[0])
            if any("Collecting fastapi" in line for line in described[0].get("tail") or []):
                break
        time_module.sleep(0.1)

    assert seen, "the command should be reported while it runs"
    assert any(
        any("Collecting fastapi" in line for line in item.get("tail") or []) for item in seen
    ), seen[-1]
    assert seen[-1]["bytes"] > 0  # output seen live, not only at the end
    from wqagent.tools.bash_tool import kill_running_commands

    kill_running_commands()
    worker.join(timeout=20)
    assert "Collecting fastapi" in result["value"].stdout  # the full output is kept


def test_stdin_is_not_inherited(state: RuntimeState) -> None:
    """A hidden prompt must fail fast instead of blocking the run for ever."""
    code = "import sys; print('reading'); sys.stdin.readline(); print('done')"
    outcome = BashTool(state).execute(f'python -c "{code}"')

    assert outcome.timed_out is False, "reading stdin must not hang"
    assert outcome.exit_code == 0
    assert "done" in outcome.stdout


@pytest.mark.skipif(os.name != "nt", reason="Windows-only pipe inheritance check")
def test_a_child_holding_the_output_open_does_not_hang_the_run(
    state: RuntimeState,
) -> None:
    """Regression: a finished install hung for ever because a grandchild held the pipes.

    The command starts a detached process that inherits stdout, then exits at once. The
    tool must return immediately (with a note) instead of waiting for that pipe to close.
    """
    import time as time_module

    command = (
        "$child = Start-Process -FilePath powershell -PassThru -NoNewWindow "
        "-ArgumentList '-Command','Start-Sleep -Seconds 30'; "
        "Write-Output \"detached pid=$($child.Id)\""
    )
    started = time_module.monotonic()

    outcome = BashTool(state).execute(command, timeout_seconds=60)
    elapsed = time_module.monotonic() - started

    try:
        assert elapsed < 25, f"the tool waited for the detached child ({elapsed:.1f}s)"
        assert outcome.output_open is True
        assert "still holds its output open" in outcome.format()
        assert "detached pid=" in outcome.stdout
    finally:
        for line in outcome.stdout.splitlines():  # clean up the detached process
            if "detached pid=" in line:
                pid = line.split("detached pid=")[-1].strip()
                if pid.isdigit():
                    os.system(f"taskkill /F /PID {pid} > NUL 2>&1")


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
