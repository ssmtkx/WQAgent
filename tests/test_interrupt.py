"""Tests for Ctrl+C handling: the guard, the watcher and the kill registry.

The interesting case is *not* a KeyboardInterrupt raised inside a node (that works and
is covered in ``test_stream_entry.py``): langgraph runs each sync node in a worker
thread and the main thread waits for it with ``concurrent.futures.wait``, which on
Windows cannot run Python-level signal handlers. The guard therefore reacts from a
watcher thread reading ``signal.set_wakeup_fd``, kills the running shell command, and
only then does the pending KeyboardInterrupt travel through the main thread.
"""

from __future__ import annotations

import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import wqagent.cli.app as cli_module
from wqagent.core.state import RuntimeState
from wqagent.tools import bash_tool as bash_module
from wqagent.tools.bash_tool import (
    BashTool,
    kill_running_commands,
    running_command_count,
)

# -- the guard ---------------------------------------------------------------
def _capture_errors(monkeypatch: Any) -> Any:
    """Point the guard's console at a buffer; read it back with ``.getvalue()``."""
    from io import StringIO

    from rich.console import Console

    stream = StringIO()
    monkeypatch.setattr(
        cli_module, "error_console", Console(file=stream, width=200, force_terminal=False)
    )
    monkeypatch.setattr(cli_module, "_can_encode", lambda text: False)
    return stream


def test_first_interrupt_kills_the_command_and_explains(monkeypatch: Any) -> None:
    printed = _capture_errors(monkeypatch)
    killed: list[int] = []
    guard = cli_module._InterruptGuard(killer=lambda: (killed.append(1), 3)[1])

    with guard:
        guard._on_signal()

    assert killed == [1] and guard.presses == 1
    text = printed.getvalue()
    assert "interrupt received" in text
    assert "stopped 3 running command(s)" in text
    assert "press Ctrl+C again to exit immediately" in text


def test_first_interrupt_without_a_running_command(monkeypatch: Any) -> None:
    printed = _capture_errors(monkeypatch)
    guard = cli_module._InterruptGuard(killer=lambda: 0)

    guard._on_signal()

    assert guard.presses == 1
    text = printed.getvalue()
    assert "stopped" not in text
    assert "interrupt received" in text


def test_second_interrupt_exits_immediately(monkeypatch: Any) -> None:
    printed = _capture_errors(monkeypatch)
    codes: list[int] = []
    monkeypatch.setattr(cli_module.os, "_exit", lambda code: codes.append(code))
    guard = cli_module._InterruptGuard(killer=lambda: 0)

    guard._on_signal()  # first: explain
    assert codes == []
    guard._on_signal()  # second: leave now

    assert codes == [cli_module.EXIT_INTERRUPTED]
    assert "exiting now" in printed.getvalue()


def test_a_broken_killer_never_raises_from_the_handler(monkeypatch: Any) -> None:
    printed = _capture_errors(monkeypatch)

    def boom() -> int:
        raise RuntimeError("taskkill exploded")

    guard = cli_module._InterruptGuard(killer=boom)

    guard._on_signal()  # must not raise

    assert guard.presses == 1
    assert "interrupt received" in printed.getvalue()


def test_python_handler_raises_and_leaves_the_work_to_the_watcher() -> None:
    guard = cli_module._InterruptGuard(killer=lambda: 0)
    guard._watcher = object()  # pretend the watcher is running

    with pytest.raises(KeyboardInterrupt):
        guard._handle(signal.SIGINT, None)

    assert guard.presses == 0  # the watcher owns the reaction (it runs earlier)


def test_python_handler_acts_when_there_is_no_watcher(monkeypatch: Any) -> None:
    _capture_errors(monkeypatch)
    killed: list[int] = []
    guard = cli_module._InterruptGuard(killer=lambda: (killed.append(1), 1)[1])
    guard._watcher = None

    with pytest.raises(KeyboardInterrupt):
        guard._handle(signal.SIGINT, None)

    assert killed == [1] and guard.presses == 1  # fallback path still kills


def test_watcher_reacts_to_the_wakeup_socket(monkeypatch: Any) -> None:
    """The C-level path: a byte on the wakeup fd must trigger the kill at once."""
    _capture_errors(monkeypatch)
    killed: list[int] = []
    guard = cli_module._InterruptGuard(killer=lambda: (killed.append(1), 1)[1])

    with guard:
        assert guard._watcher is not None, "socketpair unavailable on this platform"
        guard._right.send(b"\x02")  # what the C-level handler writes for SIGINT
        for _ in range(60):
            if killed:
                break
            time.sleep(0.05)

    assert killed == [1] and guard.presses == 1


def test_guard_restores_the_previous_state(monkeypatch: Any) -> None:
    original = signal.getsignal(signal.SIGINT)
    original_wakeup = signal.set_wakeup_fd(-1)
    signal.set_wakeup_fd(original_wakeup)

    with cli_module._InterruptGuard(killer=lambda: 0):
        assert signal.getsignal(signal.SIGINT) != original

    assert signal.getsignal(signal.SIGINT) is original
    assert signal.set_wakeup_fd(original_wakeup) == -1  # restored, and no fd left behind


# -- the kill registry -------------------------------------------------------
def _slow_command(marker: Path) -> str:
    if os.name == "nt":  # pragma: no cover - platform specific
        return f"Start-Sleep -Seconds 5; Set-Content -Path '{marker}' -Value done"
    return f"sleep 5; touch {marker}"  # pragma: no cover - platform specific


def test_kill_running_commands_stops_a_live_command(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    marker = workspace / "finished.txt"
    tool = BashTool(RuntimeState(workspace=workspace))
    outcome: dict[str, Any] = {}

    worker = threading.Thread(
        target=lambda: outcome.setdefault("value", tool.execute(_slow_command(marker), timeout_seconds=30))
    )
    worker.start()
    for _ in range(100):
        if running_command_count():
            break
        time.sleep(0.05)
    assert running_command_count() == 1, "the command should be registered while it runs"

    assert kill_running_commands() == 1
    worker.join(timeout=20)

    assert not worker.is_alive(), "the tool must return once its command is killed"
    assert not marker.exists(), "the command was killed, so it never reached its last step"
    assert outcome["value"].exit_code != 0
    assert running_command_count() == 0  # the registry is clean again


def test_kill_running_commands_is_safe_when_idle() -> None:
    assert kill_running_commands() == 0
    assert running_command_count() == 0


def test_execute_kills_the_child_when_the_wait_is_interrupted(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A KeyboardInterrupt during the wait must not leave the command running."""
    kills: list[str] = []

    class FakeProcess:
        pid = 4242
        returncode = 0
        stdout = None  # no pipes to read
        stderr = None

        def wait(self, timeout: float | None = None) -> int:
            raise KeyboardInterrupt

        def poll(self) -> None:
            return None

        def kill(self) -> None:  # pragma: no cover - _kill_tree is patched
            kills.append("kill")

    monkeypatch.setattr(bash_module.subprocess, "Popen", lambda *a, **k: FakeProcess())
    monkeypatch.setattr(
        BashTool, "_kill_tree", staticmethod(lambda process: kills.append("tree"))
    )
    tool = BashTool(RuntimeState(workspace=tmp_path / "ws"))

    with pytest.raises(KeyboardInterrupt):
        tool.execute("echo hi")

    assert kills == ["tree"]
    assert running_command_count() == 0
