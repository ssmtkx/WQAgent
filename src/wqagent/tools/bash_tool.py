"""Shell command execution inside the workspace, with a hard timeout.

The command runs with the current user's rights - this tool is *not* a sandbox.
What it does guarantee: the working directory is the workspace, output is
UTF-8 decoded, the process is killed (whole tree) when the timeout expires, and
a small set of destructive patterns is refused up front.

:meth:`BashTool.execute` returns the raw :class:`ShellOutcome` (exit code,
stdout, stderr) for callers that need structured results - the graph's verifier
node uses it to build ``VerificationResult`` entries; :meth:`BashTool.run`
renders the same outcome as text for the model.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from wqagent.core.approval import (
    classify_command_risk,
    new_approval_request,
    normalize_approval_mode,
)
from wqagent.core.state import RuntimeState

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.tools import StructuredTool

__all__ = [
    "BashInput",
    "BashTool",
    "MAX_TIMEOUT_SECONDS",
    "SHELL_NAME",
    "ShellOutcome",
    "shell_guidance",
]

MAX_TIMEOUT_SECONDS = 3_600.0


def _shell_executable() -> str | None:
    """Full path of the shell used on this platform, if one can be found."""
    if os.name == "nt":  # pragma: no cover - platform specific
        return shutil.which("pwsh") or shutil.which("powershell")
    return shutil.which("bash")


def _shell_name() -> str:
    """Human name of the shell, used in the tool description shown to the model."""
    if os.name == "nt":  # pragma: no cover - platform specific
        executable = _shell_executable()
        if not executable:
            return "cmd.exe"
        return "PowerShell 7 (pwsh)" if Path(executable).stem.lower() == "pwsh" else "Windows PowerShell"
    return "bash" if _shell_executable() else "/bin/sh"


#: Resolved once at import; the model is told which shell its commands run in.
SHELL_NAME = _shell_name()


def shell_guidance() -> str:
    """What to watch out for when writing commands for this machine's shell.

    The bash tool passes its own command to a real shell, and the *planner* writes
    verification commands without running them - it never sees the tool
    description, so the runtime has to tell it what this shell accepts.
    """
    if os.name == "nt":  # pragma: no cover - platform specific
        if Path(_shell_executable() or "").stem.lower() == "pwsh":
            return (
                "PowerShell 7 (pwsh): `&&` and `||` work, but POSIX tools "
                "(`test`, `cat -A`, `grep -q`, process substitution `<(...)`) do not; "
                "prefer `Test-Path`, `Select-String` and `;` between statements."
            )
        return (
            "Windows PowerShell 5.1: `&&`, `||`, `test`, `cat -A`, `grep -q` and process "
            "substitution `<(...)` are NOT available; use `;`, `if ($?) { ... }`, "
            "`Test-Path`, `Select-String`, and `python` (not `python3`) unless you know "
            "the interpreter exists."
        )
    return "POSIX shell (bash/sh): `&&`, `||`, `test`, `grep`, process substitution all work."


@dataclass(frozen=True)
class ShellOutcome:
    """Raw result of one shell command."""

    command: str
    timeout: float | None = None
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    blocked_by: str | None = None
    error: str | None = None
    requires_approval: bool = False
    approval_note: str = ""
    #: True when the command exited but a child process still holds its output open
    #: (the remaining output was not captured - the run does not wait for it).
    output_open: bool = False

    @property
    def ok(self) -> bool:
        """True when the command ran and exited with status 0."""
        return (
            self.error is None
            and self.blocked_by is None
            and not self.timed_out
            and self.exit_code == 0
        )

    def format(self) -> str:
        """Render the outcome the way the model sees it."""
        if self.error is not None:
            return f"Error: {self.error}"
        if self.blocked_by is not None:
            return (
                f"Error: command refused by the safety filter (pattern {self.blocked_by!r}). "
                "It is far more destructive than the workspace; ask the user to run it "
                "manually if it really is intended."
            )
        lines = [
            f"$ {self.command}",
            f"exit_code: {self.exit_code if self.exit_code is not None else 'unknown'}",
        ]
        if self.requires_approval:
            # the marker the approval policy leaves on a command that was allowed
            lines.append(f"requires_approval: True ({self.approval_note or 'approved'})")
        if self.timed_out:
            timeout = f"{self.timeout:g}" if self.timeout is not None else "?"
            lines.append(
                f"[timed out after {timeout}s - the process tree was terminated; if this "
                "command legitimately needs longer, call it again with a larger "
                f"`timeout_seconds` (up to {MAX_TIMEOUT_SECONDS:g}) or split it up. If it "
                "printed nothing at all, it was most likely stuck on the network (slow or "
                "unreachable package index/proxy - a mirror usually fixes it). If it may "
                "be waiting for input it cannot show, pass the tool's non-interactive "
                "flag instead (pip: `--no-input`, git: `GIT_TERMINAL_PROMPT=0`)]"
            )
        if self.output_open:
            lines.append(
                "[the command exited, but a child process still holds its output open - "
                "whatever it printed afterwards is not in this result]"
            )
        out = self.stdout.strip("\n")
        err = self.stderr.strip("\n")
        if out:
            lines.append(f"stdout:\n{out}")
        if err:
            lines.append(f"stderr:\n{err}")
        if not out and not err:
            lines.append("(no output)")
        return "\n".join(lines)


class BashInput(BaseModel):
    """Arguments accepted by :class:`BashTool`."""

    command: str = Field(
        description="Shell command to run in the workspace root, e.g. 'pytest -q'."
    )
    timeout_seconds: float | None = Field(
        default=None,
        ge=0.1,
        le=MAX_TIMEOUT_SECONDS,
        description="Kill the command after this many seconds; defaults to the runtime timeout (120).",
    )


#: Bytes read per chunk from a child's pipe.
CHUNK_BYTES = 8_192
#: Characters kept per stream (the model only ever sees the clipped result anyway).
CAPTURE_CHARS = 200_000
#: Output lines of a running command shown by status lines / heartbeats.
TAIL_LINES = 3
#: Seconds to wait for the readers after the command exited (see `output_open`).
DRAIN_GRACE_SECONDS = 2.0


class _CommandCapture:
    """Live output of one child: bounded buffers plus a tail for status lines.

    Reading happens in threads (see :func:`_start_readers`) so that the *whole* output
    is available even when a grandchild keeps the pipes open after the command itself
    has finished.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.stdout = ""
        self.stderr = ""
        self.lines: list[str] = []
        self.bytes_seen = 0

    def add(self, text: str, *, stream: str) -> None:
        with self._lock:
            self.bytes_seen += len(text)
            if stream == "out":
                self.stdout = (self.stdout + text)[-CAPTURE_CHARS:]
            else:
                self.stderr = (self.stderr + text)[-CAPTURE_CHARS:]
            normalised = text.replace("\r\n", "\n").replace("\r", "\n")
            for piece in normalised.split("\n"):
                piece = piece.strip()
                if piece:
                    self.lines.append(piece)
            if len(self.lines) > TAIL_LINES * 4:
                del self.lines[: -TAIL_LINES * 4]

    def tail(self, count: int = TAIL_LINES) -> list[str]:
        """The last few non-empty lines the command printed."""
        with self._lock:
            return list(self.lines[-count:])


def _pump(stream: Any, capture: _CommandCapture, name: str) -> None:
    """Reader thread: append chunks to *capture* until the pipe closes."""
    try:
        while True:
            reader = getattr(stream, "read1", None) or stream.read
            chunk = reader(CHUNK_BYTES)
            if not chunk:
                break
            if isinstance(chunk, bytes):
                text = chunk.decode("utf-8", "replace")
            else:  # pragma: no cover - only when a caller swaps in text pipes
                text = str(chunk)
            capture.add(text, stream=name)
    except (OSError, ValueError):  # pragma: no cover - pipe closed under us
        pass
    finally:
        try:
            stream.close()
        except (OSError, ValueError):  # pragma: no cover
            pass


def _start_readers(
    process: subprocess.Popen[bytes], capture: _CommandCapture
) -> list[threading.Thread]:
    """Start one reader per pipe; they are daemons, so they never block exit."""
    readers: list[threading.Thread] = []
    for stream, name in ((process.stdout, "out"), (process.stderr, "err")):
        if stream is None:  # pragma: no cover - pipes are always requested here
            continue
        thread = threading.Thread(
            target=_pump, args=(stream, capture, name), name=f"wq-bash-{name}", daemon=True
        )
        thread.start()
        readers.append(thread)
    return readers


def _popen_kwargs() -> dict[str, Any]:
    """Start the child in its own process group so the whole tree can be killed."""
    if os.name == "nt":  # pragma: no cover - platform specific
        return {
            "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        }
    return {"start_new_session": True}


# Children currently running, so a Ctrl+C can stop them - and so the CLI can say what
# is taking so long. The graph runs every node in a worker thread (langgraph's
# BackgroundExecutor) while the interrupt is delivered to the *main* thread, so the
# worker's `communicate()` never raises on its own: the command would keep running
# until its timeout. The CLI's signal handler calls :func:`kill_running_commands` to
# break that deadlock.
_RUNNING_LOCK = threading.Lock()
_RUNNING_COMMANDS: dict[int, "_RunningCommand"] = {}

@dataclass(frozen=True)
class _RunningCommand:
    """One shell command in flight (enough to describe it while it runs)."""

    process: subprocess.Popen[bytes]
    command: str
    timeout: float | None
    started_at: float
    capture: "_CommandCapture | None" = None


def _register_command(
    process: subprocess.Popen[bytes],
    command: str = "",
    timeout: float | None = None,
    capture: "_CommandCapture | None" = None,
) -> None:
    with _RUNNING_LOCK:
        _RUNNING_COMMANDS[id(process)] = _RunningCommand(
            process=process,
            command=command,
            timeout=timeout,
            started_at=time.monotonic(),
            capture=capture,
        )


def _unregister_command(process: subprocess.Popen[bytes]) -> None:
    with _RUNNING_LOCK:
        _RUNNING_COMMANDS.pop(id(process), None)


def _live_commands() -> list[_RunningCommand]:
    with _RUNNING_LOCK:
        return [
            entry for entry in _RUNNING_COMMANDS.values() if entry.process.poll() is None
        ]


def running_command_count() -> int:
    """How many shell commands this process is currently running."""
    return len(_live_commands())


def describe_running_commands() -> list[dict[str, Any]]:
    """What is running right now: command text, elapsed seconds, timeout and tail."""
    described: list[dict[str, Any]] = []
    for entry in _live_commands():
        described.append(
            {
                "command": " ".join(str(entry.command).split())[:160],
                "elapsed": max(0.0, time.monotonic() - entry.started_at),
                "timeout": entry.timeout,
                "tail": entry.capture.tail() if entry.capture is not None else [],
                "bytes": entry.capture.bytes_seen if entry.capture is not None else 0,
            }
        )
    return sorted(described, key=lambda item: item["elapsed"], reverse=True)


def kill_running_commands() -> int:
    """Kill every shell command in flight; returns how many were still running.

    Safe to call from a signal handler (Python runs those in the main thread): it
    never raises and never blocks for long.
    """
    killed = 0
    for entry in _live_commands():
        try:
            BashTool._kill_tree(entry.process)
            killed += 1
        except Exception:  # noqa: BLE001 - a signal handler must not raise
            continue
    return killed


class BashTool:
    """Run a shell command in the workspace and capture its output."""

    name: str = "bash"
    description: str = (
        "Run a shell command in the workspace root and return its exit code, stdout and "
        f"stderr. On this machine the shell is {SHELL_NAME}, so write commands in that "
        "shell's syntax: on Windows use ';' to chain commands and avoid POSIX-only "
        "operators such as '&&', '||' and 'cat'. The command is killed after "
        "`timeout_seconds` (default 120) together with its child processes. Use it for "
        "builds, tests, git and running the programs you write. Installing packages, "
        "downloading from the network and starting long-running servers need approval: "
        "such a command is denied unless the run allows it, and its result then carries "
        "`requires_approval: True`."
    )
    args_schema: type[BaseModel] = BashInput

    def __init__(self, state: RuntimeState) -> None:
        self.state = state

    def as_tool(self) -> "StructuredTool":
        """Wrap this implementation in a LangChain ``StructuredTool``."""
        from langchain_core.tools import StructuredTool

        return StructuredTool.from_function(
            func=self.run,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
        )

    def run_bash(self, command: str, timeout_seconds: float | None = None) -> str:
        """Run *command* through the approval policy and render the outcome.

        Risky commands (see :data:`~wqagent.core.approval.RISK_PATTERNS`) are
        resolved by the run's approval mode before anything is executed.
        """
        return self.state.clip(self.execute(command, timeout_seconds).format(), label="bash output")

    def run(self, command: str, timeout_seconds: float | None = None) -> str:
        """Run *command* and return its rendered output; never raises."""
        return self.run_bash(command, timeout_seconds)

    def execute(
        self, command: str, timeout_seconds: float | None = None
    ) -> ShellOutcome:
        """Run *command* and return the raw outcome (exit code, stdout, stderr).

        The approval gate lives here - the one place a command is really started -
        so verification commands run by the verifier node cannot bypass it either.
        """
        state = self.state
        command = (command or "").strip()
        if not command:
            return ShellOutcome(command=command, error="command must not be empty.")
        blocked = self._blocked_by(command)
        if blocked is not None:
            return ShellOutcome(command=command, blocked_by=blocked)

        risk = classify_command_risk(command)
        approval_note = ""
        requires_approval = risk is not None
        if risk is not None:
            denied, approval_note = self._decide_approval(command, risk)
            if denied is not None:
                return denied

        if timeout_seconds is None:
            timeout = float(state.timeout_seconds)
        else:
            timeout = max(0.1, min(float(timeout_seconds), MAX_TIMEOUT_SECONDS))

        argv = self._shell_argv(command)
        try:
            process = subprocess.Popen(
                argv,
                cwd=str(state.workspace),
                env=self._child_env(),
                # stdin is NOT inherited: a tool that decides to ask a question (pip on a
                # private index, git for credentials) must fail fast and visibly instead
                # of blocking on a prompt nobody can see.
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **_popen_kwargs(),
            )
        except OSError as exc:
            return ShellOutcome(
                command=command,
                timeout=timeout,
                error=f"cannot start command: {exc}",
                requires_approval=requires_approval,
                approval_note=approval_note,
            )

        timed_out = False
        capture = _CommandCapture()
        readers = _start_readers(process, capture)
        _register_command(process, command, timeout, capture)
        leaked_output = False
        try:
            try:
                # Wait for the *process*, not for the pipes: a grandchild that inherited
                # the pipes keeps them open after the command itself is done, and waiting
                # for EOF would hang a finished install for ever.
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                self._kill_tree(process)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:  # pragma: no cover - unkillable child
                    pass
            except BaseException:
                # Ctrl+C (or anything else) while the command runs: the child lives in
                # its own process group, so it never sees the interrupt - kill it here
                # instead of leaving it running until the timeout.
                self._kill_tree(process)
                try:
                    process.wait(timeout=5)
                except BaseException:  # pragma: no cover - best effort only
                    pass
                raise
            # give the readers a moment to drain what the command already wrote
            for reader in readers:
                reader.join(timeout=DRAIN_GRACE_SECONDS)
            leaked_output = any(reader.is_alive() for reader in readers)
        finally:
            _unregister_command(process)

        stdout, stderr = capture.stdout, capture.stderr

        return ShellOutcome(
            command=command,
            timeout=timeout,
            exit_code=process.returncode,
            stdout=stdout or "",
            stderr=stderr or "",
            timed_out=timed_out,
            output_open=leaked_output,
            requires_approval=requires_approval,
            approval_note=approval_note,
        )

    # -- approval ------------------------------------------------------------
    def _decide_approval(self, command: str, risk: str) -> tuple[ShellOutcome | None, str]:
        """Apply the approval mode: ``(outcome_to_return, marker_note)``.

        ``outcome_to_return`` is ``None`` when the command may run; the note is then
        attached to the outcome as the ``requires_approval`` marker.
        """
        mode = normalize_approval_mode(getattr(self.state, "approval_mode", None))
        if mode == "auto":
            return None, f"auto-approved: {risk}"
        if mode == "deny":
            return self._denied(command, risk, "approval mode is 'deny'"), ""

        request = new_approval_request(command, risk)
        handler = getattr(self.state, "approval_handler", None)
        if handler is None:
            return (
                self._denied(command, risk, "no approval handler is available (mode=inline)"),
                "",
            )
        try:
            decision = handler(request)
        except Exception as exc:  # noqa: BLE001 - a broken approver must not crash the run
            return (
                self._denied(
                    command, risk, f"the approval handler failed ({type(exc).__name__}: {exc})"
                ),
                "",
            )
        approved = bool(getattr(decision, "approved", False))
        reason = str(getattr(decision, "reason", "") or "").strip()
        if not approved:
            return (
                self._denied(command, risk, "denied by the user", reason),
                "",
            )
        note = f"approved by the user: {risk}"
        return None, f"{note} ({reason})" if reason else note

    @staticmethod
    def _denied(command: str, risk: str, why: str, detail: str = "") -> ShellOutcome:
        """The outcome of a command the approval policy refused to run."""
        text = (
            f"command not run - approval required ({risk}); {why}. "
            "Ask the user to run it themselves, or start the run with approval mode "
            "'auto' if it is intended."
        )
        if detail:
            text = f"{text} [{detail}]"
        return ShellOutcome(command=command, error=text, requires_approval=True)

    # -- internals -----------------------------------------------------------
    def _blocked_by(self, command: str) -> str | None:
        for pattern in self.state.blocked_command_patterns:
            try:
                if re.search(pattern, command, re.IGNORECASE):
                    return pattern
            except re.error:  # pragma: no cover - user supplied bad pattern
                continue
        return None

    @staticmethod
    def _shell_argv(command: str) -> list[str]:
        """Build the shell invocation for this platform."""
        if os.name == "nt":  # pragma: no cover - platform specific
            shell = _shell_executable()
            if shell:
                # Force UTF-8 so the parent can decode output reliably.
                wrapped = (
                    "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
                    "$OutputEncoding = [System.Text.Encoding]::UTF8; " + command
                )
                return [
                    shell,
                    "-NoLogo",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    wrapped,
                ]
            comspec = os.environ.get("COMSPEC", "cmd.exe")
            return [comspec, "/d", "/s", "/c", command]
        shell = _shell_executable()
        if shell:
            return [shell, "-lc", command]
        return ["/bin/sh", "-c", command]

    def _child_env(self) -> dict[str, str]:
        """Environment for the child, with a few things fixed up.

        * ``PYTHONIOENCODING``/``PYTHONUTF8``/``PYTHONUNBUFFERED`` so Python children
          speak UTF-8 and print as they go (a buffered child looks like a hung one).
        * a **relative** ``TEMP``/``TMP``/``TMPDIR`` is replaced by an absolute path
          inside the workspace. Tools create their temporary directories relative to
          those variables, so ``TEMP=.`` (as some terminals and wrappers set) makes
          ``pip`` litter the workspace with ``pip-install-*``/``pip-unpack-*``
          directories - and they survive when the command is killed.
        * pip defaults that keep it non-interactive and skip its startup version check
          (one extra network round-trip before anything useful happens). Only set when
          the user has not chosen a value themselves.
        """
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env["WQ_WORKSPACE"] = str(self.state.workspace)
        env.setdefault("PIP_NO_INPUT", "1")
        env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        for name in ("TEMP", "TMP", "TMPDIR"):
            value = env.get(name, "").strip()
            if not value or not os.path.isabs(value):
                env[name] = str(self._temp_dir())
        return env

    def _temp_dir(self) -> Path:
        """Absolute per-workspace temp directory (created on demand)."""
        path = self.state.workspace / ".wqagent" / "tmp"
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:  # pragma: no cover - read-only workspace
            return Path(tempfile.gettempdir())
        return path

    @staticmethod
    def _kill_tree(process: subprocess.Popen[str]) -> None:
        """Terminate the command and everything it started."""
        if os.name == "nt":  # pragma: no cover - platform specific
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                    capture_output=True,
                    timeout=20,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except (OSError, subprocess.SubprocessError):
                pass
        else:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        try:
            process.kill()
        except OSError:  # pragma: no cover - already gone
            pass
