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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

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
        if self.timed_out:
            timeout = f"{self.timeout:g}" if self.timeout is not None else "?"
            lines.append(f"[timed out after {timeout}s - the process tree was terminated]")
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


def _popen_kwargs() -> dict[str, Any]:
    """Start the child in its own process group so the whole tree can be killed."""
    if os.name == "nt":  # pragma: no cover - platform specific
        return {
            "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        }
    return {"start_new_session": True}


class BashTool:
    """Run a shell command in the workspace and capture its output."""

    name: str = "bash"
    description: str = (
        "Run a shell command in the workspace root and return its exit code, stdout and "
        f"stderr. On this machine the shell is {SHELL_NAME}, so write commands in that "
        "shell's syntax: on Windows use ';' to chain commands and avoid POSIX-only "
        "operators such as '&&', '||' and 'cat'. The command is killed after "
        "`timeout_seconds` (default 120) together with its child processes. Use it for "
        "builds, tests, git and running the programs you write."
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

    def run(self, command: str, timeout_seconds: float | None = None) -> str:
        """Run *command* and return its rendered output; never raises."""
        outcome = self.execute(command, timeout_seconds)
        return self.state.clip(outcome.format(), label="bash output")

    def execute(
        self, command: str, timeout_seconds: float | None = None
    ) -> ShellOutcome:
        """Run *command* and return the raw outcome (exit code, stdout, stderr)."""
        state = self.state
        command = (command or "").strip()
        if not command:
            return ShellOutcome(command=command, error="command must not be empty.")
        blocked = self._blocked_by(command)
        if blocked is not None:
            return ShellOutcome(command=command, blocked_by=blocked)

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
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                **_popen_kwargs(),
            )
        except OSError as exc:
            return ShellOutcome(command=command, timeout=timeout, error=f"cannot start command: {exc}")

        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill_tree(process)
            try:
                stdout, stderr = process.communicate(timeout=15)
            except subprocess.TimeoutExpired:  # pragma: no cover - unkillable child
                stdout, stderr = "", ""

        return ShellOutcome(
            command=command,
            timeout=timeout,
            exit_code=process.returncode,
            stdout=stdout or "",
            stderr=stderr or "",
            timed_out=timed_out,
        )

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
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env["WQ_WORKSPACE"] = str(self.state.workspace)
        return env

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
