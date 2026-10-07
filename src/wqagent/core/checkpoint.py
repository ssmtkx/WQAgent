"""Checkpointing: keep enough of a run to resume it after a crash.

Three modes (``RuntimeState.checkpoint_mode`` / ``WQ_CHECKPOINT_MODE``):

``light`` (default)
    After every node switch: ``checkpoint.json`` (metadata + state summary),
    ``RECOVERY.md`` (human-readable guide) and a git snapshot of the workspace.
    Resume keeps the plan, todos, criteria, counters and the compressed context -
    not the raw transcript.
``strict``
    Everything ``light`` does, plus ``state.json`` (the full state, transcript
    included) and ``events.jsonl`` (every streamed event, appended as it arrives).
``off``
    Nothing is written.

Everything lives inside the workspace, in ``.wqagent/checkpoints``::

    <workspace>/.wqagent/checkpoints/
        checkpoint.json     metadata + state summary (what resume reads)
        RECOVERY.md         how to pick this run up again
        state.json          strict only: the full state
        events.jsonl        strict only: one streamed event per line
        git/                internal snapshot repository (never the user's .git)

The git snapshot uses ``git --git-dir=<root>/git --work-tree=<workspace>``, so the
workspace's own repository (if any) is never touched, and ``.wqagent/`` is excluded
from the snapshot. Any git problem degrades to "checkpoint without a snapshot"
instead of failing the run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

__all__ = [
    "CHECKPOINT_DIR",
    "CHECKPOINT_MODE_ENV",
    "DEFAULT_CHECKPOINT_MODE",
    "MANIFEST_LIMIT",
    "VALID_CHECKPOINT_MODES",
    "CheckpointManager",
    "build_recovery_markdown",
    "checkpoint_mode_from_env",
    "message_from_dict",
    "message_to_dict",
    "normalize_checkpoint_mode",
    "resume_command",
    "state_of",
    "state_to_payload",
]

#: Directory (inside the workspace) that holds every checkpoint artifact.
CHECKPOINT_DIR = ".wqagent"
#: Sub-directory of :data:`CHECKPOINT_DIR` that holds the checkpoints themselves.
CHECKPOINT_SUBDIR = "checkpoints"
#: Environment variable that chooses the checkpoint mode.
CHECKPOINT_MODE_ENV = "WQ_CHECKPOINT_MODE"
#: Modes a run may use.
VALID_CHECKPOINT_MODES: frozenset[str] = frozenset({"light", "strict", "off"})
#: Mode used when none was chosen (and for unusable values).
DEFAULT_CHECKPOINT_MODE = "light"
#: Files listed in the workspace manifest before it is marked truncated.
MANIFEST_LIMIT = 500
#: Characters kept per free-text field of a checkpoint summary.
SUMMARY_CHARS = 1_500
#: Seconds a single git command may take.
GIT_TIMEOUT_SECONDS = 60

#: State keys that are not serialised (the runtime is rebuilt by the caller).
_SKIPPED_STATE_KEYS = frozenset({"runtime"})
#: Environment variables that must never leak a parent repository into the snapshot.
_GIT_ENV_DROP = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
)


def normalize_checkpoint_mode(mode: str | None) -> str:
    """The mode to use: missing, blank or unknown values fall back to ``"light"``."""
    text = str(mode or "").strip().lower()
    return text if text in VALID_CHECKPOINT_MODES else DEFAULT_CHECKPOINT_MODE


def checkpoint_mode_from_env(default: str = DEFAULT_CHECKPOINT_MODE) -> str:
    """The mode from ``WQ_CHECKPOINT_MODE``, or *default* (itself normalized)."""
    raw = os.environ.get(CHECKPOINT_MODE_ENV, "")
    return normalize_checkpoint_mode(raw) if raw.strip() else normalize_checkpoint_mode(default)


def resume_command(workspace: Path | str) -> str:
    """The command that picks a checkpointed run up again."""
    return f"wq --resume {workspace}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _short(value: Any, limit: int = SUMMARY_CHARS) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _json_default(value: Any) -> Any:
    """Encoder for the values a checkpoint payload may still contain.

    Chat messages end up in node updates, so they become the same dict
    :func:`message_to_dict` produces; anything else falls back to ``str()``.
    """
    if isinstance(value, BaseMessage):
        return message_to_dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        try:
            return asdict(value)
        except Exception:  # noqa: BLE001 - fall back to str()
            return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(str(item) for item in value)
    return str(value)


def _json_safe(value: Any) -> Any:
    """Best-effort JSON value: anything unserialisable becomes its ``str()``.

    This checks *native* JSON only, so a state summary is always plain data;
    :func:`_json_default` is the looser safety net used when writing files.
    """
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        if is_dataclass(value) and not isinstance(value, type):
            try:
                return _json_safe(asdict(value))
            except Exception:  # noqa: BLE001 - fall through to str()
                return str(value)
        return str(value)
    return value


def message_to_dict(message: Any) -> dict[str, Any]:
    """Serialize one chat message (the transcript is not JSON by itself)."""
    data: dict[str, Any] = {
        "type": str(getattr(message, "type", "ai") or "ai"),
        "content": _json_safe(getattr(message, "content", "")),
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        data["tool_calls"] = _json_safe(list(tool_calls))
    for field in ("tool_call_id", "name", "id"):
        value = getattr(message, field, None)
        if value:
            data[field] = str(value)
    return data


def message_from_dict(data: Mapping[str, Any]) -> BaseMessage:
    """Rebuild a chat message from :func:`message_to_dict` output."""
    kind = str(data.get("type") or "ai").lower()
    content = data.get("content", "")
    if kind == "human":
        return HumanMessage(content=content, id=data.get("id"))
    if kind == "system":
        return SystemMessage(content=content, id=data.get("id"))
    if kind == "tool":
        return ToolMessage(
            content=content,
            tool_call_id=str(data.get("tool_call_id") or "unknown"),
            name=data.get("name"),
            id=data.get("id"),
        )
    tool_calls = data.get("tool_calls")
    return AIMessage(
        content=content,
        tool_calls=list(tool_calls) if isinstance(tool_calls, Sequence) else [],
        id=data.get("id"),
    )


def state_to_payload(state: Mapping[str, Any], *, include_messages: bool = True) -> dict[str, Any]:
    """The JSON-safe form of a graph state.

    With ``include_messages=False`` the transcript is replaced by
    ``message_count`` - that is what a *light* checkpoint keeps.
    """
    payload: dict[str, Any] = {}
    for key, value in state.items():
        if key in _SKIPPED_STATE_KEYS:
            continue
        if key == "messages":
            messages = list(value or []) if isinstance(value, (list, tuple)) else []
            if include_messages:
                payload["messages"] = [message_to_dict(message) for message in messages]
            else:
                payload["message_count"] = len(messages)
            continue
        payload[key] = _json_safe(value)
    return payload


def payload_to_state(payload: Mapping[str, Any], runtime: Any) -> dict[str, Any]:
    """Rebuild ``inputs`` for a resumed run from :func:`state_to_payload` output."""
    state: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "message_count":
            continue
        if key == "messages":
            messages = value if isinstance(value, list) else []
            state["messages"] = [
                message_from_dict(item) for item in messages if isinstance(item, Mapping)
            ]
            continue
        state[key] = value
    state["runtime"] = runtime
    return state


def state_of(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """The state part of a checkpoint payload (tolerates the older flat shape)."""
    state = payload.get("state")
    return state if isinstance(state, Mapping) else payload


def build_recovery_markdown(payload: Mapping[str, Any]) -> str:
    """Render ``RECOVERY.md``: task, status, files, git commit and resume command."""
    state = state_of(payload)
    task = _short(payload.get("task"), 400) or "(no task recorded)"
    status = str(payload.get("status") or "unknown")
    node = str(payload.get("latest_node") or "-")
    workspace = str(payload.get("workspace") or "")
    manifest = payload.get("workspace_manifest")
    manifest = manifest if isinstance(manifest, Mapping) else {}
    git = payload.get("git")
    git = git if isinstance(git, Mapping) else {}

    lines = [
        "# Recovery",
        "",
        f"**Task:** {task}",
        f"**Status:** {status} (latest node: {node})",
        f"**Saved:** {payload.get('saved_at') or 'unknown'} "
        f"(mode: {payload.get('mode') or 'unknown'})",
        f"**Workspace:** {workspace or '(unknown)'}",
        f"**Attempts:** {state.get('attempts', 0)}/{state.get('max_attempts', 0)}",
        "",
        "## Resume",
        "",
        "```",
        str(
            payload.get("resume_command")
            or (resume_command(workspace) if workspace else "wq --resume <workspace>")
        ),
        "```",
        "",
    ]

    if state.get("plan_summary"):
        lines += ["## Plan", "", _short(state.get("plan_summary")), ""]
    todos = state.get("todos")
    if isinstance(todos, Sequence) and todos:
        lines += ["## Todos", ""]
        for todo in todos:
            if not isinstance(todo, Mapping):
                continue
            mark = {
                "completed": "[x]",
                "in_progress": "[~]",
                "blocked": "[!]",
            }.get(str(todo.get("status") or "pending"), "[ ]")
            lines.append(f"- {mark} {todo.get('id')}: {todo.get('content')}")
        lines.append("")
    if state.get("last_error"):
        lines += ["## Last error", "", _short(state.get("last_error")), ""]
    if state.get("context_summary"):
        lines += ["## Compressed context", "", _short(state.get("context_summary")), ""]

    files = manifest.get("files")
    files = files if isinstance(files, Sequence) else []
    lines += ["## Workspace files", ""]
    if files:
        lines.append(f"{manifest.get('count', len(files))} file(s)")
        if manifest.get("truncated"):
            lines.append(f"(listing truncated to the first {MANIFEST_LIMIT})")
        lines.append("")
        for item in files:
            if isinstance(item, Mapping):
                lines.append(f"- {item.get('path')} ({item.get('size', 0)} bytes)")
    else:
        lines.append("(the workspace was empty)")
    lines.append("")

    lines += ["## Snapshot", ""]
    if git.get("ok"):
        lines += [
            f"- git commit: `{git.get('commit')}`",
            f"- files in snapshot: {git.get('files', 0)}",
            "- restore: `git --git-dir .wqagent/checkpoints/git --work-tree . checkout "
            f"{git.get('commit')} -- .`",
        ]
    else:
        lines.append(f"- no git snapshot ({git.get('error') or 'git is unavailable'})")
    if payload.get("state_file"):
        lines.append(f"- full state: `{payload['state_file']}`")
    if payload.get("events_file"):
        lines.append(f"- event log: `{payload['events_file']}`")
    lines.append("")
    return "\n".join(lines)


class CheckpointManager:
    """Save and load the checkpoints of one workspace."""

    def __init__(self, runtime: Any, task: str = "") -> None:
        self.workspace = Path(getattr(runtime, "workspace", "."))
        self.mode = normalize_checkpoint_mode(getattr(runtime, "checkpoint_mode", None))
        self.task = str(task or "")
        self.root = self.workspace / CHECKPOINT_DIR / CHECKPOINT_SUBDIR
        self.git_dir = self.root / "git"

    # -- paths ---------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """False in ``off`` mode: nothing is written and nothing is loaded."""
        return self.mode != "off"

    @property
    def strict(self) -> bool:
        return self.mode == "strict"

    @property
    def checkpoint_path(self) -> Path:
        return self.root / "checkpoint.json"

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    @property
    def events_path(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def recovery_path(self) -> Path:
        return self.root / "RECOVERY.md"

    # -- saving --------------------------------------------------------------
    def save(
        self,
        state: Mapping[str, Any],
        *,
        status: str = "running",
        latest_node: str | None = None,
        event: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Write one checkpoint; returns the event to publish, or ``None`` when off.

        Steps: create the directory, (strict) append the event and the full state,
        list the workspace, snapshot it with git, write ``checkpoint.json`` and
        render ``RECOVERY.md``.
        """
        if not self.enabled:
            return None
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return self._failed(f"cannot create {self.root}: {exc}")

        if self.strict and event is not None:
            self.append_event(event)
        if self.strict:
            self._write_json(self.state_path, state_to_payload(state, include_messages=True))

        manifest = self.workspace_manifest()
        git = self.snapshot_workspace_git()
        summary = state_to_payload(state, include_messages=False)
        payload: dict[str, Any] = {
            "version": 2,
            "mode": self.mode,
            "workspace": str(self.workspace),
            "task": self.task or str(state.get("task") or ""),
            "status": str(status),
            "latest_node": str(latest_node or ""),
            "saved_at": _now(),
            "resume_command": resume_command(self.workspace),
            "checkpoint_dir": str(self.root),
            "state_file": "state.json" if self.strict else "",
            "events_file": "events.jsonl" if self.strict else "",
            "message_count": int(summary.get("message_count") or 0),
            "workspace_manifest": manifest,
            "git": git,
            # the state lives under its own key: metadata must never mix with it
            "state": summary,
        }
        if not self._write_json(self.checkpoint_path, payload):
            return self._failed("cannot write checkpoint.json")
        try:
            self.recovery_path.write_text(
                build_recovery_markdown(payload), encoding="utf-8"
            )
        except OSError as exc:
            return self._failed(f"cannot write RECOVERY.md: {exc}")

        return {
            "type": "checkpoint_saved",
            "mode": self.mode,
            "path": str(self.checkpoint_path),
            "recovery_path": str(self.recovery_path),
            "status": payload["status"],
            "latest_node": payload["latest_node"],
            "files": int(manifest.get("count") or 0),
            "git_commit": str(git.get("commit") or ""),
            "git_ok": bool(git.get("ok")),
            "strict": self.strict,
            "resume_command": payload["resume_command"],
        }

    def append_event(self, event: Mapping[str, Any]) -> bool:
        """Append one streamed event to ``events.jsonl`` (strict mode only)."""
        if self.mode != "strict":
            return False
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        dict(event), default=_json_default, ensure_ascii=False
                    )
                    + "\n"
                )
        except (OSError, TypeError, ValueError):
            return False
        return True

    def workspace_manifest(self) -> dict[str, Any]:
        """The workspace file listing (relative paths, sizes, count)."""
        files: list[dict[str, Any]] = []
        try:
            for path in sorted(self.workspace.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(self.workspace).as_posix()
                if relative.startswith(f"{CHECKPOINT_DIR}/") or relative == CHECKPOINT_DIR:
                    continue
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                files.append({"path": relative, "size": size})
        except OSError:
            pass
        return {
            "count": len(files),
            "truncated": len(files) > MANIFEST_LIMIT,
            "files": files[:MANIFEST_LIMIT],
        }

    # -- git snapshot --------------------------------------------------------
    def _git(
        self, args: Sequence[str], *, config: Sequence[str] = ()
    ) -> tuple[int, str, str]:
        """Run one git command against the snapshot repo: ``(returncode, out, err)``.

        ``returncode`` is ``-1`` when git could not be started at all.
        """
        if shutil.which("git") is None:
            return -1, "", "git is not installed"
        env = {key: value for key, value in os.environ.items() if key not in _GIT_ENV_DROP}
        env["GIT_TERMINAL_PROMPT"] = "0"
        command = [
            "git",
            f"--git-dir={self.git_dir}",
            f"--work-tree={self.workspace}",
            *config,
            *args,
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=str(self.workspace),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=GIT_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return -1, "", f"{type(exc).__name__}: {exc}"
        return completed.returncode, completed.stdout or "", (completed.stderr or "").strip()

    @staticmethod
    def _git_error(returncode: int, stderr: str, stdout: str = "") -> str:
        detail = (stderr or stdout or "").strip().splitlines()
        if detail:
            return detail[-1]
        return f"git exited {returncode}"

    @property
    def _commit_config(self) -> tuple[str, ...]:
        """Config that keeps commits unattended and reproducible."""
        return (
            "-c",
            "user.name=wq-agent",
            "-c",
            "user.email=wq-agent@localhost",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.autocrlf=false",
            "-c",
            "core.hooksPath=" + str(self.root / "no-hooks"),
            "-c",
            "gc.auto=0",
            "-c",
            "advice.detachedHead=false",
        )

    def _ensure_git(self) -> tuple[bool, str]:
        if (self.git_dir / "HEAD").is_file():
            self._write_exclude()
            return True, ""
        try:
            self.git_dir.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return False, f"cannot create {self.git_dir}: {exc}"
        env = {key: value for key, value in os.environ.items() if key not in _GIT_ENV_DROP}
        command = ["git", f"--git-dir={self.git_dir}", "init", "--quiet"]
        try:
            completed = subprocess.run(
                command,
                cwd=str(self.workspace),
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=GIT_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"{type(exc).__name__}: {exc}"
        if completed.returncode != 0:
            return False, (completed.stderr or "").strip() or "git init failed"
        self._write_exclude()
        return True, ""

    def _write_exclude(self) -> None:
        """Keep the checkpoint dir and the workspace's own repo out of the snapshot."""
        try:
            exclude = self.git_dir / "info" / "exclude"
            exclude.parent.mkdir(parents=True, exist_ok=True)
            wanted = f"{CHECKPOINT_DIR}/\n.git/\n"
            if not exclude.is_file() or exclude.read_text(encoding="utf-8") != wanted:
                exclude.write_text(wanted, encoding="utf-8")
        except OSError:
            pass

    def _head_commit(self, config: Sequence[str]) -> str:
        """The current commit, or ``""`` when the snapshot repo has none yet."""
        code, output, _ = self._git(["rev-parse", "HEAD"], config=config)
        return output.strip() if code == 0 else ""

    def snapshot_workspace_git(self) -> dict[str, Any]:
        """Commit the workspace into the internal repository; never raises."""
        record: dict[str, Any] = {"ok": False, "commit": "", "files": 0, "error": ""}
        ready, error = self._ensure_git()
        if not ready:
            record["error"] = error
            return record
        config = self._commit_config
        code, _, err = self._git(["add", "-A"], config=config)
        if code != 0:
            record["error"] = self._git_error(code, err)
            return record
        # exit 1 = there are staged changes; 0 = nothing new to commit
        code, _, err = self._git(["diff", "--cached", "--quiet"], config=config)
        if code not in (0, 1):
            record["error"] = self._git_error(code, err)
            return record
        head = self._head_commit(config)
        if code == 1 or not head:
            # an empty workspace still gets one commit, so HEAD always exists
            message = f"wq checkpoint {_now()}"
            code, _, err = self._git(
                ["commit", "--quiet", "--no-verify", "--allow-empty", "-m", message],
                config=config,
            )
            if code != 0:
                record["error"] = self._git_error(code, err)
                return record
            head = self._head_commit(config)
        if not head:
            record["error"] = "the snapshot has no commit"
            return record
        record["ok"] = True
        record["commit"] = head
        code, output, _ = self._git(["ls-files"], config=config)
        record["files"] = (
            len([line for line in output.splitlines() if line.strip()]) if code == 0 else 0
        )
        return record

    def restore_workspace(self, commit: str) -> dict[str, Any]:
        """Restore the workspace files recorded in *commit*; never raises."""
        record: dict[str, Any] = {
            "ok": False,
            "commit": str(commit or ""),
            "files": 0,
            "modified_before": 0,
            "error": "",
        }
        if not record["commit"]:
            record["error"] = "no commit recorded"
            return record
        config = self._commit_config
        code, output, _ = self._git(["status", "--porcelain"], config=config)
        if code == 0:
            record["modified_before"] = len(
                [line for line in output.splitlines() if line.strip()]
            )
        code, _, err = self._git(["checkout", record["commit"], "--", "."], config=config)
        if code != 0:
            record["error"] = self._git_error(code, err)
            return record
        record["ok"] = True
        code, output, _ = self._git(
            ["ls-tree", "-r", "--name-only", record["commit"]], config=config
        )
        record["files"] = (
            len([line for line in output.splitlines() if line.strip()]) if code == 0 else 0
        )
        return record

    # -- loading -------------------------------------------------------------
    def load_checkpoint(self) -> dict[str, Any] | None:
        """Read ``checkpoint.json``; ``None`` when there is none (or mode is off)."""
        if not self.enabled:
            return None
        try:
            payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    @classmethod
    def load_resume_inputs(
        cls,
        runtime: Any,
        task: str | None = None,
        max_attempts: int = 3,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Rebuild ``inputs`` (and report) for a resumed run.

        Returns ``(inputs, resume_event)``; *inputs* is ``None`` when the workspace
        holds no usable checkpoint. Workspace files recorded in the snapshot are
        restored first.
        """
        manager = cls(runtime, task or "")
        payload = manager.load_checkpoint()
        event: dict[str, Any] = {
            "type": "resume",
            "ok": False,
            "mode": manager.mode,
            "checkpoint_path": str(manager.checkpoint_path),
            "workspace": str(manager.workspace),
            "status": "",
            "restored_files": 0,
            "message_count": 0,
            "warnings": [],
            "task": "",
        }
        if not manager.enabled:
            event["warnings"].append("checkpointing is off; nothing to resume")
            return None, event
        if payload is None:
            event["warnings"].append(f"no checkpoint.json in {manager.root}")
            return None, event

        checkpoint_task = str(payload.get("task") or "").strip()
        requested_task = str(task or "").strip()
        resolved_task = checkpoint_task or requested_task
        event["task"] = resolved_task
        event["status"] = str(payload.get("status") or "")
        event["latest_node"] = str(payload.get("latest_node") or "")
        event["saved_at"] = str(payload.get("saved_at") or "")
        if requested_task and checkpoint_task and requested_task != checkpoint_task:
            event["warnings"].append(
                f"the checkpoint belongs to a different task ({checkpoint_task!r}); "
                "the checkpoint's task is used"
            )

        git = payload.get("git")
        if isinstance(git, Mapping) and git.get("commit"):
            restored = manager.restore_workspace(str(git["commit"]))
            event["git_commit"] = restored["commit"]
            event["restored_files"] = int(restored.get("files") or 0)
            if restored.get("modified_before"):
                event["warnings"].append(
                    f"{restored['modified_before']} workspace file(s) differed from the "
                    "snapshot and were overwritten (extra files were kept)"
                )
            if not restored.get("ok"):
                event["warnings"].append(f"could not restore the snapshot: {restored['error']}")
        else:
            event["warnings"].append("the checkpoint has no git snapshot; files were left as they are")

        state_payload: Mapping[str, Any] = state_of(payload)
        if manager.strict and manager.state_path.is_file():
            try:
                full = json.loads(manager.state_path.read_text(encoding="utf-8"))
                if isinstance(full, Mapping):
                    state_payload = full
            except (OSError, ValueError):
                event["warnings"].append("state.json is unreadable; resumed from the summary")

        inputs = payload_to_state(state_payload, runtime)
        inputs["task"] = resolved_task
        inputs.setdefault("max_attempts", max(1, int(max_attempts)))
        messages = inputs.get("messages")
        event["message_count"] = len(messages) if isinstance(messages, list) else 0
        if not messages:
            event["warnings"].append(
                "the transcript was not checkpointed (light mode); resuming from the "
                "summary and the plan"
            )
        event["ok"] = True
        return inputs, event

    # -- internals -----------------------------------------------------------
    def _write_json(self, path: Path, payload: Any) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError):
            return False
        return True

    def _failed(self, error: str) -> dict[str, Any]:
        return {
            "type": "checkpoint_saved",
            "ok": False,
            "mode": self.mode,
            "error": str(error),
            "path": str(self.checkpoint_path),
        }
