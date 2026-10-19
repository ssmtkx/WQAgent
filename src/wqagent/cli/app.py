"""``wq`` - the WQ Agent command line entry point.

Usage::

    wq "create a hello.py that prints Hello, World" --workspace ./demo
    wq "why does test_math.py fail?" -w . --max-steps 40
    wq --list-tools
    wq --show-config

The workspace defaults to ``./workspace`` and is created automatically, so the
agent can always be pointed at a fresh, safe directory.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, Mapping, NoReturn, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from wqagent import __version__
from wqagent.core.agent import DEFAULT_MAX_LOOPS, stream_agent_events
from wqagent.core.approval import (
    APPROVAL_MODE_ENV,
    APPROVAL_TIMEOUT_ENV,
    ApprovalDecision,
    ApprovalRequest,
    approval_mode_from_env,
    approval_timeout_from_env,
    normalize_approval_mode,
)
from wqagent.core.checkpoint import (
    CHECKPOINT_MODE_ENV,
    CheckpointManager,
    checkpoint_mode_from_env,
    normalize_checkpoint_mode,
    resume_command,
)
from wqagent.core.paths import WorkspacePathError, resolve_workspace
from wqagent.core.session import (
    append_assistant_turn,
    append_user_turn,
    load_or_create_session,
    save_session,
    session_path,
)
from wqagent.core.trace import (
    TRACE_MODE_ENV,
    normalize_trace_mode,
    trace_mode_from_env,
)
from wqagent.core.state import DEFAULT_TIMEOUT_SECONDS, RuntimeState
from wqagent.graph.memory import CONTEXT_TOKEN_LIMIT_ENV, context_token_limit_from_env
from wqagent.graph.state import DEFAULT_MAX_ATTEMPTS
from wqagent.providers import (
    DEFAULT_MODEL_TIMEOUT_SECONDS as DEFAULT_REQUEST_TIMEOUT_SECONDS,
)
from wqagent.providers import ProviderConfigError, create_model, describe_config
from wqagent.tools import describe_tools, tool_names
from wqagent.tools.bash_tool import describe_running_commands, kill_running_commands

app = typer.Typer(
    add_completion=False,
    rich_markup_mode="rich",
    help="WQ Agent - a workspace-scoped tool-calling coding agent.",
)
console = Console()
error_console = Console(stderr=True)

#: Accepted ``--approval-mode`` values (see :mod:`wqagent.core.approval`).
ApprovalMode = Literal["inline", "auto", "deny"]
#: Accepted ``--checkpoint-mode`` values (see :mod:`wqagent.core.checkpoint`).
CheckpointMode = Literal["light", "strict", "off"]
#: Accepted ``--trace-mode`` values: ``on`` is the friendly spelling of ``summary``.
TraceMode = Literal["on", "summary", "full", "off"]
#: Seconds of silence (no events at all) before the CLI says what it is waiting for.
HEARTBEAT_SECONDS = 30.0
#: Environment variable that sets the per-request model timeout.
REQUEST_TIMEOUT_ENV = "WQ_REQUEST_TIMEOUT"


class _nullcontext:
    """Minimal stand-in for :func:`contextlib.nullcontext` (kept explicit here)."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc_info: Any) -> None:
        return None


def request_timeout_from_env(default: float) -> float:
    """Per-request model timeout, from ``--request-timeout`` or ``WQ_REQUEST_TIMEOUT``."""
    raw = os.environ.get(REQUEST_TIMEOUT_ENV, "")
    if not raw.strip():
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return value if value > 0 else float(default)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130


# -- small helpers -----------------------------------------------------------
def _configure_output() -> None:
    """Write UTF-8 even when the console/pipe uses a legacy code page."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            if getattr(stream, "encoding", "utf-8").lower().replace("-", "") != "utf8":
                reconfigure(encoding="utf-8")
        except (ValueError, OSError):  # pragma: no cover - exotic streams
            pass


def _fail(message: str, code: int = EXIT_ERROR) -> NoReturn:
    """Print an error and exit."""
    error_console.print(f"[bold red]error:[/] {escape(message)}")
    raise typer.Exit(code)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"wq {__version__}")
        raise typer.Exit(EXIT_OK)


def _first_sentence(text: str, limit: int = 110) -> str:
    sentence = text.split(". ")[0].strip().rstrip(".")
    return sentence if len(sentence) <= limit else f"{sentence[: limit - 3]}..."


def _type_label(spec: dict[str, Any]) -> str:
    if "type" in spec:
        return str(spec["type"])
    for key in ("anyOf", "oneOf"):
        if key in spec:
            labels = [
                _type_label(option)
                for option in spec[key]
                if option.get("type") != "null"
            ]
            return " | ".join(dict.fromkeys(labels)) or "any"
    return "object"


def _format_schema(schema_class: Any) -> str:
    schema = schema_class.model_json_schema()
    required = set(schema.get("required", ()))
    lines: list[str] = []
    for name, spec in (schema.get("properties") or {}).items():
        label = f"{name}: {_type_label(spec)}"
        if name not in required:
            label += f" = {spec.get('default')!r}"
        lines.append(label)
    return "\n".join(lines)


def _summarize_args(args: dict[str, Any], limit: int = 120) -> str:
    for key in ("file_path", "command", "pattern", "path"):
        value = args.get(key)
        if isinstance(value, str) and value:
            text = value.replace("\n", " ")
            return text if len(text) <= limit else f"{text[: limit - 3]}..."
    text = json.dumps(args, ensure_ascii=False)
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _preview(text: str, *, lines: int = 8, limit: int = 1_200) -> str:
    body = text.strip()
    if not body:
        return "(no output)"
    all_lines = body.splitlines()
    shown = all_lines[:lines]
    if len(all_lines) > lines:
        shown.append(f"... ({len(all_lines) - lines} more lines, use --verbose)")
    result = "\n".join(shown)
    if len(result) > limit:
        result = f"{result[:limit]}..."
    return result


def _model_label(model: Any) -> str:
    name = getattr(model, "model_name", None) or getattr(model, "model", "unknown")
    base = getattr(model, "openai_api_base", None) or getattr(model, "base_url", None)
    return f"{name} @ {base}" if base else str(name)


class _EventSummary:
    """Counters accumulated while streaming graph events, for the closing summary."""

    def __init__(self) -> None:
        self.attempts = 0
        self.tool_calls = 0
        self.failures = 0
        self.files: list[str] = []
        self.nodes: list[str] = []
        self.passed = False

    def observe(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "tool_call":
            self.tool_calls += 1
            name = str(event.get("name") or "")
            path = str((event.get("args") or {}).get("file_path") or "")
            if name in {"file_write", "file_edit"} and path and path not in self.files:
                self.files.append(path)
        elif kind == "tool_result":
            result = event.get("result")
            ok = result.get("ok", True) if isinstance(result, Mapping) else event.get("ok", True)
            if ok is False:
                self.failures += 1
        elif kind == "node_end":
            node = str(event.get("node") or "")
            update = dict(event.get("update") or {})
            self.nodes.append(node)
            if node == "verifier":
                self.attempts = max(self.attempts, int(update.get("attempts") or 0))
                self.passed = bool(update.get("passed"))


#: Node headers: the emoji form, plus an ASCII fallback for streams that cannot
#: encode it (a GBK console/pipe raises UnicodeEncodeError deep inside rich).
_NODE_LABELS: dict[str, tuple[str, str]] = {
    "planner": ("📋 Planner", "Planner"),
    "actor": ("🔧 Actor", "Actor"),
    "searchAgent": ("🔍 searchAgent", "searchAgent"),
    "codeAgent": ("🛠 codeAgent", "codeAgent"),
    "context_monitor": ("🧭 Context", "Context"),
    "context_compressor": ("🗜 Compressor", "Compressor"),
    "verifier": ("✅ Verifier", "Verifier"),
    "final": ("📝 Final", "Final"),
}
_FAILED_LABELS: dict[str, tuple[str, str]] = {"verifier": ("❌ Verifier", "Verifier (failed)")}

#: Other non-ASCII glyphs, with ASCII fallbacks.
_GLYPHS: dict[str, tuple[str, str]] = {
    "to": ("→", "->"),
    "handoff": ("↪", ">>"),
    "result": ("↳", "->"),
    "ok": ("✓", "OK"),
    "fail": ("✗", "X"),
    "route": ("🧭", "route:"),
    "chat": ("💬", "chat:"),
}


def _can_encode(text: str) -> bool:
    """True when the console stream can write *text* as-is."""
    encoding = getattr(console.file, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError, TypeError):
        return False
    return True


def _glyph(name: str) -> str:
    fancy, plain = _GLYPHS[name]
    return fancy if _can_encode(fancy) else plain


def _node_header(node: str, update: dict[str, Any]) -> str:
    if node == "verifier" and not update.get("passed"):
        fancy, plain = _FAILED_LABELS["verifier"]
    else:
        fancy, plain = _NODE_LABELS.get(node, (node or "node", node or "node"))
    return fancy if _can_encode(fancy) else plain


def _messages_label(count: Any) -> str:
    try:
        number = int(count)
    except (TypeError, ValueError):
        return "?"
    return f"{number} msg" if number == 1 else f"{number} msgs"


def _context_pair(event: Mapping[str, Any]) -> str:
    """``"24 msgs/412000 tokens -> 1 msg/3100 tokens"`` for a compression.

    Falls back to the token-only form for events recorded before message counts
    were tracked.
    """
    before_tokens = event.get("tokens_before")
    after_tokens = event.get("tokens_after")
    before_messages = event.get("messages_before")
    after_messages = event.get("messages_after")
    if before_messages is None and after_messages is None:
        return f"{before_tokens} -> {after_tokens} tokens"
    return (
        f"{_messages_label(before_messages)}/{before_tokens} tokens -> "
        f"{_messages_label(after_messages)}/{after_tokens} tokens"
    )


def _format_memory(event: Mapping[str, Any]) -> str:
    """Pretty-print the snapshot carried by a ``memory`` event (verbose mode)."""
    memory = event.get("memory")
    if not isinstance(memory, Mapping) or not memory:
        return ""
    import json as _json

    try:
        return _json.dumps(memory, indent=2, ensure_ascii=False, default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return str(memory)


def _render_node_update(
    node: str, update: dict[str, Any], *, verbose: bool = False
) -> None:
    """Print what one graph node produced (the header is printed by the caller)."""
    if node == "planner":
        summary = str(update.get("plan_summary") or "").strip()
        if summary:
            console.print(Text(_preview(summary, lines=3, limit=400), style="dim"))
        for todo in update.get("todos") or []:
            console.print(
                f"   [ ] {escape(str(todo.get('id')))}: {escape(str(todo.get('content')))}"
            )
        for command in update.get("verification_commands") or []:
            console.print(f"   [dim]$ {escape(str(command))}[/]")
        for handoff in update.get("agent_handoffs") or []:
            console.print(
                f"   [dim]{_glyph('handoff')} {escape(str(handoff.get('to_agent')))}: "
                f"{escape(str(handoff.get('instruction')))}[/]"
            )
        supervisor = str(update.get("last_actor_summary") or "").strip()
        if supervisor:
            console.print(Text(_preview(supervisor, lines=3, limit=400), style="dim"))
    elif node == "context_monitor":
        count = int(update.get("context_token_count") or 0)
        limit = int(update.get("context_token_limit") or 0)
        budget = f"/{limit}" if limit else ""
        state_text = "over budget" if update.get("context_should_compress") else "ok"
        console.print(
            Text(
                f"   {_glyph('result')} {count}{budget} tokens ({state_text})"
                f" -> next: {update.get('context_next_node') or 'verifier'}",
                style="dim",
            )
        )
    elif node == "context_compressor":
        events = update.get("compression_events") or []
        last = events[-1] if events and isinstance(events[-1], Mapping) else {}
        console.print(
            Text(
                f"   {_glyph('result')} {_context_pair(last)} (limit {last.get('token_limit')})"
                "; history + notepad written",
                style="dim",
            )
        )
        summary = str(update.get("context_summary") or "").strip()
        if summary and verbose:
            console.print(Text(_preview(summary, lines=6, limit=800), style="dim"))
    elif node == "actor":
        summary = str(update.get("last_actor_summary") or "").strip()
        if summary:
            console.print(Text(_preview(summary, lines=4, limit=500), style="dim"))
    elif node == "verifier":
        for result in update.get("verification_results") or []:
            ok = bool(result.get("ok"))
            console.print(
                Text(
                    f"   {_glyph('ok') if ok else _glyph('fail')} {result.get('command')} "
                    f"-> exit code {result.get('exit_code')}",
                    style="dim" if ok else "red",
                )
            )
        if not update.get("verification_results"):
            console.print(Text("   (no verification commands)", style="dim"))
        error = str(update.get("last_error") or "").strip()
        if error:
            console.print(Text(_preview(error, lines=5, limit=700), style="red"))


class _NodeRenderer:
    """Prints the unified event stream, one header per node or specialist.

    Nodes stream their tool events *before* the graph reports their state update,
    so a header is opened by the first event and not repeated by ``node_end``.
    """

    def __init__(self, *, verbose: bool) -> None:
        self.verbose = verbose
        self._opened: set[str] = set()

    def render(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "ai_message":
            names = [str(name) for name in event.get("tool_calls") or []]
            if not self.verbose or not names:
                return
            self._open(str(event.get("node") or ""))
            content = str(event.get("content") or "").strip()
            if content:
                console.print(Text(content, style="dim"))
            console.print(f"[dim]   calling: {escape(', '.join(names))}[/]")
        elif kind == "handoff":
            target = str(event.get("to") or "")
            console.print(
                f"[bold magenta]{_glyph('handoff')} {escape(str(event.get('from') or ''))} "
                f"{_glyph('to')} {escape(_node_header(target, {}))}[/]"
            )
            instruction = str(event.get("instruction") or "").strip()
            if instruction:
                console.print(Text(f"   {instruction}", style="dim"))
            self._opened.discard(target)  # let the specialist introduce itself again
        elif kind == "tool_call":
            self._open(str(event.get("node") or ""))
            console.print(
                f"[bold cyan]-> {escape(str(event.get('name')))}[/] "
                f"[dim]{escape(_summarize_args(dict(event.get('args') or {})))}[/]"
            )
        elif kind == "tool_result":
            result = event.get("result")
            if not isinstance(result, Mapping):
                console.print(Text(_preview(str(result or "")), style="dim"))
            elif result.get("ok", True):
                output = str(result.get("result") or "")
                console.print(Text(output if self.verbose else _preview(output), style="dim"))
            else:
                error = str(result.get("error") or "")
                console.print(
                    f"[red]x {escape(str(event.get('name')))} failed:[/] "
                    f"{escape(error if self.verbose else _preview(error))}"
                )
        elif kind == "intent":
            route = str(event.get("intent_route") or "")
            confidence = event.get("intent_confidence")
            label = "chat (answer directly)" if route == "chat" else "workflow (plan it)"
            if self.verbose:
                console.print(
                    Text(
                        f"   {_glyph('route')} route: {label} "
                        f"[{float(confidence or 0):.2f}] {event.get('intent_reason') or ''}".rstrip(),
                        style="dim",
                    )
                )
            else:
                console.print(Text(f"   {_glyph('route')} {label}", style="dim"))
        elif kind == "chat_response":
            if self.verbose:
                console.print(
                    Text(f"   {_glyph('chat')} chat reply ({len(str(event.get('chat_response') or ''))} chars)", style="dim")
                )
        elif kind == "trace_saved":
            if self.verbose and event.get("mode") != "off":
                visits = event.get("node_visits") or {}
                nodes = ", ".join(f"{name} x{count}" for name, count in visits.items()) or "-"
                console.print(
                    Text(
                        f"   trace {event.get('trace_id')}: "
                        f"{event.get('tool_calls')} tool calls "
                        f"({event.get('failed_tool_calls')} failed), "
                        f"{event.get('handoff_count')} handoff(s), "
                        f"{event.get('checkpoint_count')} checkpoint(s), "
                        f"{event.get('duration_ms')} ms; nodes: {nodes}\n"
                        f"   -> {event.get('path')}",
                        style="dim",
                    )
                )
        elif kind == "checkpoint_saved":
            _render_checkpoint(event, verbose=self.verbose)
        elif kind == "memory":
            node = str(event.get("node") or "")
            self._open(node)
            digest = str(event.get("digest") or "")
            chars = int(event.get("chars") or 0)
            console.print(
                Text(
                    f"   {_glyph('result')} memory[{node or 'graph'}]: {digest} ({chars} chars)",
                    style="dim",
                )
            )
            if self.verbose:
                # the whole snapshot, so a run can be inspected (and diffed) in detail
                console.print(
                    Text(
                        _preview(
                            str(event.get("formatted") or "") or _format_memory(event),
                            lines=60,
                            limit=6000,
                        ),
                        style="dim",
                    )
                )
        elif kind == "context_compressed":
            persisted = event.get("persisted") or {}
            notepad = event.get("notepad") or {}
            ok = bool(event.get("ok", True))
            written = [
                name
                for record, name in (
                    (persisted, event.get("history_path")),
                    (notepad, event.get("notepad_path")),
                )
                if isinstance(record, Mapping) and record.get("ok") and name
            ]
            note = (
                f"  [{', '.join(str(name) for name in written)}]"
                if written
                else "  (history not persisted)"
            )
            console.print(
                Text(
                    f"   {_glyph('result')} compressed {_context_pair(event)}{note}",
                    style="dim" if ok else "yellow",
                )
            )
            if not ok and event.get("error"):
                console.print(Text(f"   fallback summary: {event['error']}", style="yellow"))
        elif kind == "search_results":
            count = len(event.get("results") or [])
            query = str(event.get("query") or "").strip()
            style = "dim" if event.get("ok") else "red"
            console.print(Text(f"   {_glyph('result')} {count} result(s) for {query!r}", style=style))
        elif kind == "todo_update":
            todos = event.get("todos") or []
            summary = ", ".join(
                f"{todo.get('id')}={todo.get('status')}" for todo in todos
            )
            console.print(Text(f"   {_glyph('result')} plan: {summary}", style="dim"))
        elif kind == "node_end":
            self._render_node_end(
                str(event.get("node") or ""), dict(event.get("update") or {})
            )

    def _open(self, node: str) -> None:
        if not node or node in self._opened:
            return
        console.print(f"[bold]{escape(_node_header(node, {}))}[/]")
        self._opened.add(node)

    def _render_node_end(self, node: str, update: dict[str, Any]) -> None:
        if node == "final":
            # the answer is printed by the closing panel in _render_result
            return
        if node in self._opened:
            self._opened.discard(node)  # the header was already printed for this round
        else:
            console.print(f"[bold]{escape(_node_header(node, update))}[/]")
        _render_node_update(node, update, verbose=self.verbose)


# -- informational commands --------------------------------------------------
def _print_tools(*, verbose: bool) -> None:
    table = Table(title=f"wq {__version__} - registered tools", show_lines=verbose)
    table.add_column("tool", style="bold cyan", no_wrap=True)
    table.add_column("description")
    if verbose:
        table.add_column("arguments", style="dim")
    for info in describe_tools():
        row = [info["name"], _first_sentence(info["description"])]
        if verbose:
            row.append(_format_schema(info["args_schema"]))
        table.add_row(*row)
    console.print(table)


def _print_config(
    *,
    workspace: Path,
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    env_file: Optional[Path],
    max_steps: int,
    max_attempts: int,
    timeout: int,
    approval_mode: str,
    checkpoint_mode: str,
    trace_mode: str,
    request_timeout: float,
    sources: Optional[Mapping[str, str]] = None,
) -> None:
    """Print the resolved configuration as plain, copy-paste friendly lines."""
    settings = describe_config(
        model=model, api_key=api_key, base_url=base_url, env_file=env_file
    )
    try:
        resolved_workspace = resolve_workspace(workspace, create=False)
        workspace_note = "exists" if resolved_workspace.is_dir() else "will be created"
    except WorkspacePathError as exc:
        resolved_workspace = Path(workspace)
        workspace_note = f"invalid: {exc}"

    def with_source(name: str, value: str) -> str:
        source = (sources or {}).get(name, "")
        return f"{value} ({source})" if source else value

    console.print(f"[bold]wq {__version__} - configuration[/]")
    key_row = (
        f"{settings['api_key']} (set)"
        if settings["api_key_present"]
        else "missing (set API_KEY in .env, or use --base-url for a local model)"
    )
    rows = [
        ("env file", settings["env_file"] or "(none found)"),
        ("model", settings["model"]),
        ("base url", settings["base_url"]),
        ("api key", key_row),
        ("workspace", with_source("workspace", f"{resolved_workspace} ({workspace_note})")),
        ("max attempts", str(max_attempts)),
        ("actor loops", f"{max_steps} per attempt"),
        ("context budget", f"{context_token_limit_from_env()} tokens"),
        ("approval", with_source("approval_mode", approval_mode)),
        ("checkpoints", with_source("checkpoint_mode", checkpoint_mode)),
        ("traces", with_source("trace_mode", trace_mode)),
        ("model/request", f"{request_timeout:g}s per attempt ({REQUEST_TIMEOUT_ENV})"),
        ("bash timeout", f"{timeout}s"),
        ("tools", ", ".join(tool_names())),
    ]
    for key, value in rows:
        console.print(Text(f"{key:<12}: {value}"))


# -- approval ----------------------------------------------------------------
def _parameter_sources(ctx: typer.Context, names: tuple[str, ...]) -> dict[str, str]:
    """Where click got each option from: ``cli``, ``env``, ``default`` or ``prompt``.

    Used by ``--show-config`` (so a surprising value can be traced back to the
    environment) and to notice ``--resume`` overriding ``--workspace``.
    """
    labels = {
        "COMMANDLINE": "cli",
        "ENVIRONMENT": "env",
        "DEFAULT": "default",
        "DEFAULT_MAP": "default",
        "PROMPT": "prompt",
    }
    sources: dict[str, str] = {}
    for name in names:
        try:
            source = ctx.get_parameter_source(name)
        except Exception:  # pragma: no cover - only outside a click context
            source = None
        sources[name] = labels.get(getattr(source, "name", ""), "")
    return sources


def _read_approval_answer(timeout: float) -> tuple[str, bool]:
    """Read one answer from stdin: ``(answer, timed_out)``.

    The blocking read happens in a daemon thread, so an unanswered prompt cannot hang
    the run forever. ``timeout <= 0`` waits forever; EOF and a timeout both come back
    as an empty answer, which the caller treats as "no".
    """
    if timeout <= 0:
        try:
            return (sys.stdin.readline() or "").strip().lower(), False
        except (EOFError, OSError, ValueError):
            return "", False

    import queue
    import threading

    answers: queue.Queue[str] = queue.Queue(maxsize=1)

    def read() -> None:
        try:
            answers.put((sys.stdin.readline() or "").strip().lower())
        except BaseException:  # noqa: BLE001 - any failure on stdin means "no"
            answers.put("")

    threading.Thread(target=read, name="wq-approval-read", daemon=True).start()
    try:
        return answers.get(timeout=timeout), False
    except queue.Empty:
        return "", True


def _approve_interactively(request: ApprovalRequest) -> ApprovalDecision:
    """Ask the human before a risky command runs (``--approval-mode inline``).

    Everything - panel *and* question - goes to **stderr**, because stdout may be
    redirected or captured (``wq "..." | tee log``): a question written to stdout looks
    exactly like the agent hanging. The answer is read from stdin with a bound
    (:data:`~wqagent.core.approval.APPROVAL_TIMEOUT_ENV`), so a run nobody is watching
    turns into a "no" instead of waiting forever, and a closed stdin means "no"
    instead of killing the run with an ``Abort``.
    """
    timeout = approval_timeout_from_env()
    console_note = (
        f"no answer within {timeout:g}s counts as no"
        if timeout > 0
        else "waiting for your answer (Ctrl+C stops the run)"
    )
    error_console.print(
        Panel(
            f"[bold]{escape(request.command)}[/]\n\n"
            f"risk     : {escape(request.risk_reason)}\n"
            f"tool     : {escape(request.tool_name)}\n"
            f"approval : {escape(request.id)}",
            title="[bold yellow]approval required[/]",
            border_style="yellow",
        )
    )
    error_console.print(
        f"[bold yellow]the agent is paused[/] - run this command? \\[y/N] "
        f"[dim]({console_note})[/]"
    )
    try:
        error_console.file.flush()
    except Exception:  # noqa: BLE001 - a console without a flushable file is fine
        pass

    answer, timed_out = _read_approval_answer(timeout)
    if timed_out:
        return ApprovalDecision(approved=False, reason=f"no answer within {timeout:g}s")
    if answer in ("y", "yes"):
        return ApprovalDecision(approved=True, reason="approved at the prompt")
    if not answer:
        return ApprovalDecision(
            approved=False, reason="no answer at the prompt (stdin closed?)"
        )
    return ApprovalDecision(approved=False, reason="declined at the prompt")


# -- checkpoints -------------------------------------------------------------
def _render_resume(event: Mapping[str, Any]) -> None:
    """Print what a ``--resume`` run picked up."""
    if not event.get("ok"):
        return
    lines = [
        f"[bold]resumed[/] from {escape(str(event.get('checkpoint_path')))}",
        f"task       : {escape(str(event.get('task') or ''))}",
        f"status     : {escape(str(event.get('status') or 'unknown'))} "
        f"(latest node: {escape(str(event.get('latest_node') or '-'))})",
        f"saved      : {escape(str(event.get('saved_at') or 'unknown'))}",
        f"restored   : {event.get('restored_files', 0)} file(s) from "
        f"{escape(str(event.get('git_commit') or 'no commit'))}",
        f"transcript : {event.get('message_count', 0)} message(s)",
    ]
    for warning in event.get("warnings") or []:
        lines.append(f"warning    : {escape(str(warning))}")
    console.print(Panel.fit("\n".join(lines), title="resume", border_style="cyan"))


def _render_checkpoint(event: Mapping[str, Any], *, verbose: bool) -> None:
    """Print one checkpoint line (the whole line is noise unless something failed)."""
    if event.get("ok") is False:
        console.print(
            f"[yellow]   checkpoint failed: {escape(str(event.get('error') or ''))}[/]"
        )
        return
    if not verbose:
        return
    git_note = (
        f", git {str(event.get('git_commit'))[:8]}" if event.get("git_ok") else ", no git"
    )
    console.print(
        Text(
            f"   checkpoint {event.get('status')}/{event.get('latest_node')} "
            f"-> {event.get('files')} file(s){git_note}, {event.get('mode')} mode",
            style="dim",
        )
    )


class _Heartbeat:
    """Say something while a node is in flight, so silence is never a mystery.

    A stalled provider (``--request-timeout`` seconds, retried a couple of times) can
    keep a run quiet for minutes: no events, no output, nothing on disk. This prints a
    dim line on **stderr** whenever nothing has happened for :data:`HEARTBEAT_SECONDS`,
    naming the node it is waiting on.
    """

    def __init__(self, interval: float = HEARTBEAT_SECONDS, *, enabled: bool = True) -> None:
        self.interval = interval
        self.enabled = enabled
        self.node = "the model"
        self._last = time.monotonic()
        self._stop = False
        self._thread: Any = None

    def __enter__(self) -> "_Heartbeat":
        import threading

        if self.enabled:
            self._thread = threading.Thread(target=self._run, name="wq-heartbeat", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._stop = True

    def touch(self, node: str | None = None) -> None:
        """Called on every event: the run is alive."""
        self._last = time.monotonic()
        if node:
            self.node = node

    def _run(self) -> None:
        while not self._stop:
            time.sleep(min(2.0, self.interval))
            if self._stop:
                return
            idle = time.monotonic() - self._last
            if idle < self.interval:
                continue
            self._last = time.monotonic()  # one line per interval, not per second
            error_console.print(self._explain(idle))

    def _explain(self, idle: float) -> str:
        """Name what the run is actually waiting for.

        A long shell command is the usual reason for silence (pip installing for two
        minutes), so report *it* - command text, elapsed time and its timeout - instead
        of blaming the model.
        """
        try:
            commands = describe_running_commands()
        except Exception:  # noqa: BLE001 - never break the run over a status line
            commands = []
        if commands:
            first = commands[0]
            limit = f"{first['timeout']:g}s" if first.get("timeout") else "no timeout"
            extra = f" (+{len(commands) - 1} more)" if len(commands) > 1 else ""
            tail = [str(line) for line in (first.get("tail") or [])][-2:]
            detail = "".join(f"\n    | {escape(line[:120])}" for line in tail)
            if not tail and float(first.get("elapsed") or 0) >= 30:
                # no output at all is itself information: say what it usually means
                detail = (
                    "\n    | (no output at all yet - usually a stalled network request, "
                    "e.g. a slow package index, or a tool waiting on input it cannot show)"
                )
            return (
                f"[dim]... bash has been running for {first['elapsed']:.0f}s "
                f"(timeout {limit}, {first.get('bytes', 0)} chars of output){extra}: "
                f"{escape(str(first['command']))}{detail}\n"
                "    Ctrl+C stops the run; if it needs longer, raise --timeout[/]"
            )
        return (
            f"[dim]... still in {escape(self.node)}: no events for {idle:.0f}s "
            "(waiting for the model? Ctrl+C stops the run)[/]"
        )


class _InterruptGuard:
    """Make Ctrl+C work even while the graph waits for a node.

    langgraph runs every sync node in a worker thread (``BackgroundExecutor``) and the
    main thread waits for it with ``concurrent.futures.wait``. On Windows that wait
    cannot run Python-level signal handlers (the lock wait is not interruptible), so a
    plain ``KeyboardInterrupt`` handler only fires *after* the node returns - pressing
    Ctrl+C during a 2-minute shell command looks like nothing happened at all.

    This guard therefore watches the signal from a small daemon thread through
    ``signal.set_wakeup_fd`` (the C-level handler writes the number immediately, before
    any Python code runs), which lets it:

    * stop the running shell command, so the node returns at once;
    * say what is going on, instead of appearing to ignore the key;
    * exit immediately on a second Ctrl+C (no waiting for threads or subprocesses).

    The Python-level handler still raises ``KeyboardInterrupt`` for the normal path, so
    the run is marked ``interrupted`` and its checkpoint/trace stay complete.
    """

    def __init__(
        self,
        exit_code: int = EXIT_INTERRUPTED,
        *,
        killer: Callable[[], int] = kill_running_commands,
    ) -> None:
        self.presses = 0
        self.exit_code = exit_code
        self.killer = killer
        self._previous: Any = None
        self._previous_wakeup: Any = None
        self._reader: Any = None
        self._left: Any = None
        self._right: Any = None
        self._watcher: Any = None

    # -- lifecycle -----------------------------------------------------------
    def __enter__(self) -> "_InterruptGuard":
        try:
            self._previous = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._handle)
        except (ValueError, OSError):  # pragma: no cover - not the main thread
            self._previous = None
        self._start_watcher()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._stop_watcher()
        if self._previous is not None:
            try:
                signal.signal(signal.SIGINT, self._previous)
            except (ValueError, OSError):  # pragma: no cover - interpreter shutting down
                pass

    def _start_watcher(self) -> None:
        """Watch the signal from a thread; needs a socketpair (Windows: sockets only)."""
        try:
            import select
            import socket
            import threading

            self._select = select
            left, right = socket.socketpair()
            left.setblocking(False)
            right.setblocking(False)
            self._left, self._right = left, right
            self._previous_wakeup = signal.set_wakeup_fd(right.fileno())
            self._watcher = threading.Thread(
                target=self._watch, name="wq-interrupt-watch", daemon=True
            )
            self._watcher.start()
        except Exception:  # noqa: BLE001 - the plain handler below is still installed
            self._close_sockets()

    def _stop_watcher(self) -> None:
        if self._previous_wakeup is not None or self._reader is not None:
            try:
                signal.set_wakeup_fd(self._previous_wakeup if self._previous_wakeup is not None else -1)
            except (ValueError, OSError):  # pragma: no cover - not the main thread
                pass
            self._previous_wakeup = None
        self._close_sockets()

    def _close_sockets(self) -> None:
        for sock in (self._left, self._right):
            try:
                if sock is not None:
                    sock.close()
            except OSError:  # pragma: no cover - already closed
                pass
        self._left = self._right = None
        if self._reader is not None:
            try:
                self._reader.close()
            except OSError:  # pragma: no cover
                pass
            self._reader = None

    # -- the two interrupt paths --------------------------------------------
    def _watch(self) -> None:
        """Daemon thread: react to the signal as soon as the OS reports it."""
        assert self._left is not None
        while True:
            try:
                ready, _, _ = self._select.select([self._left], [], [], 0.2)
            except (OSError, ValueError):  # pragma: no cover - sockets closed
                return
            if not ready:
                if self._left.fileno() < 0:  # pragma: no cover - closed
                    return
                continue
            try:
                self._left.recv(4096)
            except (OSError, ValueError):  # pragma: no cover - closed
                return
            try:
                self._on_signal()
            except SystemExit:  # pragma: no cover - os._exit does not raise
                return

    def _handle(self, signum: int, frame: Any) -> None:
        """Python-level handler (main thread): raise once the node has returned."""
        if self._watcher is None:
            self._on_signal()  # no watcher: do the work here, as a fallback
        raise KeyboardInterrupt

    def _on_signal(self) -> None:
        self.presses += 1
        if self.presses == 1:
            # the shell command lives in a worker thread and never sees the interrupt
            try:
                killed = self.killer()
            except Exception:  # noqa: BLE001 - never let the handler raise
                killed = 0
            note = f" - stopped {killed} running command(s)" if killed else ""
            error_console.print(
                "[yellow]interrupt received[/]"
                f"{note}: finishing the current step, then stopping "
                "(press Ctrl+C again to exit immediately)"
            )
            return
        error_console.print(
            "[yellow]exiting now[/] - the last checkpoint and trace are on disk"
        )
        os._exit(self.exit_code)


# -- the command -------------------------------------------------------------
@app.command()
def run_task(
    ctx: typer.Context,
    task: Annotated[
        Optional[str], typer.Argument(help="What the agent should do, in natural language.")
    ] = None,
    workspace: Annotated[
        Optional[Path],
        typer.Option(
            "--workspace",
            "-w",
            envvar="WQ_WORKSPACE",
            help="Workspace directory; created automatically when missing.",
            show_default="workspace",
        ),
    ] = None,
    resume: Annotated[
        Optional[Path],
        typer.Option(
            "--resume",
            help=(
                "Resume the last checkpointed run in this workspace: restores the "
                "snapshot and continues from the saved plan (prints the command in "
                "RECOVERY.md). Replaces --workspace."
            ),
        ),
    ] = None,
    model: Annotated[
        Optional[str],
        typer.Option("--model", "-m", envvar="WQ_MODEL", help="Chat model (overrides $MODEL)."),
    ] = None,
    base_url: Annotated[
        Optional[str],
        typer.Option(
            "--base-url",
            envvar="WQ_BASE_URL",
            help="OpenAI-compatible base URL (overrides $BASE_URL).",
        ),
    ] = None,
    api_key: Annotated[
        Optional[str],
        typer.Option("--api-key", envvar="WQ_API_KEY", help="API key (overrides $API_KEY)."),
    ] = None,
    env_file: Annotated[
        Optional[Path], typer.Option("--env-file", help="Explicit .env file to load.")
    ] = None,
    max_steps: Annotated[
        int,
        typer.Option(
            "--max-steps",
            min=1,
            max=200,
            help="Tool rounds the planner may use per attempt.",
            show_default=True,
        ),
    ] = DEFAULT_MAX_LOOPS,
    max_attempts: Annotated[
        int,
        typer.Option(
            "--max-attempts",
            min=1,
            max=20,
            help="Plan/act/verify rounds before the run stops.",
            show_default=True,
        ),
    ] = DEFAULT_MAX_ATTEMPTS,
    approval_mode: Annotated[
        ApprovalMode,
        typer.Option(
            "--approval-mode",
            envvar=APPROVAL_MODE_ENV,
            case_sensitive=False,
            help=(
                "How to treat risky shell commands (pip/npm install, curl, uvicorn): "
                "inline asks you first, auto runs and marks them, deny refuses them."
            ),
            show_default=True,
        ),
    ] = "inline",
    checkpoint_mode: Annotated[
        CheckpointMode,
        typer.Option(
            "--checkpoint-mode",
            envvar=CHECKPOINT_MODE_ENV,
            case_sensitive=False,
            help=(
                "What to save for a resume: light (metadata + recovery guide + git "
                "snapshot per node), strict (plus full state and every event), off."
            ),
            show_default=True,
        ),
    ] = "light",
    trace_mode: Annotated[
        TraceMode,
        typer.Option(
            "--trace-mode",
            envvar=TRACE_MODE_ENV,
            case_sensitive=False,
            help=(
                "What to record about the run under .wqagent/traces/<id>/: on/summary "
                "(counters + timeline), full (plus events.jsonl), off."
            ),
            show_default=True,
        ),
    ] = "on",
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature",
            "-t",
            min=0.0,
            max=2.0,
            help="Sampling temperature.",
            show_default=True,
        ),
    ] = 0.0,
    timeout: Annotated[
        int,
        typer.Option(
            "--timeout",
            min=1,
            max=3600,
            help="Default seconds allowed per shell command.",
            show_default=True,
        ),
    ] = DEFAULT_TIMEOUT_SECONDS,
    request_timeout: Annotated[
        Optional[float],
        typer.Option(
            "--request-timeout",
            min=1.0,
            max=3600.0,
            help=(
                "Seconds to wait for one model response before retrying or failing "
                f"(also ${REQUEST_TIMEOUT_ENV}). The provider is retried twice, so this "
                "bounds how long a stalled model can keep a run quiet."
            ),
            show_default="120",
        ),
    ] = None,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Print only the final answer.")
    ] = False,
    no_router: Annotated[
        bool,
        typer.Option(
            "--no-router",
            help=(
                "Skip the entry router and always run the workflow. By default a "
                "conversational input (greeting, thanks, a question) is answered "
                "directly by one cheap model call instead of the whole pipeline."
            ),
        ),
    ] = False,
    tui: Annotated[
        bool,
        typer.Option(
            "--tui",
            help=(
                "Open the interactive Textual UI (plan panel, event stream, approvals "
                'in a modal) instead of the plain CLI; no TASK is needed - type your '
                'turns inside the UI. Needs: pip install "wqagent\\[tui]".'
            ),
        ),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show every step, argument and full output.")
    ] = False,
    show_config: Annotated[
        bool, typer.Option("--show-config", help="Print the resolved configuration and exit.")
    ] = False,
    list_tools: Annotated[
        bool, typer.Option("--list-tools", help="Print the registered tools and exit.")
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            is_eager=True,
            callback=_version_callback,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Run TASK with the agent, confined to the workspace."""
    # Annotated options above are validated by click (`Literal` -> choices); the
    # normalize_* helpers keep the values canonical (e.g. `--trace-mode ON`).
    approval_mode = normalize_approval_mode(approval_mode)
    checkpoint_mode = normalize_checkpoint_mode(checkpoint_mode)
    trace_mode = normalize_trace_mode(trace_mode)
    workspace_path = workspace if workspace is not None else Path("workspace")
    resolved_request_timeout = request_timeout_from_env(
        request_timeout if request_timeout is not None else DEFAULT_REQUEST_TIMEOUT_SECONDS
    )
    sources = _parameter_sources(
        ctx, ("workspace", "approval_mode", "checkpoint_mode", "trace_mode", "resume")
    )
    if list_tools:
        _print_tools(verbose=verbose)
        raise typer.Exit(EXIT_OK)
    if show_config:
        _print_config(
            workspace=workspace_path,
            model=model,
            base_url=base_url,
            api_key=api_key,
            env_file=env_file,
            max_steps=max_steps,
            max_attempts=max_attempts,
            timeout=timeout,
            approval_mode=approval_mode,
            checkpoint_mode=checkpoint_mode,
            trace_mode=trace_mode,
            request_timeout=resolved_request_timeout,
            sources=sources,
        )
        raise typer.Exit(EXIT_OK)
    if not task or not task.strip():
        # `--tui` brings its own input loop, and `--resume` takes the task from the
        # checkpoint, so neither needs a task on the command line
        if resume is None and not tui:
            error_console.print(
                '[bold red]error:[/] missing task.\n'
                'usage: [bold]wq "<task>" [--workspace PATH][/]   '
                "(see [bold]wq --help[/])"
            )
            raise typer.Exit(EXIT_USAGE)
    if resume is not None and sources.get("workspace") == "cli":
        error_console.print(
            "[yellow]note:[/] --resume replaces --workspace; using the checkpoint's workspace."
        )

    try:
        state = RuntimeState(
            workspace=resume if resume is not None else workspace_path,
            timeout_seconds=timeout,
            approval_mode=approval_mode,
            approval_handler=_approve_interactively,
            checkpoint_mode=checkpoint_mode,
            trace_mode=trace_mode,
        )
    except (WorkspacePathError, ValueError) as exc:
        _fail(str(exc))

    # the TUI takes over: it records the session itself, one turn per submitted line
    if tui:
        try:
            tui_model = create_model(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                env_file=env_file,
                timeout=resolved_request_timeout,
            )
        except ProviderConfigError as exc:
            _fail(str(exc))
        from wqagent.cli.tui import run_tui

        raise typer.Exit(
            run_tui(
                workspace=state.workspace,
                model=tui_model,
                approval_mode=approval_mode,
                checkpoint_mode=checkpoint_mode,
                trace_mode=trace_mode,
                max_attempts=max_attempts,
                max_loops=max_steps,
                entry_router=not no_router,
            )
        )

    # the conversation session: what the user asked before, and what came of it
    session = load_or_create_session(state.workspace)
    turn = append_user_turn(session, task if task and task.strip() else "(resume)")
    state.session_id = str(session.get("session_id") or "")
    state.session_turn = turn

    inputs: Optional[dict[str, Any]] = None
    resume_event: Optional[dict[str, Any]] = None
    if resume is not None:
        inputs, resume_event = CheckpointManager.load_resume_inputs(
            state, task=(task or None), max_attempts=max_attempts
        )
        if inputs is None:
            _fail(
                f"no checkpoint to resume in {state.workspace}"
                f" ({'; '.join(str(item) for item in resume_event.get('warnings') or []) or 'not found'})"
            )
        task = str(inputs.get("task") or task or "")
        if not quiet:
            _render_resume(resume_event)

    try:
        chat = create_model(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            env_file=env_file,
            timeout=resolved_request_timeout,
        )
    except ProviderConfigError as exc:
        _fail(str(exc))

    if not quiet:
        console.print(
            Panel.fit(
                f"[bold]WQ Agent[/] v{__version__}\n"
                f"task      : {escape(task.strip())}\n"
                f"workspace : {escape(str(state.workspace))}\n"
                f"model     : {escape(_model_label(chat))} "
                f"(request timeout {resolved_request_timeout:g}s)\n"
                f"workflow  : planner -> verifier -> final "
                f"(planner delegates to searchAgent/codeAgent; "
                f"max {max_attempts} attempts, {max_steps} planner rounds each)\n"
                f"context   : {context_token_limit_from_env()} token budget "
                f"({CONTEXT_TOKEN_LIMIT_ENV})\n"
                f"approval  : {escape(str(state.approval_mode))} "
                f"({APPROVAL_MODE_ENV}, prompt waits "
                f"{approval_timeout_from_env():g}s / {APPROVAL_TIMEOUT_ENV})\n"
                f"checkpoint: {escape(str(state.checkpoint_mode))} "
                f"({CHECKPOINT_MODE_ENV})\n"
                f"trace     : {escape(str(state.trace_mode))} ({TRACE_MODE_ENV}), "
                f"id {escape(str(state.trace_id))}\n"
                f"tools     : {escape(', '.join(tool_names()))}",
                border_style="cyan",
                title="run",
            )
        )

    summary = _EventSummary()
    renderer = None if quiet else _NodeRenderer(verbose=verbose)
    heartbeat = _Heartbeat(enabled=not quiet)
    final: dict[str, Any] = {}
    try:
        with _InterruptGuard(), heartbeat:
            for event in stream_agent_events(
                task.strip(),
                workspace=state.workspace,
                max_attempts=max_attempts,
                max_loops=max_steps,
                model=chat,
                state=state,
                inputs=inputs,
                resume_event=resume_event,
                entry_router=not no_router and resume is None,
            ):
                heartbeat.touch(str(event.get("node") or "") or None)
                summary.observe(event)
                if event.get("type") == "final_answer":
                    final = event
                if renderer is not None:
                    renderer.render(event)
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        error_console.print("[yellow]interrupted[/]")
        save_session(state.workspace, session)  # keep the user's turn
        if state.checkpoint_mode != "off":
            error_console.print(f"[yellow]resume with:[/] {resume_command(state.workspace)}")
        raise typer.Exit(EXIT_INTERRUPTED) from None
    except Exception as exc:  # API errors, connection failures, ...
        save_session(state.workspace, session)
        if state.checkpoint_mode != "off":
            error_console.print(f"[yellow]resume with:[/] {resume_command(state.workspace)}")
        _fail(_failure_hint(exc, resolved_request_timeout))

    if not final:
        _fail("the agent stopped without producing a final answer")

    _record_exchange(state, session, turn, final, verbose=verbose)
    _render_result(final, summary, state, quiet=quiet)
    if not final.get("passed") and not quiet and state.checkpoint_mode != "off":
        console.print(f"[yellow]resume with:[/] {resume_command(state.workspace)}")
    raise typer.Exit(EXIT_OK if final.get("passed") else EXIT_ERROR)


def _failure_hint(exc: BaseException, request_timeout: float) -> str:
    """Explain a failure, adding what to do about a stalled/timed-out provider."""
    text = f"{type(exc).__name__}: {exc}"
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    if "timeout" in name or "timed out" in message:
        text += (
            f"\n[the model did not answer within {request_timeout:g}s per attempt "
            "(it is retried twice). Check the endpoint/network, or lower it with "
            f"--request-timeout / ${REQUEST_TIMEOUT_ENV} to fail faster. The run is "
            "resumable: the checkpoint and trace up to the last node are on disk.]"
        )
    return text


def _record_exchange(
    state: RuntimeState,
    session: dict[str, Any],
    turn: int,
    final: Mapping[str, Any],
    *,
    verbose: bool,
) -> None:
    """Append the assistant turn to the session and write both session files."""
    content = str(final.get("content") or "")
    route = str(final.get("route") or "workflow")
    append_assistant_turn(
        session,
        turn=turn,
        route=route,
        content=content,
        summary=_first_sentence(content),
    )
    save_session(state.workspace, session)
    if verbose:
        target = session_path(state.workspace)
        console.print(
            Text(
                f"   session turn {turn} recorded ({route}) -> {target}",
                style="dim",
            )
        )


def _render_result(
    final: dict[str, Any],
    summary: _EventSummary,
    state: RuntimeState,
    *,
    quiet: bool,
) -> None:
    """Print the final answer event plus the run summary."""
    passed = bool(final.get("passed"))
    chat = str(final.get("route") or "") == "chat"
    answer = str(final.get("content") or "").strip() or "(the model returned an empty answer)"
    if not quiet:
        console.print(f"[bold]{_node_header('final', {})}[/]")
    if chat:
        console.print(
            Panel(
                Markdown(answer),
                title="chat" if passed else "not verified",
                border_style="cyan",
            )
        )
    else:
        console.print(
            Panel(
                Markdown(answer),
                title="final answer" if passed else "not verified",
                border_style="green" if passed else "red",
            )
        )
    if quiet:
        return
    if chat:
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold")
        table.add_column()
        table.add_row("route", "chat (no tools, no workspace changes)")
        table.add_row(
            "why",
            f"{final.get('intent_reason') or 'conversational input'} "
            f"[{float(final.get('intent_confidence') or 0):.2f}]",
        )
        console.print(table)
        console.print(
            "[dim]give it a task (\"...\") if you wanted work done - or pass --no-router "
            "to always run the workflow[/]"
        )
        return
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("verdict", "passed" if passed else "failed")
    table.add_row("attempts", str(summary.attempts or 1))
    calls = str(summary.tool_calls)
    if summary.failures:
        calls += f" ({summary.failures} failed)"
    table.add_row("tool calls", calls)
    if summary.files:
        listed = ", ".join(summary.files[:10]) + (" ..." if len(summary.files) > 10 else "")
        table.add_row("files written", Text(listed))
    if summary.nodes:
        table.add_row("nodes", " -> ".join(summary.nodes))
    table.add_row("workspace", Text(str(state.workspace)))
    console.print(table)


def main() -> None:
    """Console script entry point registered as ``wq``."""
    _configure_output()
    try:
        app(prog_name="wq")
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        error_console.print("[yellow]interrupted[/]")
        raise SystemExit(EXIT_INTERRUPTED) from None


if __name__ == "__main__":  # pragma: no cover
    main()
