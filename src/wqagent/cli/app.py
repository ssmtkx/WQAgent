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
import sys
from pathlib import Path
from typing import Any, Mapping, NoReturn, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from wqagent import __version__
from wqagent.core.agent import DEFAULT_MAX_LOOPS, stream_agent_events
from wqagent.core.paths import WorkspacePathError, resolve_workspace
from wqagent.core.state import DEFAULT_TIMEOUT_SECONDS, RuntimeState
from wqagent.graph.state import DEFAULT_MAX_ATTEMPTS
from wqagent.providers import ProviderConfigError, create_model, describe_config
from wqagent.tools import describe_tools, tool_names

app = typer.Typer(
    add_completion=False,
    rich_markup_mode="rich",
    help="WQ Agent - a workspace-scoped tool-calling coding agent.",
)
console = Console()
error_console = Console(stderr=True)

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


def _render_node_update(node: str, update: dict[str, Any]) -> None:
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
        _render_node_update(node, update)


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
        ("workspace", f"{resolved_workspace} ({workspace_note})"),
        ("max attempts", str(max_attempts)),
        ("actor loops", f"{max_steps} per attempt"),
        ("bash timeout", f"{timeout}s"),
        ("tools", ", ".join(tool_names())),
    ]
    for key, value in rows:
        console.print(Text(f"{key:<12}: {value}"))


# -- the command -------------------------------------------------------------
@app.command()
def run_task(
    task: Optional[str] = typer.Argument(
        None, help="What the agent should do, in natural language."
    ),
    workspace: Path = typer.Option(
        Path("workspace"),
        "--workspace",
        "-w",
        envvar="WQ_WORKSPACE",
        help="Workspace directory; created automatically when missing.",
        show_default=True,
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", envvar="WQ_MODEL", help="Chat model (overrides $MODEL)."
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "--base-url",
        envvar="WQ_BASE_URL",
        help="OpenAI-compatible base URL (overrides $BASE_URL).",
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", envvar="WQ_API_KEY", help="API key (overrides $API_KEY)."
    ),
    env_file: Optional[Path] = typer.Option(
        None, "--env-file", help="Explicit .env file to load."
    ),
    max_steps: int = typer.Option(
        DEFAULT_MAX_LOOPS,
        "--max-steps",
        min=1,
        max=200,
        help="Tool rounds the planner may use per attempt.",
        show_default=True,
    ),
    max_attempts: int = typer.Option(
        DEFAULT_MAX_ATTEMPTS,
        "--max-attempts",
        min=1,
        max=20,
        help="Plan/act/verify rounds before the run stops.",
        show_default=True,
    ),
    temperature: float = typer.Option(
        0.0,
        "--temperature",
        "-t",
        min=0.0,
        max=2.0,
        help="Sampling temperature.",
        show_default=True,
    ),
    timeout: int = typer.Option(
        DEFAULT_TIMEOUT_SECONDS,
        "--timeout",
        min=1,
        max=3600,
        help="Default seconds allowed per shell command.",
        show_default=True,
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Print only the final answer."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show every step, argument and full output."
    ),
    show_config: bool = typer.Option(
        False, "--show-config", help="Print the resolved configuration and exit."
    ),
    list_tools: bool = typer.Option(
        False, "--list-tools", help="Print the registered tools and exit."
    ),
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        is_eager=True,
        callback=_version_callback,
        help="Show the version and exit.",
    ),
) -> None:
    """Run TASK with the agent, confined to the workspace."""
    if list_tools:
        _print_tools(verbose=verbose)
        raise typer.Exit(EXIT_OK)
    if show_config:
        _print_config(
            workspace=workspace,
            model=model,
            base_url=base_url,
            api_key=api_key,
            env_file=env_file,
            max_steps=max_steps,
            max_attempts=max_attempts,
            timeout=timeout,
        )
        raise typer.Exit(EXIT_OK)
    if not task or not task.strip():
        error_console.print(
            '[bold red]error:[/] missing task.\n'
            'usage: [bold]wq "<task>" [--workspace PATH][/]   '
            "(see [bold]wq --help[/])"
        )
        raise typer.Exit(EXIT_USAGE)

    try:
        state = RuntimeState(workspace=workspace, timeout_seconds=timeout)
    except (WorkspacePathError, ValueError) as exc:
        _fail(str(exc))

    try:
        chat = create_model(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            env_file=env_file,
        )
    except ProviderConfigError as exc:
        _fail(str(exc))

    if not quiet:
        console.print(
            Panel.fit(
                f"[bold]WQ Agent[/] v{__version__}\n"
                f"task      : {escape(task.strip())}\n"
                f"workspace : {escape(str(state.workspace))}\n"
                f"model     : {escape(_model_label(chat))}\n"
                f"workflow  : planner -> verifier -> final "
                f"(planner delegates to searchAgent/codeAgent; "
                f"max {max_attempts} attempts, {max_steps} planner rounds each)\n"
                f"tools     : {escape(', '.join(tool_names()))}",
                border_style="cyan",
                title="run",
            )
        )

    summary = _EventSummary()
    renderer = None if quiet else _NodeRenderer(verbose=verbose)
    final: dict[str, Any] = {}
    try:
        for event in stream_agent_events(
            task.strip(),
            workspace=state.workspace,
            max_attempts=max_attempts,
            max_loops=max_steps,
            model=chat,
            state=state,
        ):
            summary.observe(event)
            if event.get("type") == "final_answer":
                final = event
            if renderer is not None:
                renderer.render(event)
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        error_console.print("[yellow]interrupted[/]")
        raise typer.Exit(EXIT_INTERRUPTED) from None
    except Exception as exc:  # API errors, connection failures, ...
        _fail(f"{type(exc).__name__}: {exc}")

    if not final:
        _fail("the agent stopped without producing a final answer")

    _render_result(final, summary, state, quiet=quiet)
    raise typer.Exit(EXIT_OK if final.get("passed") else EXIT_ERROR)


def _render_result(
    final: dict[str, Any],
    summary: _EventSummary,
    state: RuntimeState,
    *,
    quiet: bool,
) -> None:
    """Print the final answer event plus the run summary."""
    passed = bool(final.get("passed"))
    answer = str(final.get("content") or "").strip() or "(the model returned an empty answer)"
    if not quiet:
        console.print(f"[bold]{_node_header('final', {})}[/]")
    console.print(
        Panel(
            Markdown(answer),
            title="final answer" if passed else "not verified",
            border_style="green" if passed else "red",
        )
    )
    if quiet:
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
