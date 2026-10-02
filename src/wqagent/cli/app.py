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
from typing import Any, NoReturn, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from wqagent import __version__
from wqagent.core.agent import DEFAULT_MAX_STEPS, stream_agent_events
from wqagent.core.paths import WorkspacePathError, resolve_workspace
from wqagent.core.state import DEFAULT_TIMEOUT_SECONDS, RuntimeState
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
    """Counters accumulated while streaming ReAct events, for the closing summary."""

    def __init__(self) -> None:
        self.loops = 0
        self.tool_calls = 0
        self.failures = 0
        self.files: list[str] = []

    def observe(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "ai_message":
            self.loops = max(self.loops, int(event.get("loop") or 0))
        elif kind == "tool_call":
            self.tool_calls += 1
            name = str(event.get("name") or "")
            path = str((event.get("args") or {}).get("file_path") or "")
            if name in {"file_write", "file_edit"} and path and path not in self.files:
                self.files.append(path)
        elif kind == "tool_result":
            if not (event.get("result") or {}).get("ok", True):
                self.failures += 1


def _render_event(event: dict[str, Any], *, verbose: bool) -> None:
    """Print one ReAct event as soon as it arrives."""
    kind = event.get("type")
    if kind == "ai_message":
        # Only intermediate reasoning is interesting here: the turn without tool
        # calls is the final answer, and it gets its own panel below.
        names = [str(name) for name in event.get("tool_calls") or []]
        if not verbose or not names:
            return
        content = str(event.get("content") or "").strip()
        if content:
            console.print(Text(content, style="dim"))
        console.print(f"[dim]   calling: {escape(', '.join(names))}[/]")
    elif kind == "tool_call":
        console.print(
            f"[bold cyan]-> {escape(str(event.get('name')))}[/] "
            f"[dim]{escape(_summarize_args(dict(event.get('args') or {})))}[/]"
        )
    elif kind == "tool_result":
        result = dict(event.get("result") or {})
        if result.get("ok", True):
            output = str(result.get("result") or "")
            console.print(Text(output if verbose else _preview(output), style="dim"))
        else:
            error = str(result.get("error") or "")
            console.print(
                f"[red]x {escape(str(event.get('name')))} failed:[/] "
                f"{escape(error if verbose else _preview(error))}"
            )


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
        ("max steps", str(max_steps)),
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
        DEFAULT_MAX_STEPS,
        "--max-steps",
        min=1,
        max=500,
        help="Model/tool rounds before the run stops.",
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
                f"tools     : {escape(', '.join(tool_names()))}",
                border_style="cyan",
                title="run",
            )
        )

    summary = _EventSummary()
    final: dict[str, Any] = {}
    try:
        for event in stream_agent_events(
            task.strip(),
            workspace=state.workspace,
            max_loops=max_steps,
            model=chat,
            state=state,
        ):
            summary.observe(event)
            if event.get("type") == "final_answer":
                final = event
            if not quiet:
                _render_event(event, verbose=verbose)
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        error_console.print("[yellow]interrupted[/]")
        raise typer.Exit(EXIT_INTERRUPTED) from None
    except Exception as exc:  # API errors, connection failures, ...
        _fail(f"{type(exc).__name__}: {exc}")

    if not final:
        _fail("the agent stopped without producing a final answer")

    _render_result(final, summary, state, quiet=quiet)
    raise typer.Exit(EXIT_OK if not final.get("stopped_early") else EXIT_ERROR)


def _render_result(
    final: dict[str, Any],
    summary: _EventSummary,
    state: RuntimeState,
    *,
    quiet: bool,
) -> None:
    """Print the final answer event plus the run summary."""
    stopped_early = bool(final.get("stopped_early"))
    answer = str(final.get("content") or "").strip() or "(the model returned an empty answer)"
    console.print(
        Panel(
            Markdown(answer),
            title="stopped early" if stopped_early else "final answer",
            border_style="yellow" if stopped_early else "green",
        )
    )
    if quiet:
        return
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("loops", str(summary.loops))
    calls = str(summary.tool_calls)
    if summary.failures:
        calls += f" ({summary.failures} failed)"
    table.add_row("tool calls", calls)
    if summary.files:
        listed = ", ".join(summary.files[:10]) + (" ..." if len(summary.files) > 10 else "")
        table.add_row("files written", Text(listed))
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
