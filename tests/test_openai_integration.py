"""End-to-end tests over real HTTP against a fake OpenAI-compatible server.

These tests exercise the whole chain - ``create_model()`` (langchain-openai),
``bind_tools()``, the OpenAI request/response protocol, the agent loop and the
real tools - without needing credentials or internet access.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

from typer.testing import CliRunner

from wqagent.cli.app import app
from wqagent.core.agent import Agent
from wqagent.core.state import RuntimeState
from wqagent.providers import create_model
from wqagent.tools import build_tools

runner = CliRunner()


def _tool_call_response(name: str, arguments: dict[str, Any], call_id: str = "call_1") -> dict[str, Any]:
    return {
        "id": "chatcmpl-tool",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-test",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _answer_response(text: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-final",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
    }


class _FakeOpenAIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    responses: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            type(self).requests.append(json.loads(raw or b"{}"))
        except json.JSONDecodeError:  # pragma: no cover - should not happen
            type(self).requests.append({})
        payload = type(self).responses.pop(0)
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:  # silence the test output
        return


@contextmanager
def fake_openai(responses: list[dict[str, Any]]) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """Serve *responses* on a throwaway localhost port; yield (base_url, requests)."""
    _FakeOpenAIHandler.responses = list(responses)
    _FakeOpenAIHandler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", _FakeOpenAIHandler.requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_agent_round_trip_over_http(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    workspace = tmp_path / "ws"
    scripted = [
        _tool_call_response(
            "file_write", {"file_path": "hello.py", "content": "print('hi')\n"}
        ),
        _answer_response("Created hello.py."),
    ]

    with fake_openai(scripted) as (base_url, requests):
        model = create_model(
            model="gpt-test",
            api_key="test-key",
            base_url=base_url,
            env_file=tmp_path / "no-such.env",
        )
        state = RuntimeState(workspace=workspace, timeout_seconds=30)
        result = Agent(model, build_tools(state), state).run("create hello.py")

    assert result.text == "Created hello.py."
    assert result.completed
    assert (workspace / "hello.py").read_text(encoding="utf-8") == "print('hi')\n"

    # the model really received all five tool schemas and the tool result
    assert len(requests) == 2
    tools_sent = requests[0]["tools"]
    assert sorted(tool["function"]["name"] for tool in tools_sent) == [
        "bash",
        "file_edit",
        "file_read",
        "file_write",
        "grep",
    ]
    assert requests[0]["messages"][0]["role"] == "system"
    assert str(workspace) in requests[0]["messages"][0]["content"]
    assert requests[1]["messages"][-1]["role"] == "tool"
    assert "Created hello.py" in requests[1]["messages"][-1]["content"]


def test_cli_round_trip_over_http(tmp_path: Path, monkeypatch: Any) -> None:
    """The whole workflow over real HTTP: planner -> codeAgent -> verifier -> final."""
    monkeypatch.chdir(tmp_path)
    workspace = tmp_path / "cli-ws"
    scripted = [
        _tool_call_response(
            "todo_write",
            {
                "plan_summary": "write notes/todo.md",
                "todos": [{"id": "t1", "content": "write notes/todo.md"}],
                "acceptance_criteria": ["the file exists"],
                "verification_commands": ["echo ok"],
            },
        ),
        _tool_call_response(
            "call_code_agent", {"instruction": "write notes/todo.md"}, "call_2"
        ),
        _tool_call_response(
            "file_write",
            {"file_path": "notes/todo.md", "content": "- [ ] ship it\n"},
            "call_3",
        ),
        _answer_response("Wrote notes/todo.md."),
        _answer_response("Delegated the implementation."),
        _answer_response("Inspected the file."),
        _tool_call_response(
            "verification_report",
            {
                "passed": True,
                "reason": "the file exists and the command succeeded",
                "checks": [{"name": "the file exists", "passed": True, "detail": "read it"}],
                "recommended_next_instruction": "",
            },
            "call_9",
        ),
        _answer_response("Done: notes/todo.md is written and verified."),
    ]
    for name in ("API_KEY", "OPENAI_API_KEY", "MODEL", "OPENAI_MODEL", "BASE_URL", "OPENAI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)

    with fake_openai(scripted) as (base_url, requests):
        monkeypatch.setenv("API_KEY", "test-key")
        monkeypatch.setenv("MODEL", "gpt-test")
        monkeypatch.setenv("BASE_URL", base_url)
        result = runner.invoke(app, ["write the todo file", "-w", str(workspace)])

    assert result.exit_code == 0, result.output
    assert (workspace / "notes" / "todo.md").read_text(encoding="utf-8") == "- [ ] ship it\n"
    assert "Done: notes/todo.md is written and verified." in result.output
    # eight calls: planner x3, codeAgent x2, verifier x2 (inspect + report), final
    assert len(requests) == 8
    assert sorted(
        tool["function"]["name"] for tool in requests[0]["tools"]
    ) == ["call_code_agent", "call_search_agent", "todo_write"]
    assert sorted(
        tool["function"]["name"] for tool in requests[2]["tools"]
    ) == ["bash", "file_edit", "file_read", "file_write", "grep", "todo_update"]
    assert sorted(
        tool["function"]["name"] for tool in requests[5]["tools"]
    ) == ["file_read", "grep"]
