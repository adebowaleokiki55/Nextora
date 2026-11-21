"""Run each real provider client through the tool loop against a local mock
server, checking what actually goes over the wire. No network access needed."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from backend.app.agents.runner import AgentTool, run_tool_loop
from backend.app.core.llm import ToolSpec
from backend.app.core.llm.anthropic_provider import FALLBACK_BETA, AnthropicClient
from backend.app.core.llm.ollama_provider import OllamaClient
from backend.app.core.llm.openai_provider import OpenAICompatibleClient


class MockServer:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", 0))
                server.requests.append(
                    {
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": json.loads(self.rfile.read(length) or b"{}"),
                    }
                )
                payload = json.dumps(server.responses.pop(0)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def mock_server():
    servers = []

    def start(responses):
        server = MockServer(responses)
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.close()


def lookup_tools():
    return [AgentTool(ToolSpec("lookup", "Look something up."), lambda _a: {"value": 42})]


def test_anthropic_client_over_the_wire(mock_server):
    usage = {"input_tokens": 1, "output_tokens": 1}
    server = mock_server(
        [
            {
                "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "sig-abc"},
                    {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}},
                ],
                "stop_reason": "tool_use", "stop_sequence": None, "usage": usage,
            },
            {
                "id": "msg_2", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
                "content": [{"type": "text", "text": "It is 42."}],
                "stop_reason": "end_turn", "stop_sequence": None, "usage": usage,
            },
        ]
    )
    client = AnthropicClient(api_key="test-key", model="claude-opus-5-5", base_url=server.url)
    result = run_tool_loop(client, "sys", "q", lookup_tools(), required_tools={"lookup"})

    assert result.answer == "It is 42."
    first, second = server.requests
    assert FALLBACK_BETA in first["headers"]["anthropic-beta"]
    assert first["body"]["fallbacks"] == "default"
    assert first["body"]["tools"][0]["input_schema"]["type"] == "object"
    assert "temperature" not in first["body"]

    replayed = second["body"]["messages"][1]["content"]
    assert replayed[0] == {"type": "thinking", "thinking": "", "signature": "sig-abc"}
    assert replayed[1]["type"] == "tool_use" and replayed[1]["id"] == "toolu_1"
    tool_result = second["body"]["messages"][2]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "toolu_1"
    assert json.loads(tool_result["content"]) == {"value": 42}


def test_anthropic_model_without_fallback_uses_plain_endpoint(mock_server):
    server = mock_server(
        [
            {
                "id": "m", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
                "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
                "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        ]
    )
    client = AnthropicClient(api_key="k", model="claude-haiku-5-5", base_url=server.url)
    assert client.chat("sys", [{"role": "user", "content": "q"}]).text == "hi"
    assert "fallbacks" not in server.requests[0]["body"]


def test_openai_compatible_client_over_the_wire(mock_server):
    def completion(message, finish):
        return {
            "id": "c", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
        }

    server = mock_server(
        [
            completion(
                {
                    "role": "assistant", "content": None,
                    "tool_calls": [
                        {"id": "call_1", "type": "function",
                         "function": {"name": "lookup", "arguments": "{}"}}
                    ],
                },
                "tool_calls",
            ),
            completion({"role": "assistant", "content": "It is 42."}, "stop"),
        ]
    )
    client = OpenAICompatibleClient(api_key="k", model="m", base_url=server.url + "/v1")
    result = run_tool_loop(client, "sys", "q", lookup_tools(), required_tools={"lookup"})

    assert result.answer == "It is 42."
    assert server.requests[0]["path"] == "/v1/chat/completions"
    assert server.requests[0]["headers"]["authorization"] == "Bearer k"
    messages = server.requests[1]["body"]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
    assert messages[3]["tool_call_id"] == "call_1"


def test_ollama_client_over_the_wire(mock_server):
    server = mock_server(
        [
            {"message": {"role": "assistant", "content": "",
                         "tool_calls": [{"function": {"name": "lookup", "arguments": {}}}]}},
            {"message": {"role": "assistant", "content": "It is 42."}},
        ]
    )
    client = OllamaClient(api_key=None, model="qwen2.5", url=server.url + "/api/chat")
    result = run_tool_loop(client, "sys", "q", lookup_tools(), required_tools={"lookup"})

    assert result.answer == "It is 42."
    first, second = server.requests
    assert "authorization" not in first["headers"]
    assert first["body"]["options"] == {"temperature": 0, "num_predict": 500}
    assert second["body"]["messages"][-1] == {
        "role": "tool", "tool_name": "lookup", "content": '{"value": 42}'
    }
