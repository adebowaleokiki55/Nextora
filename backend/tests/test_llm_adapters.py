import json
from types import SimpleNamespace as NS

import pytest

from backend.app.core.llm import LLMError, ToolCall, ToolSpec
from backend.app.core.llm.anthropic_provider import (
    parse_anthropic_response,
    to_anthropic_messages,
    to_anthropic_tools,
)
from backend.app.core.llm.ollama_provider import parse_ollama_message, to_ollama_messages
from backend.app.core.llm.openai_provider import (
    parse_openai_response,
    to_openai_messages,
    to_openai_tools,
)

CALLS = [ToolCall("t1", "a", {"x": 1}), ToolCall("t2", "b", {})]
CONVERSATION = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "checking", "tool_calls": CALLS},
    {"role": "tool", "tool_call_id": "t1", "name": "a", "content": '{"ok": 1}'},
    {"role": "tool", "tool_call_id": "t2", "name": "b", "content": '{"ok": 2}'},
]


def test_anthropic_merges_parallel_tool_results():
    out = to_anthropic_messages(CONVERSATION)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert [b["type"] for b in out[1]["content"]] == ["text", "tool_use", "tool_use"]
    assert [b["tool_use_id"] for b in out[2]["content"]] == ["t1", "t2"]


def test_anthropic_replays_native_content_unchanged():
    raw = [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "tool_use", "id": "t1", "name": "a", "input": {}},
    ]
    out = to_anthropic_messages(
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [], "raw": raw, "provider": "anthropic"},
        ]
    )
    assert out[1]["content"] is raw


def test_raw_from_another_provider_is_not_replayed():
    out = to_anthropic_messages(
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "tool_calls": [], "raw": {"x": 1}, "provider": "ollama"},
        ]
    )
    assert out[1]["content"] == [{"type": "text", "text": "a"}]


def test_anthropic_tool_schema():
    spec = ToolSpec("a", "desc")
    assert to_anthropic_tools([spec]) == [
        {"name": "a", "description": "desc", "input_schema": spec.parameters}
    ]


def test_anthropic_parse_and_refusal():
    resp = NS(
        stop_reason="tool_use",
        model="claude-opus-5-5",
        content=[
            NS(type="thinking", thinking=""),
            NS(type="text", text="Let me look."),
            NS(type="tool_use", id="t9", name="a", input={"k": "v"}),
        ],
    )
    parsed = parse_anthropic_response(resp, "claude-opus-5-5")
    assert parsed.text == "Let me look."
    assert parsed.tool_calls == [ToolCall("t9", "a", {"k": "v"})]
    assert parsed.raw_message is resp.content

    with pytest.raises(LLMError):
        parse_anthropic_response(NS(stop_reason="refusal", content=[], model="m"), "m")


def test_openai_round_trip():
    out = to_openai_messages("sys", CONVERSATION)
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[2]["tool_calls"][0]["function"] == {"name": "a", "arguments": json.dumps({"x": 1})}
    assert out[3] == {"role": "tool", "tool_call_id": "t1", "content": '{"ok": 1}'}
    assert to_openai_tools([ToolSpec("a", "d")])[0]["type"] == "function"

    resp = NS(
        model="m",
        choices=[
            NS(
                finish_reason="tool_calls",
                message=NS(
                    content=None,
                    tool_calls=[
                        NS(id="c1", function=NS(name="a", arguments='{"x": 2}')),
                        NS(id=None, function=NS(name="b", arguments="not json")),
                    ],
                ),
            )
        ],
    )
    parsed = parse_openai_response(resp, "m")
    assert parsed.tool_calls == [ToolCall("c1", "a", {"x": 2}), ToolCall("call_1", "b", {})]
    assert parsed.raw_message["tool_calls"][0]["id"] == "c1"


def test_ollama_format_matches_original_client():
    out = to_ollama_messages("sys", CONVERSATION)
    assert out[-1] == {"role": "tool", "tool_name": "b", "content": '{"ok": 2}'}

    native = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "a", "arguments": {}}}],
    }
    parsed = parse_ollama_message(native, "gpt-oss:20b")
    assert parsed.tool_calls[0].name == "a"
    assert parsed.raw_message is native
