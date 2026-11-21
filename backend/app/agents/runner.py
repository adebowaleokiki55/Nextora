"""A provider-agnostic tool-calling loop.

The model may only call tools in the approved list; handlers are plain
Python functions, so all numbers come from code, not the model.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from backend.app.core.llm import LLMClient, LLMError, ToolSpec

# Called with (event_kind, payload) for each step; used later for the live feed.
EventHook = Callable[[str, dict[str, Any]], None]


@dataclass
class AgentTool:
    spec: ToolSpec
    handler: Callable[[dict[str, Any]], Any]


@dataclass
class AgentResult:
    answer: str
    tools_executed: list[str] = field(default_factory=list)
    tool_results: dict[str, Any] = field(default_factory=dict)
    rounds: int = 0


def run_tool_loop(
    client: LLMClient,
    system: str,
    user_message: str,
    tools: list[AgentTool],
    required_tools: set[str] | None = None,
    max_rounds: int = 6,
    reminder: Callable[[set[str]], str] | None = None,
    exhausted_detail: str = "The agent could not finish within the allowed rounds.",
    on_event: EventHook | None = None,
    max_tokens: int | None = None,
) -> AgentResult:
    """Run the model until it answers, enforcing required and approved tools.

    A repeated call with identical arguments gets the cached result; a call
    with different arguments (e.g. a corrected draft) runs again.
    """
    approved = {tool.spec.name: tool for tool in tools}
    required = set(required_tools or set())
    specs = [tool.spec for tool in tools]
    emit = on_event or (lambda kind, payload: None)

    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
    result = AgentResult(answer="")
    cache: dict[tuple[str, str], Any] = {}

    for round_number in range(1, max_rounds + 1):
        result.rounds = round_number
        response = client.chat(system, messages, specs, max_tokens=max_tokens) if max_tokens else client.chat(system, messages, specs)

        if not response.tool_calls:
            if not response.text:
                raise LLMError(502, "The AI provider returned an empty response.")

            missing = required - set(result.tools_executed)
            if missing:
                messages.append(response.as_message())
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            reminder(missing)
                            if reminder
                            else "Before answering, call the required tools that have not "
                            "yet executed: " + ", ".join(sorted(missing))
                        ),
                    }
                )
                continue

            result.answer = response.text
            emit("final", {"text": response.text})
            return result

        if response.text:
            emit("thought", {"text": response.text})

        messages.append(response.as_message())

        for call in response.tool_calls:
            tool = approved.get(call.name)
            if tool is None:
                emit("error", {"text": f"Blocked unapproved tool: {call.name}"})
                raise LLMError(502, "The agent requested an unapproved tool.")

            emit("tool_call", {"tool": call.name, "arguments": call.arguments})

            key = (call.name, json.dumps(call.arguments, sort_keys=True, default=str))
            if key not in cache:
                cache[key] = tool.handler(call.arguments)
                if call.name not in result.tools_executed:
                    result.tools_executed.append(call.name)
            output = cache[key]
            result.tool_results[call.name] = output
            emit("tool_result", {"tool": call.name, "result": output})

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": json.dumps(output, ensure_ascii=False),
                }
            )

    raise LLMError(502, exhausted_detail)
