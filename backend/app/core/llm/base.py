"""Provider-neutral types for chat + tool calling.

Conversation messages are plain dicts in one neutral shape:

    {"role": "user", "content": str}
    {"role": "assistant", "content": str, "tool_calls": [ToolCall, ...],
     "raw": <provider-native message, replayed unchanged>, "provider": str}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str}

Each provider adapter converts these to and from its own wire format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMError(Exception):
    """An AI-provider failure, carrying an HTTP-style status for the API layer."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []}
    )


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str | None
    provider: str
    model: str
    raw_message: Any = None

    def as_message(self) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": self.text,
            "tool_calls": list(self.tool_calls),
            "raw": self.raw_message,
            "provider": self.provider,
        }


class LLMClient(Protocol):
    provider: str
    model: str
    runtime_label: str

    def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse: ...


def native_raw(message: dict[str, Any], provider: str) -> Any:
    """The provider-native copy of an assistant message, if it came from `provider`."""
    if message.get("provider") == provider:
        return message.get("raw")
    return None
