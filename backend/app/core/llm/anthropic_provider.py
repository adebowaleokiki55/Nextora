"""Claude via the official Anthropic Python SDK."""

from __future__ import annotations

from typing import Any

from backend.app.core.llm.base import LLMError, LLMResponse, ToolCall, ToolSpec, native_raw

# Models that accept the server-side refusal fallback (beta). On a policy
# decline the API re-runs the request on a fallback model in the same call.
FALLBACK_MODELS = {
    "claude-fable-5-2",
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
}
FALLBACK_BETA = "server-side-fallback-2026-07-01"

MAX_TOKENS = 16000


def to_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Neutral messages -> Messages API format.

    Consecutive tool results are merged into one user message, as the API
    expects all results for one assistant turn together.
    """
    converted: list[dict[str, Any]] = []

    for message in messages:
        role = message["role"]

        if role == "user":
            converted.append({"role": "user", "content": message["content"]})

        elif role == "assistant":
            raw = native_raw(message, "anthropic")
            if raw is not None:
                # Replay the exact content blocks (including thinking blocks).
                converted.append({"role": "assistant", "content": raw})
                continue

            blocks: list[dict[str, Any]] = []
            if message.get("content"):
                blocks.append({"type": "text", "text": message["content"]})
            for call in message.get("tool_calls") or []:
                blocks.append(
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                )
            converted.append({"role": "assistant", "content": blocks or ""})

        elif role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": message["tool_call_id"],
                "content": message["content"],
            }
            previous = converted[-1] if converted else None
            if (
                previous is not None
                and previous["role"] == "user"
                and isinstance(previous["content"], list)
                and all(b.get("type") == "tool_result" for b in previous["content"])
            ):
                previous["content"].append(block)
            else:
                converted.append({"role": "user", "content": [block]})

        else:
            raise ValueError(f"Unknown message role: {role}")

    return converted


def to_anthropic_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.parameters}
        for t in tools
    ]


def parse_anthropic_response(response: Any, model: str) -> LLMResponse:
    if getattr(response, "stop_reason", None) == "refusal":
        raise LLMError(502, "The AI model declined to answer this request.")

    texts: list[str] = []
    calls: list[ToolCall] = []

    for block in response.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            texts.append(block.text)
        elif block_type == "tool_use":
            calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {})))

    return LLMResponse(
        text="\n".join(texts).strip(),
        tool_calls=calls,
        stop_reason=response.stop_reason,
        provider="anthropic",
        model=getattr(response, "model", model) or model,
        raw_message=response.content,
    )


class AnthropicClient:
    provider = "anthropic"
    runtime_label = "Anthropic Claude API"

    def __init__(self, api_key: str | None, model: str, base_url: str | None = None):
        import anthropic

        self._anthropic = anthropic
        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key, base_url=base_url, timeout=180.0)

    def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        anthropic = self._anthropic
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max(max_tokens or MAX_TOKENS, MAX_TOKENS),
            "system": system,
            "messages": to_anthropic_messages(messages),
        }
        if tools:
            kwargs["tools"] = to_anthropic_tools(tools)

        try:
            if self.model in FALLBACK_MODELS:
                response = self._client.beta.messages.create(
                    betas=[FALLBACK_BETA], fallbacks="default", **kwargs
                )
            else:
                response = self._client.messages.create(**kwargs)

        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise LLMError(502, "Anthropic authentication failed. Check the API key.") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(502, f"Anthropic could not find the model '{self.model}'.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError(503, "Anthropic rate limit reached. Try again shortly.") from exc
        except anthropic.APITimeoutError as exc:
            raise LLMError(504, "The Anthropic API timed out.") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(503, "Cannot connect to the Anthropic API.") from exc
        except anthropic.APIStatusError as exc:
            print(f"Anthropic API error {exc.status_code}: {exc.message}")
            raise LLMError(502, "The Anthropic API returned an error. Check backend logs.") from exc

        return parse_anthropic_response(response, self.model)
