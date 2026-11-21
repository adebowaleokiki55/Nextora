"""Any OpenAI-compatible chat-completions API.

Works with OpenAI, Groq, OpenRouter, Together, Mistral, DeepSeek, Google
Gemini's OpenAI endpoint, Ollama's /v1 endpoint, LM Studio, vLLM, and others:
set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

from backend.app.core.llm.base import LLMError, LLMResponse, ToolCall, ToolSpec, native_raw


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def to_openai_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = [{"role": "system", "content": system}]

    for message in messages:
        role = message["role"]

        if role == "user":
            converted.append({"role": "user", "content": message["content"]})

        elif role == "assistant":
            raw = native_raw(message, "openai")
            if raw is not None:
                converted.append(raw)
                continue
            entry: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
            calls = message.get("tool_calls") or []
            if calls:
                entry["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                    }
                    for call in calls
                ]
            converted.append(entry)

        elif role == "tool":
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": message["tool_call_id"],
                    "content": message["content"],
                }
            )

        else:
            raise ValueError(f"Unknown message role: {role}")

    return converted


def to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in tools
    ]


def parse_openai_response(response: Any, model: str) -> LLMResponse:
    if not getattr(response, "choices", None):
        raise LLMError(502, "The AI provider returned no choices.")

    choice = response.choices[0]
    message = choice.message
    calls = [
        ToolCall(
            id=call.id or f"call_{index}",
            name=call.function.name,
            arguments=_parse_arguments(call.function.arguments),
        )
        for index, call in enumerate(message.tool_calls or [])
    ]

    # Replayable assistant message in wire format.
    raw: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
    if calls:
        raw["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in calls
        ]

    return LLMResponse(
        text=(message.content or "").strip(),
        tool_calls=calls,
        stop_reason=choice.finish_reason,
        provider="openai",
        model=getattr(response, "model", None) or model,
        raw_message=raw,
    )


class OpenAICompatibleClient:
    provider = "openai"

    def __init__(self, api_key: str | None, model: str, base_url: str):
        import openai

        self._openai = openai
        self.model = model
        self.base_url = base_url
        host = urlparse(base_url).hostname or "unknown host"
        self.runtime_label = f"OpenAI-compatible API ({host})"
        # Local servers ignore the key, but the SDK requires a value.
        self._client = openai.OpenAI(
            api_key=api_key or "not-needed", base_url=base_url, timeout=180.0
        )

    def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        openai = self._openai
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": to_openai_messages(system, messages),
        }
        if tools:
            kwargs["tools"] = to_openai_tools(tools)

        try:
            response = self._client.chat.completions.create(**kwargs)
        except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
            raise LLMError(502, "AI provider authentication failed. Check the API key.") from exc
        except openai.NotFoundError as exc:
            raise LLMError(
                502, f"The AI provider could not find the model '{self.model}' or the API URL."
            ) from exc
        except openai.RateLimitError as exc:
            raise LLMError(503, "AI provider rate or usage limit reached.") from exc
        except openai.APITimeoutError as exc:
            raise LLMError(504, "The AI provider timed out.") from exc
        except openai.APIConnectionError as exc:
            raise LLMError(503, "Cannot connect to the AI provider. Check LLM_BASE_URL.") from exc
        except openai.APIStatusError as exc:
            print(f"OpenAI-compatible API error {exc.status_code}: {exc.message}")
            raise LLMError(502, "The AI provider returned an error. Check backend logs.") from exc

        return parse_openai_response(response, self.model)
