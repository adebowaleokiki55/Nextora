"""Native Ollama /api/chat (local Ollama or Ollama Cloud).

This keeps the behaviour of the original Nextora agent client: temperature 0,
500 predicted tokens, Bearer auth for Ollama Cloud.
"""

from __future__ import annotations

import json
from typing import Any

import requests

from backend.app.core.llm.base import LLMError, LLMResponse, ToolCall, ToolSpec, native_raw


def to_ollama_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = [{"role": "system", "content": system}]

    for message in messages:
        role = message["role"]
        if role == "user":
            converted.append({"role": "user", "content": message["content"]})
        elif role == "assistant":
            raw = native_raw(message, "ollama")
            if raw is not None:
                converted.append(raw)
                continue
            entry: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
            calls = message.get("tool_calls") or []
            if calls:
                entry["tool_calls"] = [
                    {"function": {"name": c.name, "arguments": c.arguments}} for c in calls
                ]
            converted.append(entry)
        elif role == "tool":
            converted.append(
                {"role": "tool", "tool_name": message["name"], "content": message["content"]}
            )
        else:
            raise ValueError(f"Unknown message role: {role}")

    return converted


def to_ollama_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in tools
    ]


def parse_ollama_message(message: dict[str, Any], model: str) -> LLMResponse:
    calls: list[ToolCall] = []
    for index, call in enumerate(message.get("tool_calls") or []):
        function = call.get("function", {}) if isinstance(call, dict) else {}
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {}
        calls.append(
            ToolCall(
                id=call.get("id") or f"call_{index}",
                name=function.get("name") or "",
                arguments=arguments if isinstance(arguments, dict) else {},
            )
        )

    return LLMResponse(
        text=(message.get("content") or "").strip(),
        tool_calls=calls,
        stop_reason=None,
        provider="ollama",
        model=model,
        raw_message=message,
    )


class OllamaClient:
    provider = "ollama"

    def __init__(self, api_key: str | None, model: str, url: str):
        self.api_key = api_key
        self.model = model
        self.url = url
        self.runtime_label = "Ollama Cloud" if "ollama.com" in url else "Ollama"

    def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None = None,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": to_ollama_messages(system, messages),
            "tools": to_ollama_tools(tools or []),
            "stream": False,
            "options": {"temperature": 0, "num_predict": 500},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        label = self.runtime_label
        try:
            response = requests.post(self.url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()

        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            print(
                f"{label} HTTP error:",
                status,
                exc.response.text[:1000] if exc.response is not None else str(exc),
            )
            if status in (401, 403):
                raise LLMError(502, f"{label} authentication failed. Check the API key and model access.") from exc
            if status == 429:
                raise LLMError(503, f"{label} rate limit or usage limit reached.") from exc
            if status == 404:
                raise LLMError(
                    502, f"{label} could not find the requested model or API endpoint. Verify the URL and model."
                ) from exc
            raise LLMError(502, f"{label} returned an HTTP error. Check backend logs.") from exc

        except requests.Timeout as exc:
            raise LLMError(504, f"{label} timed out.") from exc

        except requests.ConnectionError as exc:
            raise LLMError(503, f"Cannot connect to {label}. Check the API URL and network.") from exc

        except requests.RequestException as exc:
            print(f"{label} request error: {type(exc).__name__}: {exc}")
            raise LLMError(502, f"{label} request failed.") from exc

        except ValueError as exc:
            raise LLMError(502, f"{label} returned invalid JSON.") from exc

        message = data.get("message")
        if not isinstance(message, dict):
            raise LLMError(502, f"{label} returned an invalid chat response.")

        return parse_ollama_message(message, self.model)
