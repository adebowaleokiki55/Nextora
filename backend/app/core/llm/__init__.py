"""Provider-agnostic LLM access: Anthropic, any OpenAI-compatible API, or Ollama."""

from __future__ import annotations

from backend.app.config import (
    PROVIDER_ANTHROPIC,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
    LLMSettings,
)
from backend.app.core.llm.base import LLMClient, LLMError, LLMResponse, ToolCall, ToolSpec

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMResponse",
    "ToolCall",
    "ToolSpec",
    "create_llm_client",
]


def create_llm_client(settings: LLMSettings) -> LLMClient:
    """Build a client for the configured provider, or raise LLMError(503)."""
    if not settings.configured:
        raise LLMError(
            503,
            "No AI provider is configured. Set LLM_PROVIDER (anthropic, openai or ollama), "
            "LLM_API_KEY and LLM_MODEL in the backend environment.",
        )

    if settings.provider == PROVIDER_ANTHROPIC:
        from backend.app.core.llm.anthropic_provider import AnthropicClient

        return AnthropicClient(settings.api_key, settings.model, settings.base_url)

    if settings.provider == PROVIDER_OPENAI:
        from backend.app.core.llm.openai_provider import OpenAICompatibleClient

        return OpenAICompatibleClient(settings.api_key, settings.model, settings.base_url)

    if settings.provider == PROVIDER_OLLAMA:
        from backend.app.core.llm.ollama_provider import OllamaClient

        return OllamaClient(settings.api_key, settings.model, settings.base_url)

    raise LLMError(503, f"Unsupported AI provider: {settings.provider}")
