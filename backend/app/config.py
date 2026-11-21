"""Application settings, read from environment variables (and .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://nextora-a5yj6qcjy-adebowale-okikis-projects.vercel.app",
    "https://nextora-zeta.vercel.app",
)

# --------------------------------------------------------------------------
# AI provider settings
# --------------------------------------------------------------------------

PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI = "openai"  # any OpenAI-compatible API
PROVIDER_OLLAMA = "ollama"  # native Ollama /api/chat (local or Ollama Cloud)
PROVIDER_NONE = "none"
SUPPORTED_PROVIDERS = (PROVIDER_ANTHROPIC, PROVIDER_OPENAI, PROVIDER_OLLAMA, PROVIDER_NONE)

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OLLAMA_URL = "https://ollama.com/api/chat"
DEFAULT_OLLAMA_MODEL = "gpt-oss:20b"


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    api_key: str | None = None
    model: str | None = None
    base_url: str | None = None

    @property
    def is_local(self) -> bool:
        host = (urlparse(self.base_url or "").hostname or "").lower()
        return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}

    @property
    def configured(self) -> bool:
        if self.provider == PROVIDER_NONE or not self.model:
            return False
        # Local servers (Ollama, LM Studio, vLLM...) do not need a key.
        return bool(self.api_key) or self.is_local

    @property
    def runtime_label(self) -> str:
        if self.provider == PROVIDER_ANTHROPIC:
            return "Anthropic Claude API"
        if self.provider == PROVIDER_OLLAMA:
            return "Ollama Cloud" if "ollama.com" in (self.base_url or "") else "Ollama"
        if self.provider == PROVIDER_OPENAI:
            host = urlparse(self.base_url or "").hostname or "unknown host"
            return f"OpenAI-compatible API ({host})"
        return "Not configured"


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def resolve_llm_settings(env: Mapping[str, str]) -> LLMSettings:
    """Work out which AI provider to use from environment variables.

    Explicit: LLM_PROVIDER + LLM_API_KEY (+ LLM_MODEL, LLM_BASE_URL).
    Otherwise the first provider-specific key found wins:
    ANTHROPIC_API_KEY, OPENAI_API_KEY, OLLAMA_API_KEY.
    """
    get = lambda name: _clean(env.get(name))  # noqa: E731

    provider = (get("LLM_PROVIDER") or "").lower() or None
    generic_key = get("LLM_API_KEY")

    if provider is None:
        if generic_key:
            provider = (
                PROVIDER_ANTHROPIC if generic_key.startswith("sk-ant-") else PROVIDER_OPENAI
            )
        elif get("ANTHROPIC_API_KEY"):
            provider = PROVIDER_ANTHROPIC
        elif get("OPENAI_API_KEY"):
            provider = PROVIDER_OPENAI
        elif get("OLLAMA_API_KEY"):
            provider = PROVIDER_OLLAMA
        else:
            provider = PROVIDER_NONE

    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{provider}'. "
            f"Use one of: {', '.join(SUPPORTED_PROVIDERS)}."
        )

    if provider == PROVIDER_ANTHROPIC:
        return LLMSettings(
            provider=provider,
            api_key=generic_key or get("ANTHROPIC_API_KEY"),
            model=get("LLM_MODEL") or get("CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL,
            base_url=get("LLM_BASE_URL") or get("ANTHROPIC_BASE_URL"),
        )

    if provider == PROVIDER_OPENAI:
        return LLMSettings(
            provider=provider,
            api_key=generic_key or get("OPENAI_API_KEY"),
            model=get("LLM_MODEL") or get("OPENAI_MODEL"),
            base_url=(
                get("LLM_BASE_URL") or get("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL
            ).rstrip("/"),
        )

    if provider == PROVIDER_OLLAMA:
        return LLMSettings(
            provider=provider,
            api_key=generic_key or get("OLLAMA_API_KEY"),
            model=get("LLM_MODEL") or get("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL,
            base_url=(get("LLM_BASE_URL") or get("OLLAMA_URL") or DEFAULT_OLLAMA_URL).rstrip("/"),
        )

    return LLMSettings(provider=PROVIDER_NONE)


# --------------------------------------------------------------------------
# Database settings
# --------------------------------------------------------------------------

def normalize_database_url(url: str) -> str:
    """Accept the postgres:// URLs that hosts such as Render hand out."""
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


def default_database_url() -> str:
    return f"sqlite:///{(DATA_DIR / 'nextora.db').as_posix()}"


@dataclass(frozen=True)
class Settings:
    database_url: str
    cors_origins: tuple[str, ...]
    lab_model_path: Path
    llm: LLMSettings = field(default_factory=lambda: LLMSettings(provider=PROVIDER_NONE))


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env

    cors = _clean(env.get("CORS_ORIGINS"))
    cors_origins = (
        tuple(origin.strip() for origin in cors.split(",") if origin.strip())
        if cors
        else DEFAULT_CORS_ORIGINS
    )

    model_path = _clean(env.get("LAB_MODEL_PATH"))

    return Settings(
        database_url=normalize_database_url(
            _clean(env.get("DATABASE_URL")) or default_database_url()
        ),
        cors_origins=cors_origins,
        lab_model_path=Path(model_path) if model_path else PROJECT_ROOT / "model.pth",
        llm=resolve_llm_settings(env),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
