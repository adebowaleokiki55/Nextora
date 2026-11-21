import pytest

from backend.app.config import (
    DEFAULT_CLAUDE_MODEL,
    load_settings,
    normalize_database_url,
    resolve_llm_settings,
)


def test_no_keys_means_not_configured():
    s = resolve_llm_settings({})
    assert s.provider == "none"
    assert not s.configured


def test_anthropic_key_autodetected_with_claude_model_env():
    s = resolve_llm_settings({"ANTHROPIC_API_KEY": "k", "CLAUDE_MODEL": "claude-sonnet-5-5"})
    assert (s.provider, s.model, s.configured) == ("anthropic", "claude-sonnet-5-5", True)


def test_anthropic_default_model():
    assert resolve_llm_settings({"ANTHROPIC_API_KEY": "k"}).model == DEFAULT_CLAUDE_MODEL


def test_generic_key_prefix_picks_provider():
    assert resolve_llm_settings({"LLM_API_KEY": "sk-ant-xyz"}).provider == "anthropic"
    assert resolve_llm_settings({"LLM_API_KEY": "gsk_abc"}).provider == "openai"


def test_openai_compatible_needs_model():
    s = resolve_llm_settings({"LLM_PROVIDER": "openai", "LLM_API_KEY": "k"})
    assert not s.configured  # no model given

    s = resolve_llm_settings(
        {
            "LLM_PROVIDER": "openai",
            "LLM_API_KEY": "k",
            "LLM_MODEL": "llama-3.3-70b-versatile",
            "LLM_BASE_URL": "https://api.groq.com/openai/v1/",
        }
    )
    assert s.configured
    assert s.base_url == "https://api.groq.com/openai/v1"
    assert s.runtime_label == "OpenAI-compatible API (api.groq.com)"


def test_existing_ollama_env_still_works():
    s = resolve_llm_settings({"OLLAMA_API_KEY": "k"})
    assert (s.provider, s.model, s.runtime_label) == ("ollama", "gpt-oss:20b", "Ollama Cloud")


def test_local_server_needs_no_key():
    s = resolve_llm_settings(
        {
            "LLM_PROVIDER": "ollama",
            "LLM_BASE_URL": "http://localhost:11434/api/chat",
            "LLM_MODEL": "qwen2.5",
        }
    )
    assert s.configured
    assert s.runtime_label == "Ollama"


def test_explicit_provider_overrides_other_keys():
    s = resolve_llm_settings(
        {"LLM_PROVIDER": "anthropic", "OPENAI_API_KEY": "o", "ANTHROPIC_API_KEY": "a"}
    )
    assert (s.provider, s.api_key) == ("anthropic", "a")


def test_unknown_provider_rejected():
    with pytest.raises(ValueError):
        resolve_llm_settings({"LLM_PROVIDER": "skynet"})


def test_database_url_normalisation():
    assert normalize_database_url("postgres://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert normalize_database_url("sqlite:///x.db") == "sqlite:///x.db"


def test_cors_override():
    s = load_settings({"CORS_ORIGINS": "https://a.example, https://b.example"})
    assert s.cors_origins == ("https://a.example", "https://b.example")
