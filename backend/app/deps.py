"""FastAPI dependencies shared by the module routers."""

from __future__ import annotations

from fastapi import HTTPException

from backend.app.config import get_settings
from backend.app.core.llm import LLMClient, LLMError, create_llm_client


def get_llm_client() -> LLMClient:
    """The configured AI client. Tests override this with a fake."""
    try:
        return create_llm_client(get_settings().llm)
    except LLMError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


def http_error(exc: LLMError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.detail)
