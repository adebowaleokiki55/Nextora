"""Health and diagnostics endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.config import get_settings
from backend.app.core.lab.concurrency import INFERENCE_SLOT
from backend.app.core.lab.inference import DEVICE, is_model_loaded
from backend.app.core.system_info import read_process_memory
from backend.app.db import check_database

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
def health():
    llm = get_settings().llm
    return {
        "status": "ok",
        "model_loaded": is_model_loaded(),
        "device": str(DEVICE),
        "agent_provider": llm.provider,
        "agent_model": llm.model,
        "agent_runtime": llm.runtime_label,
        "agent_configured": llm.configured,
        "database_ok": check_database(),
        "clinical_validation": False,
    }


@router.get("/diagnostics/memory")
def memory_diagnostics():
    return {
        "status": "ok",
        "memory": read_process_memory(),
        "inference_busy": INFERENCE_SLOT.busy,
        "device": str(DEVICE),
    }
