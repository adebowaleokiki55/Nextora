"""Treatment decision support (side page; outside the five core modules).

Preliminary protocol-aware decision support. Not a prescription or a
clinically validated system.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.app.agents.treatment_explainer import build_treatment_report
from backend.app.core.llm import LLMClient, LLMError
from backend.app.core.treatment_engine import TreatmentRequest, assess_treatment
from backend.app.deps import get_llm_client, http_error

router = APIRouter(prefix="/api", tags=["treatment"])


def _assess(request: TreatmentRequest) -> dict:
    try:
        return assess_treatment(request)
    except Exception as exc:
        print(f"Treatment assessment error: {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail="Treatment assessment failed.") from exc


@router.post("/treatment/recommend")
def recommend_treatment(request: TreatmentRequest):
    return _assess(request)


@router.post("/agent/treatment-report")
def treatment_agent_report(
    request: TreatmentRequest,
    client: LLMClient = Depends(get_llm_client),
):
    assessment = _assess(request)

    try:
        return build_treatment_report(client, assessment, request.model_dump())
    except LLMError as exc:
        raise http_error(exc) from exc
    except Exception as exc:
        print(f"Treatment agent error: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500, detail="The treatment agent failed. Check backend logs."
        ) from exc
