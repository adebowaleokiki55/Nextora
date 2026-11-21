"""Module 5: AI laboratory intelligence (blood-smear CNN).

/api/predict, /api/analyze-smear and /api/agent/explain are the original
stateless Nextora endpoints. /api/lab/... stores results for review:
every submission starts PENDING_REVIEW and only a qualified reviewer's
VERIFIED decision lets it feed surveillance.
"""

from __future__ import annotations

import base64
import gc
import json
from datetime import date
from typing import Any, Literal

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.agents.lab_explainer import explain_analysis
from backend.app.config import PROJECT_ROOT, get_settings
from backend.app.core.lab.concurrency import INFERENCE_SLOT
from backend.app.core.lab.inference import (
    ALLOWED_CONTENT_TYPES,
    DEVICE,
    MAX_UPLOAD_BYTES,
    MODEL_NAME,
    ImageInputError,
    classify_cell_image,
    get_model,
    get_model_version,
    validate_image_dimensions,
)
from backend.app.core.lab.records import (
    LabRecordError,
    create_submission,
    list_submissions,
    review_submission,
    submission_detail,
    submission_summary,
)
from backend.app.core.lab.smear_analysis import analyze_smear
from backend.app.core.llm import LLMClient, LLMError
from backend.app.db import get_db
from backend.app.deps import get_llm_client, http_error
from backend.app.models import DataOrigin, LabSubmission, ReviewerResult, VerificationStatus

router = APIRouter(prefix="/api", tags=["lab"])


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _model():
    return get_model(get_settings().lab_model_path)


def _acquire_inference_slot() -> None:
    if not INFERENCE_SLOT.try_acquire():
        raise HTTPException(
            status_code=503,
            detail=(
                "The image-analysis service is busy. "
                "Please wait a few seconds and try again."
            ),
            headers={"Retry-After": "5"},
        )


def _read_upload(file: UploadFile, empty_detail: str, too_large_detail: str) -> bytes:
    """Read at most the configured upload limit plus one byte."""
    image_bytes = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail=empty_detail)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=too_large_detail)
    return image_bytes


def _validate(image_bytes: bytes) -> None:
    try:
        validate_image_dimensions(image_bytes)
    except ImageInputError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


# --------------------------------------------------------------------------
# Single-cell prediction
# --------------------------------------------------------------------------

@router.post("/predict")
def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image.")

    image_bytes = _read_upload(
        file,
        empty_detail="The uploaded file is empty.",
        too_large_detail="Image exceeds the 10 MB upload limit.",
    )
    _validate(image_bytes)
    model = _model()
    _acquire_inference_slot()

    try:
        return classify_cell_image(image_bytes, model)

    except ImageInputError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    except Exception as exc:
        print(f"Prediction error: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500, detail="Prediction failed. Check the backend logs."
        ) from exc

    finally:
        INFERENCE_SLOT.release()


# --------------------------------------------------------------------------
# Whole-smear analysis
# --------------------------------------------------------------------------

def _run_smear_analysis(file: UploadFile) -> tuple[dict, bytes | None]:
    """Validate the upload and run whole-smear analysis.

    Returns the analysis dict and the annotated PNG bytes (if any).
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415, detail="Upload a JPEG, PNG, or WebP microscopy image."
        )

    image_bytes = _read_upload(
        file,
        empty_detail="The uploaded image is empty.",
        too_large_detail="Image must be 10 MB or smaller.",
    )
    _validate(image_bytes)
    model = _model()
    _acquire_inference_slot()

    try:
        result = analyze_smear(image_bytes=image_bytes, model=model, device=DEVICE)
        annotated_image = result.pop("annotated_image", None)

        if annotated_image is not None and not isinstance(annotated_image, bytes):
            raise RuntimeError("analyze_smear returned an unexpected annotated image type.")

        return result, annotated_image

    except (ValueError, cv2.error) as exc:
        raise HTTPException(status_code=400, detail=f"Could not analyze this image: {exc}") from exc

    except Exception as exc:
        print(f"Smear analysis error: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500, detail="Smear analysis failed. Check the backend logs."
        ) from exc

    finally:
        image_bytes = b""
        gc.collect()
        INFERENCE_SLOT.release()


def _with_image(result: dict, annotated_image: bytes | None) -> dict:
    if annotated_image is not None:
        result["annotated_image_base64"] = base64.b64encode(annotated_image).decode("ascii")
        result["annotated_image_mime_type"] = "image/png"
    return result


@router.post("/analyze-smear")
def analyze_smear_endpoint(file: UploadFile = File(...)):
    """Stateless analysis: nothing is stored."""
    # A plain (sync) endpoint: FastAPI runs it in a worker thread, so CPU-heavy
    # inference does not block the event loop.
    result, annotated_image = _run_smear_analysis(file)
    return _with_image(result, annotated_image)


# --------------------------------------------------------------------------
# Analysis explanation agent
# --------------------------------------------------------------------------

class AgentExplainRequest(BaseModel):
    mode: Literal["single_cell", "whole_smear"]
    analysis: dict[str, Any]


@router.post("/agent/explain")
def explain(request: AgentExplainRequest, client: LLMClient = Depends(get_llm_client)):
    try:
        return explain_analysis(client, request.mode, request.analysis)
    except LLMError as exc:
        raise http_error(exc) from exc
    except Exception as exc:
        print(f"Analysis agent error: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500, detail="The analysis agent failed. Check backend logs."
        ) from exc


# --------------------------------------------------------------------------
# Lab submissions: stored results awaiting qualified review
# --------------------------------------------------------------------------

def _record_error(exc: LabRecordError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.detail)


def _get_submission(db: Session, submission_id: int) -> LabSubmission:
    submission = db.get(LabSubmission, submission_id)
    if submission is None:
        raise HTTPException(status_code=404, detail=f"Lab submission {submission_id} was not found.")
    return submission


@router.post("/lab/submissions", status_code=201)
def submit_smear(
    file: UploadFile = File(...),
    facility_id: int = Form(...),
    sample_date: date = Form(...),
    submitted_by: str = Form(...),
    sample_reference: str | None = Form(None),
    # Default SYNTHETIC: an upload only counts as a real facility sample when
    # the submitter says so explicitly.
    data_origin: DataOrigin = Form(DataOrigin.SYNTHETIC),
    db: Session = Depends(get_db),
):
    """Analyse a whole-smear image and store it as PENDING_REVIEW."""
    result, annotated_image = _run_smear_analysis(file)

    try:
        submission = create_submission(
            db,
            facility_id=facility_id,
            sample_date=sample_date,
            submitted_by=submitted_by,
            sample_reference=sample_reference,
            analysis=result,
            annotated_png=annotated_image,
            model_name=MODEL_NAME,
            model_version=get_model_version(get_settings().lab_model_path),
            data_origin=data_origin,
            image_filename=file.filename,
        )
    except LabRecordError as exc:
        raise _record_error(exc) from exc

    body = submission_detail(submission)
    body["disclaimer"] = result.get("disclaimer")
    return _with_image(body, annotated_image)


@router.get("/lab/submissions")
def get_submissions(
    status: VerificationStatus | None = Query(None),
    location_id: int | None = Query(None),
    facility_id: int | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    submissions = list_submissions(db, status, location_id, facility_id, limit)
    return {"submissions": [submission_summary(s) for s in submissions]}


@router.get("/lab/submissions/{submission_id}")
def get_submission(submission_id: int, db: Session = Depends(get_db)):
    return submission_detail(_get_submission(db, submission_id))


@router.get("/lab/submissions/{submission_id}/annotated.png")
def get_annotated_image(submission_id: int, db: Session = Depends(get_db)):
    png = _get_submission(db, submission_id).annotated_png
    if not png:
        raise HTTPException(status_code=404, detail="No annotated image is stored for this submission.")
    return Response(content=png, media_type="image/png")


class ReviewRequest(BaseModel):
    decision: Literal["VERIFIED", "REJECTED"]
    reviewer_name: str
    reviewer_result: Literal["POSITIVE", "NEGATIVE"] | None = None
    notes: str | None = None


@router.post("/lab/submissions/{submission_id}/review")
def review(submission_id: int, request: ReviewRequest, db: Session = Depends(get_db)):
    try:
        submission = review_submission(
            db,
            submission_id,
            decision=VerificationStatus(request.decision),
            reviewer_name=request.reviewer_name,
            reviewer_result=(
                ReviewerResult(request.reviewer_result) if request.reviewer_result else None
            ),
            notes=request.notes,
        )
    except LabRecordError as exc:
        raise _record_error(exc) from exc
    return submission_detail(submission)


@router.get("/lab/model-card")
def model_card():
    card = json.loads((PROJECT_ROOT / "ml" / "lab_model_card.json").read_text(encoding="utf-8"))
    card.pop("_how_to_add_metrics", None)
    card["model_version"] = get_model_version(get_settings().lab_model_path)
    return card
