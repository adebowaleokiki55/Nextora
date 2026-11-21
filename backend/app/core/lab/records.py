"""Lab submissions: save, list and review. Pure Python + SQLAlchemy, no FastAPI.

Rules enforced here:
- every submission starts as PENDING_REVIEW;
- only a PENDING_REVIEW submission can be reviewed, exactly once;
- only VERIFIED submissions produce observations (and alerts) that
  surveillance and prediction may use;
- the observation records the reviewer's reading, not the model's.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models import (
    Alert,
    AlertSeverity,
    DataOrigin,
    Facility,
    LabCell,
    LabSubmission,
    Observation,
    ReviewerResult,
    VerificationStatus,
)

LAB_SOURCE = "Nextora lab module upload"

METRIC_SMEARS_REVIEWED = "lab_smears_reviewed"
METRIC_SMEARS_POSITIVE = "lab_smears_positive"


class LabRecordError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _clean_text(value: str | None, field: str, max_length: int, required: bool) -> str | None:
    value = (value or "").strip()
    if not value:
        if required:
            raise LabRecordError(422, f"{field} is required.")
        return None
    if len(value) > max_length:
        raise LabRecordError(422, f"{field} must be at most {max_length} characters.")
    return value


def create_submission(
    session: Session,
    *,
    facility_id: int,
    sample_date: date,
    submitted_by: str,
    analysis: dict[str, Any],
    annotated_png: bytes | None,
    model_name: str,
    model_version: str,
    data_origin: DataOrigin,
    sample_reference: str | None = None,
    image_filename: str | None = None,
    today: date | None = None,
) -> LabSubmission:
    """Store a whole-smear analysis as a PENDING_REVIEW submission."""
    facility = session.get(Facility, facility_id)
    if facility is None:
        raise LabRecordError(404, f"Facility {facility_id} was not found.")
    if not facility.has_laboratory:
        raise LabRecordError(422, f"{facility.name} is not registered as a laboratory.")

    if sample_date > (today or date.today()):
        raise LabRecordError(422, "The sample date cannot be in the future.")

    submitted_by = _clean_text(submitted_by, "Submitted by", 120, required=True)
    sample_reference = _clean_text(sample_reference, "Sample reference", 60, required=False)

    source = LAB_SOURCE
    if data_origin is DataOrigin.SYNTHETIC:
        source += " (demo/test image, not a patient sample from this facility)"

    submission = LabSubmission(
        facility_id=facility.id,
        location_id=facility.location_id,
        sample_date=sample_date,
        sample_reference=sample_reference,
        submitted_by=submitted_by,
        image_filename=(image_filename or "")[:255] or None,
        image_width=int(analysis["image_width"]),
        image_height=int(analysis["image_height"]),
        model_name=model_name,
        model_version=model_version,
        candidate_count=int(analysis["candidate_count"]),
        candidate_infected_count=int(analysis["candidate_infected_count"]),
        candidate_uninfected_count=int(analysis["candidate_uninfected_count"]),
        annotated_png=annotated_png,
        status=VerificationStatus.PENDING_REVIEW,
        data_origin=data_origin,
        source=source,
        cells=[
            LabCell(
                cell_index=int(cell["cell_id"]),
                x=int(cell["box"]["x"]),
                y=int(cell["box"]["y"]),
                width=int(cell["box"]["width"]),
                height=int(cell["box"]["height"]),
                predicted_class=int(cell["predicted_class"]),
                prediction=str(cell["prediction"]),
                confidence=float(cell["confidence"]),
            )
            for cell in analysis.get("cells", [])
        ],
    )
    session.add(submission)
    session.commit()
    return submission


def list_submissions(
    session: Session,
    status: VerificationStatus | None = None,
    location_id: int | None = None,
    facility_id: int | None = None,
    limit: int = 100,
) -> list[LabSubmission]:
    query = select(LabSubmission).order_by(LabSubmission.created_at.desc(), LabSubmission.id.desc())
    if status is not None:
        query = query.where(LabSubmission.status == status)
    if location_id is not None:
        query = query.where(LabSubmission.location_id == location_id)
    if facility_id is not None:
        query = query.where(LabSubmission.facility_id == facility_id)
    return list(session.scalars(query.limit(max(1, min(limit, 500)))))


def review_submission(
    session: Session,
    submission_id: int,
    *,
    decision: VerificationStatus,
    reviewer_name: str,
    reviewer_result: ReviewerResult | None = None,
    notes: str | None = None,
    now: datetime | None = None,
) -> LabSubmission:
    """Record a qualified reviewer's decision.

    VERIFIED needs the reviewer's own reading (POSITIVE or NEGATIVE) and
    creates observations; REJECTED (unusable or inconclusive image) needs a
    note and creates nothing.
    """
    submission = session.get(LabSubmission, submission_id)
    if submission is None:
        raise LabRecordError(404, f"Lab submission {submission_id} was not found.")
    if submission.status is not VerificationStatus.PENDING_REVIEW:
        raise LabRecordError(
            409, f"Lab submission {submission_id} was already reviewed ({submission.status.value})."
        )

    reviewer_name = _clean_text(reviewer_name, "Reviewer name", 120, required=True)
    notes = _clean_text(notes, "Review notes", 2000, required=False)

    if decision is VerificationStatus.VERIFIED:
        if reviewer_result is None:
            raise LabRecordError(
                422, "Verifying a result requires the reviewer's own reading: POSITIVE or NEGATIVE."
            )
    elif decision is VerificationStatus.REJECTED:
        if reviewer_result is not None:
            raise LabRecordError(422, "A rejected submission has no reviewer reading.")
        if not notes:
            raise LabRecordError(422, "Explain why the submission is rejected.")
    else:
        raise LabRecordError(422, "The decision must be VERIFIED or REJECTED.")

    reviewed_at = now or datetime.now(timezone.utc)
    submission.status = decision
    submission.reviewer_result = reviewer_result
    submission.reviewed_by = reviewer_name
    submission.reviewed_at = reviewed_at
    submission.review_notes = notes

    if decision is VerificationStatus.VERIFIED:
        _record_verified_observations(session, submission)
        if reviewer_result is ReviewerResult.POSITIVE:
            _raise_positive_alert(session, submission)

    session.commit()
    return submission


def _record_verified_observations(session: Session, submission: LabSubmission) -> None:
    source = f"lab_submission:{submission.id} reviewed by {submission.reviewed_by}"
    positive = 1.0 if submission.reviewer_result is ReviewerResult.POSITIVE else 0.0

    for metric, value in (
        (METRIC_SMEARS_REVIEWED, 1.0),
        (METRIC_SMEARS_POSITIVE, positive),
    ):
        session.add(
            Observation(
                location_id=submission.location_id,
                facility_id=submission.facility_id,
                metric=metric,
                value=value,
                unit="smears",
                period="day",
                period_start=submission.sample_date,
                period_end=submission.sample_date,
                data_origin=submission.data_origin,
                source=source,
                verification_status=VerificationStatus.VERIFIED,
            )
        )


def _raise_positive_alert(session: Session, submission: LabSubmission) -> None:
    facility = submission.facility
    session.add(
        Alert(
            alert_type="LAB_VERIFIED_POSITIVE",
            source_module="lab",
            severity=AlertSeverity.MEDIUM,
            location_id=submission.location_id,
            facility_id=submission.facility_id,
            title=f"Verified malaria-positive smear at {facility.name}",
            message=(
                f"A blood smear from {facility.name} (sample date "
                f"{submission.sample_date.isoformat()}) was reviewed as positive by "
                f"{submission.reviewed_by}. The AI model flagged "
                f"{submission.candidate_infected_count} of {submission.candidate_count} "
                "candidate cells as possibly parasitized; the reviewer's reading is the "
                "recorded result."
            ),
            payload={"lab_submission_id": submission.id},
            data_origin=submission.data_origin,
        )
    )


def submission_summary(submission: LabSubmission) -> dict[str, Any]:
    return {
        "id": submission.id,
        "created_at": submission.created_at.isoformat(),
        "status": submission.status.value,
        "facility": {"id": submission.facility_id, "name": submission.facility.name},
        "location": {"id": submission.location_id, "name": submission.location.name},
        "sample_date": submission.sample_date.isoformat(),
        "sample_reference": submission.sample_reference,
        "submitted_by": submission.submitted_by,
        "image_filename": submission.image_filename,
        "image_width": submission.image_width,
        "image_height": submission.image_height,
        "model_name": submission.model_name,
        "model_version": submission.model_version,
        "candidate_count": submission.candidate_count,
        "candidate_infected_count": submission.candidate_infected_count,
        "candidate_uninfected_count": submission.candidate_uninfected_count,
        "reviewer_result": submission.reviewer_result.value if submission.reviewer_result else None,
        "reviewed_by": submission.reviewed_by,
        "reviewed_at": submission.reviewed_at.isoformat() if submission.reviewed_at else None,
        "review_notes": submission.review_notes,
        "data_origin": submission.data_origin.value,
        "source": submission.source,
        "has_annotated_image": True,
    }


def submission_detail(submission: LabSubmission) -> dict[str, Any]:
    detail = submission_summary(submission)
    detail["cells"] = [
        {
            "cell_id": cell.cell_index,
            "box": {"x": cell.x, "y": cell.y, "width": cell.width, "height": cell.height},
            "predicted_class": cell.predicted_class,
            "prediction": cell.prediction,
            "confidence": cell.confidence,
        }
        for cell in submission.cells
    ]
    return detail
