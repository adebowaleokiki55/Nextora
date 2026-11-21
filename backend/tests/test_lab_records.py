from datetime import date

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.demo_seed import reset_and_seed, seed_if_empty
from backend.app.core.lab.records import (
    METRIC_SMEARS_POSITIVE,
    METRIC_SMEARS_REVIEWED,
    LabRecordError,
    create_submission,
    list_submissions,
    review_submission,
)
from backend.app.db import init_db, make_engine
from backend.app.models import (
    Alert,
    DataOrigin,
    Facility,
    LabSubmission,
    Location,
    Observation,
    ReviewerResult,
    VerificationStatus,
)

ANALYSIS = {
    "image_width": 800,
    "image_height": 600,
    "candidate_count": 2,
    "candidate_infected_count": 1,
    "candidate_uninfected_count": 1,
    "cells": [
        {"cell_id": 1, "box": {"x": 1, "y": 2, "width": 30, "height": 31},
         "predicted_class": 0, "prediction": "Parasitized (Infected)", "confidence": 91.2},
        {"cell_id": 2, "box": {"x": 50, "y": 60, "width": 29, "height": 30},
         "predicted_class": 1, "prediction": "Uninfected", "confidence": 88.0},
    ],
}


@pytest.fixture
def session():
    engine = make_engine("sqlite://")
    init_db(engine)
    with Session(engine) as s:
        seed_if_empty(s)
        yield s


def lab(session, lga="Ikorodu"):
    return session.scalar(
        select(Facility).join(Location).where(Location.name == lga, Facility.has_laboratory)
    )


def submit(session, **overrides):
    args = dict(
        facility_id=lab(session).id,
        sample_date=date(2026, 11, 20),
        submitted_by="Lab tech",
        analysis=ANALYSIS,
        annotated_png=b"png",
        model_name="PlasmoSENet",
        model_version="sha256:abc",
        data_origin=DataOrigin.SYNTHETIC,
        today=date(2026, 11, 21),
    )
    args.update(overrides)
    return create_submission(session, **args)


def observations(session):
    return list(session.scalars(select(Observation).order_by(Observation.metric)))


def test_seed_creates_lagos_scenario(session):
    assert session.query(Location).count() == 20
    ikorodu = session.scalar(select(Location).where(Location.name == "Ikorodu"))
    assert len([f for f in ikorodu.facilities if not f.has_laboratory]) == 5
    assert all(f.data_origin is DataOrigin.SYNTHETIC for f in session.scalars(select(Facility)))
    assert not seed_if_empty(session)  # idempotent


def test_new_submission_is_pending_and_feeds_nothing(session):
    sub = submit(session)
    assert sub.status is VerificationStatus.PENDING_REVIEW
    assert sub.location.name == "Ikorodu"  # location comes from the facility
    assert [c.cell_index for c in sub.cells] == [1, 2]
    assert "demo/test image" in sub.source
    assert observations(session) == []
    assert session.query(Alert).count() == 0


def test_verified_positive_creates_observations_and_alert(session):
    sub = submit(session)
    review_submission(session, sub.id, decision=VerificationStatus.VERIFIED,
                      reviewer_name="Dr Ade", reviewer_result=ReviewerResult.POSITIVE)

    obs = observations(session)
    assert [(o.metric, o.value) for o in obs] == [
        (METRIC_SMEARS_POSITIVE, 1.0), (METRIC_SMEARS_REVIEWED, 1.0)
    ]
    assert all(o.verification_status is VerificationStatus.VERIFIED for o in obs)
    assert all(o.data_origin is DataOrigin.SYNTHETIC for o in obs)
    assert all(o.period_start == date(2026, 11, 20) for o in obs)

    alert = session.scalar(select(Alert))
    assert alert.alert_type == "LAB_VERIFIED_POSITIVE"
    assert alert.payload == {"lab_submission_id": sub.id}
    assert alert.data_origin is DataOrigin.SYNTHETIC


def test_verified_negative_records_zero_positive_and_no_alert(session):
    sub = submit(session)
    review_submission(session, sub.id, decision=VerificationStatus.VERIFIED,
                      reviewer_name="Dr Ade", reviewer_result=ReviewerResult.NEGATIVE)
    assert [(o.metric, o.value) for o in observations(session)] == [
        (METRIC_SMEARS_POSITIVE, 0.0), (METRIC_SMEARS_REVIEWED, 1.0)
    ]
    assert session.query(Alert).count() == 0


def test_rejected_needs_reason_and_feeds_nothing(session):
    sub = submit(session)
    with pytest.raises(LabRecordError, match="why"):
        review_submission(session, sub.id, decision=VerificationStatus.REJECTED, reviewer_name="R")
    review_submission(session, sub.id, decision=VerificationStatus.REJECTED,
                      reviewer_name="R", notes="Out of focus")
    assert sub.status is VerificationStatus.REJECTED
    assert observations(session) == []


def test_verify_requires_reviewer_reading(session):
    sub = submit(session)
    with pytest.raises(LabRecordError) as err:
        review_submission(session, sub.id, decision=VerificationStatus.VERIFIED, reviewer_name="R")
    assert err.value.status_code == 422


def test_cannot_review_twice(session):
    sub = submit(session)
    review_submission(session, sub.id, decision=VerificationStatus.VERIFIED,
                      reviewer_name="R", reviewer_result=ReviewerResult.NEGATIVE)
    with pytest.raises(LabRecordError) as err:
        review_submission(session, sub.id, decision=VerificationStatus.REJECTED,
                          reviewer_name="R", notes="changed my mind")
    assert err.value.status_code == 409


@pytest.mark.parametrize(
    "overrides, status",
    [
        ({"facility_id": 99999}, 404),
        ({"sample_date": date(2026, 11, 22)}, 422),  # future
        ({"submitted_by": "   "}, 422),
        ({"sample_reference": "x" * 61}, 422),
    ],
)
def test_invalid_submissions_rejected(session, overrides, status):
    with pytest.raises(LabRecordError) as err:
        submit(session, **overrides)
    assert err.value.status_code == status


def test_non_laboratory_facility_rejected(session):
    clinic = session.scalar(select(Facility).where(Facility.has_laboratory.is_(False)))
    with pytest.raises(LabRecordError, match="not registered as a laboratory"):
        submit(session, facility_id=clinic.id)


def test_list_filters(session):
    a = submit(session)
    submit(session, facility_id=lab(session, "Epe").id)
    review_submission(session, a.id, decision=VerificationStatus.REJECTED, reviewer_name="R", notes="blurry")
    pending = list_submissions(session, status=VerificationStatus.PENDING_REVIEW)
    assert [s.location.name for s in pending] == ["Epe"]
    assert len(list_submissions(session, location_id=a.location_id)) == 1


def test_reset_clears_lab_data(session):
    submit(session)
    reset_and_seed(session)
    assert session.query(LabSubmission).count() == 0
    assert session.query(Location).count() == 20
