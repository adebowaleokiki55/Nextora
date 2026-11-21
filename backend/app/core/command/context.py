"""Read-only views over the other modules, scoped to one location.

These are what the agents' tools return. Every number comes from the
deterministic module code; agents only read and explain it.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.core.prediction.forecast import forecast_to_dict, latest_forecasts, summarise
from backend.app.core.supply.calculations import plan_redistribution, priority_key
from backend.app.core.supply.service import assess_all, assessment_to_dict, plan_to_dict
from backend.app.core.surveillance.metrics import CASES, HUMIDITY, LAB_POSITIVE, LAB_REVIEWED, RAINFALL, TEMP_MEAN
from backend.app.core.surveillance.repository import load_observations, origins_by_metric, wide_for_location
from backend.app.core.surveillance.signals import compute_signals
from backend.app.models import (
    Alert,
    AlertStatus,
    LabSubmission,
    Location,
    Proposal,
    ProposalStatus,
    VerificationStatus,
)


class ContextError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def get_location(session: Session, location_id: int) -> Location:
    location = session.get(Location, location_id)
    if location is None:
        raise ContextError(404, f"Location {location_id} was not found.")
    return location


def surveillance_signals(session: Session, location_id: int) -> dict[str, Any]:
    location = get_location(session, location_id)
    long = load_observations(
        session,
        location_ids=[location_id],
        metrics=[CASES, RAINFALL, TEMP_MEAN, HUMIDITY, LAB_POSITIVE, LAB_REVIEWED],
    )
    origins = origins_by_metric(long)
    result = compute_signals(wide_for_location(long, location_id), origins)
    return {
        "location": location.name,
        **result,
        "data_origin": {"cases": origins.get(CASES), "weather": origins.get(RAINFALL)},
    }


def risk_forecast(session: Session, location_id: int) -> dict[str, Any]:
    location = get_location(session, location_id)
    rows = latest_forecasts(session).get(location_id, [])
    if not rows:
        return {"location": location.name, "available": False, "note": "No forecast has been run for this location."}
    forecasts = [forecast_to_dict(r) for r in rows]
    summary = summarise(forecasts)
    return {
        "location": location.name,
        "available": True,
        "risk_score": summary["risk_score"],
        "risk_level": summary["risk_level"],
        "early_warning": summary["early_warning"],
        "first_warning_week": summary["first_warning_week"].isoformat() if summary["first_warning_week"] else None,
        "model_version": rows[0].model_version,
        "data_origin": rows[0].data_origin.value,
        "forecasts": [
            {
                "target_week": f["target_week"].isoformat(),
                "predicted_cases": f["predicted_cases"],
                "range_80pct": [f["lower_cases"], f["upper_cases"]],
                "alert_threshold": f["alert_threshold"],
                "probability_reaching_threshold": f["exceedance_probability"],
                "drivers_log_scale": f["drivers"],
            }
            for f in forecasts
        ],
        "risk_score_meaning": "Highest estimated probability (x100) that weekly cases reach the alert threshold in the next 4 weeks.",
    }


def lab_summary(session: Session, location_id: int, weeks: int = 4) -> dict[str, Any]:
    location = get_location(session, location_id)
    latest = session.scalar(select(func.max(LabSubmission.sample_date)))
    counts = dict(
        session.execute(
            select(LabSubmission.status, func.count())
            .where(LabSubmission.location_id == location_id)
            .group_by(LabSubmission.status)
        ).all()
    )
    since = (latest - timedelta(weeks=weeks)) if latest else None
    verified = session.scalars(
        select(LabSubmission).where(
            LabSubmission.location_id == location_id,
            LabSubmission.status == VerificationStatus.VERIFIED,
            *( [LabSubmission.sample_date > since] if since else [] ),
        )
    ).all()
    return {
        "location": location.name,
        "verified_last_weeks": weeks,
        "verified_smears": len(verified),
        "verified_positive_smears": sum(1 for s in verified if s.reviewer_result and s.reviewer_result.value == "POSITIVE"),
        "pending_review": counts.get(VerificationStatus.PENDING_REVIEW, 0),
        "rejected": counts.get(VerificationStatus.REJECTED, 0),
        "any_synthetic": any(s.data_origin.value == "SYNTHETIC" for s in verified),
        "note": "Only results verified by a qualified reviewer count as evidence. Pending results are not used.",
    }


def open_alerts(session: Session, location_id: int) -> list[dict[str, Any]]:
    get_location(session, location_id)
    alerts = session.scalars(
        select(Alert)
        .where(Alert.location_id == location_id, Alert.status != AlertStatus.RESOLVED)
        .order_by(Alert.created_at.desc(), Alert.id.desc())
    ).all()
    return [alert_to_dict(a) for a in alerts]


def alert_to_dict(alert: Alert) -> dict[str, Any]:
    return {
        "id": alert.id,
        "created_at": alert.created_at.isoformat(),
        "alert_type": alert.alert_type,
        "source_module": alert.source_module,
        "severity": alert.severity.value,
        "status": alert.status.value,
        "location_id": alert.location_id,
        "facility_id": alert.facility_id,
        "title": alert.title,
        "message": alert.message,
        "data_origin": alert.data_origin.value,
        "payload": alert.payload,
    }


def supply_shortages(session: Session, location_id: int) -> dict[str, Any]:
    location = get_location(session, location_id)
    shortages = sorted(
        (a for a in assess_all(session, location_id=location_id) if a.is_shortage), key=priority_key
    )
    return {"location": location.name, "shortages": [assessment_to_dict(a) for a in shortages], "data_origin": "SYNTHETIC"}


def redistribution_plan(session: Session, location_id: int) -> dict[str, Any]:
    """Plan for recipients in this location; donors may be anywhere within range."""
    location = get_location(session, location_id)
    assessments = [
        a for a in assess_all(session)
        if a.position.location_id == location_id or a.transferable_quantity > 0
    ]
    return {"location": location.name, **plan_to_dict(plan_redistribution(assessments))}


def pending_proposals(session: Session, location_id: int) -> list[dict[str, Any]]:
    from backend.app.core.command.proposals import proposal_to_dict

    rows = session.scalars(
        select(Proposal)
        .where(Proposal.location_id == location_id, Proposal.status == ProposalStatus.PENDING)
        .order_by(Proposal.id)
    ).all()
    return [proposal_to_dict(p) for p in rows]


def situation_snapshot(session: Session, location_id: int) -> dict[str, Any]:
    """Everything the command centre knows about one location, compactly."""
    signals = surveillance_signals(session, location_id)
    shortages = supply_shortages(session, location_id)
    return {
        "location": signals["location"],
        "surveillance": {
            "as_of_week": signals["as_of_week"],
            "level": signals["level"],
            "signals": [
                {"label": s["label"], "status": s["status"], "explanation": s["explanation"]}
                for s in signals["signals"]
            ],
            "data_origin": signals["data_origin"],
        },
        "risk_forecast": risk_forecast(session, location_id),
        "verified_lab": lab_summary(session, location_id),
        "supply_shortages": [
            {
                "facility": s["facility"]["name"],
                "item": s["item"]["code"],
                "status": s["status"],
                "on_hand": s["on_hand"],
                "weeks_of_stock": s["weeks_of_stock"],
                "shortage_quantity": s["shortage_quantity"],
            }
            for s in shortages["shortages"]
        ],
        "open_alerts": [
            {"id": a["id"], "type": a["alert_type"], "severity": a["severity"], "title": a["title"]}
            for a in open_alerts(session, location_id)
        ],
        "pending_proposals": [
            {"id": p["id"], "type": p["proposal_type"], "title": p["title"]}
            for p in pending_proposals(session, location_id)
        ],
    }
