"""Module 4: medical supply and facility intelligence API.

Read-only plans and scans. Moving stock happens only when a human approves
a transfer proposal in the command centre (Phase 5).
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.core.supply.calculations import DEFAULT_POLICY, STATUS_RANK, plan_redistribution, priority_key
from backend.app.core.supply.items import ITEMS
from backend.app.core.supply.service import (
    SupplyError,
    assess_all,
    assessment_to_dict,
    facility_detail,
    plan_to_dict,
    run_supply_scan,
)
from backend.app.db import get_db

router = APIRouter(prefix="/api/supply", tags=["supply"])


def _check_item(item_code: str | None) -> None:
    if item_code is not None and item_code not in ITEMS:
        raise HTTPException(status_code=422, detail=f"Unknown item '{item_code}'. Known: {', '.join(ITEMS)}.")


@router.get("/items")
def items():
    return {"items": [asdict(item) for item in ITEMS.values()]}


@router.get("/policy")
def policy():
    return {
        **asdict(DEFAULT_POLICY),
        "explanation": (
            "Weeks of stock = on hand / expected weekly demand. Expected demand = average weekly "
            "consumption over the last 8 weeks x the location's forecast ratio (next 4 weeks of "
            "forecast cases / last 8 weeks of cases), bounded 0.5-3.0, and never below 1.0 where "
            "the malaria risk score is 50 or more. STOCKOUT = none left; CRITICAL < 1 week; "
            "LOW < 2 weeks; SURPLUS > 10 weeks. Shortages are topped up to 6 weeks (rounded up to "
            "whole packs); donors keep at least 6 weeks of their own demand and only give whole "
            "packs within 40 km, nearest first."
        ),
    }


@router.get("/status")
def status(
    item_code: str | None = Query(None),
    location_id: int | None = Query(None),
    status: str | None = Query(None, description="STOCKOUT, CRITICAL, LOW, OK or SURPLUS"),
    db: Session = Depends(get_db),
):
    _check_item(item_code)
    if status is not None and status not in STATUS_RANK:
        raise HTTPException(status_code=422, detail=f"Unknown status '{status}'.")
    assessments = assess_all(db, item_code=item_code, location_id=location_id)
    if status is not None:
        assessments = [a for a in assessments if a.status == status]
    return {"stock": [assessment_to_dict(a) for a in sorted(assessments, key=priority_key)]}


@router.get("/shortages")
def shortages(db: Session = Depends(get_db)):
    """Facilities at risk of running out, most urgent first (higher-risk areas first)."""
    assessments = sorted((a for a in assess_all(db) if a.is_shortage), key=priority_key)
    return {"shortages": [assessment_to_dict(a) for a in assessments]}


@router.get("/redistribution")
def redistribution(
    item_code: str | None = Query(None),
    location_id: int | None = Query(None, description="Only plan for recipients in this location"),
    db: Session = Depends(get_db),
):
    """Recommended transfers from facilities with surplus, plus central orders.

    A recommendation only: nothing moves until a human approves it.
    """
    _check_item(item_code)
    assessments = assess_all(db, item_code=item_code)
    if location_id is not None:
        assessments = [
            a for a in assessments
            if a.position.location_id == location_id or a.transferable_quantity > 0
        ]
    plan = plan_redistribution(assessments)
    return {**plan_to_dict(plan), "requires_human_approval": True}


@router.post("/scan")
def scan(db: Session = Depends(get_db)):
    """Re-assess all stock and raise, update or resolve shortage alerts."""
    return run_supply_scan(db)


@router.get("/facilities/{facility_id}")
def facility(facility_id: int, db: Session = Depends(get_db)):
    try:
        return facility_detail(db, facility_id)
    except SupplyError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
